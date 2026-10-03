(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The implicit gradient of the iLQR solve.

   The solve is idealized as the argmin of the local QP built at the final
   trajectory: minimize Σ ½τᵀCτ + cᵀτ subject to x_{k+1} = A x_k + B u_k + a_k
   (τ = (x,u), C the loss-Hessian blocks, c = l_τ(τ̄) − Cτ̄, a the dynamics
   constant terms). Its KKT system is solved by the Riccati machinery, and —
   the KKT matrix being symmetric — the same machinery solves the adjoint
   system needed for the implicit-function-theorem VJP:

   1. given the cotangent ȳ on the trajectory, run a perturbed Riccati pass
      whose linear terms are ȳ, then a homogeneous linearized rollout from
      δx₀ = 0; this yields the solution perturbation (δx, δu) =: δτ and, via
      the costate recursion, δλ;
   2. assemble block cotangents on (A, B, a, C, c) — e.g.
      ∂/∂a_k = −δλ_{k+1}, ∂/∂c_k = −δτ_k, ∂/∂C_k = −sym(δτ_k τ̄_kᵀ),
      ∂/∂A_k = −(δλ_{k+1} x̄_kᵀ + δx_k δλ_{k+1}ᵀ), grad x₀ = ȳ_x,0 − A₀ᵀδλ₁;
   3. chain them to the user parameters θ with a nested [Rune.grad] over a
      scalar surrogate Σ ⟨cotangent, block(θ)⟩, recomputing the blocks from θ
      exactly as the forward pass does (so derived quantities — a_k, c_k —
      propagate their couplings).

   The cotangents are stacked along the horizon and the surrogate sums over
   them with batched products, so one traced step stands for the whole
   horizon. The gradient of the initial control guess [us0] is zero: the
   idealized solve is a fixed point and does not depend on it. *)

open Base

module Make (C : Lqr.Ctx) = struct
  module Solver = Solver.Make (C)

  type value = C.value

  (* The problem parameters at [value]: what [Rune.grad] takes. *)
  let params = Nx.Ptree.instantiate (module C.Params)

  (* The solve inputs: problem parameters, initial state, initial control
     guess. A structure of tensors, so that a differentiation with respect to
     the solve inputs sees [theta], [x0] and every control of [us0] as its
     own leaf with its own name. *)
  module Inputs = struct
    type 'a t =
      { theta : 'a C.Params.t
      ; x0 : 'a
      ; us0 : 'a array
      }

    let walk c { theta; x0; us0 } =
      let open Nx.Ptree.Walk in
      let theta = field c "theta" C.Params.walk theta in
      let x0 = field c "x0" leaf x0 in
      let us0 =
        field
          c
          "us0"
          (fun c us0 ->
             ignore (int c (Array.length us0));
             Array.mapi us0 ~f:(fun i u -> index c i leaf u))
          us0
      in
      { theta; x0; us0 }
  end

  let inputs_ptree : value Inputs.t Nx.Ptree.t = Nx.Ptree.instantiate (module Inputs)

  type residual =
    { theta : value C.Params.t (* problem parameters the solve was seeded with *)
    ; x0 : value (* the initial state the solve was seeded with *)
    ; us0 : value array (* the initial control guess *)
    ; xs : value array (* solution trajectory, length horizon + 1 *)
    ; us : value array (* solution controls, length horizon *)
    }

  let horizon = C.horizon
  let n = C.n
  let m = C.m

  (* [taus_of xs us] stacks the trajectory into a [(K+1); n+m] tensor: row k is
     (x_k; u_k) for k < K, row K is (x_K; 0). This is the output of the
     differentiable solve; a loss on the solve takes its cotangent from here. *)
  let taus_of (xs : value array) (us : value array) : value =
    let zero_u = Nx.zeros_like us.(0) in
    let rows =
      Array.init (horizon + 1) ~f:(fun k ->
        if k < horizon
        then Nx.concatenate ~axis:0 [ xs.(k); us.(k) ]
        else Nx.concatenate ~axis:0 [ xs.(k); zero_u ])
    in
    Nx.stack ~axis:0 (Array.to_list rows)

  (* [split_taus taus] is ([xs], [us]) with [xs] of length [horizon + 1] and
     [us] of length [horizon]. *)
  let split_taus (taus : value) : value array * value array =
    let row k = Nx.slice [ I k ] taus in
    let xs = Array.create ~len:(horizon + 1) taus in
    let us = Array.create ~len:horizon taus in
    for k = 0 to horizon - 1 do
      xs.(k) <- Nx.slice [ R (0, n) ] (row k);
      us.(k) <- Nx.slice [ R (n, n + m) ] (row k)
    done;
    xs.(horizon) <- Nx.slice [ R (0, n) ] (row horizon);
    xs, us

  (* A differentiable loss evaluated on the solve output: the problem's own
     running + final losses along the trajectory. Useful for self-derivation
     checks and as a building block for user losses. *)
  let loss_of_taus (theta : value C.Params.t) (taus : value) : value =
    let xs, us = split_taus taus in
    let acc = ref (C.final_loss theta ~x:xs.(horizon)) in
    for k = 0 to horizon - 1 do
      acc
      := Nx.add
           !acc
           (C.running_loss
              theta
              ~k:(Nx.scalar C.dtype (Float.of_int k))
              ~x:xs.(k)
              ~u:us.(k))
    done;
    !acc

  (* [outer a b] is [a bᵀ], batched over every leading axis. *)
  let outer a b = Nx.matmul (Nx.unsqueeze ~axes:[ -1 ] a) (Nx.unsqueeze ~axes:[ -2 ] b)

  (* [matvec m v] is [m @ v] for a batch of matrices and a batch of vectors. *)
  let matvec m v = Nx.squeeze ~axes:[ -1 ] (Nx.matmul m (Nx.unsqueeze ~axes:[ -1 ] v))

  (* [row t k] is the k-th step of a stacked tensor. *)
  let row (t : value) k = Nx.slice [ I k ] t

  (* [stack_array ts] stacks the steps of [ts] along a new leading axis. *)
  let stack_array (ts : value array) : value = Nx.stack ~axis:0 (Array.to_list ts)

  (* The implicit VJP. [ybar] is the cotangent on the stacked trajectory; the
     result holds cotangents on (theta, x0, us0). *)
  let bwd (res : residual) (ybar : value) : value Inputs.t =
    let { theta; x0 = _; us0 = _; xs; us } = res in
    let xs_t = stack_array xs
    and us_t = stack_array us in
    let tape = Solver.blocks_t theta xs_t us_t in
    let zero_n = Nx.zeros_like xs.(0) in
    (* Cotangent slices: ȳ_x,k. *)
    let yx k = Nx.slice [ I k; R (0, n) ] ybar in
    (* Step 1: solve the adjoint KKT system M z = (ȳ; 0) with the Riccati
       machinery. Linear cost terms = ȳ (except ȳ_x,0: x₀ is not a free
       variable), homogeneous linearized rollout from δx₀ = 0. *)
    let lin =
      { Lqr.rlx =
          Nx.concatenate
            ~axis:0
            [ Nx.zeros C.dtype [| 1; n |]; Nx.slice [ R (1, horizon); R (0, n) ] ybar ]
      ; rlu = Nx.slice [ R (0, horizon); R (n, n + m) ] ybar
      ; rfx = yx horizon
      }
    in
    let gains, _, _ = Lqr.backward ~mu:0.0 ~tape ~lin in
    let dx, du = Lqr.rollout_linearized ~tape ~gains ~x0:zero_n in
    (* Perturbed costates: δλ_K = C_xx,K δx_K + ȳ_x,K;
       δλ_k = Aᵀδλ_{k+1} + C_xx δx_k + C_uxᵀ δu_k + ȳ_x,k. Only λ₁..λ_K are
       needed. *)
    let dl = Array.create ~len:(horizon + 1) zero_n in
    dl.(horizon) <- Nx.add (Nx.matmul tape.fxx (row dx horizon)) (yx horizon);
    for k = horizon - 1 downto 0 do
      dl.(k)
      <- Nx.add
           (Nx.matmul (Nx.matrix_transpose (row tape.Lqr.blocks.a k)) dl.(k + 1))
           (Nx.add
              (Nx.matmul (row tape.Lqr.blocks.lxx k) (row dx k))
              (Nx.add
                 (Nx.matmul (Nx.matrix_transpose (row tape.Lqr.blocks.lux k)) (row du k))
                 (yx k)))
    done;
    (* Step 2: block cotangents, following the reference formulation (the
       VJP of the fixed-point map). λ* are the solution costates of the
       problem's own losses; δτ = (δx, δu) and δλ the perturbation solve's
       response. Every cotangent is a stacked tensor: one batched product per
       block, whatever the horizon. *)
    let lam = Array.create ~len:(horizon + 1) zero_n in
    lam.(horizon) <- tape.fx;
    for k = horizon - 1 downto 0 do
      lam.(k)
      <- Nx.add
           (Nx.matmul (Nx.matrix_transpose (row tape.Lqr.blocks.a k)) lam.(k + 1))
           (row tape.Lqr.blocks.lx k)
    done;
    let dl1 = stack_array (Array.init horizon ~f:(fun k -> dl.(k + 1))) in
    let lm1 = stack_array (Array.init horizon ~f:(fun k -> lam.(k + 1))) in
    let x = Nx.slice [ R (0, horizon) ] xs_t in
    let dxk = Nx.slice [ R (0, horizon) ] dx in
    let dx_last = row dx horizon in
    let cot_A = Nx.add (outer dl1 x) (outer lm1 dxk) in
    let cot_B = Nx.add (outer dl1 us_t) (outer lm1 du) in
    let cot_cx = dxk
    and cot_cu = du in
    let cot_lxx = Lqr.sym (outer dxk x) in
    let cot_luu = Lqr.sym (outer du us_t) in
    let cot_lux = Nx.add (outer du x) (Nx.matrix_transpose (outer dxk us_t)) in
    let cot_fx = dx_last in
    let cot_fxx = Lqr.sym (outer dx_last xs.(horizon)) in
    let grad_x0 = dl.(0) in
    (* Step 3: chain the block cotangents to θ with a nested grad over a
       scalar surrogate that recomputes the blocks from θ exactly like the
       forward pass, so derived quantities (a_k, c_k) propagate couplings. *)
    let ks = Nx.arange C.dtype 0 horizon 1 in
    let grad_theta =
      Rune.grad
        params
        (fun theta ->
           let tape = Solver.blocks_t theta xs_t us_t in
           (* f_k, the dynamics at each nominal step. *)
           let f =
             snd
               (Rune.scan
                  Nx.Ptree.unit
                  Nx.Ptree.(pair (pair tensor tensor) tensor)
                  Nx.Ptree.tensor
                  ~f:(fun () ((x, u), k) -> (), C.dynamics theta ~k ~x ~u)
                  ~init:()
                  ((x, us_t), ks))
           in
           let acc = ref (Nx.scalar_like xs.(0) 0.0) in
           let term t cot = acc := Nx.add !acc (Nx.sum (Nx.mul t cot)) in
           term tape.Lqr.blocks.a cot_A;
           term tape.Lqr.blocks.b cot_B;
           term f dl1;
           (* a_k = f_k − A x̄ − B ū *)
           term
             (Nx.add (matvec tape.Lqr.blocks.a x) (matvec tape.Lqr.blocks.b us_t))
             (Nx.neg dl1);
           term tape.Lqr.blocks.lxx cot_lxx;
           term tape.Lqr.blocks.luu cot_luu;
           term tape.Lqr.blocks.lux cot_lux;
           (* c_k = l_τ(τ̄) − C τ̄ *)
           term tape.Lqr.blocks.lx cot_cx;
           term tape.Lqr.blocks.lu cot_cu;
           term
             (Nx.add
                (matvec tape.Lqr.blocks.lxx x)
                (matvec (Nx.matrix_transpose tape.Lqr.blocks.lux) us_t))
             (Nx.neg cot_cx);
           term
             (Nx.add (matvec tape.Lqr.blocks.luu us_t) (matvec tape.Lqr.blocks.lux x))
             (Nx.neg cot_cu);
           (* Terminal: c_K = fx − fxx x̄_K, C_K,xx = fxx. *)
           term tape.fx cot_fx;
           term tape.fxx cot_fxx;
           term (Nx.matmul tape.fxx xs.(horizon)) (Nx.neg cot_fx);
           !acc)
        res.theta
    in
    { theta = grad_theta; x0 = grad_x0; us0 = Array.map ~f:Nx.zeros_like res.us0 }
end
