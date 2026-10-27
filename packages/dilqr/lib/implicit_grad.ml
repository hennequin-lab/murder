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

type ('p, 'a) residual =
  { theta : 'p (* problem parameters the solve was seeded with *)
  ; x0 : 'a (* the initial state the solve was seeded with *)
  ; us0 : 'a array (* the initial control guess *)
  ; xs : 'a array (* solution trajectory, length horizon + 1 *)
  ; us : 'a array (* solution controls, length horizon *)
  }

(* [taus_of_stacked p xs us] is [Problem.taus_of] at stacked tensors: what a
   forward-mode solve returns. *)
let taus_of_stacked p (xs : 'w Problem.value) (us : 'w Problem.value) : 'w Problem.value =
  let horizon = p.Problem.horizon in
  let last =
    Nx.concatenate
      ~axis:(-1)
      [ Nx.slice [ I horizon ] xs; Nx.zeros_like (Nx.slice [ I 0 ] us) ]
  in
  let body = Nx.concatenate ~axis:(-1) [ Nx.slice [ R (0, horizon) ] xs; us ] in
  Nx.concatenate ~axis:0 [ body; Nx.unsqueeze ~axes:[ 0 ] last ]

(* [outer a b] is [a bᵀ], batched over every leading axis. *)
let outer a b = Nx.matmul (Nx.unsqueeze ~axes:[ -1 ] a) (Nx.unsqueeze ~axes:[ -2 ] b)

(* [matvec m v] is [m @ v] for a batch of matrices and a batch of vectors. *)
let matvec m v = Nx.squeeze ~axes:[ -1 ] (Nx.matmul m (Nx.unsqueeze ~axes:[ -1 ] v))

(* [row t k] is the k-th step of a stacked tensor. *)
let row (t : ('a, 'b) Nx.t) k = Nx.slice [ I k ] t

(* [flip x] reverses the horizon axis. *)
let flip x = Nx.flip ~axes:[ 0 ] x

(* [stack_array ts] stacks the steps of [ts] along a new leading axis. *)
let stack_array (ts : ('a, 'b) Nx.t array) : ('a, 'b) Nx.t =
  Nx.stack ~axis:0 (Array.to_list ts)

(* The implicit VJP. [ybar] is the cotangent on the stacked trajectory; the
   result holds cotangents on (theta, x0, us0). *)
let bwd p (res : ('p, 'w Problem.value) residual) (ybar : 'w Problem.value)
  : ('p, 'w Problem.value) Problem.Inputs.t
  =
  let { theta; x0 = _; us0 = _; xs; us } = res in
  let horizon = p.Problem.horizon
  and n = p.Problem.n
  and m = p.Problem.m in
  let xs_t = stack_array xs
  and us_t = stack_array us in
  let tape = Solver.blocks_t p theta xs_t us_t in
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
          [ Nx.zeros p.Problem.dtype [| 1; n |]
          ; Nx.slice [ R (1, horizon); R (0, n) ] ybar
          ]
    ; rlu = Nx.slice [ R (0, horizon); R (n, n + m) ] ybar
    ; rfx = yx horizon
    }
  in
  let gains, _, _ = Lqr.backward tape ~lin in
  let dx, du = Lqr.rollout_linearized ~tape ~gains ~x0:zero_n in
  (* Perturbed costates: δλ_K = C_xx,K δx_K + ȳ_x,K;
     δλ_k = Aᵀδλ_{k+1} + C_xx δx_k + C_uxᵀ δu_k + ȳ_x,k. One scan over the
     horizon, emitting the incoming carry so the stack is δλ_K … δλ_1 and flips
     to δλ_1 … δλ_K; the final carry is δλ_0, the gradient on x₀. *)
  let dl0, dl1 =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(
        pair (pair tensor tensor) (pair tensor (pair tensor (pair tensor tensor))))
      Nx.Ptree.tensor
      ~f:(fun dl ((a, lxx), (lux, (dx_k, (du_k, yx_k)))) ->
        let dl_k =
          Nx.add
            (matvec (Nx.matrix_transpose a) dl)
            (Nx.add
               (matvec lxx dx_k)
               (Nx.add (matvec (Nx.matrix_transpose lux) du_k) yx_k))
        in
        dl_k, dl)
      ~init:(Nx.add (matvec tape.fxx (row dx horizon)) (yx horizon))
      ( (flip tape.Lqr.blocks.a, flip tape.Lqr.blocks.lxx)
      , ( flip tape.Lqr.blocks.lux
        , ( flip (Nx.slice [ R (0, horizon) ] dx)
          , (flip du, flip (Nx.slice [ R (0, horizon); R (0, n) ] ybar)) ) ) )
  in
  let dl1 = flip dl1 in
  (* Step 2: block cotangents, following the reference formulation (the VJP of
     the fixed-point map). λ* are the solution costates of the problem's own
     losses; δτ = (δx, δu) and δλ the perturbation solve's response. Every
     cotangent is a stacked tensor: one batched product per block, whatever the
     horizon. *)
  let lm1 = Lqr.costate tape in
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
  let grad_x0 = dl0 in
  (* Step 3: chain the block cotangents to θ with a nested grad over a
     scalar surrogate that recomputes the blocks from θ exactly like the
     forward pass, so derived quantities (a_k, c_k) propagate couplings. *)
  let ks = Nx.arange p.Problem.dtype 0 horizon 1 in
  let grad_theta =
    Rune.grad
      p.Problem.params
      (fun theta ->
         let tape = Solver.blocks_t p theta xs_t us_t in
         (* f_k, the dynamics at each nominal step. *)
         let f =
           snd
             (Rune.scan
                Nx.Ptree.unit
                Nx.Ptree.(pair (pair tensor tensor) tensor)
                Nx.Ptree.tensor
                ~f:(fun () ((x, u), k) -> (), p.Problem.dynamics theta ~k ~x ~u)
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
      theta
  in
  { Problem.Inputs.theta = grad_theta
  ; x0 = grad_x0
  ; us0 = Array.map ~f:Nx.zeros_like res.us0
  }

(* The adaptive solve, wrapped with its implicit gradient. Eager only — do
   not jit. *)
let solve_ilqr
      p
      ?max_iter
      ?conv_threshold
      ?c1
      ?ls_depth
      ?reg0
      (inputs : ('p, 'w Problem.value) Problem.Inputs.t)
  : 'w Problem.value
  =
  let reg = Option.value reg0 ~default:Regularisation.init in
  Rune.custom_vjp
    (Problem.inputs p)
    Nx.Ptree.tensor
    (fun { Problem.Inputs.theta; x0; us0 } ->
       let sol =
         Solver.solve p ?max_iter ?conv_threshold ?c1 ?ls_depth ~reg theta x0 us0
       in
       ( taus_of_stacked p (Solver.stack sol.xs) (Solver.stack sol.us)
       , fun ybar -> bwd p { theta; x0; us0; xs = sol.xs; us = sol.us } ybar ))
    inputs

(* The traceable solve, wrapped with its implicit gradient: put this inside
   [Rune.jit] (with a [Rune.grad] inside it) to compile the solve and its
   gradient together. *)
let solve_ilqr_fixed_unbatched
      p
      ~(iters : int)
      ?alphas
      ?mu
      ?c1
      (inputs : ('p, 'w Problem.value) Problem.Inputs.t)
  : 'w Problem.value
  =
  Rune.custom_vjp
    (Problem.inputs p)
    Nx.Ptree.tensor
    (fun { Problem.Inputs.theta; x0; us0 } ->
       let sol = Solver.solve_fixed_unbatched p ?alphas ?mu ?c1 ~iters theta x0 us0 in
       ( taus_of_stacked p (Solver.stack sol.xs) (Solver.stack sol.us)
       , fun ybar -> bwd p { theta; x0; us0; xs = sol.xs; us = sol.us } ybar ))
    inputs

(* Batched inputs (a leading batch axis on the parameters, the initial state
   and every initial control) are solved by vmapping the unbatched solve, its
   implicit VJP included, over the batch: each member gets its own trajectory
   and its own gradient. *)
let solve_ilqr_fixed p ~iters ?alphas ?mu ?c1 inputs =
  if not (Solver.is_batched inputs.Problem.Inputs.x0 inputs.us0)
  then solve_ilqr_fixed_unbatched p ~iters ?alphas ?mu ?c1 inputs
  else
    Rune.vmap
      Nx.Ptree.(Problem.inputs p @-> returns Nx.Ptree.tensor)
      (fun i -> solve_ilqr_fixed_unbatched p ~iters ?alphas ?mu ?c1 i)
      inputs

(* ──────────────────────────────────────────────────────────────────────────
   Forward mode
   ────────────────────────────────────────────────────────────────────────── *)

(* The tangent map of a solve: the local model at the solution, and one
   streaming tangent solve. The factors and the tape are computed here, in the
   map, so that a solve whose derivative nobody asks for does not pay for them —
   the same bargain the VJP's pullback makes. *)
let tangent_of_solve p ~theta ~(sol : 'w Problem.value Solver.t) ~dx0 ~dtheta
  : 'w Problem.value
  =
  let dsol = Solver.tangent_solve p theta ~dtheta ~dx0 sol.xs sol.us in
  taus_of_stacked p dsol.Lqr.xs dsol.Lqr.us

(* The adaptive solve with a custom forward-mode rule. Eager only: the primal
   solve reads values on the host. *)
let solve_ilqr_jvp
      p
      ?max_iter
      ?conv_threshold
      ?c1
      ?ls_depth
      ?reg0
      (inputs : ('p, 'w Problem.value) Problem.Inputs.t)
  : 'w Problem.value
  =
  let reg = Option.value reg0 ~default:Regularisation.init in
  Rune.custom_jvp
    (Problem.inputs p)
    Nx.Ptree.tensor
    (fun { Problem.Inputs.theta; x0; us0 } ->
       let sol =
         Solver.solve p ?max_iter ?conv_threshold ?c1 ?ls_depth ~reg theta x0 us0
       in
       ( taus_of_stacked p (Solver.stack sol.xs) (Solver.stack sol.us)
       , fun { Problem.Inputs.theta = dtheta; x0 = dx0; us0 = _ } ->
           tangent_of_solve p ~theta ~sol ~dx0 ~dtheta ))
    inputs

(* The traceable solve with a custom forward-mode rule. *)
let solve_ilqr_fixed_jvp_unbatched
      p
      ~(iters : int)
      ?alphas
      ?mu
      ?c1
      (inputs : ('p, 'w Problem.value) Problem.Inputs.t)
  : 'w Problem.value
  =
  Rune.custom_jvp
    (Problem.inputs p)
    Nx.Ptree.tensor
    (fun { Problem.Inputs.theta; x0; us0 } ->
       let sol = Solver.solve_fixed_unbatched p ?alphas ?mu ?c1 ~iters theta x0 us0 in
       ( taus_of_stacked p (Solver.stack sol.xs) (Solver.stack sol.us)
       , fun { Problem.Inputs.theta = dtheta; x0 = dx0; us0 = _ } ->
           tangent_of_solve p ~theta ~sol ~dx0 ~dtheta ))
    inputs

(* Batched inputs are solved member by member, the primal and the tangents
   alike: each member gets its own trajectory and its own derivative. *)
let solve_ilqr_fixed_jvp p ~iters ?alphas ?mu ?c1 inputs =
  if not (Solver.is_batched inputs.Problem.Inputs.x0 inputs.us0)
  then solve_ilqr_fixed_jvp_unbatched p ~iters ?alphas ?mu ?c1 inputs
  else
    Rune.vmap
      Nx.Ptree.(Problem.inputs p @-> returns Nx.Ptree.tensor)
      (fun i -> solve_ilqr_fixed_jvp_unbatched p ~iters ?alphas ?mu ?c1 i)
      inputs
