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
   2. assemble block cotangents on the first-order data. The quadratic blocks'
      cotangents — ∂/∂C_k = −sym(δτ_k τ̄_kᵀ), and the matching ∂/∂A_k term
      δλ_{k+1} x̄_kᵀ — cancel identically against the model's own constants
      c = l_τ(τ̄) − Cτ̄ and a = f − Ax̄ − Bū, whose derivatives carry the same
      cotangents with the opposite sign. What survives is the loss gradients
      (∂/∂l_τ,k = −δτ_k, ∂/∂φ_x = −δτ_K) and the dynamics (∂/∂f_k = −δλ_{k+1},
      ∂/∂A_k = −δx_k δλ_{k+1}ᵀ, and the same for B_k with δu_k), plus
      grad x₀ = ȳ_x,0 − A₀ᵀδλ₁.
   3. chain them to the user parameters θ with a nested [Rune.grad] over a
      scalar surrogate Σ ⟨cotangent, first-order block(θ)⟩, recomputing the
      blocks from θ exactly as the forward pass does.

   Because the Hessians drop out of step 2, step 3 never differentiates one:
   the surrogate traces dynamics, dynamics Jacobians and loss gradients, and
   no third-order derivative is taken. The result is exactly the derivative of
   the idealized argmin — an argmin's derivative goes through its KKT matrix,
   which is the primal's.

   The cotangents are stacked along the horizon and the surrogate sums over
   them with batched products, so one traced step stands for the whole
   horizon. The gradient of the initial control guess [us0] is zero: the
   idealized solve is a fixed point and does not depend on it.

   Forward mode runs the transpose of that same linear map with the streaming
   tangent solve. The two rules are attached together: the solve is one
   [Rune.custom_jvp] whose tangent map is a [Rune.custom_transpose] carrying
   [bwd] as its transpose, so [Rune.grad] and [Rune.jvp] both apply to the
   unchanged solve function. *)

open Base
open Nx

type ('p, 'a) residual =
  { theta : 'p (* problem parameters the solve was seeded with *)
  ; x0 : 'a (* the initial state the solve was seeded with *)
  ; us0 : 'a array (* the initial control guess *)
  ; xs : 'a array (* solution trajectory, length horizon + 1 *)
  ; us : 'a array (* solution controls, length horizon *)
  }

(* [taus_of_stacked p xs us] is [Problem.taus_of] at stacked tensors: the
   stacked trajectory a differentiable solve returns. *)
let taus_of_stacked p (xs : ('w, 'd) Problem.value) (us : ('w, 'd) Problem.value)
  : ('w, 'd) Problem.value
  =
  let horizon = p.Problem.horizon in
  let last =
    concatenate ~axis:(-1) [ slice [ I horizon ] xs; zeros_like (slice [ I 0 ] us) ]
  in
  let body = concatenate ~axis:(-1) [ slice [ R (0, horizon) ] xs; us ] in
  concatenate ~axis:0 [ body; unsqueeze ~axes:[ 0 ] last ]

(* [outer a b] is [a bᵀ], batched over every leading axis. *)
let outer a b = Infix.(unsqueeze ~axes:[ -1 ] a *@ unsqueeze ~axes:[ -2 ] b)

(* [matvec m v] is [m @ v] for a batch of matrices and a batch of vectors. *)
let matvec m v = squeeze ~axes:[ -1 ] Infix.(m *@ unsqueeze ~axes:[ -1 ] v)

(* [row t k] is the k-th step of a stacked tensor. *)
let row (t : ('a, 'b, 'd) Nx.t) k = slice [ I k ] t

(* [flip x] reverses the horizon axis. *)
let flip x = Nx.flip ~axes:[ 0 ] x

(* The per-step first-order blocks differentiated w.r.t. the parameters in
   Step 3 of {!bwd}: dynamics value, its Jacobians, and the loss gradients.
   Hessian blocks do not enter the solution derivative at a fixed point, so
   computing them inside [Rune.grad] would only trace 3rd-order derivatives
   that cancel out. *)
module First_order_rows = struct
  type ('a, _) t =
    { a : 'a
    ; b : 'a
    ; f : 'a
    ; lx : 'a
    ; lu : 'a
    }
  [@@deriving ptree]
end

let first_order_rows () = Nx.Ptree.instantiate (module First_order_rows)

(* [stack_array ts] stacks the steps of [ts] along a new leading axis. *)
let stack_array (ts : ('a, 'b, 'd) Nx.t array) : ('a, 'b, 'd) Nx.t =
  stack ~axis:0 (Array.to_list ts)

(* The implicit VJP. [ybar] is the cotangent on the stacked trajectory; the
   result holds cotangents on (theta, x0, us0). *)
let bwd p (res : ('p, ('w, 'd) Problem.value) residual) (ybar : ('w, 'd) Problem.value)
  : ('p, ('w, 'd) Problem.value) Problem.Inputs.t
  =
  let { theta; x0 = _; us0 = _; xs; us } = res in
  let horizon = p.Problem.horizon
  and n = p.Problem.n
  and m = p.Problem.m in
  let xs_t = stack_array xs
  and us_t = stack_array us in
  let tape = Solver.blocks_t p theta xs_t us_t in
  let zero_n = zeros_like xs.(0) in
  (* Cotangent slices: ȳ_x,k. *)
  let yx k = slice [ I k; R (0, n) ] ybar in
  (* Step 1: solve the adjoint KKT system M z = (ȳ; 0) with the Riccati
     machinery. Linear cost terms = ȳ (except ȳ_x,0: x₀ is not a free
     variable), homogeneous linearized rollout from δx₀ = 0. *)
  let lin =
    { Lqr.rlx =
        concatenate
          ~axis:0
          [ zeros_at (beside ybar) p.Problem.dtype [| 1; n |]
          ; slice [ R (1, horizon); R (0, n) ] ybar
          ]
    ; rlu = slice [ R (0, horizon); R (n, n + m) ] ybar
    ; rfx = yx horizon
    }
  in
  let common = Lqr.backward_common tape in
  let gains, _, _, _ = Lqr.backward_lin ~common ~tape ~lin in
  let dx, du = Lqr.rollout_linearized ~tape ~gains ~x0:zero_n in
  (* Perturbed costates: δλ_K = C_xx,K δx_K + ȳ_x,K;
     δλ_k = Aᵀδλ_{k+1} + C_xx δx_k + C_uxᵀ δu_k + ȳ_x,k. One scan over the
     horizon, emitting the incoming carry so the stack is δλ_K … δλ_1 and flips
     to δλ_1 … δλ_K; the final carry is δλ_0, the gradient on x₀. *)
  let costates ~dx ~du =
    let dl0, dl1 =
      Rune.scan
        Nx.Ptree.tensor
        Nx.Ptree.(
          pair (pair tensor tensor) (pair tensor (pair tensor (pair tensor tensor))))
        Nx.Ptree.tensor
        (fun dl ((a, lxx), (lux, (dx_k, (du_k, yx_k)))) ->
           let open Infix in
           let dl_k =
             matvec (matrix_transpose a) dl
             + (matvec lxx dx_k + (matvec (matrix_transpose lux) du_k + yx_k))
           in
           dl_k, dl)
        ~init:Infix.(matvec tape.fxx (row dx horizon) + yx horizon)
        ( (flip tape.Lqr.blocks.a, flip tape.Lqr.blocks.lxx)
        , ( flip tape.Lqr.blocks.lux
          , ( flip (slice [ R (0, horizon) ] dx)
            , (flip du, flip (slice [ R (0, horizon); R (0, n) ] ybar)) ) ) )
    in
    dl0, flip dl1
  in
  let dl0, dl1 = costates ~dx ~du in
  (* A free initial state: the perturbation must also satisfy its stationarity,
     which the pinned system leaves as [dl0]. The correction is the model's step
     in [x₀] and the closed loop's response, and the costates are recomputed on
     the corrected perturbation — the recursion is linear, so this is the pinned
     costates plus the homogeneous response's. *)
  let dx, du, dl0, dl1 =
    if not p.Problem.free_initial
    then dx, du, dl0, dl1
    else (
      let dx1, du1 = Lqr.initial_response ~common ~tape ~residual:dl0 in
      let dx = Infix.(dx + dx1)
      and du = Infix.(du + du1) in
      let dl0, dl1 = costates ~dx ~du in
      dx, du, dl0, dl1)
  in
  (* Step 2: block cotangents for the first-order terms. λ* (`lm1`) are the
     solution costates of the problem's own losses; δτ = (dx, du) and δλ (`dl1`)
     the perturbation solve's response. The quadratic-block cotangents cancel
     algebraically against `-C τ̄` in `c_k = l_τ - C τ̄` and `A_k x̄_k + B_k ū_k`
     in `a_k = f_k - A_k x̄_k - B_k ū_k`, leaving only first-order sensitivities. *)
  let lm1 = Lqr.costate tape in
  let x = slice [ R (0, horizon) ] xs_t in
  let dxk = slice [ R (0, horizon) ] dx in
  let dx_last = row dx horizon in
  let cot_A = outer lm1 dxk in
  let cot_B = outer lm1 du in
  let grad_x0 = dl0 in
  (* Step 3: chain the net block cotangents to θ with a nested grad over one
     scan of first-order blocks (no 2nd-order loss Hessians inside AD). *)
  let rows_ptree = first_order_rows () in
  let ks = arange_at (beside us_t) p.Problem.dtype 0 horizon 1 in
  let grad_theta =
    Rune.grad
      p.Problem.params
      (fun theta ->
         let _, rows =
           Rune.scan
             Nx.Ptree.unit
             Nx.Ptree.(pair (pair tensor tensor) tensor)
             rows_ptree
             (fun () ((x_k, u_k), k) ->
                let f = p.Problem.dynamics theta ~k ~x:x_k ~u:u_k in
                let a =
                  Rune.jacfwd' (fun x -> p.Problem.dynamics theta ~k ~x ~u:u_k) x_k
                in
                let b =
                  Rune.jacfwd' (fun u -> p.Problem.dynamics theta ~k ~x:x_k ~u) u_k
                in
                let lx =
                  Rune.grad' (fun x -> p.Problem.running_loss theta ~k ~x ~u:u_k) x_k
                in
                let lu =
                  Rune.grad' (fun u -> p.Problem.running_loss theta ~k ~x:x_k ~u) u_k
                in
                (), { First_order_rows.a; b; f; lx; lu })
             ~init:()
             ((x, us_t), ks)
         in
         let fx = Rune.grad' (fun x -> p.Problem.final_loss theta ~x) xs.(horizon) in
         let open Infix in
         sum (rows.First_order_rows.a * cot_A)
         + sum (rows.b * cot_B)
         + sum (rows.f * dl1)
         + sum (rows.lx * dxk)
         + sum (rows.lu * du)
         + sum (fx * dx_last))
      theta
  in
  { Problem.Inputs.theta = grad_theta
  ; x0 = (if p.Problem.free_initial then zeros_like grad_x0 else grad_x0)
  ; us0 = Array.map ~f:zeros_like res.us0
  }

(* [adaptive_solver p ?beam ?parallel] is the adaptive solve. Supplying a
   [Rune.jit] search option asks for the compiled kernels; without one the
   solve runs eagerly, as it did before the differentiable entry point grew
   its JIT options. The host keeps the control flow either way. *)
let adaptive_solver p ?beam ?parallel =
  match beam, parallel with
  | None, None -> Solver.solve p
  | _ -> Solver.compile ?beam ?parallel p

(* The tangent map of a solve: the local model at the solution, and one
   streaming tangent solve. The factors and the tape are computed here, in the
   map, so that a solve whose derivative nobody asks for does not pay for them —
   the same bargain the VJP's pullback makes. *)
let tangent_of_solve p ~theta ~(sol : ('w, 'd) Problem.value Solver.t) ~dx0 ~dtheta
  : ('w, 'd) Problem.value
  =
  let dsol = Solver.tangent_solve p theta ~dtheta ~dx0 sol.xs sol.us in
  taus_of_stacked p dsol.Lqr.xs dsol.Lqr.us

(* The implicit tangent map of a solve. It is linear in the tangent of the
   inputs, and its transpose is the implicit VJP: marking it with
   [Rune.custom_transpose] lets one [Rune.custom_jvp] carry both rules. Forward
   mode runs the streaming tangent solve; reverse mode applies [bwd] instead of
   transposing the sweeps operation by operation. *)
let tangent_rule
      p
      ~theta
      ~x0
      ~us0
      ~(sol : ('w, 'd) Problem.value Solver.t)
      (dinputs : ('p, ('w, 'd) Problem.value) Problem.Inputs.t)
  : ('w, 'd) Problem.value
  =
  Rune.custom_transpose
    (Problem.inputs p)
    Nx.Ptree.tensor
    (fun dinputs ->
       tangent_of_solve
         p
         ~theta
         ~sol
         ~dx0:dinputs.Problem.Inputs.x0
         ~dtheta:dinputs.Problem.Inputs.theta)
    ~transpose:(fun ybar -> bwd p { theta; x0; us0; xs = sol.xs; us = sol.us } ybar)
    dinputs

(* The adaptive solve with its implicit derivative, in both modes. The
   returned closure owns the eager or compiled kernels once per problem, so it
   can serve many solves and many regularization or line-search settings
   without retracing. The primal still reads scalars on the host for its
   convergence and line-search control flow, so the closure is eager. *)
let solve_adaptive (p : ('w, 'p, 'd) Problem.t) ?beam ?parallel =
  let solve = adaptive_solver p ?beam ?parallel in
  fun ?max_iter
    ?conv_threshold
    ?c1
    ?ls_depth
    ?damping
    (inputs : ('p, ('w, 'd) Problem.value) Problem.Inputs.t)
    : ('w, 'd) Problem.value ->
    Rune.custom_jvp
      (Problem.inputs p)
      Nx.Ptree.tensor
      (fun { Problem.Inputs.theta; x0; us0 } ->
         let sol = solve ?max_iter ?conv_threshold ?c1 ?ls_depth ?damping theta x0 us0 in
         ( taus_of_stacked p (Solver.stack sol.xs) (Solver.stack sol.us)
         , tangent_rule p ~theta ~x0 ~us0 ~sol ))
      inputs

(* The traceable fixed-iteration solve with its implicit derivative, in both
   modes. Put this inside [Rune.jit] (with a [Rune.grad] or a [Rune.jvp]
   inside it) to compile the solve and its derivative together. *)
let solve_fixed_unbatched
      p
      ~(max_iters : int)
      ?alphas
      ?damping
      ?c1
      ?conv_threshold
      (inputs : ('p, ('w, 'd) Problem.value) Problem.Inputs.t)
  : ('w, 'd) Problem.value
  =
  Rune.custom_jvp
    (Problem.inputs p)
    Nx.Ptree.tensor
    (fun { Problem.Inputs.theta; x0; us0 } ->
       let sol =
         Solver.solve_fixed p ?alphas ?damping ?c1 ?conv_threshold ~max_iters theta x0 us0
       in
       ( taus_of_stacked p (Solver.stack sol.xs) (Solver.stack sol.us)
       , tangent_rule p ~theta ~x0 ~us0 ~sol ))
    inputs

(* A batch's solve is one solve over the batch, so its derivative is taken
   member by member: the tangent map maps each member's tangent solve over the
   batch, and the transpose each member's implicit VJP. The batch axis is a map
   axis like any other, and the primal — with the shared step and damping the
   batch's members took — is outside both.

   The derivative is still the implicit one at the trajectory the solve
   returned, per member: the step and the damping are devices that find the
   trajectory, not part of the map being differentiated. *)
let batched_tangent p ~batch_rank ~(theta : 'p) ~(sol : ('w, 'd) Problem.value Solver.t) =
  let taus = Problem.taus_of p sol.xs sol.us in
  let signature =
    Nx.Ptree.(
      pair p.Problem.params (pair p.Problem.params (pair tensor tensor))
      @-> returns tensor)
  in
  fun (dinputs : ('p, ('w, 'd) Problem.value) Problem.Inputs.t) ->
    Solver.vmap_fn
      batch_rank
      signature
      (fun (theta_i, (dtheta_i, (dx0_i, taus_i))) ->
         let xs_i, us_i = Problem.split_taus p taus_i in
         let dsol =
           Solver.tangent_solve p theta_i ~dtheta:dtheta_i ~dx0:dx0_i xs_i us_i
         in
         taus_of_stacked p dsol.Lqr.xs dsol.Lqr.us)
      (theta, (dinputs.theta, (dinputs.x0, taus)))

let batched_transpose
      p
      ~batch_rank
      ~(sol : ('w, 'd) Problem.value Solver.t)
      (inputs : ('p, ('w, 'd) Problem.value) Problem.Inputs.t)
  =
  let taus = Problem.taus_of p sol.xs sol.us in
  let signature =
    Nx.Ptree.(pair (Problem.inputs p) (pair tensor tensor) @-> returns (Problem.inputs p))
  in
  fun (ybar : ('w, 'd) Problem.value) ->
    Solver.vmap_fn
      batch_rank
      signature
      (fun ((i : ('p, ('w, 'd) Problem.value) Problem.Inputs.t), (taus_i, ybar_i)) ->
         let xs_i, us_i = Problem.split_taus p taus_i in
         bwd p { theta = i.theta; x0 = i.x0; us0 = i.us0; xs = xs_i; us = us_i } ybar_i)
      (inputs, (taus, ybar))

(* Batched inputs — leading batch axes on the parameters, the initial state
   and every initial control — are solved as one batch, so the batch's step and
   damping schedule are the solve's, and the batch stops when its slowest
   member has converged. The derivative is taken member by member, which is
   what the implicit function theorem says: each member's solution does not
   depend on the others. *)
let solve_fixed p ?alphas ?damping ?c1 ?conv_threshold ~max_iters inputs =
  let batch_rank =
    Array.length (Solver.check_batch_shape inputs.Problem.Inputs.x0 inputs.us0)
  in
  if batch_rank = 0
  then solve_fixed_unbatched p ~max_iters ?alphas ?damping ?c1 ?conv_threshold inputs
  else
    Rune.custom_jvp
      (Problem.inputs p)
      Nx.Ptree.tensor
      (fun { Problem.Inputs.theta; x0; us0 } ->
         (* The rule's own inputs, built from its arguments: the closure below
            must not hold the caller's tracked leaves. *)
         let inputs = { Problem.Inputs.theta; x0; us0 } in
         let sol =
           Solver.solve_fixed
             p
             ?alphas
             ?damping
             ?c1
             ?conv_threshold
             ~max_iters
             theta
             x0
             us0
         in
         let taus = Problem.taus_of p sol.xs sol.us in
         (* Built per tangent, as {!tangent_rule} is: the rule's own result
            must not be traced while the primal is. *)
         let tangent dinputs =
           Rune.custom_transpose
             (Problem.inputs p)
             Nx.Ptree.tensor
             (fun dinputs -> batched_tangent p ~batch_rank ~theta ~sol dinputs)
             ~transpose:(fun ybar -> batched_transpose p ~batch_rank ~sol inputs ybar)
             dinputs
         in
         taus, tangent)
      inputs
