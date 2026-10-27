(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Differentiable iLQR for discrete-time optimal control on Nx tensors.

    Dilqr solves discrete-time optimal control problems

    {[
      min_{u_0..u_{K-1}}  φ(x_K) + Σ_{k<K} l(x_k, u_k)
        s.t. x_{k+1} = f(x_k, u_k)
    ]}

    with iterative LQR on Nx tensors, and differentiates through the solve.
    The derivative is implicit: it is the derivative of the map that takes the
    local quadratic model at the solution to its minimizer, not the derivative
    of the iLQR iterations. Both modes are available and are transposes of
    each other in the sense that [⟨ȳ, JVP v⟩ = ⟨VJP ȳ, v⟩] for every cotangent
    [ȳ] and every tangent [v] (within the idealization both modes share).

    - {b Reverse mode.} A perturbed Riccati pass and a linearized rollout
      solve the adjoint KKT system of the local QP, and a nested [Rune.grad]
      chains the resulting block cotangents back to the problem parameters.
      {!solve_adaptive}, {!solve_fixed}.
    - {b Forward mode.} One Riccati backward pass at the solution yields the
      factors that the primal and every tangent share — the Cholesky factor of
      the regularized [Q_uu], the feedback gain, the value Hessian — and each
      tangent then costs one cheap backward pass and one rollout.
      {!Jvp.solve_adaptive}, {!Jvp.solve_fixed}.

    Rune cannot attach both a custom VJP and a custom JVP to one function, so
    the reverse rule is the default and the forward rule lives in
    {!module:Jvp}. Each pair shares its primal solve exactly; only the
    attached derivative rule differs.

    A problem is a value ({!Problem.t}): its dtype, dimensions, horizon,
    dynamics and losses are record fields, and its parameter structure is an
    [Nx.Ptree.t]. Everything below the solve level is a pure tensor
    computation with static control flow, and every time loop is a
    {!Rune.scan}: one traced step stands for the whole horizon, which
    [Rune.jit] compiles as a loop. A fixed-iteration solve, its implicit
    derivative in either mode, and the tangent sweeps of a forward-mode batch
    therefore trace and compile with [Rune.jit] on CPU or GPU into programs
    whose size grows with the iteration count, not with the horizon. The
    adaptive solve ({!solve_adaptive}) uses host-side control flow
    (regularization retries, Armijo backtracking) and is not itself traceable;
    with a JIT option, its tensor kernels are compiled by [Rune.jit] while
    that control flow stays on the host. Its returned closure is therefore
    eager.

    {b Batches.} {!solve_fixed} and {!Jvp.solve_fixed} solve a batch of
    problems at once: give every leaf of the parameters, the initial state
    and every initial control a leading batch axis, and each member gets its
    own trajectory, line search and derivative. The dynamics and the losses
    stay written for one member; the solver maps them over the batch. The
    adaptive solves ({!solve_adaptive}, {!Jvp.solve_adaptive}) are
    deliberately unbatched: their control flow is host-side, and a batch
    would need one of each per member.

    The record-returning solves and the tensor-level helpers remain in
    {!module:Solver} for advanced use.

    {b Batched forward mode.} A batch of {e tangents} is a different axis from
    a batch of problems, and it composes with it: map over the tangents with
    [Rune.vmap] and [Rune.jvp], and the primal solve — with the factors every
    tangent reuses — is computed once, outside the lanes.

    {[
      let solve = Dilqr.Jvp.solve_fixed problem in
      let taus = solve ~iters:5 inputs in
      let dtaus =
        Rune.vmap
          Nx.Ptree.(inputs problem @-> returns Nx.Ptree.tensor)
          (fun dinputs ->
             snd
               (Rune.jvp
                  (inputs problem)
                  Nx.Ptree.tensor
                  (fun i -> solve ~iters:5 i)
                  inputs
                  dinputs))
          dirs
      in
      (* [taus] is [K+1; n+m]; [dtaus] is [k; K+1; n+m] for [k] directions. *)
    ]}

    Inside the map the lane axis is invisible, so the tangent code is the
    tangent code of a single tangent, and every [k]-lane tensor is one batched
    tensor: the tangents of the problem's parameters, of every [A_k] and
    [B_k], of the local models' constant terms, and of the loss gradients. The
    quadratic blocks ([lxx], [luu], [lux], [φ_xx]) contribute no tangent at
    all — see {!module:Lqr} — and the tangents of the first-order data are
    computed step by step as the tangent sweeps reach them rather than
    materialized ({!Solver.tangent_solve}), so a direction costs
    [O (horizon · (n + m))] of memory on top of the primal solve's tape, and
    one backward sweep of [n]-vectors plus one rollout of compute. *)

(** Levenberg-Marquardt regularization schedule. *)
module Regularisation : module type of Regularisation

(** Line-search trial schedules and the Armijo predicate. *)
module Linesearch : module type of Linesearch

(** Tensor-math core: the linearization tape, the Riccati backward pass, the
    factors a solve shares with its tangents, the LQR solve and its JVP, gain
    rollouts, and the perturbed sweeps used by the implicit VJP. *)
include module type of Lqr

module Lqr : module type of Lqr

(** The problem description, as a value. *)
module Problem : module type of Problem

(** The solve inputs as a structure of tensors. *)
module Inputs = Problem.Inputs

(** [inputs problem] is the structure of the solve inputs: the problem's
    parameters at [theta], the initial state [x0], and every control of the
    initial guess [us0]. It is what {!Rune.grad}, {!Rune.jvp},
    {!Rune.custom_vjp}, {!Rune.custom_jvp} and {!Rune.jit} take. *)
val inputs : ('w, 'p) Problem.t -> ('p, 'w Problem.value) Inputs.t Nx.Ptree.t

(** [taus_of problem xs us] stacks the trajectory into a [(K+1); n+m] tensor:
    row k is (x_k; u_k) for k < K, row K is (x_K; 0). The batch axes lead the
    horizon, so a batched trajectory stacks to [batch...; K+1; n+m]. *)
val taus_of
  :  ('w, 'p) Problem.t
  -> 'w Problem.value array
  -> 'w Problem.value array
  -> 'w Problem.value

(** [split_taus problem taus] is ([xs], [us]). *)
val split_taus
  :  ('w, 'p) Problem.t
  -> 'w Problem.value
  -> 'w Problem.value array * 'w Problem.value array

(** [loss_of_taus problem theta taus] evaluates the problem's own running +
    final losses along the trajectory. Differentiable, traceable. A batched
    [theta] and trajectory are evaluated member by member. *)
val loss_of_taus : ('w, 'p) Problem.t -> 'p -> 'w Problem.value -> 'w Problem.value

(** The iLQR solvers. *)
module Solver : sig
  type 'a t =
    { xs : 'a array (** states, length [horizon + 1] *)
    ; us : 'a array (** controls, length [horizon] *)
    ; cost : 'a (** scalar tensor: running + final loss *)
    }

  (** [rollout problem theta x0 us] integrates the dynamics from [x0] under
      [us] and returns the state path and total cost. Batched inputs are
      rolled out member by member. *)
  val rollout
    :  ('w, 'p) Problem.t
    -> 'p
    -> 'w Problem.value
    -> 'w Problem.value array
    -> 'w Problem.value array * 'w Problem.value

  (** [blocks problem theta xs us] linearizes dynamics and losses along the
      trajectory. A batched trajectory is linearized member by member. *)
  val blocks
    :  ('w, 'p) Problem.t
    -> 'p
    -> 'w Problem.value array
    -> 'w Problem.value array
    -> (float, 'w) Lqr.tape

  (** [tangent problem theta dtheta xs us] is the derivative of the tape's
      first-order data along [dtheta], at the trajectory: the tangents of
      [∂f/∂x], [∂f/∂u], of the local models' constant term
      [f_k - A_k x_k - B_k u_k], of [∂l/∂x], of [∂l/∂u] and of [∂φ/∂x].

      It is the one thing a forward-mode solve needs from the problem: the
      tangents of the loss Hessians and of [∂²φ/∂x²] do not appear.

      The trajectory is one member's, as the solve's own time loops see it: a
      batched problem is mapped over the batch by the solve that calls this. A
      batch of {e tangents} is a different axis and does not belong here — map
      the whole solve over the directions ([Rune.vmap] around [Rune.jvp]), so
      that the primal solve and its factorization are shared. *)
  val tangent
    :  ('w, 'p) Problem.t
    -> 'p
    -> dtheta:'p
    -> 'w Problem.value array
    -> 'w Problem.value array
    -> (float, 'w) Lqr.tangent

  (** [tangent_solve problem theta ~dtheta ~dx0 xs us] is the tangent of the
      local model's LQR solve along the trajectory: the derivative of
      {!Lqr.solve} at the linearization of the problem at [xs], [us], in the
      direction [dtheta] of the problem's parameters, with initial state
      tangent [dx0].

      The three sweeps of {!Lqr.solve_jvp}, with the tape's Riccati factors and
      costate shared and the tangents of the first-order data computed step by
      step as the sweeps reach them, never stored. The direction-multiplied
      tensors are then only the tangent trajectory, the tangent feedforward and
      the tangent control — [O (horizon · (n + m))] per direction against the
      [O (horizon · (n² + n·m))] a materialized {!Lqr.tangent} would take — and
      everything that does not depend on [dtheta] is a constant of any
      [Rune.vmap] around it, so a batch of directions pays for the tape, the
      Hessian recursion and the factor once. This is what the forward-mode
      solves run.

      The trajectory is one member's, as the solve's own time loops see it: a
      batched problem is mapped over the batch by the solve that calls this. *)

  val tangent_solve
    :  ('w, 'p) Problem.t
    -> 'p
    -> dtheta:'p
    -> dx0:'w Problem.value
    -> 'w Problem.value array
    -> 'w Problem.value array
    -> (float, 'w) Lqr.solution

  (** Adaptive iLQR. Host-side control flow: unbatched (a batch would need one
      line search per member; {!solve_fixed} has it), and not itself traceable
      by [Rune.jit]. Defaults: [max_iter = 20], [conv_threshold = 1e-6],
      [c1 = 1e-2], [ls_depth = 8]. Fails when no step can be regularized into
      descent. {!compile} is the same solve with the tensor work compiled. *)
  val solve
    :  ('w, 'p) Problem.t
    -> ?max_iter:int
    -> ?conv_threshold:float
    -> ?c1:float
    -> ?ls_depth:int
    -> ?reg:Regularisation.t
    -> 'p
    -> 'w Problem.value
    -> 'w Problem.value array
    -> 'w Problem.value t

  (** The adaptive solve with its tensor work compiled. The type of the
      returned function is {!solve} minus the problem. *)
  type ('w, 'p) compiled =
    ?max_iter:int
    -> ?conv_threshold:float
    -> ?c1:float
    -> ?ls_depth:int
    -> ?reg:Regularisation.t
    -> 'p
    -> 'w Problem.value
    -> 'w Problem.value array
    -> 'w Problem.value t

  (** [compile ?beam ?parallel problem] is {!solve} with each traceable kernel
      — the dynamics and cost scans, the linearization and Riccati pass, and
      one line-search rollout — compiled by [Rune.jit]. The host still owns
      the control flow: regularization retries, Armijo backtracking and the
      convergence test read scalars, so the result matches the eager solve.
      The problem's dynamics and losses must be traceable (no host read of a
      value that depends on a traced argument); {!solve_fixed} has the same
      requirement.

      Build the function once and reuse it. Its programs are keyed by the
      shapes, dtypes and placements of the inputs, and the regularization
      strength and line-search step arrive as tensor arguments, so a new
      [mu] or [alpha] does not retrace. [beam] and [parallel] are
      [Rune.jit]'s kernel-search options; with [beam] each kernel is
      searched for its fastest variant, and [parallel] compiles the
      candidates on that many domains. *)
  val compile : ?beam:int -> ?parallel:int -> ('w, 'p) Problem.t -> ('w, 'p) compiled

  (** Fixed-iteration iLQR with tensor-only control flow: traceable and
      JIT-compilable. Every time loop is a [Rune.scan], compiled as a loop,
      so a program costs its iterations, not its horizon times its
      iterations. [alphas] is the trial schedule (default [1; 1/2; …], 8
      trials); the line search is a masked argmin over the trials, evaluated
      in-graph. [mu] fixes the regularization (default none).

      Batched inputs — a leading batch axis on the parameters, the initial
      state and every initial control — are solved member by member (the
      single-member solve is mapped over the batch): [cost] is then [B] and
      every state and control is [B; n] or [B; m]. *)
  val solve_fixed
    :  ('w, 'p) Problem.t
    -> ?alphas:float list
    -> ?mu:float
    -> ?c1:float
    -> iters:int
    -> 'p
    -> 'w Problem.value
    -> 'w Problem.value array
    -> 'w Problem.value t
end

(** Opaque data the solve's forward pass hands to its backward pass. *)
type ('p, 'a) residual

(** The implicit VJP: cotangent on the stacked trajectory in, gradients on
    the solve inputs out. *)
val bwd
  :  ('w, 'p) Problem.t
  -> ('p, 'w Problem.value) residual
  -> 'w Problem.value
  -> ('p, 'w Problem.value) Inputs.t

(** {2 Differentiable solves} *)

(** A differentiable solve: the solve inputs in, the stacked trajectory out.
    Both the reverse-mode and the forward-mode entry points have this type. *)
type ('w, 'p) solve = ('p, 'w Problem.value) Inputs.t -> 'w Problem.value

(** [solve_adaptive problem ?beam ?parallel] is the adaptive iLQR solve with
    its implicit reverse-mode derivative. The returned function takes the
    adaptive options and the solve inputs, and returns the stacked trajectory.

    [beam] and [parallel] are [Solver.compile]'s kernel-search options.
    Supplying either one compiles the tensor kernels once per problem and
    reuses them across calls; pass [~parallel:1] (or [~beam:0]) to compile
    with the default search settings. Without either, the solve runs eagerly.
    The host still owns the regularization retries, line search and
    convergence test, so the returned closure is eager either way and must not
    be placed inside [Rune.jit]. Unbatched; use {!Jvp.solve_adaptive} for
    forward mode.

    The derivative is taken at the undamped KKT system of the local model at
    the solution: the solve's own regularization is a device that finds the
    solution, not a part of the map being differentiated. If the loss
    curvature leaves [Q_uu] of the model at the solution indefinite, that
    factor is NaN and so is the gradient. *)
val solve_adaptive
  :  ('w, 'p) Problem.t
  -> ?beam:int
  -> ?parallel:int
  -> ?max_iter:int
  -> ?conv_threshold:float
  -> ?c1:float
  -> ?ls_depth:int
  -> ?reg:Regularisation.t
  -> ('w, 'p) solve

(** [solve_fixed problem] is the fixed-iteration iLQR solve with its implicit
    reverse-mode derivative. The returned function takes the fixed-solve
    options and the solve inputs, and returns the stacked trajectory.

    Batched inputs (a leading batch axis on every leaf) are solved member by
    member, the implicit VJP included; the stacked trajectory is then
    [B; K+1; n+m]. The solve is traceable, so put the returned function inside
    [Rune.jit] with a [Rune.grad] inside:

    {[
    let solve = solve_fixed problem in
    let step =
      Rune.jit
        Nx.Ptree.(inputs problem @-> returns Nx.Ptree.tensor)
        (fun inputs ->
           Rune.grad
             (inputs problem)
             (fun inputs -> loss (solve ~iters:3 inputs))
             inputs)
    ]}

    The derivative is taken at the undamped KKT system, as in
    {!solve_adaptive}. Use {!Jvp.solve_fixed} for forward mode. *)
val solve_fixed
  :  ('w, 'p) Problem.t
  -> ?alphas:float list
  -> ?mu:float
  -> ?c1:float
  -> iters:int
  -> ('w, 'p) solve

(** The forward-mode forms of the two solves. Each returns the same primal
    trajectory as its reverse-mode counterpart; only the attached rule
    differs. *)
module Jvp : sig
  (** [Jvp.solve_adaptive problem ?beam ?parallel] is {!solve_adaptive} with a
      custom forward-mode rule: [Rune.jvp] applied to it is the derivative of
      the converged solution along the tangent of the solve inputs, computed
      from one factorization of the local model at the solution rather than by
      differentiating the iteration. The tangent of [us0] does not reach the
      result — the idealized solve does not depend on its initial guess — and
      neither does its reverse-mode cotangent.

      Use it when the number of directions is small and the output is large, or
      when a whole batch of directions has to share one solve: map [Rune.jvp]
      over the tangent seeds with [Rune.vmap] and the primal solve, with its
      factorization, is computed once (see the module documentation). The
      returned closure is eager and unbatched, like {!solve_adaptive}. *)
  val solve_adaptive
    :  ('w, 'p) Problem.t
    -> ?beam:int
    -> ?parallel:int
    -> ?max_iter:int
    -> ?conv_threshold:float
    -> ?c1:float
    -> ?ls_depth:int
    -> ?reg:Regularisation.t
    -> ('w, 'p) solve

  (** [Jvp.solve_fixed problem] is {!solve_fixed} with a custom forward-mode
      rule. Same primal, same batched inputs, same traceability: the tangent
      map is pure tensor code whose time loops are scans, so the solve and its
      JVP compile together. The batched forward-mode recipe in the module
      documentation is this function:

      {[
      let solve = Dilqr.Jvp.solve_fixed problem in
      let dtaus =
        Rune.vmap
          Nx.Ptree.(inputs problem @-> returns Nx.Ptree.tensor)
          (fun dinputs ->
             snd
               (Rune.jvp
                  (inputs problem)
                  Nx.Ptree.tensor
                  (fun i -> solve ~iters:5 i)
                  inputs
                  dinputs))
          dirs
      ]}

      Two notes on composing it. [Rune.vmap] around [Rune.jvp] keeps one solve
      for all the directions, because the primal is a constant of the map; a
      [vmap] that also varies the problem's parameters re-solves per lane,
      which is what it should do. And a {e reverse} differentiation of this
      function would transpose the tangent map, which [Rune] supports only
      degraded (its scans are written out and a value beside a tangent is
      taken as zero): take gradients with {!solve_fixed}, which is built for
      them. *)
  val solve_fixed
    :  ('w, 'p) Problem.t
    -> ?alphas:float list
    -> ?mu:float
    -> ?c1:float
    -> iters:int
    -> ('w, 'p) solve
end
