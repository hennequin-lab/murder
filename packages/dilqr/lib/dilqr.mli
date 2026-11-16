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
    - {b Forward mode.} One Riccati backward pass at the solution yields the
      factors that the primal and every tangent share — the Cholesky factor of
      the regularized [Q_uu], the feedback gain, the value Hessian — and each
      tangent then costs one cheap backward pass and one rollout.

    Both rules are attached to the same solve. {!solve_adaptive} and
    {!solve_fixed} wrap their primal in one {!Rune.custom_jvp}: forward mode
    runs its tangent map, and reverse mode applies {!bwd}'s pullback as that
    map's transpose (a {!Rune.custom_transpose}). There is no separate
    forward-mode entry point: [Rune.jvp] and [Rune.grad] both applied to the
    same function return the two derivatives.

    A problem is a value ({!Problem.t}): its dtype, dimensions, horizon,
    dynamics and losses are record fields, and its parameter structure is an
    [Nx.Ptree.t]. Everything below the solve level is a pure tensor
    computation, and every time loop is a {!Rune.scan}: one traced step stands
    for the whole horizon, which [Rune.jit] compiles as a loop. The fixed
    solve's control flow is data-dependent but stays in the graph — its
    iteration budget and damping retries are {!Rune.iterate}s and its line
    search {!Rune.cond} branches — so a solve, its implicit derivative in
    either mode, and the tangent sweeps of a forward-mode batch all trace and
    compile with [Rune.jit] on CPU or GPU into programs whose size grows with
    neither the horizon nor the budget. The
    adaptive solve ({!solve_adaptive}) uses host-side control flow
    (regularization retries, Armijo backtracking) and is not itself traceable;
    with a JIT option, its tensor kernels are compiled by [Rune.jit] while
    that control flow stays on the host. Its returned closure is therefore
    eager.

    {b The initial state.} A problem either starts from the state its inputs
    give it, or — with {!Problem.t}'s [free_initial] — prices that state in its
    step-0 running loss and optimizes it with the controls. The second is what
    an encoder that infers an initial condition needs: the solve then returns
    the joint mode, its trajectory's step 0 is the state it found, and both
    implicit derivatives carry the initial state's own dependence on the
    parameters. The step is Newton's, the local model being quadratic in
    [x₀], so one solve finds it; a model without a proper prior on [x₀] leaves
    the block singular.

    {b Batches.} {!solve_fixed} solves a batch of problems at once: give every
    leaf of the parameters, the initial state and every initial control a
    leading batch axis, and each member gets its own trajectory, cost and
    derivative. The dynamics and the losses stay written for one member; the
    solver maps them over the batch. Two things are the batch's: the step —
    a trial is accepted when every member that has not converged satisfies
    Armijo at it, so the batch moves at the step its slowest member can take —
    and the iteration budget, which is spent until the last member has
    converged. Damping is one schedule for the batch, of a strength relative to
    each member's own value-Hessian scale. {!solve_adaptive} is deliberately
    unbatched: its control flow is host-side, and a batch would need one of
    each per member.

    The record-returning solves and the tensor-level helpers remain in
    {!module:Solver} for advanced use.

    {b Batched forward mode.} A batch of {e tangents} is a different axis from
    a batch of problems. Map over the tangents with [Rune.vmap] and [Rune.jvp]
    around one problem's solve, and the primal — with the factors every tangent
    reuses — is computed once, outside the lanes. A batched problem's
    derivative is already taken member by member, so a direction axis belongs
    around a single problem: both would be leading axes.

    {[
      let solve = Dilqr.solve_fixed problem in
      let taus = solve ~max_iters:5 inputs in
      let dtaus =
        Rune.vmap
          Nx.Ptree.(inputs problem @-> returns Nx.Ptree.tensor)
          (fun dinputs ->
             snd
               (Rune.jvp
                  (inputs problem)
                  Nx.Ptree.tensor
                  (fun i -> solve ~max_iters:5 i)
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

(** What a solve does about the local model's curvature: no damping, one fixed
    strength, or a schedule that adapts it. Adaptive by default. *)
module Damping : module type of Damping

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
    {!Rune.custom_jvp} and {!Rune.jit} take. *)
val inputs
  :  ('w, 'p, 'd) Problem.t
  -> (('p, ('w, 'd) Problem.value) Inputs.t, 'd) Nx.Ptree.t

(** [taus_of problem xs us] stacks the trajectory into a [(K+1); n+m] tensor:
    row k is (x_k; u_k) for k < K, row K is (x_K; 0). The batch axes lead the
    horizon, so a batched trajectory stacks to [batch...; K+1; n+m]. *)
val taus_of
  :  ('w, 'p, 'd) Problem.t
  -> ('w, 'd) Problem.value array
  -> ('w, 'd) Problem.value array
  -> ('w, 'd) Problem.value

(** [split_taus problem taus] is ([xs], [us]). *)
val split_taus
  :  ('w, 'p, 'd) Problem.t
  -> ('w, 'd) Problem.value
  -> ('w, 'd) Problem.value array * ('w, 'd) Problem.value array

(** [loss_of_taus problem theta taus] evaluates the problem's own running +
    final losses along the trajectory. Differentiable, traceable. A batched
    [theta] and trajectory are evaluated member by member. *)
val loss_of_taus
  :  ('w, 'p, 'd) Problem.t
  -> 'p
  -> ('w, 'd) Problem.value
  -> ('w, 'd) Problem.value

(** The iLQR solvers. *)
module Solver : sig
  type 'a t =
    { xs : 'a array (** states, length [horizon + 1] *)
    ; us : 'a array (** controls, length [horizon] *)
    ; cost : 'a (** scalar tensor: running + final loss *)
    ; iters : 'a
      (** the number of iterations the solve ran, as a scalar of the problem's
          dtype. A batch runs the iterations its slowest member needs, so this
          is how much of a [max_iters] budget it spent. *)
    }

  (** [rollout problem theta x0 us] integrates the dynamics from [x0] under
      [us] and returns the state path and total cost. Batched inputs are
      rolled out member by member. *)
  val rollout
    :  ('w, 'p, 'd) Problem.t
    -> 'p
    -> ('w, 'd) Problem.value
    -> ('w, 'd) Problem.value array
    -> ('w, 'd) Problem.value array * ('w, 'd) Problem.value

  (** [blocks problem theta xs us] linearizes dynamics and losses along the
      trajectory. A batched trajectory is linearized member by member. *)
  val blocks
    :  ('w, 'p, 'd) Problem.t
    -> 'p
    -> ('w, 'd) Problem.value array
    -> ('w, 'd) Problem.value array
    -> (float, 'w, 'd) Lqr.tape

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
    :  ('w, 'p, 'd) Problem.t
    -> 'p
    -> dtheta:'p
    -> ('w, 'd) Problem.value array
    -> ('w, 'd) Problem.value array
    -> (float, 'w, 'd) Lqr.tangent

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
    :  ('w, 'p, 'd) Problem.t
    -> 'p
    -> dtheta:'p
    -> dx0:('w, 'd) Problem.value
    -> ('w, 'd) Problem.value array
    -> ('w, 'd) Problem.value array
    -> (float, 'w, 'd) Lqr.solution

  (** Adaptive iLQR. Host-side control flow: unbatched (a batch would need one
      line search per member; {!solve_fixed} has it), and not itself traceable
      by [Rune.jit]. Defaults: [max_iter = 20], [conv_threshold = 1e-6],
      [c1 = 1e-2], [ls_depth = 8], [damping = Damping.default]. Fails when no
      step can be regularized into descent — with [Damping.Off] or a
      [Damping.Fixed] strength that is already the case when the model's
      [Q_uu] is not positive definite, and the failure says so. {!compile} is
      the same solve with the tensor work compiled. *)
  val solve
    :  ('w, 'p, 'd) Problem.t
    -> ?max_iter:int
    -> ?conv_threshold:float
    -> ?c1:float
    -> ?ls_depth:int
    -> ?damping:Damping.t
    -> 'p
    -> ('w, 'd) Problem.value
    -> ('w, 'd) Problem.value array
    -> ('w, 'd) Problem.value t

  (** The adaptive solve with its tensor work compiled. The type of the
      returned function is {!solve} minus the problem. *)
  type ('w, 'p, 'd) compiled =
    ?max_iter:int
    -> ?conv_threshold:float
    -> ?c1:float
    -> ?ls_depth:int
    -> ?damping:Damping.t
    -> 'p
    -> ('w, 'd) Problem.value
    -> ('w, 'd) Problem.value array
    -> ('w, 'd) Problem.value t

  (** [compile ?beam ?parallel problem] is {!solve} with each traceable kernel
      — the dynamics and cost scans, the linearization and Riccati pass, and
      one line-search rollout — compiled by [Rune.jit]. The host still owns
      the control flow: regularization retries, Armijo backtracking and the
      convergence test read scalars, so the result matches the eager solve.
      The problem's dynamics and losses must be traceable (no host read of a
      value that depends on a traced argument); {!solve_fixed} has the same
      requirement.

      Build the function once and reuse it. Its programs are keyed by the
      shapes, dtypes and placements of the inputs, and the damping strength and
      line-search step arrive as tensor arguments, so a new strength or [alpha]
      does not retrace. [beam] and [parallel] are
      [Rune.jit]'s kernel-search options; with [beam] each kernel is
      searched for its fastest variant, and [parallel] compiles the
      candidates on that many domains. *)
  val compile
    :  ?beam:int
    -> ?parallel:int
    -> ('w, 'p, 'd) Problem.t
    -> ('w, 'p, 'd) compiled

  (** Fixed-iteration iLQR with tensor-only control flow: traceable and
      JIT-compilable. Every time loop is a [Rune.iterate] with a tensor test,
      compiled as a loop, so a program costs its code, not its horizon or its
      iteration budget.

      One iteration is a Riccati pass, then a backtracking line search that
      tries [alphas] in order (default [1; 1/2; …], 8 trials) and stops at the
      first trial the batch accepts: [alphas] is a host-side schedule, and
      each level is a branch, so a level the batch never reaches computes
      nothing. A trial is accepted when every member that has not converged
      satisfies Armijo at it, so the batch moves at the step its slowest
      member can take. A member whose modelled decrease has become negligible
      is converged: it is frozen and does not vote, so it cannot veto the
      others. The loop stops once every member has converged, or once an
      iteration could not buy a step at any damping — so the batch spends the
      iterations of its slowest member out of [max_iters], and [iters] reports
      how many.

      [damping] is [Damping.default] — adaptive, from the schedule's own start
      — unless the caller picks another: [Damping.Off] or [Damping.Fixed]
      leave nothing to raise when a pass cannot buy a step, so such an
      iteration ends the solve. An adaptive strength is a fraction of each
      member's own value-Hessian scale, measured by the pass, so members whose
      curvature differs are damped alike; the first pass uses the schedule's
      units.

      Batched inputs — a leading batch axis on the parameters, the initial
      state and every initial control — are solved as one batch: [cost] is
      then [B] and every state and control is [B; n] or [B; m]. The members
      share the step and the damping schedule.

      [alphas] must be positive (a rejected batch is reported as a zero step),
      and [max_iters] not negative ([Rune.iterate] does not take one). *)
  val solve_fixed
    :  ('w, 'p, 'd) Problem.t
    -> ?alphas:float list
    -> ?damping:Damping.t
    -> ?c1:float
    -> ?conv_threshold:float
    -> max_iters:int
    -> 'p
    -> ('w, 'd) Problem.value
    -> ('w, 'd) Problem.value array
    -> ('w, 'd) Problem.value t
end

(** Opaque data the solve's forward pass hands to its backward pass. *)
type ('p, 'a) residual

(** The implicit VJP: cotangent on the stacked trajectory in, gradients on
    the solve inputs out. *)
val bwd
  :  ('w, 'p, 'd) Problem.t
  -> ('p, ('w, 'd) Problem.value) residual
  -> ('w, 'd) Problem.value
  -> ('p, ('w, 'd) Problem.value) Inputs.t

(** {2 Differentiable solves} *)

(** A differentiable solve: the solve inputs in, the stacked trajectory out.
    Both {!solve_adaptive} and {!solve_fixed} have this type, and both modes'
    derivatives are attached to it. *)
type ('w, 'p, 'd) solve = ('p, ('w, 'd) Problem.value) Inputs.t -> ('w, 'd) Problem.value

(** [solve_adaptive problem ?beam ?parallel] is the adaptive iLQR solve with
    its implicit derivative in both modes. The returned function takes the
    adaptive options and the solve inputs, and returns the stacked trajectory.

    [beam] and [parallel] are [Solver.compile]'s kernel-search options.
    Supplying either one compiles the tensor kernels once per problem and
    reuses them across calls; pass [~parallel:1] (or [~beam:0]) to compile
    with the default search settings. Without either, the solve runs eagerly.
    The host still owns the regularization retries, line search and
    convergence test, so the returned closure is eager either way and must not
    be placed inside [Rune.jit]. Unbatched.

    [Rune.grad] of the returned function is the implicit VJP; [Rune.jvp] of it
    is the implicit JVP, computed from one factorization of the local model at
    the solution rather than by differentiating the iteration. The tangent of
    [us0] does not reach the result — the idealized solve does not depend on
    its initial guess — and neither does its reverse-mode cotangent.

    The derivative is taken at the undamped KKT system of the local model at
    the solution: the solve's own regularization is a device that finds the
    solution, not a part of the map being differentiated. If the loss
    curvature leaves [Q_uu] of the model at the solution indefinite, that
    factor is NaN and so is the gradient. *)
val solve_adaptive
  :  ('w, 'p, 'd) Problem.t
  -> ?beam:int
  -> ?parallel:int
  -> ?max_iter:int
  -> ?conv_threshold:float
  -> ?c1:float
  -> ?ls_depth:int
  -> ?damping:Damping.t
  -> ('w, 'p, 'd) solve

(** [solve_fixed problem] is the fixed-iteration iLQR solve with its implicit
    derivative in both modes. The returned function takes the fixed-solve
    options and the solve inputs, and returns the stacked trajectory.

    Batched inputs (a leading batch axis on every leaf) are solved as one
    batch — the step and the damping schedule are the batch's, and the solve
    stops when its slowest member has converged — and the implicit derivative
    is taken member by member, which is what the implicit function theorem
    says: one member's solution does not depend on the others. The stacked
    trajectory is then [B; K+1; n+m]. The solve is traceable, so put the
    returned function inside [Rune.jit] with a [Rune.grad] or a [Rune.jvp]
    inside:

    {[
    let solve = solve_fixed problem in
    let step =
      Rune.jit
        Nx.Ptree.(inputs problem @-> returns Nx.Ptree.tensor)
        (fun inputs ->
           Rune.grad
             (inputs problem)
             (fun inputs -> loss (solve ~max_iters:3 inputs))
             inputs)
    ]}

    The derivative is taken at the undamped KKT system, as in
    {!solve_adaptive}: the damping, the step and the iteration count are
    devices that find the trajectory, not part of the map being
    differentiated. [Rune.grad] and [Rune.jvp] both apply to the same returned
    function; the forward tangent map is pure tensor code whose time loops are
    scans, so the solve and its JVP compile together. *)
val solve_fixed
  :  ('w, 'p, 'd) Problem.t
  -> ?alphas:float list
  -> ?damping:Damping.t
  -> ?c1:float
  -> ?conv_threshold:float
  -> max_iters:int
  -> ('w, 'p, 'd) solve
