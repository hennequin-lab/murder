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
    The implicit gradient is computed with the LQR machinery itself: a
    perturbed Riccati pass and linearized rollout solve the adjoint KKT
    system of the local QP, and a nested [Rune.grad] chains the resulting
    block cotangents back to the problem parameters.

    A problem is a value ({!Problem.t}): its dtype, dimensions, horizon,
    dynamics and losses are record fields, and its parameter structure is an
    [Nx.Ptree.t]. Everything below the solve level is a pure tensor
    computation with static control flow, and every time loop is a
    {!Rune.scan}: one traced step stands for the whole horizon, which
    [Rune.jit] compiles as a loop. A fixed-iteration solve and its implicit
    gradient therefore trace and compile with [Rune.jit] on CPU or GPU in a
    program whose size grows with the iteration count, not with the horizon.
    The adaptive solver ({!Solver.solve}) uses host-side control flow
    (regularization retries, Armijo backtracking) and runs eagerly only. *)

(** Levenberg-Marquardt regularization schedule. *)
module Regularisation : module type of Regularisation

(** Line-search trial schedules and the Armijo predicate. *)
module Linesearch : module type of Linesearch

(** Tensor-math core: the linearization tape, the Riccati backward pass, gain
    rollouts, and the perturbed sweeps used by the implicit gradient. *)
include module type of Lqr

module Lqr : module type of Lqr

(** The problem description, as a value. *)
module Problem : module type of Problem

(** The solve inputs as a structure of tensors. *)
module Inputs = Problem.Inputs

(** [inputs problem] is the structure of the solve inputs: the problem's
    parameters at [theta], the initial state [x0], and every control of the
    initial guess [us0]. It is what {!Rune.grad}, {!Rune.custom_vjp} and
    {!Rune.jit} take. *)
val inputs : ('w, 'p) Problem.t -> ('p, 'w Problem.value) Inputs.t Nx.Ptree.t

(** [taus_of problem xs us] stacks the trajectory into a [(K+1); n+m] tensor:
    row k is (x_k; u_k) for k < K, row K is (x_K; 0). *)
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
    final losses along the trajectory. Differentiable, traceable. *)
val loss_of_taus : ('w, 'p) Problem.t -> 'p -> 'w Problem.value -> 'w Problem.value

(** The iLQR solvers. *)
module Solver : sig
  type 'a t =
    { xs : 'a array (** states, length [horizon + 1] *)
    ; us : 'a array (** controls, length [horizon] *)
    ; cost : 'a (** scalar tensor: running + final loss *)
    }

  (** [rollout problem theta x0 us] integrates the dynamics from [x0] under
      [us] and returns the state path and total cost. *)
  val rollout
    :  ('w, 'p) Problem.t
    -> 'p
    -> 'w Problem.value
    -> 'w Problem.value array
    -> 'w Problem.value array * 'w Problem.value

  (** [blocks problem theta xs us] linearizes dynamics and losses along the
      trajectory. *)
  val blocks
    :  ('w, 'p) Problem.t
    -> 'p
    -> 'w Problem.value array
    -> 'w Problem.value array
    -> (float, 'w) Lqr.tape

  (** Adaptive iLQR. Host-side control flow: eager only, do not jit.
      Defaults: [max_iter = 20], [conv_threshold = 1e-6], [c1 = 1e-2],
      [ls_depth = 8]. Fails when no step can be regularized into descent. *)
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

  (** Fixed-iteration iLQR with tensor-only control flow: traceable and
      JIT-compilable. Every time loop is a [Rune.scan], compiled as a loop,
      so a program costs its iterations, not its horizon times its
      iterations. [alphas] is the trial schedule (default [1; 1/2; …], 8
      trials); the line search is a masked argmin over the trials, evaluated
      in-graph. [mu] fixes the regularization (default none). *)
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

(** [solve_ilqr problem inputs] runs the adaptive solve and returns the
    stacked trajectory, with the implicit gradient attached through a custom
    VJP. Eager only — under [Rune.grad] the backward pass computes the
    implicit gradient; do not jit (use {!solve_ilqr_fixed}). *)
val solve_ilqr
  :  ('w, 'p) Problem.t
  -> ?max_iter:int
  -> ?conv_threshold:float
  -> ?c1:float
  -> ?ls_depth:int
  -> ?reg0:Regularisation.t
  -> ('p, 'w Problem.value) Inputs.t
  -> 'w Problem.value

(** The traceable solve with its implicit gradient. Put inside [Rune.jit]
    with a [Rune.grad] inside:

    {[
    let step =
      Rune.jit
        Nx.Ptree.(inputs problem @-> returns Nx.Ptree.tensor)
        (fun inputs ->
           Rune.grad
             (inputs problem)
             (fun inputs -> loss (solve_ilqr_fixed problem ~iters:3 inputs))
             inputs)
    ]} *)
val solve_ilqr_fixed
  :  ('w, 'p) Problem.t
  -> iters:int
  -> ?alphas:float list
  -> ?mu:float
  -> ?c1:float
  -> ('p, 'w Problem.value) Inputs.t
  -> 'w Problem.value
