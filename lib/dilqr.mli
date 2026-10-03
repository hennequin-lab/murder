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

    Structure: everything below the solve level — {!module-type:Lqr.Ctx},
    {!Lqr.tape}, {!Lqr.backward}, the rollouts — is a pure tensor computation
    with static control flow, and every time loop is a {!Rune.scan}: one
    traced step stands for the whole horizon, which [Rune.jit] compiles as a
    loop. A fixed-iteration solve and its implicit gradient therefore trace
    and compile with [Rune.jit] on CPU or GPU in a program whose size grows
    with the iteration count, not with the horizon. The adaptive solver
    ({!Make.val-solve}) uses host-side control flow (regularization retries,
    Armijo backtracking) and runs eagerly only. *)

(** Levenberg-Marquardt regularization schedule. *)
module Regularisation : module type of Regularisation

(** Line-search trial schedules and the Armijo predicate. *)
module Linesearch : module type of Linesearch

(** Tensor-math core: the linearization tape, the Riccati backward pass, gain
    rollouts, and the problem signature {!module-type:Lqr.Ctx}. *)
include module type of Lqr

module Lqr : module type of Lqr

module Make (C : Ctx) : sig
  type value = C.value

  module Solver : sig
    type t =
      { xs : value array (** states, length [horizon + 1] *)
      ; us : value array (** controls, length [horizon] *)
      ; cost : value (** scalar tensor: running + final loss *)
      }

    (** [rollout theta x0 us] integrates the dynamics from [x0] under [us] and
        returns the state path and total cost. *)
    val rollout : value C.Params.t -> value -> value array -> value array * value

    (** [blocks theta xs us] linearizes dynamics and losses along the
        trajectory. *)
    val blocks : value C.Params.t -> value array -> value array -> (float, C.w) Lqr.tape

    (** Adaptive iLQR. Host-side control flow: eager only, do not jit.
        Defaults: [max_iter = 20], [conv_threshold = 1e-6], [c1 = 1e-2],
        [ls_depth = 8]. Fails when no step can be regularized into descent. *)
    val solve
      :  ?max_iter:int
      -> ?conv_threshold:float
      -> ?c1:float
      -> ?ls_depth:int
      -> ?reg:Regularisation.t
      -> value C.Params.t
      -> value
      -> value array
      -> t

    (** Fixed-iteration iLQR with tensor-only control flow: traceable and
        JIT-compilable. Every time loop is a [Rune.scan], compiled as a loop,
        so a program costs its iterations, not its horizon times its
        iterations. [alphas] is the trial schedule (default [1; 1/2; …], 8
        trials); the line search is a masked argmin over the trials, evaluated
        in-graph. [mu] fixes the regularization (default none). *)
    val solve_fixed
      :  ?alphas:float list
      -> ?mu:float
      -> ?c1:float
      -> iters:int
      -> value C.Params.t
      -> value
      -> value array
      -> t
  end

  (** The solve inputs as a structure of tensors. *)
  module Inputs : sig
    type 'a t =
      { theta : 'a C.Params.t (** problem parameters *)
      ; x0 : 'a (** initial state *)
      ; us0 : 'a array (** initial control guess, length [horizon] *)
      }

    (** The structure of the solve inputs: the tensors of [theta], [x0] and of
        every control of [us0] are leaves, and [us0]'s length is reported. *)
    val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
  end

  (** {!Inputs} at [value]: the structure [Rune.grad], [Rune.custom_vjp] and
      [Rune.jit] take. *)
  val inputs_ptree : value Inputs.t Nx.Ptree.t

  (** Opaque data the solve's forward pass hands to its backward pass. *)
  type residual

  (** [taus_of xs us] stacks the trajectory into a [(K+1); n+m] tensor: row k
      is (x_k; u_k) for k < K, row K is (x_K; 0). *)
  val taus_of : value array -> value array -> value

  (** [split_taus taus] is ([xs], [us]). *)
  val split_taus : value -> value array * value array

  (** [loss_of_taus theta taus] evaluates the problem's own running + final
      losses along the trajectory. Differentiable, traceable. *)
  val loss_of_taus : value C.Params.t -> value -> value

  (** The implicit VJP: cotangent on the stacked trajectory in, gradients on
      the solve inputs out. *)
  val bwd : residual -> value -> value Inputs.t

  (** [solve_ilqr inputs] runs the adaptive solve and returns the stacked
      trajectory, with the implicit gradient attached through a custom VJP.
      Eager only — under [Rune.grad] the backward pass computes the implicit
      gradient; do not jit (use {!val-solve_ilqr_fixed}). *)
  val solve_ilqr
    :  ?max_iter:int
    -> ?conv_threshold:float
    -> ?c1:float
    -> ?ls_depth:int
    -> ?reg0:Regularisation.t
    -> value Inputs.t
    -> value

  (** The traceable solve with its implicit gradient. Put inside [Rune.jit]
      with a [Rune.grad] inside:

      {[
      let step =
        Rune.jit
          Nx.Ptree.(inputs_ptree @-> returns Nx.Ptree.tensor)
          (fun inputs ->
             Rune.grad
               inputs_ptree
               (fun inputs -> loss (solve_ilqr_fixed ~iters:3 inputs))
               inputs)
      ]} *)
  val solve_ilqr_fixed
    :  iters:int
    -> ?alphas:float list
    -> ?mu:float
    -> ?c1:float
    -> value Inputs.t
    -> value
end
