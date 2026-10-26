(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Inverse kinematics by damped Gauss-Newton.

    A solver is a policy: how much to damp, when to accept a step, when to stop.
    The robotics layer's job is to make the policy short and replaceable, so
    this module is one loop over {!Task} residuals and nothing else — no task
    knows it exists, and a demo can drive the same iteration by hand.

    The step is Levenberg-Marquardt with per-sample damping and a gain ratio: a
    candidate is kept where the residual falls by a fair fraction of what the
    linear model promised, and the damping relaxes there; elsewhere the
    configuration stands and the damping stiffens. The gain ratio is not a
    detail — comparing two residual values alone lets floating-point noise accept
    steps that change nothing, which relaxes the damping, which enlarges the next
    step, which also changes nothing: a stall that looks like progress. Every
    configuration of a batch is therefore its own solver, with its own damping and
    its own convergence, and the whole thing is one tensor computation per
    iteration — no loop over targets.

    {2:trustregion The trust region is not optional}

    A pose inside the workspace usually has a solution that winds a joint through
    several turns. Undamped Gauss-Newton finds it happily: it is a correct answer
    and a useless trajectory, and a rendered demo shows it at once where a
    residual does not. [step_limit] caps the largest single-coordinate step, and
    [lower]/[upper] project onto the joint limits.

    {2:convergence What this does not do}

    There is no line search, no restart, no nullspace projection and no
    hierarchy. Those are the research questions §4.5 of the design note lists;
    what is here is the part that is not a question, and {!iterate} is the hook
    for a caller that wants to answer them.

    In particular this is a {e local} solver: pose IK has local minima of the
    pose error, and a start far outside the target's basin will settle into one.
    In the tests' six-joint arm, starts within about a radian all converge, and
    around 1.7 rad roughly a third do not — no damping schedule changes that,
    and a restart is the caller's decision to make. *)

(** {1:state State} *)

(** The type for a solver state: a configuration per sample and the damping
    that produced it. *)
type state

(** [init ?lambda ~q ()] is the state at [q] with damping [lambda] (default
    [1e-3]). [q] has shape [batch_shape @ [|dof|]] and [lambda] is a scalar or
    has shape [batch_shape]; it is broadcast. *)
val init : ?lambda:float -> q:Nx.float32_t -> unit -> state

(** [q s] is the configuration of [s], of shape [batch_shape @ [|dof|]]. *)
val q : state -> Nx.float32_t

(** [lambda s] is the damping of [s], of shape [batch_shape]. *)
val lambda : state -> Nx.float32_t

(** [iterations s] is the number of steps taken since {!init}. *)
val iterations : state -> int

(** {1:steps Steps} *)

(** [iterate ?step_limit ?lower ?upper task s] is one Levenberg-Marquardt step
    on [task] from [s].

    [step_limit] (default [0.5]) caps the largest single-coordinate change, in
    the coordinates' units. [lower] and [upper] are joint limits of shape
    [|dof|] or [batch_shape @ [|dof|]]; when given, the candidate is projected
    onto them. A step that does not lower [task]'s residual is rejected for its
    sample and the damping is raised instead, so the sequence of residuals is
    non-increasing. *)
val iterate
  :  ?step_limit:float
  -> ?lower:Nx.float32_t
  -> ?upper:Nx.float32_t
  -> Task.t
  -> state
  -> state

(** {1:solving Solving} *)

(** The result of {!solve}. *)
type result =
  { q : Nx.float32_t (** the configuration, of shape [batch_shape @ [|dof|]] *)
  ; residual : Nx.float32_t (** the residual's norm at [q], of shape [batch_shape] *)
  ; iterations : int (** steps taken for the whole batch *)
  ; lambda : Nx.float32_t (** the damping reached, of shape [batch_shape] *)
  ; converged : Nx.bool_t
    (** where the residual norm fell below the tolerance, of shape
        [batch_shape] *)
  }

(** [solve ?max_iter ?tolerance ?lambda ?step_limit ?lower ?upper task ~q] runs
    {!iterate} until every sample has converged or [max_iter] steps have been
    taken.

    [max_iter] defaults to [100], [tolerance] to [1e-6], and the damping,
    step limit and limits are {!init}/{!iterate}'s. This is the eager driver: it
    reads whether the batch has converged on the host, so it must not be called
    inside a traced function — use {!iterate} there, in a loop of the caller's
    choosing. *)
val solve
  :  ?max_iter:int
  -> ?tolerance:float
  -> ?lambda:float
  -> ?step_limit:float
  -> ?lower:Nx.float32_t
  -> ?upper:Nx.float32_t
  -> Task.t
  -> q:Nx.float32_t
  -> result
