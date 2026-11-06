(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Levenberg-Marquardt damping, as a choice.

    A linearization's control Hessian [Q_uu] need not be positive definite, and
    the Riccati pass then has no gains to offer. Damping shifts the local
    model's value Hessian at every step to [V + mu I], which is Tassa's
    [Q_uu + mu BᵀB], [Q_ux + mu BᵀA] and [Q_xx + mu AᵀA] at once, and makes the
    pass stable as [mu] grows.

    A solve takes one of these values, and nothing else decides how it damps.
    The default is adaptive. *)

type t =
  | Off
  (** No damping. A pass whose [Q_uu] is not positive definite has no step
      to offer, and an iteration that cannot buy one is the solve's last. *)
  | Fixed of float
  (** One absolute strength, in the units of the value Hessian: cost per
      state squared. Nothing raises or lowers it, so an iteration that cannot
      buy a step is the last. [Fixed 0.] damps nothing, as {!Off} does. *)
  | Adaptive of Regularisation.t
  (** A schedule: the strength is raised when an iteration cannot buy a step
      and lowered after a full one. What the schedule holds is a *fraction*
      of each member's own value-Hessian scale, as the pass measures it, so
      the members of a batch whose curvature differs by orders of magnitude
      are damped alike; the first pass of a solve uses the schedule's own
      units. Failing to buy a step at any strength leaves the trajectory
      where it is. *)

(** [Adaptive Regularisation.init]: damped adaptively, from the schedule's own
    default, which is where the adaptive and the fixed solves both start. *)
val default : t
