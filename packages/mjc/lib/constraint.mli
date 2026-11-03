(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Constraint rows.

    Every constraint is a row of one set, with a Jacobian, a position, the
    distance at which it activates, and a type that says how to project its
    force. v1 assembles joint limits: a hinge or slide joint whose position
    leaves its range contributes one row per violated side, in the order
    MuJoCo uses (lower before upper, joints in id order). Ball and free joint
    limits, equality, friction loss and contact arrive later.

    The rows are written to a fixed-size buffer and their count is a tensor,
    because which limits are violated is data. Rows that are not active are
    scattered past the buffer and dropped, so the layout is the same whether
    a step has one violated limit or none. *)

(** {1:rows Rows} *)

(** The type codes of [efc_type], matching MuJoCo's [mjtConstraint]. *)
val type_equality : int

val type_friction_dof : int
val type_friction_tendon : int
val type_limit_joint : int
val type_limit_tendon : int
val type_contact_frictionless : int
val type_contact_pyramidal : int

(** [limits m d] is [d] with the joint-limit rows assembled into [efc_type],
    [efc_id], [efc_pos], [efc_margin] and [efc_J], and [nefc] set to their
    count. It reads [d.qpos] and the model's ranges, margins and limit flags.
    A row that does not fit in the model's [nefcmax] budget raises through
    [Nx.check] rather than being dropped silently. *)
val limits : Model.host -> Data.host -> Data.host
