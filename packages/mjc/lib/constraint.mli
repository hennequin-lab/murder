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

(** {1:structure Structure}

    What decides which rows exist and which objects they couple, as opposed to
    what their numbers are: the cone option, the timestep the refsafe clamp reads,
    and the equality table. A traced program cannot read those out of its tensors
    ([Nx.item] returns an OCaml value), so they are read once, on the host, and
    carried -- the same structure/parameters split as {!Step.config}. *)
type config

(** [config m] is [m]'s structure, read from it once. *)
val config : 'a Model.model -> config

(** [equality m d] writes the rows of every active equality constraint, from
    row 0, and sets [nefc] to their number. It is the first stage to run: the
    limit and contact stages append after it. *)
val equality : config -> 'a Model.model -> 'a Data.data -> 'a Data.data

(** [limits m d] appends one row per active joint limit after the rows already
    in the buffer, and adds their count to [nefc]. It reads [d.qpos] and the
    model's ranges, margins and limit flags. A row that does not fit in the
    model's [nefcmax] budget raises through [Nx.check] rather than being
    dropped silently. *)
val limits : 'a Model.model -> 'a Data.data -> 'a Data.data

(** [friction m d] appends one row per dof whose friction loss is non-zero,
    after the rows already in the buffer. The force on such a row is bounded
    by the loss, so the solver treats it as linear outside the bound. *)
val friction : config -> 'a Model.model -> 'a Data.data -> 'a Data.data

(** [contact m d] appends one row per contact, after the rows already in the
    buffer. v1 computes frictionless contacts only, so a model whose geoms ask
    for one with friction is refused here rather than solved with rows that do
    not match MuJoCo's. *)
val contact : config -> 'a Model.model -> 'a Data.data -> 'a Data.data

(** {1:impedance Impedance} *)

(** [impedance m d] is [d] with [efc_vel], [efc_R], [efc_D] and [efc_aref]
    computed for the assembled rows, following [mj_constraintUpdate]'s
    classical (non-discrete) path:

    {v
      efc_vel = J qvel
      R       = max(1e-15, (1 - imp) * diagA / imp),  diagA = dof_invweight0
      D       = 1 / R
      aref    = -B * efc_vel - K * I * (efc_pos - efc_margin)
    v}

    [imp] and its derivative come from the row's [solimp] evaluated at its
    position, and [K], [B] from its [solref], with MuJoCo's refsafe clamp
    [solref.0 >= 2 dt] and its standard-versus-direct reference rule. Rows
    beyond [nefc] are given [R = 1] and [D = 0] so that a fixed-size system
    stays positive definite. *)
val tendon_limits : 'a Model.model -> 'a Data.data -> 'a Data.data

val impedance : ?dt:float -> config -> 'a Model.model -> 'a Data.data -> 'a Data.data
