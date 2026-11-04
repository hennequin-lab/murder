(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Joint control: the forward-kinematics debugging tool, without a window.

    V1 asks for "number keys select a joint, [[]/[]] move it, [0] zeroes it".
    The interesting part of that is not the keyboard: it is the write to
    [qpos] and the forward pass that makes the links follow, and it is worth
    testing without a display. So it lives here, as pure functions of a model
    and a state, and the window only maps keys onto them.

    A hinge and a slide have one position coordinate, so a keystroke means
    exactly one thing for them. A ball has four and a free joint seven, and
    which of those a key press should write is a question the keyboard cannot
    answer, so {!driven} is [false] for them and the window says so instead of
    guessing. *)

(** {1:joints Joints} *)

(** The type for the kinds of joint, which is the model's [jnt_type]. *)
type kind =
  | Hinge
  | Slide
  | Ball
  | Free

(** The type for a joint a viewer can select: what it is called, where it
    lives in [qpos], and what range it is allowed if it has one. *)
type joint =
  { index : int (** the joint's index in the model. *)
  ; name : string (** the model's name, or [""]. *)
  ; kind : kind
  ; qposadr : int (** the first address in [qpos]. *)
  ; limits : (float * float) option (** the range, when [jnt_limited]. *)
  }

(** [joints m] is every joint of [m], in index order. *)
val joints : Mjc.Model.host -> joint array

(** [driven j] is [true] iff [j] has a single position coordinate, so that
    moving it by a scalar is well defined. *)
val driven : joint -> bool

(** [step j] is the distance one keystroke moves [j]: about three degrees for
    a hinge, and one centimetre — in the model's units — for a slide. [0] for
    a joint that is not driven. *)
val step : joint -> float

(** [position d j] is [j]'s current position. It reads [qpos] whatever the
    joint's kind; for a ball or a free joint that is the first coordinate,
    which is a quaternion component or a translation, not an angle. *)
val position : Mjc.Data.host -> joint -> float

(** [label j] is how the joint is named in a HUD: its name, or its index when
    the model has none. *)
val label : joint -> string

(** [describe d j] is one line saying what [j] is and where it is now: its
    value and range for a driven joint, and the reason it is not driven
    otherwise. This is the line the window shows. *)
val describe : Mjc.Data.host -> joint -> string

(** {1:moving Moving} *)

(** [set m d j v] is [d] with [j]'s position written as [v], clamped into its
    range, and kinematics re-run so every frame follows. The second component
    is the value actually written, which differs from [v] when the joint is at
    its limit and the write was clamped.

    For a joint that is not {!driven} it is [d] and the position now, which is
    how a caller learns the write did not happen. *)
val set : Mjc.Model.host -> Mjc.Data.host -> joint -> float -> Mjc.Data.host * float

(** [nudge m d j k] is [set] at the current position plus [k * step j], so
    that [k] is [-1.] or [1.] for one keystroke and a caller can use other
    multipliers for a finer or coarser move. *)
val nudge : Mjc.Model.host -> Mjc.Data.host -> joint -> float -> Mjc.Data.host * float

(** [zero m d j] is [set] at the joint's reference position from [m.qpos0]. *)
val zero : Mjc.Model.host -> Mjc.Data.host -> joint -> Mjc.Data.host * float

(** {1:conversion Conversion} *)

(** [kind_of_jnt_type t] is the kind of a model's joint type. *)
val kind_of_jnt_type : Mjc.Topo.jnt_type -> kind
