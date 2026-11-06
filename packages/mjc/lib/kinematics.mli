(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Forward kinematics: every frame in the model, from [qpos].

    The body pass walks bodies in topological order, which the topology
    guarantees: a body's parent has a smaller index. Each body composes its
    fixed offset in the parent frame, then applies its joints in order. A
    second pass places the inertial, geom and site frames from their body's
    pose.

    The arithmetic follows MuJoCo's [mj_kinematics] statement by statement,
    because the parities this milestone gates on are bit-level differences:
    the same expressions, in the same order, with joints applied on the right
    of the accumulated rotation and anchors stored before their joint's own
    rotation. A model with a free joint takes its pose straight from [qpos]
    and ignores the body's fixed offset, as MuJoCo does.

    The [sameframe] shortcuts MuJoCo uses for geoms and sites whose offset is
    zero and whose rotation is the identity are not reproduced: the general
    expression gives the same value, because multiplying by an exact zero and
    a unit quaternion is exact. *)

(** {1:forward Forward kinematics} *)

(** [forward m d] is [d] with [xpos xquat xanchor xaxis xipos ximat geom_xpos geom_xmat site_xpos site_xmat] computed from [d.qpos]. The body pass reads
    [m.body_pos body_quat jnt_axis jnt_pos jnt_type qpos0] and [d.qpos]; the
    second pass reads [m.body_ipos body_iquat geom_pos geom_quat site_pos site_quat].

    Leaves that a later stage computes ([cvel], [com], the force vectors,
    [M]) are left as they were. *)
val forward : 'a Model.model -> 'a Data.data -> 'a Data.data

(** {1:checks Checks} *)

(** [check m] is [Error message] when the model has a kinematic feature this
    implementation does not handle, and [Ok ()] otherwise. The only such
    feature in v1 is a free joint on a body whose parent is not the world.
    {!Model.validate} catches the malformed cases; this catches the ones that
    are legal in MuJoCo and out of scope here. *)
val check : 'a Model.model -> (unit, string) result
