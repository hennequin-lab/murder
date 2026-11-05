(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Joint coordinates: the layout of [qpos] and [qvel], and position
    integration.

    A hinge or slide joint owns one [qpos] and one [qvel] entry; a ball joint
    owns a quaternion and three velocities; a free joint owns a translation, a
    quaternion and six velocities. The addresses come from
    {!Topo.t.jnt_qposadr} and {!Topo.t.jnt_dofadr}, which the converter fixes
    once.

    These functions update a whole [qpos] vector from a whole [qvel], one
    joint at a time, with tensor writes. The loop runs at trace time, so the
    number of writes is a property of the model and not of the data. *)

(** {1:integration Integration} *)

(** [integrate_pos t ~qpos ~qvel ~dt] is [qpos] advanced by [qvel] for [dt],
    the map MuJoCo calls [mj_integratePos]. A hinge or slide entry moves by
    [dt] times its velocity; a ball quaternion is advanced by the angular
    velocity with {!Rigid.quat_integrate}; a free joint translates its
    position and then advances its quaternion the same way.

    [qpos] has shape [[nq]] and [qvel] shape [[nv]]. *)
val integrate_pos
  :  Topo.t
  -> qpos:('s, 'd) Rigid.f
  -> qvel:('s, 'd) Rigid.f
  -> dt:float
  -> ('s, 'd) Rigid.f

(** [normalize_quat t qpos] is [qpos] with every ball and free quaternion
    normalized, the map MuJoCo calls [mj_normalizeQuat]. *)
val normalize_quat : Topo.t -> ('s, 'd) Rigid.f -> ('s, 'd) Rigid.f

(** {1:accessors Accessors} *)

(** [qpos_width t j] is the number of [qpos] entries joint [j] owns: 1 for a
    hinge or slide, 4 for a ball, 7 for a free joint. *)
val qpos_width : Topo.t -> int -> int

(** [dof_width t j] is the number of [qvel] entries joint [j] owns: 1, 3 or
    6. *)
val dof_width : Topo.t -> int -> int
