(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Passive forces: joint springs and joint dampers.

    A hinge or slide joint pulls its position towards [qpos_spring] with its
    stiffness, and every degree of freedom resists its velocity with its
    damping, both with the sign MuJoCo gives them: the force is
    [-k (q - q_spring)] and [-b v]. A ball or free joint's spring is not
    implemented in v1; its [qpos_spring] would need a quaternion difference,
    and no model in the corpus uses one. Damping is implemented for every
    joint type, because it is a per-dof coefficient.

    Fluid forces, gravity compensation and adhesion are out of v1
    (`MILESTONES.md` D12). *)

(** {1:stage The stage} *)

(** [forward m d] is [d] with [qfrc_passive] computed from [d.qpos] and
    [d.qvel]. *)
val forward : Model.host -> Data.host -> Data.host
