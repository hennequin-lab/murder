(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** One forward evaluation: poses, inertia, passive forces, the bias force
    and the smooth acceleration.

    This is [mj_forward] for a model with no constraints, which is what a
    model without contacts, limits or equalities has. It composes the stages
    rather than merging them, so a caller debugging a number can call them in
    turn and compare each with the C engine. *)

(** {1:forward Forward} *)

(** [forward m d] is [d] with every derived quantity computed from [d.qpos],
    [d.qvel] and the applied forces: the pose and the inertia of the position
    stage, [qfrc_passive], [cvel], [cdof_dot], [qfrc_bias], then

    {[
    qfrc_smooth = qfrc_passive - qfrc_bias + qfrc_applied + qfrc_actuator
    ]}

    and [qacc = mass_matrix^-1 qfrc_smooth].

    With constraints, M3's solver replaces [qacc]; without them it is the
    whole answer, as [qacc_smooth] is in the C engine. *)
val forward : Model.host -> Data.host -> Data.host
