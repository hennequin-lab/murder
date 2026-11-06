(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The velocity stage: spatial velocities, motion-subspace derivatives and
    the bias force.

    The body pass is MuJoCo's [mj_comVel]: a body's spatial velocity is its
    parent's plus its own degrees of freedom, and each degree of freedom's
    motion axis changes at the rate the accumulated velocity rotates it. The
    bias force is the abbreviation of the recursive Newton–Euler algorithm
    that MuJoCo calls with no joint acceleration ([mj_rne] with [flg_acc] 0):
    it carries Coriolis, centrifugal and gravity forces, and nothing else. *)

(** {1:stage The velocity stage} *)

(** [forward m d] is [d] with [cvel], [cdof_dot] and [qfrc_bias] computed. It
    needs [d.cinert] and [d.cdof] from {!Mass.forward}, and the poses from
    {!Kinematics.forward}. Gravity enters through the world body's
    acceleration, as in the C engine, so [qfrc_bias] contains it. *)
val forward : 'a Model.model -> 'a Data.data -> 'a Data.data
