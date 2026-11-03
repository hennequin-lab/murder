(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Integration: one step of the semi-implicit Euler scheme.

    The order is MuJoCo's [mj_advance]: the velocity is advanced with the
    acceleration the forward pass just computed, then the position with the
    new velocity, then the clock. A model whose joints have no limits, no
    contacts and no equalities has no constraints, so its [qacc] is the whole
    answer and this is [mj_step]. *)

(** {1:step Step} *)

(** [advance m d] is [d] one time step forward in time: [qvel] gains
    [dt * qacc], [qpos] then gains [dt] times the new [qvel] through
    {!Pose.integrate_pos}, and [time] gains [dt]. *)
val advance : Model.host -> Data.host -> Data.host

(** [step m d] is [advance m (Forward.forward m d)]. *)
val step : Model.host -> Data.host -> Data.host
