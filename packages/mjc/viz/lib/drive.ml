(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base

type t =
  { amplitude : float
  ; period : float
  ; offset : float
    (** The pose the sine is added to: 0 for a gripper, a stance for a leg. *)
  ; phase_step : float
    (** Radians of phase added per actuator. 0 moves them all together, which
        is what a mirrored pair of tendons wants; pi alternates them, which is
        what keeps a legged model on the floor instead of pogoing. *)
  }

let default = { amplitude = 1.2; period = 0.9; offset = 0.; phase_step = 0. }

let make ~amplitude ~period ~offset ~phase_step =
  { amplitude; period; offset; phase_step }

let ctrl (t : t) ~nu ~time =
  let w = 2. *. Float.pi /. t.period in
  Array.init nu ~f:(fun i ->
    t.offset
    +. (t.amplitude *. Float.sin ((w *. time) +. (t.phase_step *. Float.of_int i))))
