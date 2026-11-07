(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A scripted control signal, so a demo moves without a keyboard.

    V1's controls are forward-kinematics tools: they write [qpos] and the model
    follows. That says nothing about actuators, and an actuator demo that needs
    a human to hold a key cannot be photographed. This is the smallest thing
    that replaces the human: a sine, applied to every actuator, which is what
    makes a pair of mirrored tendons open and close together. *)

(** The type for a scripted signal. *)
type t =
  { amplitude : float (** The control at a quarter period, in the model's units. *)
  ; period : float (** Seconds for a full cycle. *)
  ; offset : float
    (** The pose the signal is added to. A gripper wants 0; a legged model
        wants the stance its servos should hold, with the sine read as a
        wiggle about it. *)
  ; pattern : float array
    (** Per-actuator weights, cycled; empty means every actuator. A legged
        model wants its knees driven and not its hips, which no amplitude can
        express. *)
  ; phase_step : float
    (** Radians of phase added per actuator. Every actuator in step hops a
        legged model off the floor; alternating it (pi) rocks the body with
        the feet planted. *)
  }

(** [default] is a signal sized for the gripper demo: amplitude 1.2 and a period
    of 0.9 s. The hand closes in about a fifth of a second, holds, opens again,
    and repeats — fast enough to read as a gripper, slow enough to see the
    links. *)
val default : t

(** [make ~amplitude ~period ~offset] is the same signal with the caller's
    numbers, which is what the viewer's [--drive-*] flags build. *)
val make
  :  amplitude:float
  -> period:float
  -> offset:float
  -> phase_step:float
  -> pattern:float array
  -> t

(** [ctrl t ~nu ~time] is [nu] controls at [time]: actuator [i] follows
    [offset + amplitude * sin (2 pi time / period + phase_step * i)], so a
    mirrored pair moves together and a legged model can be driven in antiphase. *)
val ctrl : t -> nu:int -> time:float -> float array
