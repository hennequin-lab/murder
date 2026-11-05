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
  }

(** [default] is a signal sized for the gripper demo: amplitude 1.6 and a period
    of 2.4 s, which closes the hand over about half a second and opens it
    again — fast enough to watch, slow enough to see the fingers. *)
val default : t

(** [ctrl t ~nu ~time] is [nu] controls at [time]: every actuator follows
    [amplitude * sin (2 pi time / period)], so a mirrored pair moves together. *)
val ctrl : t -> nu:int -> time:float -> float array
