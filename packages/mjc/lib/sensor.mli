(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sensors.

    A sensor writes a fixed-width block of numbers at an address the loader
    assigned, so the whole block is one tensor and each sensor is a slice of
    it. The families v1 has are projections of state that the forward pass
    already computed — a joint's coordinate or velocity, a frame's pose or
    velocity, an actuator's force, a tendon's length or velocity, a subtree's
    centre of mass, velocity or angular momentum — which is why no sensor here
    needs a stage of its own.

    Refused by name, because they need machinery v1 does not have: a reference
    frame ([reftype]/[refid], which is a relative pose computation), the kinds
    that read the acceleration stage (gyro, accelerometer, velocimeter), the
    ones that need a ray-geometry query (touch, rangefinder, contact and
    tactile), and [framequat], whose matrix-to-quaternion convention has not
    been settled against the engine — a wrong quaternion is a wrong
    orientation, not a rounding difference. *)

(** [forward m d] is [d] with {!Data.sensordata} filled.

    Raises [Invalid_argument] for a sensor kind or an option v1 does not have,
    naming the sensor. *)
val forward : Model.host -> Data.host -> Data.host
