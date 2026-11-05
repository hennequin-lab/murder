(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Actuator forces.

    The engine's rule is short and worth writing down once: an actuator's
    output is [gain * input + bias], where the input is the control or, for a
    stateful actuator, its activation; the gain and bias are polynomials in
    the transmission's length and velocity; and the generalized force is the
    moment matrix applied to that scalar, [qfrc_actuator = momentᵀ force]. The
    types the lab's models use fall out of the same expression:

    - a {e motor} has a fixed gain of one and no bias, so [force = ctrl];
    - a {e position} servo has a fixed gain [kp] and an affine bias
      [(0, -kp, -kv)], so [force = kp (ctrl - length) - kv velocity];
    - a {e velocity} servo is the same with [gain = kv] and bias [(0, 0, -kv)];
    - an {e intvelocity} servo is a position servo whose setpoint is the
      integral of the control: its dynamics are [act' = ctrl], the input is
      the activation, and the same affine bias closes the loop;
    - a {e damper} has zero gain and a bias [(0, 0, -kv)].

    Control and force ranges are clamps, not rows: they scale nothing, they
    bound the value. *)

(** [forward m d] is [d] with its actuator state advanced one evaluation:
    {!Data.actuator_length}, {!Data.actuator_velocity},
    {!Data.actuator_force}, {!Data.qfrc_actuator} and {!Data.act_dot} are set
    from [d]'s [qpos], [qvel], [ctrl] and [act].

    Raises [Invalid_argument] for an actuator kind v1 does not have, naming
    it, rather than silently producing a zero force. *)
val forward : Model.host -> Data.host -> Data.host

(** [damping m d] is [[nu]]: the derivative of each actuator's force with
    respect to its transmission velocity, [biasprm.[2] + gainprm.[2] * input].
    The implicit integrator needs it because a velocity servo's [kv] is a
    damping term by another name, and integrating it explicitly is the
    difference between a servo that tracks and one that rings. *)
val damping : ?force_limited:bool -> Model.host -> Data.host -> Nx.host Nx.float64_t

(** [qfrc m d] is the generalized actuator force alone, without touching [d].
    It is what a test compares against the engine's [qfrc_actuator]. *)
val qfrc : Model.host -> Data.host -> Nx.host Nx.float64_t
