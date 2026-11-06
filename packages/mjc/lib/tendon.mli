(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Fixed tendons.

    A fixed tendon is a linear combination of joint coordinates: its length is
    [Σ coef * qpos] over the joints it wraps, so its Jacobian is a dense
    [[ntendon; nv]] matrix with one coefficient per wrapped degree of freedom,
    and its velocity is [J qvel]. That is the whole of it for the fixed case;
    the spatial case — sites, geometric wrapping, pulleys — is a path
    computation with its own unknowns, and v1 does not have it.

    The Jacobian is a tensor because it is what an actuator's moment arm
    reads, and a tendon actuator's gear multiplies it. *)

(** The type for a model's tendons.

    [jac] is [[ntendon;nv]], [length] and [velocity] are [[ntendon]]. *)
type 'a t =
  { jac : 'a Nx.float64_t
  ; length : 'a Nx.float64_t
  ; velocity : 'a Nx.float64_t
  }

(** [forward m d] is the tendon geometry at [d]'s configuration.

    Raises [Invalid_argument] for what v1 does not have: a site or geometric
    wrap, and a tendon whose range is a limit row (that arrives with the
    tendon constraint, and until it does a model that asks for it must not be
    run as if it had no limit). *)
val forward : 'a Model.model -> 'a Data.data -> 'a t
