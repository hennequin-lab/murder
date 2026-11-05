(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Transmission geometry: what an actuator drives, and how fast.

    MuJoCo describes a transmission with a moment arm per degree of freedom,
    and derives the length and velocity from it: [velocity = moment · qvel],
    and for the scalar transmissions v1 has, [length] is the joint coordinate
    or the tendon length scaled by the same gear. The moment matrix is dense
    here — [nu] and [nv] are small, and a sparse pattern is a second
    representation to keep in step for no gain at these sizes.

    The moment arm is a parameter, not geometry: [gear] is a leaf of
    {!Model.Params}, so a differentiated step sees the transmission. *)

(** The type for a model's transmissions.

    [moment] is [[nu;nv]]: row [i] is the arm actuator [i] has on each degree
    of freedom. [length] and [velocity] are [[nu]]. *)
type t =
  { moment : Nx.host Nx.float64_t
  ; length : Nx.host Nx.float64_t
  ; velocity : Nx.host Nx.float64_t
  }

(** [forward m d] is the transmission geometry at [d]'s configuration.

    Raises [Invalid_argument] for a transmission v1 does not have: a tendon
    (the moment needs [ten_J], which arrives with {!Tendon}) and anything the
    loader did not already reject. *)
val forward : Model.host -> Data.host -> t

(** [pretty t] is the transmission, for a log line or a test failure. *)
val pp : Stdlib.Format.formatter -> t -> unit
