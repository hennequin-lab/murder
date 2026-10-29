(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Shared types. *)

(** A tensor at the model's element type: [float] payload, ['w] layout. Every
    function of this library is polymorphic in ['w] and works at any float
    dtype. A {e trial} is one such tensor of shape [[n_steps; n_output]]: the
    leading axis is time, the last the observation channels. A {e mini-batch}
    of trials is one tensor with the trials on leading axes,
    [[batch…; n_steps; n_output]]. *)
type 'w value = (float, 'w) Nx.t
