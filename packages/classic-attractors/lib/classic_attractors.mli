(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Classic low-dimensional continuous-time attractors.

    Each module is an {!S} over its own vector field:
    {!S.simulate} draws independent initial conditions and {!S.snippets} cuts a long run into shuffled chunks. Both return
    [[n_steps; n; dim]] host tensors of the requested dtype, integrated with
    Jera's classical Runge-Kutta at one step per [dt]. *)

(** The interface every attractor implements. *)
module type S = Attractor.S

(** The Lorenz (1963) butterfly. *)
module Lorenz : S

(** The Rössler spiral. *)
module Rossler : S

(** Thomas' cyclically symmetric lattice. *)
module Thomas : S

(** The Halvorsen three-lobed attractor. *)
module Halvorsen : S

(** The Aizawa sphere-with-a-tube attractor. *)
module Aizawa : S

(** The periodically driven Duffing oscillator. *)
module Duffing : S
