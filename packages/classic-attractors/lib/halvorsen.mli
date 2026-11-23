(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Halvorsen attractor.

    {v
      x' = -a x - 4 y - 4 z - y^2
      y' = -a y - 4 z - 4 x - z^2
      z' = -a z - 4 x - 4 y - x^2
    v}

    with [a = 1.89]: a cyclically symmetric attractor of dimension about 2.2,
    with three spiralling lobes. The flows of `Attractor.S.simulate` start
    uniformly in [[-3.5, 3.5]]^3, inside the basin; `Attractor.S.snippets`
    starts from [(-1.48, -1.51, 2.04)]. A step of [dt = 0.005] resolves the
    flow (the coefficients are four times the Lorenz ones), and a transient of
    about 20 time units lands on the attractor. *)

include Attractor.S
