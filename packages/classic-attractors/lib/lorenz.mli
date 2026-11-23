(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Lorenz (1963) attractor.

    {v
      x' = sigma (y - x)
      y' = x (rho - z) - y
      z' = x y - beta z
    v}

    with [sigma = 10], [rho = 28] and [beta = 8/3], the parameters of the
    original paper: a chaotic attractor of dimension about 2.06, the classic
    butterfly. The flows of `Attractor.S.simulate` start uniformly in
    [[-20, 20] x [-20, 20] x [0, 50]]; `Attractor.S.snippets` starts from
    [(1, 1, 1)]. A step of [dt = 0.01] resolves the flow well, and a transient
    of about 10 time units lands on the attractor. *)

include Attractor.S
