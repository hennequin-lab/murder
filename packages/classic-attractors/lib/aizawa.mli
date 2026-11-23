(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Aizawa attractor.

    {v
      x' = (z - b) x - d y
      y' = d x + (z - b) y
      z' = c + a z - z^3 / 3 - (x^2 + y^2) (1 + e z) + f z x^3
    v}

    with [a = 0.95], [b = 0.7], [c = 0.6], [d = 3.5], [e = 0.25] and
    [f = 0.1]: a sphere with a tube through it, with rotational symmetry in
    [x] and [y]. The flows of `Attractor.S.simulate` start uniformly in
    [[-1.5, 1.5] x [-1.5, 1.5] x [-0.5, 2]]; `Attractor.S.snippets` starts
    from [(0.1, 0, 0)]. A step of [dt = 0.01] resolves the flow well, and a
    transient of about 100 time units lands on the attractor. *)

include Attractor.S
