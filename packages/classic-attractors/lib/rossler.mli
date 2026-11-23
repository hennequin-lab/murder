(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Rössler attractor.

    {v
      x' = -y - z
      y' = x + a y
      z' = b + z (x - c)
    v}

    with [a = 0.2], [b = 0.2] and [c = 5.7]: a single spiral band around a
    saddle focus, of dimension about 2.01. The flows of
    `Attractor.S.simulate` start uniformly in
    [[-8, 8] x [-8, 8] x [0, 5]]; `Attractor.S.snippets` starts from
    [(1, 1, 0)]. The basin is not the whole space, so the box stays inside it.
    A step of [dt = 0.02] resolves the flow well, and a transient of about 50
    time units lands on the attractor. *)

include Attractor.S
