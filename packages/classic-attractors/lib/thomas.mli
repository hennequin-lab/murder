(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Thomas' cyclically symmetric attractor.

    {v
      x' = sin y - b x
      y' = sin z - b y
      z' = sin x - b z
    v}

    with [b = 0.19]: a lattice of lobes with a three-fold symmetry, chaotic at
    this damping (the system is a damped particle in a periodic potential, and
    the origin is its only fixed point). The flows of `Attractor.S.simulate`
    start uniformly in [[-4, 4]]^3; `Attractor.S.snippets` starts from
    [(1, 1, 1)]. A step of [dt = 0.02] resolves the flow well, and a transient
    of about 100 time units lands on the attractor. *)

include Attractor.S
