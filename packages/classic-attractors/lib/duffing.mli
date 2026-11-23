(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The periodically driven Duffing oscillator.

    {v
      x' = v
      v' = -delta v - alpha x - beta x^3 + gamma cos (omega t)
    v}

    with [delta = 0.02], [alpha = -1], [beta = 1], [gamma = 0.5] and
    [omega = 1]: a softening spring kicked at resonance, chaotic at these
    parameters. The field is not autonomous, so the attractor is the set the
    stroboscopic map at the drive period draws; a long run crosses its lobes.
    The flows of `Attractor.S.simulate` start uniformly in
    [[-2, 2] x [-2, 2]]; `Attractor.S.snippets` starts from [(0, 0)]. A step
    of [dt = 0.01] resolves the drive, and a transient of about 100 time units
    lands on the attractor. *)

include Attractor.S
