(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The public surface: curvature descriptions, little losses, and the sketch
   driver. The mark's axis and total and the driver's internals stay private —
   a user describes losses and reads sketches. *)

module Curv = Curv

let observe = Observe.observe
let mse = Observe.mse
let sse = Observe.sse
let softmax_ce = Observe.softmax_ce

type ('p, 'd) sketch = ('p, 'd) Sketch.t =
  { k : int
  ; loss : 'd Nx.float64_t
  ; c : 'd Nx.float64_t
  ; ggn : 'd Nx.float64_t
  ; dirs : 'p
  ; apply : 'd Nx.float64_t -> 'p
  }

let sketch = Sketch.run

module Optim = Optim
