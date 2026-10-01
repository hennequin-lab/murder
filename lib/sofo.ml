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
let sum = Observe.sum
let softmax_ce = Observe.softmax_ce

type 'p sketch = 'p Sketch.t =
  { k : int
  ; loss : Nx.float64_t
  ; c : Nx.float64_t
  ; ggn : Nx.float64_t
  ; dirs : 'p
  ; apply : Nx.float64_t -> 'p
  }

let sketch = Sketch.run

module Optim = Optim
