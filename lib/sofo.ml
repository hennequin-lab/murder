(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The public surface: curvature descriptions, little losses, and the sketch
   driver. The observation effect, the collector and the driver's internals
   stay private — a user describes losses and reads sketches, and a library
   that needs different accumulation can add it here. *)

module Curv = Curv

let observe = Observe.observe
let mse = Observe.mse
let softmax_ce = Observe.softmax_ce

type diagnostics = Sketch.diagnostics =
  { observed_loss : Nx.float64_t
  ; observed_c : Nx.float64_t
  ; blocks : int
  ; skipped : int
  }

type 'p sketch = 'p Sketch.t =
  { k : int
  ; loss : Nx.float64_t
  ; c : Nx.float64_t
  ; ggn : Nx.float64_t
  ; apply : Nx.float64_t -> 'p
  ; diagnostics : diagnostics
  }

let sketch = Sketch.run
let check = Sketch.check

module Optim = Optim
