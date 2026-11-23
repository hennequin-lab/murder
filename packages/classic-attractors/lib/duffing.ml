(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx

let delta = 0.02
let alpha = -1.
let beta = 1.
let gamma = 0.5
let omega = 1.

(* The drive reads the time, so the field casts it to the state's dtype: the
   march's times are float64. *)
let field t y =
  let open Infix in
  let x = slice [ A; I 0 ] y in
  let v = slice [ A; I 1 ] y in
  let t = cast (dtype y) t in
  let drive = cos (t *$ omega) *$ gamma in
  let dx = v in
  let dv = (v *$ -.delta) - (x *$ alpha) - (x * x * x *$ beta) + drive in
  stack ~axis:1 [ dx; dv ]

include Driver.Make (struct
    let name = "Duffing"
    let dim = 2
    let bounds = [| -2., 2.; -2., 2. |]
    let init = [| 0.; 0. |]
    let field = field
  end)
