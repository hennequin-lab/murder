(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx

let a = 0.95
let b = 0.7
let c = 0.6
let d = 3.5
let e = 0.25
let f = 0.1

let field _t y =
  let open Infix in
  let x = slice [ A; I 0 ] y in
  let yc = slice [ A; I 1 ] y in
  let z = slice [ A; I 2 ] y in
  let dx = ((z -$ b) * x) - (yc *$ d) in
  let dy = (x *$ d) + ((z -$ b) * yc) in
  let r2 = (x * x) + (yc * yc) in
  let dz =
    (z *$ a)
    +$ c
    - (z * z * z *$ (1. /. 3.))
    - (r2 * ((z *$ e) +$ 1.))
    + (z *$ f * (x * x * x))
  in
  stack ~axis:1 [ dx; dy; dz ]

include Driver.Make (struct
    let name = "Aizawa"
    let dim = 3
    let bounds = [| -1.5, 1.5; -1.5, 1.5; -0.5, 2. |]
    let init = [| 0.1; 0.; 0. |]
    let field = field
  end)
