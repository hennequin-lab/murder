(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx

let a = 1.89

let field _t y =
  let open Infix in
  let x = slice [ A; I 0 ] y in
  let yc = slice [ A; I 1 ] y in
  let z = slice [ A; I 2 ] y in
  let dx = -((x *$ a) + (yc *$ 4.) + (z *$ 4.) + (yc * yc)) in
  let dy = -((yc *$ a) + (z *$ 4.) + (x *$ 4.) + (z * z)) in
  let dz = -((z *$ a) + (x *$ 4.) + (yc *$ 4.) + (x * x)) in
  stack ~axis:1 [ dx; dy; dz ]

include Driver.Make (struct
    let name = "Halvorsen"
    let dim = 3
    let bounds = [| -3.5, 3.5; -3.5, 3.5; -3.5, 3.5 |]
    let init = [| -1.48; -1.51; 2.04 |]
    let field = field
  end)
