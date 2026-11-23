(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx

let a = 0.2
let b = 0.2
let c = 5.7

let field _t y =
  let open Infix in
  let x = slice [ A; I 0 ] y in
  let yc = slice [ A; I 1 ] y in
  let z = slice [ A; I 2 ] y in
  let dx = -yc - z in
  let dy = x + (yc *$ a) in
  let dz = (z * (x -$ c)) +$ b in
  stack ~axis:1 [ dx; dy; dz ]

include Driver.Make (struct
    let name = "Rossler"
    let dim = 3
    let bounds = [| -8., 8.; -8., 8.; 0., 5. |]
    let init = [| 1.; 1.; 0. |]
    let field = field
  end)
