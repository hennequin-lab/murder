(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx

let b = 0.19

let field _t y =
  let open Infix in
  let x = slice [ A; I 0 ] y in
  let yc = slice [ A; I 1 ] y in
  let z = slice [ A; I 2 ] y in
  let dx = sin yc - (x *$ b) in
  let dy = sin z - (yc *$ b) in
  let dz = sin x - (z *$ b) in
  stack ~axis:1 [ dx; dy; dz ]

include Driver.Make (struct
    let name = "Thomas"
    let dim = 3
    let bounds = [| -4., 4.; -4., 4.; -4., 4. |]
    let init = [| 1.; 1.; 1. |]
    let field = field
  end)
