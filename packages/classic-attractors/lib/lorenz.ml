(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx

let sigma = 10.
let rho = 28.
let beta = 8. /. 3.

let field _t y =
  let open Infix in
  let x = slice [ A; I 0 ] y in
  let yc = slice [ A; I 1 ] y in
  let z = slice [ A; I 2 ] y in
  let dx = (yc - x) *$ sigma in
  let dy = (x * (-z +$ rho)) - yc in
  let dz = (x * yc) - (z *$ beta) in
  stack ~axis:1 [ dx; dy; dz ]

include Driver.Make (struct
    let name = "Lorenz"
    let dim = 3
    let bounds = [| -20., 20.; -20., 20.; 0., 50. |]
    let init = [| 1.; 1.; 1. |]
    let field = field
  end)
