(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  { page : Hugin_vg.Renderable.t
  ; density : float
  }

let create ?(density = 1.) ~width ~height picture =
  { page = Hugin_vg.Renderable.v width height picture; density }

let rgba t = Hugin_vg_raster.render ~density:t.density t.page
let png t = Hugin_vg_raster.png ~density:t.density t.page
let svg t = Hugin_vg_svg.render t.page
let pdf t = Hugin_vg_pdf.render t.page

let write path data =
  let oc = open_out_bin path in
  Fun.protect ~finally:(fun () -> close_out oc) (fun () -> output_string oc data)

let save_png t path = write path (png t)
let save_svg t path = write path (svg t)
let save_pdf t path = write path (pdf t)
