(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Primitive

type t =
  { eye : p3
  ; r0 : p3
  ; r1 : p3
  ; r2 : p3 (* rows of the world -> camera rotation *)
  ; focal : float
  ; width : float
  ; height : float
  ; near : float
  ; far : float
  ; ortho : bool
  ; ortho_scale : float
  }

let create
      ?(ortho = false)
      ?(near = 0.1)
      ?(far = 1000.)
      ~eye
      ~target
      ~up
      ~fov_y
      ~width
      ~height
      ()
  =
  let forward = normalize (sub target eye) in
  let back = scale forward (-1.) in
  let right = normalize (cross up back) in
  let up_cam = cross back right in
  let r0 = right
  and r1 = up_cam
  and r2 = back in
  let focal = height /. 2. /. tan (fov_y /. 2.) in
  let ortho_scale = height /. 4. in
  { eye; r0; r1; r2; focal; width; height; near; far; ortho; ortho_scale }

let eye c = c.eye
let near c = c.near
let width c = c.width
let height c = c.height

let project c p =
  let q = sub p c.eye in
  let xc = dot c.r0 q
  and yc = dot c.r1 q
  and zc = dot c.r2 q in
  if zc >= -.c.near || zc <= -.c.far
  then None
  else if c.ortho
  then
    Some
      ( (c.width /. 2.) +. (c.ortho_scale *. xc)
      , (c.height /. 2.) -. (c.ortho_scale *. yc)
      , -.zc )
  else
    Some
      ( (c.width /. 2.) +. (c.focal *. xc /. -.zc)
      , (c.height /. 2.) -. (c.focal *. yc /. -.zc)
      , -.zc )

let project_point c p = project c p
