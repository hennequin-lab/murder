(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Geom
open Primitive

(* Camera coordinates are +x right, +y up and +z toward the viewer. The
   camera looks down -z and the image plane is z = -focal. The rows of the
   world -> camera rotation are [right], [up] and [back]. *)

type t =
  { eye : p3
  ; right : p3
  ; up : p3
  ; back : p3
  ; focal : float
  ; width : float
  ; height : float
  ; near : float
  ; far : float
  ; ortho : bool
  ; ortho_scale : float
  ; view : Motor.t
  ; view_inv : Motor.t
  }

let create
      ?(ortho = false)
      ?(near = 0.1)
      ?(far = 1000.)
      ?ortho_scale
      ~eye
      ~target
      ~up
      ~fov_y
      ~width
      ~height
      ()
  =
  if (not (Float.is_finite fov_y)) || fov_y <= 0. || fov_y >= Float.pi
  then invalid_arg "Camera.create: fov_y must be in (0, pi)";
  if
    (not (Float.is_finite width))
    || (not (Float.is_finite height))
    || width <= 0.
    || height <= 0.
  then invalid_arg "Camera.create: width and height must be finite and positive";
  if
    (not (Float.is_finite near))
    || (not (Float.is_finite far))
    || near <= 0.
    || far <= near
  then invalid_arg "Camera.create: expected finite 0 < near < far";
  let direction = sub target eye in
  if norm direction < 1e-9 then invalid_arg "Camera.create: eye and target coincide";
  let forward = normalize direction in
  let back = scale forward (-1.) in
  let right0 = cross up back in
  if norm right0 < 1e-9
  then invalid_arg "Camera.create: up and the view direction are parallel";
  let right = normalize right0 in
  (* The second camera axis is the cross product, so it is already unit. *)
  let up_cam = cross back right in
  let focal = height /. 2. /. tan (fov_y /. 2.) in
  let ortho_scale =
    match ortho_scale with
    | Some s when Float.is_finite s && s > 0. -> s
    | Some _ -> invalid_arg "Camera.create: ortho_scale must be finite and positive"
    | None -> height /. 4.
  in
  let rows =
    Nx.create
      Nx.float32
      [| 3; 3 |]
      [| right.x
       ; right.y
       ; right.z
       ; up_cam.x
       ; up_cam.y
       ; up_cam.z
       ; back.x
       ; back.y
       ; back.z
      |]
  in
  let rotate = Motor.of_rotation rows in
  let translate =
    Motor.translate3 (-.dot right eye) (-.dot up_cam eye) (-.dot back eye)
  in
  let view = Motor.mul translate rotate in
  let view_inv = Motor.inverse view in
  { eye
  ; right
  ; up = up_cam
  ; back
  ; focal
  ; width
  ; height
  ; near
  ; far
  ; ortho
  ; ortho_scale
  ; view
  ; view_inv
  }

let eye c = c.eye
let right c = c.right
let up c = c.up
let forward c = scale c.back (-1.)
let focal c = c.focal
let width c = c.width
let height c = c.height
let near c = c.near
let far c = c.far
let is_orthographic c = c.ortho
let ortho_scale c = c.ortho_scale
let view c = c.view
let eye_point c = Pga.point3 c.eye.x c.eye.y c.eye.z
let image_plane_local c = Pga.plane3 0. 0. 1. (-.c.focal)
let image_plane c = Pga.apply_motor c.view_inv (image_plane_local c)
let eye_local = Pga.point3 0. 0. 0.

let ray c ~x ~y =
  let local =
    if c.ortho
    then (
      (* Parallel to the view axis through the pixel's image point. The far
         point comes first so that [Pga.line_direction] points into the
         scene. *)
      let s = c.ortho_scale in
      let xc = (x -. (c.width /. 2.)) /. s in
      let yc = ((c.height /. 2.) -. y) /. s in
      Pga.line_through (Pga.point3 xc yc (-1.)) (Pga.point3 xc yc 0.))
    else (
      (* The line through the eye and the pixel's image point. *)
      let xi = x -. (c.width /. 2.) in
      let yi = (c.height /. 2.) -. y in
      let image = Pga.point3 xi yi (-.c.focal) in
      Pga.line_through image eye_local)
  in
  Pga.apply_motor c.view_inv local

(* The camera-space coordinates are the coordinates used by the view motor.
   Keeping a scalar copy avoids two tensor allocations per rendered point in
   the fast path. *)
let to_camera c p =
  let q = sub p c.eye in
  dot c.right q, dot c.up q, dot c.back q

let in_range c zc = zc < -.c.near && zc > -.c.far
let page_of_image c (x, y, depth) = (c.width /. 2.) +. x, (c.height /. 2.) -. y, depth

(* The fast evaluation of the same projection: a pinhole division for the
   perspective camera, a uniform scale for the orthographic one. *)
let project_image_pinhole c p =
  let xc, yc, zc = to_camera c p in
  if not (in_range c zc)
  then None
  else if c.ortho
  then Some (c.ortho_scale *. xc, c.ortho_scale *. yc, -.zc)
  else Some (c.focal *. xc /. -.zc, c.focal *. yc /. -.zc, -.zc)

(* The reference projection: the meet of the view ray through the point with
   the image plane. This is the construction documented in VIZ.md, section
   3.2. The image coordinates fall out of [Pga.point_coords], which performs
   the perspective division. *)
let project_image_meet c p =
  if c.ortho
  then project_image_pinhole c p
  else (
    (* Cull and report depth in the same float64 coordinates as the fast
       path; only the image coordinates come from the meet. *)
    let _xc, _yc, zc = to_camera c p in
    if not (in_range c zc)
    then None
    else (
      let pcam = Pga.apply_motor c.view (Pga.point3 p.x p.y p.z) in
      let hit = Pga.meet (Pga.join eye_local pcam) (image_plane_local c) in
      let q = Pga.point_coords hit in
      Some (Nx.item [ 0 ] q, Nx.item [ 1 ] q, -.zc)))

let project_image c p = project_image_meet c p

let project c p =
  match project_image c p with
  | None -> None
  | Some image -> Some (page_of_image c image)

let project_pinhole c p =
  match project_image_pinhole c p with
  | None -> None
  | Some image -> Some (page_of_image c image)
