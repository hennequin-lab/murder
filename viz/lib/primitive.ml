(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Geom

type p3 =
  { x : float
  ; y : float
  ; z : float
  }

let p3 x y z = { x; y; z }
let add a b = { x = a.x +. b.x; y = a.y +. b.y; z = a.z +. b.z }
let sub a b = { x = a.x -. b.x; y = a.y -. b.y; z = a.z -. b.z }
let scale a s = { x = a.x *. s; y = a.y *. s; z = a.z *. s }
let dot a b = (a.x *. b.x) +. (a.y *. b.y) +. (a.z *. b.z)

let cross a b =
  { x = (a.y *. b.z) -. (a.z *. b.y)
  ; y = (a.z *. b.x) -. (a.x *. b.z)
  ; z = (a.x *. b.y) -. (a.y *. b.x)
  }

let norm a = sqrt (dot a a)

let normalize a =
  let n = norm a in
  if n = 0. then a else scale a (1. /. n)

type circle =
  { center : p3
  ; normal : p3
  ; radius : float
  }

type sphere =
  { center : p3
  ; radius : float
  }

type plane =
  { normal : p3
  ; offset : float
  ; patch : p3 list
  }

type frame =
  { origin : p3
  ; x : p3
  ; y : p3
  ; z : p3
  }

type t =
  | Point of p3
  | Segment of p3 * p3
  | Polyline of p3 list
  | Polygon of p3 list
  | Circle of circle
  | Sphere of sphere
  | Plane of plane
  | Frame of frame

let transform_point (m : Motor.t) (p : p3) =
  let x = Cga.point3 p.x p.y p.z in
  let y = Cga.apply_motor m x in
  let a = Cga.coords y |> Nx.to_array in
  { x = a.(0); y = a.(1); z = a.(2) }

let transform (m : Motor.t) = function
  | Point p -> Point (transform_point m p)
  | Segment (a, b) -> Segment (transform_point m a, transform_point m b)
  | Polyline ps -> Polyline (List.map (transform_point m) ps)
  | Polygon ps -> Polygon (List.map (transform_point m) ps)
  | Circle c ->
    let center = transform_point m c.center in
    let n = sub (transform_point m (add c.center c.normal)) center in
    Circle { center; normal = normalize n; radius = c.radius }
  | Sphere s -> Sphere { center = transform_point m s.center; radius = s.radius }
  | Plane p ->
    let patch = List.map (transform_point m) p.patch in
    let origin = p3 0. 0. 0. in
    let n = sub (transform_point m (add origin p.normal)) (transform_point m origin) in
    let offset = p.offset -. dot n origin in
    Plane { normal = normalize n; offset; patch }
  | Frame f ->
    Frame
      { origin = transform_point m f.origin
      ; x = transform_point m f.x
      ; y = transform_point m f.y
      ; z = transform_point m f.z
      }

let p3_of_tensor t =
  let a = Nx.to_array t in
  { x = a.(0); y = a.(1); z = a.(2) }

let scalar_of_tensor t = Nx.item [] t
let of_cga_point p = Point (p3_of_tensor (Cga.coords p))

let of_cga_sphere s =
  let center = p3_of_tensor (Cga.coords (Cga.sphere_center s)) in
  let radius = sqrt (scalar_of_tensor (Cga.sphere_radius2 s)) in
  Sphere { center; radius }

let plane_patch ~normal ~offset =
  let n = normalize normal in
  let a = if Float.abs n.z < 0.9 then p3 0. 0. 1. else p3 1. 0. 0. in
  let u = normalize (cross n a) in
  let v = cross n u in
  let c = scale n offset in
  [ add c (add (scale u (-2.)) (scale v (-2.)))
  ; add c (add (scale u 2.) (scale v (-2.)))
  ; add c (add (scale u 2.) (scale v 2.))
  ; add c (add (scale u (-2.)) (scale v 2.))
  ]

let of_cga_plane pi =
  let normal = p3_of_tensor (Cga.plane_normal pi) in
  let offset = scalar_of_tensor (Cga.plane_offset pi) in
  Plane { normal; offset; patch = plane_patch ~normal ~offset }

let of_cga_line ?(extent = 2.5) l =
  let d = p3_of_tensor (Cga.line_direction l) in
  let m = p3_of_tensor (Cga.line_moment l) in
  let c = cross d m in
  Segment (sub c (scale d extent), add c (scale d extent))

let of_cga_circle c =
  let center = p3_of_tensor (Cga.coords (Cga.circle_center c)) in
  let normal = p3_of_tensor (Cga.circle_normal c) in
  let radius = sqrt (scalar_of_tensor (Cga.circle_radius2 c)) in
  Circle { center; normal; radius }
