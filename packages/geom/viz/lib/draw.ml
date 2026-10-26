(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg
open Hugin_vg
open Primitive
module P = Primitive
module C = Camera

type color = Color.t

type style =
  { fill : color option
  ; stroke : color option
  ; width : float
  ; dash : float list
  ; opacity : float
  }

let default =
  { fill = Some (Color.v 0.25 0.52 0.82)
  ; stroke = Some (Color.v 0.10 0.22 0.36)
  ; width = 1.5
  ; dash = []
  ; opacity = 1.
  }

let solid ?(alpha = 1.) c = { default with fill = Some c; stroke = None; opacity = alpha }

let outline ?(width = 1.5) ?(dash = []) c =
  { default with fill = None; stroke = Some c; width; dash }

type entity =
  { primitive : P.t
  ; style : style
  ; layer : int
  ; motion : float -> P.t -> P.t
  }

type scene =
  { camera : C.t
  ; background : Color.t
  ; entities : entity list
  }

let static ?(layer = 1) ?(style = default) p =
  { primitive = p; style; layer; motion = (fun _ p -> p) }

let moving ?(layer = 1) ?(style = default) ~motion p =
  { primitive = p; style; layer; motion = (fun t p -> P.transform (motion t) p) }

let moving_with ?(layer = 1) ?(style = default) ~motion p =
  { primitive = p; style; layer; motion }

let create ?(background = Color.v 0.98 0.98 0.98) camera entities =
  { camera; background; entities }

let add e s = { s with entities = s.entities @ [ e ] }
let camera s = s.camera
let background s = s.background

let path_of_points ?(close = false) pts =
  let xs = Array.of_list (List.map (fun (p : P.p3) -> p.x) pts) in
  let ys = Array.of_list (List.map (fun (p : P.p3) -> p.y) pts) in
  if Array.length xs = 0
  then Path.empty
  else if close
  then Path.polygon xs ys
  else Path.polyline xs ys

let project_all c pts =
  let rec go acc = function
    | [] -> Some (List.rev acc)
    | p :: rest ->
      (match C.project c p with
       | None -> None
       | Some q -> go (q :: acc) rest)
  in
  go [] pts

let stroke_of_style s =
  if s.dash = [] then Stroke.v s.width else Stroke.v ~dash:s.dash s.width

let paint ?(close = false) s pts =
  let path = path_of_points ~close pts in
  if Path.is_empty path
  then Picture.empty
  else (
    let fill =
      match s.fill with
      | None -> Picture.empty
      | Some c -> Picture.fill c path
    in
    let stroke =
      match s.stroke with
      | None -> Picture.empty
      | Some c -> Picture.stroke (stroke_of_style s) c path
    in
    let p = Picture.group [ fill; stroke ] in
    if s.opacity < 1. then Picture.opacity s.opacity p else p)

let circle_points ~center ~normal ~radius =
  let n : P.p3 = P.normalize normal in
  let a = if Float.abs n.P.x < 0.9 then P.p3 1. 0. 0. else P.p3 0. 1. 0. in
  let u = P.normalize (P.cross n a) in
  let v = P.cross n u in
  let nseg = 96 in
  List.init nseg (fun i ->
    let t = 2. *. Float.pi *. float_of_int i /. float_of_int nseg in
    P.add center (P.add (P.scale u (radius *. cos t)) (P.scale v (radius *. sin t))))

let silhouette (c : C.t) (s : P.sphere) =
  let dvec = P.sub (C.eye c) s.center in
  let d = P.norm dvec in
  if d <= s.radius +. 1e-9
  then None
  else (
    let f = s.radius *. s.radius /. (d *. d) in
    let center = P.add s.center (P.scale dvec f) in
    let radius = s.radius *. sqrt (max 0. (1. -. f)) in
    Some (P.Circle { center; normal = dvec; radius }))

let representative_depth c = function
  | P.Point p -> C.project c p
  | P.Segment (a, b) -> C.project c (P.scale (P.add a b) 0.5)
  | P.Polyline ps | P.Polygon ps ->
    (match ps with
     | [] -> None
     | p :: _ ->
       let sum = List.fold_left P.add (P.p3 0. 0. 0.) ps in
       C.project c (P.scale sum (1. /. float_of_int (List.length ps))))
  | P.Circle c0 -> C.project c c0.center
  | P.Sphere s -> C.project c s.center
  | P.Plane pl ->
    (match pl.patch with
     | [] -> None
     | p :: _ ->
       let sum = List.fold_left P.add (P.p3 0. 0. 0.) pl.patch in
       C.project c (P.scale sum (1. /. float_of_int (List.length pl.patch))))
  | P.Frame f -> C.project c f.origin

let depth c prim =
  match representative_depth c prim with
  | None -> 0.
  | Some (_, _, d) ->
    (match prim with
     | P.Sphere s -> d -. s.radius
     | _ -> d)

let draw_circle c s center normal radius =
  match project_all c (circle_points ~center ~normal ~radius) with
  | None -> Picture.empty
  | Some qs -> paint ~close:true s (List.map (fun (x, y, _) -> P.p3 x y 0.) qs)

let draw_point c s p =
  match C.project c p with
  | None -> Picture.empty
  | Some (x, y, _) ->
    let radius = max 2.5 (s.width +. 1.) in
    let circle = Path.circle (P2.v x y) radius in
    let fill =
      match s.fill with
      | Some col -> Picture.fill col circle
      | None -> Picture.empty
    in
    let stroke =
      match s.stroke with
      | Some col -> Picture.stroke (Stroke.v 1.) col circle
      | None -> Picture.empty
    in
    let p = Picture.group [ fill; stroke ] in
    if s.opacity < 1. then Picture.opacity s.opacity p else p

let draw_polyline c s pts =
  match project_all c pts with
  | None -> Picture.empty
  | Some qs -> paint s (List.map (fun (x, y, _) -> P.p3 x y 0.) qs)

let draw_polygon c s pts =
  match project_all c pts with
  | None -> Picture.empty
  | Some qs -> paint ~close:true s (List.map (fun (x, y, _) -> P.p3 x y 0.) qs)

let draw_frame c s (f : P.frame) =
  let origin = f.origin
  and x = f.x
  and y = f.y
  and z = f.z in
  let axes =
    [ x, Color.v 0.85 0.20 0.15; y, Color.v 0.15 0.65 0.25; z, Color.v 0.15 0.35 0.85 ]
  in
  let lines =
    List.map
      (fun (endpoint, col) ->
         draw_polyline c (outline ~width:s.width col) [ origin; endpoint ])
      axes
  in
  let dot = draw_point c (solid (Color.v 0.1 0.1 0.1)) origin in
  Picture.group (dot :: lines)

let draw_primitive c s = function
  | P.Point p -> draw_point c s p
  | P.Segment (a, b) -> draw_polyline c s [ a; b ]
  | P.Polyline ps -> draw_polyline c s ps
  | P.Polygon ps -> draw_polygon c s ps
  | P.Circle c0 -> draw_circle c s c0.center c0.normal c0.radius
  | P.Sphere sp ->
    (match silhouette c sp with
     | None -> Picture.empty
     | Some (P.Circle ci) -> draw_circle c s ci.center ci.normal ci.radius
     | Some _ -> Picture.empty)
  | P.Plane pl -> draw_polygon c s pl.patch
  | P.Frame f -> draw_frame c s f

let picture_at scene ~time =
  let items =
    List.concat_map
      (fun e ->
         let prim = e.motion time e.primitive in
         let pic = draw_primitive scene.camera e.style prim in
         if Picture.equal pic Picture.empty
         then []
         else [ e.layer, depth scene.camera prim, pic ])
      scene.entities
  in
  let items =
    List.sort
      (fun (l1, d1, _) (l2, d2, _) ->
         let c = Stdlib.compare l1 l2 in
         if c <> 0 then c else Stdlib.compare d2 d1)
      items
  in
  let bg =
    Picture.fill
      scene.background
      (Path.rect (Box2.v 0. 0. (C.width scene.camera) (C.height scene.camera)))
  in
  Picture.group (bg :: List.map (fun (_, _, p) -> p) items)

let render scene ~time =
  Hugin_vg.Renderable.v
    (C.width scene.camera)
    (C.height scene.camera)
    (picture_at scene ~time)
