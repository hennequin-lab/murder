(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
module S = Mjc_viz.Scene

(* raylib splits its bindings: window, input, camera and colours are core, the
   3D shape calls are models. *)
module RC = Raylib_core
module RM = Raylib_models
module RT = Raylib_text

type t =
  { model : Mjc.Model.host
  ; mutable data : Mjc.Data.host
  ; mutable paused : bool
  ; mutable frames : bool
  ; mutable contacts : bool
  ; mutable forces : bool
  ; mutable grid : bool
  ; camera : RC.Camera3D.t
  }

let v3 (p : S.p3) = RC.Vector3.create p.S.x p.y p.z

let colour (c : S.colour) =
  RC.Color.create
    (Int.of_float (c.r *. 255.))
    (Int.of_float (c.g *. 255.))
    (Int.of_float (c.b *. 255.))
    255

let black = RC.Color.create 20 20 20 255
let red = RC.Color.create 210 40 40 255

let open_ (model : Mjc.Model.host) =
  let data = Mjc.Data.create model in
  { model
  ; data
  ; paused = false
  ; frames = false
  ; contacts = true
  ; forces = false
  ; grid = true
  ; camera =
      RC.Camera3D.create
        (RC.Vector3.create 1.6 (0. -. 2.2) 1.3)
        (RC.Vector3.create 0. 0. 0.5)
        (RC.Vector3.create 0. 0. 1.)
        45.
        RC.CameraProjection.Perspective
  }

(* The camera is a Ctypes structure, so it is read through the accessors and
   written back with a fresh vector. *)
let vget v = RC.Vector3.x v, RC.Vector3.y v, RC.Vector3.z v
let vmake x y z = RC.Vector3.create x y z

let orbit (c : RC.Camera3D.t) (dx : float) (dy : float) =
  let dx = dx *. 0.01
  and dy = dy *. 0.01 in
  let px, py, pz = vget (RC.Camera3D.position c)
  and tx, ty, tz = vget (RC.Camera3D.target c) in
  let ox = px -. tx
  and oy = py -. ty
  and oz = pz -. tz in
  let r = Float.sqrt ((ox *. ox) +. (oy *. oy) +. (oz *. oz)) in
  let theta = Float.atan2 oy ox +. dx in
  let phi =
    Float.max
      0.15
      (Float.min 3.0 (Float.atan2 (Float.sqrt ((ox *. ox) +. (oy *. oy))) oz +. dy))
  in
  let rxy = r *. Float.sin phi in
  RC.Camera3D.set_position
    c
    (vmake
       (tx +. (rxy *. Float.cos theta))
       (ty +. (rxy *. Float.sin theta))
       (tz +. (r *. Float.cos phi)))

let zoom (c : RC.Camera3D.t) k =
  let px, py, pz = vget (RC.Camera3D.position c)
  and tx, ty, tz = vget (RC.Camera3D.target c) in
  RC.Camera3D.set_position
    c
    (vmake (tx +. ((px -. tx) *. k)) (ty +. ((py -. ty) *. k)) (tz +. ((pz -. tz) *. k)))

(* {1 Drawing a scene}

   Edges rather than faces for boxes and cylinders, and a wireframe for the
   plane's patch, because a debugging view wants to see through a solid; the
   spheres are solid so that depth reads. *)

let draw_geom (g : S.geom) =
  let col = colour g.colour in
  match g.shape with
  | S.Sphere { centre; radius; _ } -> RM.draw_sphere (v3 centre) radius col
  | S.Capsule { a; b; radius; _ } -> RM.draw_capsule (v3 a) (v3 b) radius 8 8 col
  | S.Box { centre; axes = ux, uy, uz; half = hx, hy, hz; _ } ->
    let corner sx sy sz =
      let open S in
      ( { x = centre.x +. (ux.x *. sx *. hx) +. (uy.x *. sy *. hy) +. (uz.x *. sz *. hz)
        ; y = centre.y +. (ux.y *. sx *. hx) +. (uy.y *. sy *. hy) +. (uz.y *. sz *. hz)
        ; z = centre.z +. (ux.z *. sx *. hx) +. (uy.z *. sy *. hy) +. (uz.z *. sz *. hz)
        }
      , () )
    in
    let c sx sy sz = corner sx sy sz |> fst in
    let v =
      Array.init 8 ~f:(fun i ->
        c
          (if i land 1 = 0 then -1. else 1.)
          (if i land 2 = 0 then -1. else 1.)
          (if i land 4 = 0 then -1. else 1.))
    in
    List.iter
      [ 0, 1; 0, 2; 0, 4; 1, 3; 1, 5; 2, 3; 2, 6; 3, 7; 4, 5; 4, 6; 5, 7; 6, 7 ]
      ~f:(fun (i, j) -> RM.draw_line_3d (v3 v.(i)) (v3 v.(j)) col)
  | S.Cylinder { centre; axis; radius; half_height; _ } ->
    (* raylib's cylinder is vertical; a capsule over the same segment is the
       closest primitive it has, and rounds the rims by one radius. *)
    let at s =
      S.
        { x = centre.x +. (axis.x *. s *. half_height)
        ; y = centre.y +. (axis.y *. s *. half_height)
        ; z = centre.z +. (axis.z *. s *. half_height)
        }
    in
    RM.draw_capsule (v3 (at 1.)) (v3 (at (-1.))) radius 8 8 col
  | S.Ellipsoid { centre; radii = rx, ry, _; _ } ->
    RM.draw_sphere (v3 centre) (Float.max rx ry) col
  | S.Plane _ -> ()
  | S.Unsupported _ -> ()

let draw (t : t) =
  let scene =
    S.of_state ~frames:t.frames ~contacts:t.contacts ~forces:t.forces t.model t.data
  in
  let open RC in
  begin_mode_3d t.camera;
  if t.grid then RM.draw_grid 20 1.;
  List.iter scene.S.geoms ~f:draw_geom;
  List.iter scene.S.bodies ~f:(fun (b : S.body) ->
    RM.draw_line_3d (v3 b.frame.origin) (v3 b.frame.x) (Color.create 200 60 60 255);
    RM.draw_line_3d (v3 b.frame.origin) (v3 b.frame.y) (Color.create 60 170 60 255);
    RM.draw_line_3d (v3 b.frame.origin) (v3 b.frame.z) (Color.create 60 90 200 255));
  List.iter scene.S.contacts ~f:(fun (c : S.contact) ->
    RM.draw_sphere (v3 c.point) 0.02 red;
    RM.draw_line_3d
      (v3 c.point)
      (v3
         S.
           { x = c.point.x +. (c.normal.x *. 0.1)
           ; y = c.point.y +. (c.normal.y *. 0.1)
           ; z = c.point.z +. (c.normal.z *. 0.1)
           })
      red);
  List.iter scene.S.forces ~f:(fun (f : S.arrow) ->
    let tip =
      S.
        { x = f.from.x +. f.vector.x
        ; y = f.from.y +. f.vector.y
        ; z = f.from.z +. f.vector.z
        }
    in
    RM.draw_line_3d (v3 f.from) (v3 tip) (Color.create 40 100 220 255));
  end_mode_3d ();
  let ncon = Stdlib.Int32.to_int (Nx.item [] t.data.Mjc.Data.ncon) in
  let nefc = Stdlib.Int32.to_int (Nx.item [] t.data.Mjc.Data.nefc) in
  RT.draw_text
    (Stdlib.Printf.sprintf
       "t %.2f s   ncon %d   nefc %d   %s"
       (Nx.item [] t.data.Mjc.Data.time)
       ncon
       nefc
       (if t.paused then "PAUSED" else "running"))
    10
    10
    20
    black;
  RT.draw_text
    "space pause | . step | r reset | g grid | f frames | c contacts | x forces | esc \
     quit"
    10
    34
    18
    black;
  RT.draw_fps 10 56

let run ?(steps_per_frame = 1) ?frames (t : t) =
  let width = 1280
  and height = 720 in
  RC.init_window width height "mjc";
  RC.set_target_fps 60;
  let remaining = ref (Option.value frames ~default:Int.max_value) in
  let quit = ref false in
  while (not !quit) && !remaining > 0 do
    (* Input *)
    if RC.is_key_pressed RC.Key.Escape then quit := true;
    if RC.is_key_pressed RC.Key.Space then t.paused <- not t.paused;
    if RC.is_key_pressed RC.Key.Period
    then (
      t.paused <- true;
      t.data <- Mjc.Step.step t.model t.data);
    if RC.is_key_pressed RC.Key.R then t.data <- Mjc.Data.create t.model;
    if RC.is_key_pressed RC.Key.G then t.grid <- not t.grid;
    if RC.is_key_pressed RC.Key.F then t.frames <- not t.frames;
    if RC.is_key_pressed RC.Key.C then t.contacts <- not t.contacts;
    if RC.is_key_pressed RC.Key.X then t.forces <- not t.forces;
    if RC.is_mouse_button_down RC.MouseButton.Right
    then (
      let d = RC.get_mouse_delta () in
      orbit t.camera (0. -. RC.Vector2.x d) (RC.Vector2.y d))
    else if RC.is_mouse_button_down RC.MouseButton.Left
    then (
      let d = RC.get_mouse_delta () in
      let tx, ty, tz = vget (RC.Camera3D.target t.camera) in
      RC.Camera3D.set_target
        t.camera
        (vmake (tx -. (RC.Vector2.x d *. 0.005)) (ty +. (RC.Vector2.y d *. 0.005)) tz));
    let wheel = RC.get_mouse_wheel_move () in
    if Float.( <> ) wheel 0. then zoom t.camera (1. -. (wheel *. 0.1));
    (* Physics *)
    if not t.paused
    then
      for _ = 1 to steps_per_frame do
        t.data <- Mjc.Step.step t.model t.data
      done;
    (* Draw *)
    RC.begin_drawing ();
    RC.clear_background (RC.Color.create 247 247 250 255);
    draw t;
    RC.end_drawing ();
    Int.decr remaining
  done;
  RC.close_window ()
