(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
module S = Mjc_viz.Scene
module R = Mjc_viz.Render

type t =
  { model : Mjc.Model.host
  ; mutable data : Mjc.Data.host
  ; mutable paused : bool
  ; mutable frames : bool
  ; mutable contacts : bool
  ; mutable forces : bool
  ; mutable grid : bool
  ; mutable camera : Raylib.Camera3D.t
  ; mutable scale : float
  }

let v3 (p : S.p3) = Raylib.Vector3.create p.S.x p.y p.z

let colour (c : S.colour) =
  Raylib.Color.create
    (int_of_float (c.r *. 255.))
    (int_of_float (c.g *. 255.))
    (int_of_float (c.b *. 255.))
    255

let black = Raylib.Color.create 20 20 20 255
let red = Raylib.Color.create 210 40 40 255

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
      Raylib.Camera3D.create
        (Raylib.Vector3.create 1.6 (0. -. 2.2) 1.3)
        (Raylib.Vector3.create 0. 0. 0.5)
        (Raylib.Vector3.create 0. 0. 1.)
        45.
        Raylib.CameraProjection.perspective
  ; scale = 1.
  }

(* Orbit and zoom about the camera's target. *)
let orbit (c : Raylib.Camera3D.t) (dx : float) (dy : float) =
  let dx = dx *. 0.01
  and dy = dy *. 0.01 in
  let ox = c.position.x -. c.target.x
  and oy = c.position.y -. c.target.y
  and oz = c.position.z -. c.target.z in
  let r = Float.sqrt ((ox *. ox) +. (oy *. oy) +. (oz *. oz)) in
  let theta = Float.atan2 oy ox +. dx in
  let phi =
    Float.max
      0.15
      (Float.min 3.0 (Float.atan2 (Float.sqrt ((ox *. ox) +. (oy *. oy))) oz +. dy))
  in
  let rxy = r *. Float.sin phi in
  c.position.x <- c.target.x +. (rxy *. Float.cos theta);
  c.position.y <- c.target.y +. (rxy *. Float.sin theta);
  c.position.z <- c.target.z +. (r *. Float.cos phi)

let zoom (c : Raylib.Camera3D.t) k =
  let ox = c.position.x -. c.target.x
  and oy = c.position.y -. c.target.y
  and oz = c.position.z -. c.target.z in
  c.position.x <- c.target.x +. (ox *. k);
  c.position.y <- c.target.y +. (oy *. k);
  c.position.z <- c.target.z +. (oz *. k)

(* {1 Drawing a scene}

   Edges rather than faces for boxes and cylinders, and a wireframe for the
   plane's patch, because a debugging view wants to see through a solid; the
   spheres are solid so that depth reads. *)

let draw_geom (g : S.geom) =
  let col = colour g.colour in
  match g.shape with
  | S.Sphere { centre; radius } -> Raylib.draw_sphere v3 centre radius col
  | S.Capsule { a; b; radius } -> Raylib.draw_capsule v3 a v3 b radius 8 8 col
  | S.Box { centre; axes = ux, uy, uz; half = hx, hy, hz } ->
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
      ~f:(fun (i, j) -> Raylib.draw_line_3d (v3 v.(i)) (v3 v.(j)) col)
  | S.Cylinder { centre; axis; radius; half_height } ->
    let top =
      S.
        { x = centre.x +. (axis.x *. half_height)
        ; y = centre.y +. (axis.y *. half_height)
        ; z = centre.z +. (axis.z *. half_height)
        }
    and bot =
      S.
        { x = centre.x -. (axis.x *. half_height)
        ; y = centre.y -. (axis.y *. half_height)
        ; z = centre.z -. (axis.z *. half_height)
        }
    in
    Raylib.draw_cylinder_wires v3 top v3 bot radius 12 4 col
  | S.Ellipsoid { centre; radii = rx, ry, rz } ->
    Raylib.draw_ellipsoid_wires v3 centre rx ry rz 8 8 col
  | S.Plane _ -> ()
  | S.Unsupported _ -> ()

let draw (t : t) =
  let scene =
    S.of_state ~frames:t.frames ~contacts:t.contacts ~forces:t.forces t.model t.data
  in
  let open Raylib in
  begin_mode_3d t.camera;
  if t.grid then draw_grid 20 1.;
  List.iter scene.S.geoms ~f:draw_geom;
  List.iter scene.S.bodies ~f:(fun (b : S.body) ->
    draw_line_3d (v3 b.frame.origin) (v3 b.frame.x) (Color.create 200 60 60 255);
    draw_line_3d (v3 b.frame.origin) (v3 b.frame.y) (Color.create 60 170 60 255);
    draw_line_3d (v3 b.frame.origin) (v3 b.frame.z) (Color.create 60 90 200 255));
  List.iter scene.S.contacts ~f:(fun (c : S.contact) ->
    draw_sphere (v3 c.point) 0.02 red;
    draw_line_3d
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
    draw_line_3d (v3 f.from) (v3 tip) (Color.create 40 100 220 255));
  end_mode_3d ();
  let ncon = Stdlib.Int32.to_int (Nx.item [] t.data.Mjc.Data.ncon) in
  let nefc = Stdlib.Int32.to_int (Nx.item [] t.data.Mjc.Data.nefc) in
  draw_text
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
  draw_text
    "space pause | . step | r reset | g grid | f frames | c contacts | x forces | esc \
     quit"
    10
    34
    18
    black;
  draw_fps 10 56

let run ?(steps_per_frame = 1) ?frames (t : t) =
  let width = 1280
  and height = 720 in
  Raylib.init_window width height "mjc";
  Raylib.set_target_fps 60;
  let remaining = ref (Option.value frames ~default:max_int) in
  let quit = ref false in
  while (not !quit) && !remaining > 0 do
    (* Input *)
    if Raylib.is_key_pressed Raylib.Key.escape then quit := true;
    if Raylib.is_key_pressed Raylib.Key.space then t.paused <- not t.paused;
    if Raylib.is_key_pressed Raylib.Key.period
    then (
      t.paused <- true;
      t.data <- Mjc.Step.step t.model t.data);
    if Raylib.is_key_pressed Raylib.Key.r then t.data <- Mjc.Data.create t.model;
    if Raylib.is_key_pressed Raylib.Key.g then t.grid <- not t.grid;
    if Raylib.is_key_pressed Raylib.Key.f then t.frames <- not t.frames;
    if Raylib.is_key_pressed Raylib.Key.c then t.contacts <- not t.contacts;
    if Raylib.is_key_pressed Raylib.Key.x then t.forces <- not t.forces;
    if Raylib.is_mouse_button_down Raylib.MouseButton.right
    then (
      let d = Raylib.get_mouse_delta () in
      orbit t.camera (0. -. d.x) d.y)
    else if Raylib.is_mouse_button_down Raylib.MouseButton.left
    then (
      let d = Raylib.get_mouse_delta () in
      t.camera.target.x <- t.camera.target.x -. (d.x *. 0.005);
      t.camera.target.y <- t.camera.target.y +. (d.y *. 0.005));
    let wheel = Raylib.get_mouse_wheel_move () in
    if Float.( <> ) wheel 0. then zoom t.camera (1. -. (wheel *. 0.1));
    (* Physics *)
    if not t.paused
    then
      for _ = 1 to steps_per_frame do
        t.data <- Mjc.Step.step t.model t.data
      done;
    (* Draw *)
    Raylib.begin_drawing ();
    Raylib.clear_background (Raylib.Color.create 247 247 250 255);
    draw t;
    Raylib.end_drawing ();
    Int.decr remaining
  done;
  Raylib.close_window ()
