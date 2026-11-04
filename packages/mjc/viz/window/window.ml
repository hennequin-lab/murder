(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
module S = Mjc_viz.Scene

(* raylib splits its bindings: window, input, camera and colours are core, the
   3D shape calls are models, and text is a third library. *)
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
  ; mutable hud : bool
  ; joints : Mjc_viz.Controls.joint array
  ; mutable joint : int (** the selected joint index, [-1] for none. *)
  ; mutable scene_s : float (** seconds spent building scenes, for the bench. *)
  ; width : int
  ; height : int
  ; camera : RC.Camera3D.t
  }

let default_eye = { S.x = 1.6; y = -2.2; z = 1.3 }
let default_target = { S.x = 0.; y = 0.; z = 0.5 }

(* [Data.create] is the reference state, but it leaves the derived arrays at
   zero: the frames are what kinematics computes. A viewer that never steps
   would draw every geom at the origin, so the state it starts from is the
   reference state after one forward pass, and so is a reset. *)
let fresh (model : Mjc.Model.host) = Mjc.Kinematics.forward model (Mjc.Data.create model)

let open_
      ?(width = 1280)
      ?(height = 720)
      ?(eye = default_eye)
      ?(target = default_target)
      ?(hud = true)
      ?(frames = false)
      ?(contacts = true)
      ?(forces = false)
      ?(grid = true)
      (model : Mjc.Model.host)
  =
  let data = fresh model in
  { model
  ; data
  ; paused = false
  ; frames
  ; contacts
  ; forces
  ; grid
  ; hud
  ; joints = Mjc_viz.Controls.joints model
  ; joint = -1
  ; scene_s = 0.
  ; width
  ; height
  ; camera =
      RC.Camera3D.create
        (RC.Vector3.create eye.x eye.y eye.z)
        (RC.Vector3.create target.x target.y target.z)
        (RC.Vector3.create 0. 0. 1.)
        45.
        RC.CameraProjection.Perspective
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

(* {1 Joint control}

   The model, the state and the selected joint: the writing and the clamping
   live in [Mjc_viz.Controls], where they are tested without a display, and
   this is only the keyboard. *)

let joint_count (t : t) = Array.length t.joints

let joint (t : t) =
  if t.joint < 0 || t.joint >= joint_count t then None else Some t.joints.(t.joint)

let joint_line (t : t) =
  match joint t with
  | None -> "joint: none  (1-9 select, 0 zeroes)"
  | Some j -> Mjc_viz.Controls.describe t.data j

let nudge (t : t) dir =
  match joint t with
  | None -> ()
  | Some j ->
    let data, _ = Mjc_viz.Controls.nudge t.model t.data j dir in
    t.data <- data

let zero_joint (t : t) =
  match joint t with
  | None -> ()
  | Some j ->
    let data, _ = Mjc_viz.Controls.zero t.model t.data j in
    t.data <- data

let select_joint (t : t) i = if i < joint_count t then t.joint <- i

(* {1 The camera}

   [Camera3D] is a Ctypes structure, so it is read through the accessors and
   written back with a fresh vector. *)

let vget v = RC.Vector3.x v, RC.Vector3.y v, RC.Vector3.z v
let vmake x y z = RC.Vector3.create x y z

let look (t : t) eye target =
  RC.Camera3D.set_position t.camera (v3 eye);
  RC.Camera3D.set_target t.camera (v3 target)

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

   Edges rather than faces for boxes, and a wireframe for the plane's patch,
   because a debugging view wants to see through a solid; the spheres are
   solid so that depth reads. *)

(* A plane is drawn as the patch the model asks for: its outline and a grid
   inside it. Scene keeps the plane's normal and offset but no in-plane axes,
   so two are made from the normal here; any pair will do, and a plane with no
   edge is no more correct for being aligned with an axis. *)
let draw_patch ~normal ~offset ~half:(hx, hy) col =
  let open S in
  let norm (v : p3) = Float.sqrt ((v.x *. v.x) +. (v.y *. v.y) +. (v.z *. v.z)) in
  let smul (v : p3) k = { x = v.x *. k; y = v.y *. k; z = v.z *. k } in
  let add (a : p3) (b : p3) = { x = a.x +. b.x; y = a.y +. b.y; z = a.z +. b.z } in
  let cross (a : p3) (b : p3) =
    { x = (a.y *. b.z) -. (a.z *. b.y)
    ; y = (a.z *. b.x) -. (a.x *. b.z)
    ; z = (a.x *. b.y) -. (a.y *. b.x)
    }
  in
  let n = smul normal (1. /. norm normal) in
  let seed =
    if Float.(abs n.z < 0.9)
    then { x = 0.; y = 0.; z = 1. }
    else { x = 1.; y = 0.; z = 0. }
  in
  let u = smul (cross n seed) (1. /. norm (cross n seed)) in
  let w = cross n u in
  let centre = smul n (0. -. offset) in
  let at su sw = add centre (add (smul u (su *. hx)) (smul w (sw *. hy))) in
  (* the outline, then a grid of eighths *)
  for i = 0 to 8 do
    let t = (Int.to_float i /. 4.) -. 1. in
    RM.draw_line_3d (v3 (at t (-1.))) (v3 (at t 1.)) col
  done;
  for i = 0 to 8 do
    let t = (Int.to_float i /. 4.) -. 1. in
    RM.draw_line_3d (v3 (at (-1.) t)) (v3 (at 1. t)) col
  done

let draw_geom (g : S.geom) =
  let col = colour g.colour in
  match g.shape with
  | S.Sphere { centre; radius; _ } -> RM.draw_sphere (v3 centre) radius col
  | S.Capsule { a; b; radius; _ } -> RM.draw_capsule (v3 a) (v3 b) radius 8 8 col
  | S.Box { centre; axes = ux, uy, uz; half = hx, hy, hz; _ } ->
    let corner sx sy sz =
      let open S in
      { x = centre.x +. (ux.x *. sx *. hx) +. (uy.x *. sy *. hy) +. (uz.x *. sz *. hz)
      ; y = centre.y +. (ux.y *. sx *. hx) +. (uy.y *. sy *. hy) +. (uz.y *. sz *. hz)
      ; z = centre.z +. (ux.z *. sx *. hx) +. (uy.z *. sy *. hy) +. (uz.z *. sz *. hz)
      }
    in
    let v =
      Array.init 8 ~f:(fun i ->
        corner
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
  | S.Plane { normal; offset; half; _ } ->
    draw_patch ~normal ~offset ~half (RC.Color.create 206 206 212 255)
  | S.Unsupported _ -> ()

let draw (t : t) =
  let t0 = Unix.gettimeofday () in
  let scene =
    S.of_state ~frames:t.frames ~contacts:t.contacts ~forces:t.forces t.model t.data
  in
  t.scene_s <- t.scene_s +. (Unix.gettimeofday () -. t0);
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
  if t.hud
  then (
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
    RT.draw_text (joint_line t) 10 34 18 black;
    RT.draw_text
      "space pause | . step | r reset | g grid | f frames | n contacts | x forces | h \
       hud | c camera | esc quit"
      10
      56
      16
      black;
    RT.draw_fps 10 76)

(* [c] puts the camera back where a run starts. *)
let camera_reset (t : t) = look t default_eye default_target

(* {1 The loop} *)

let run ?(steps_per_frame = 1) ?frames ?shot ?(bench = false) (t : t) =
  RC.init_window t.width t.height "mjc";
  (* A bench or a screenshot must not be paced by the display. *)
  if not bench then RC.set_target_fps 60;
  let drawn = ref 0 in
  let last = Option.value frames ~default:Int.max_value in
  let quit = ref false in
  let digits = [ RC.Key.One; Two; Three; Four; Five; Six; Seven; Eight; Nine ] in
  let start = Unix.gettimeofday () in
  while (not !quit) && !drawn < last do
    (* Input *)
    if RC.is_key_pressed RC.Key.Escape || RC.window_should_close () then quit := true;
    if RC.is_key_pressed RC.Key.Space then t.paused <- not t.paused;
    if RC.is_key_pressed RC.Key.Period
    then (
      t.paused <- true;
      t.data <- Mjc.Step.step t.model t.data);
    if RC.is_key_pressed RC.Key.R then t.data <- fresh t.model;
    if RC.is_key_pressed RC.Key.G then t.grid <- not t.grid;
    if RC.is_key_pressed RC.Key.F then t.frames <- not t.frames;
    if RC.is_key_pressed RC.Key.N then t.contacts <- not t.contacts;
    if RC.is_key_pressed RC.Key.X then t.forces <- not t.forces;
    if RC.is_key_pressed RC.Key.H then t.hud <- not t.hud;
    if RC.is_key_pressed RC.Key.C then camera_reset t;
    if RC.is_key_pressed RC.Key.Left_bracket then nudge t (-1.);
    if RC.is_key_pressed RC.Key.Right_bracket then nudge t 1.;
    if RC.is_key_pressed RC.Key.Zero then zero_joint t;
    List.iteri digits ~f:(fun i k -> if RC.is_key_pressed k then select_joint t i);
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
    (* The screenshot reads the frame out of the buffer, so it has to happen
       between the draw calls and the swap. *)
    (match shot with
     | Some path when !drawn = last - 1 -> RC.take_screenshot path
     | _ -> ());
    RC.end_drawing ();
    Int.incr drawn
  done;
  let elapsed = Unix.gettimeofday () -. start in
  RC.close_window ();
  if bench
  then (
    let n = Float.of_int !drawn in
    Stdlib.Printf.printf
      "bench: %d frames at %dx%d in %.3f s: %.2f ms/frame (%.0f fps), of which %.3f \
       ms/frame building the scene\n\
       %!"
      !drawn
      t.width
      t.height
      elapsed
      (elapsed *. 1000. /. n)
      (n /. elapsed)
      (t.scene_s *. 1000. /. n));
  !drawn
