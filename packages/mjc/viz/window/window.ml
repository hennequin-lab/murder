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
module RL = Raylib_rlgl
module RT = Raylib_text

(** The type for a replay cursor. [playing] advances the index one frame per
    drawn frame, which is the whole of "playback": the physics has already
    happened, so a slow display drops frames rather than slowing the run. *)
type follow =
  { log : Mjc_viz.Log.t
  ; mutable index : int
  ; mutable playing : bool
  ; divergence : int option
    (** the first frame outside the model's reach, found once when the log
        is opened, so that [d] can jump to it. *)
  }

type t =
  { model : Mjc.Model.host
  ; mutable data : Mjc.Data.host
  ; mutable paused : bool
  ; mutable frames : bool
  ; mutable contacts : bool
  ; mutable forces : bool
  ; mutable grid : bool
  ; mutable hud : bool
  ; mutable inertia : bool
  ; mutable com : bool
  ; mutable joint_axes : bool
  ; mutable velocity : bool
  ; mutable energy : bool
  ; joints : Mjc_viz.Controls.joint array
  ; mutable joint : int (** the selected joint index, [-1] for none. *)
  ; mutable scene_s : float (** seconds spent building scenes, for the bench. *)
  ; log : Mjc_viz.Log.writer option (** where each frame is appended, when recording. *)
  ; follow : follow option (** the recorded run being replayed, when following. *)
  ; width : int
  ; height : int
  ; camera : RC.Camera3D.t
  ; eye : S.p3
  ; target : S.p3
  }

(* The opening view is the still backend's: {!Scene.framing}, which ignores
   planes and offsets the eye by the box's largest side. Two backends that
   framed differently would look like a disagreement about where the model
   is. *)

let fresh (model : Mjc.Model.host) = Mjc.Kinematics.forward model (Mjc.Data.create model)

let open_
      ?(width = 1280)
      ?(height = 720)
      ?eye
      ?target
      ?(hud = true)
      ?(frames = false)
      ?(contacts = true)
      ?(forces = false)
      ?(grid = true)
      ?(inertia = false)
      ?(com = false)
      ?(joint_axes = false)
      ?(velocity = false)
      ?(energy = false)
      ?log_path
      ?follow_path
      (model : Mjc.Model.host)
  =
  let data = fresh model in
  let scene_eye, scene_target = S.framing (S.of_state model data) in
  let follow =
    Option.map follow_path ~f:(fun path ->
      match Mjc_viz.Log.open_ path with
      | Error e -> Stdlib.failwith e
      | Ok log ->
        let ngeom, nconmax = Mjc_viz.Log.model_shape log in
        if ngeom <> model.topo.ngeom || nconmax <> model.topo.nconmax
        then
          Stdlib.failwith
            (Stdlib.Printf.sprintf
               "%s holds %d geoms and %d contact slots, %s has %d and %d"
               path
               ngeom
               nconmax
               model.topo.name
               model.topo.ngeom
               model.topo.nconmax);
        { log; index = 0; playing = true; divergence = Mjc_viz.Log.divergence log })
  in
  let log = Option.map log_path ~f:(fun path -> Mjc_viz.Log.create ~path model) in
  let eye = Option.value eye ~default:scene_eye
  and target = Option.value target ~default:scene_target in
  { model
  ; data
  ; paused = false
  ; frames
  ; contacts
  ; forces
  ; grid
  ; hud
  ; inertia
  ; com
  ; joint_axes
  ; velocity
  ; energy
  ; joints = Mjc_viz.Controls.joints model
  ; joint = -1
  ; scene_s = 0.
  ; log
  ; follow
  ; width
  ; height
  ; eye
  ; target
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

(* {1 Overlays}

   Each layer is one of [Scene]'s pure functions, drawn in a colour that says
   which layer it is: inertia boxes amber, centres of mass violet, joint axes
   by kind, velocities blue. *)

let amber = RC.Color.create 214 138 26 255
let violet = RC.Color.create 140 70 190 255
let hinge_col = RC.Color.create 214 138 26 255
let slide_col = RC.Color.create 30 150 160 200

let box_edges
      ~(centre : S.p3)
      ~(axes : S.p3 * S.p3 * S.p3)
      ~(half : float * float * float)
      col
  =
  let ux, uy, uz = axes in
  let hx, hy, hz = half in
  let corner sx sy sz =
    S.
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

let at (a : S.p3) (d : S.p3) k =
  S.{ x = a.x +. (d.x *. k); y = a.y +. (d.y *. k); z = a.z +. (d.z *. k) }

let draw_overlays (t : t) (s : S.t) =
  if t.inertia
  then
    List.iter s.S.inertia ~f:(fun (i : S.inertia) ->
      box_edges ~centre:i.centre ~axes:i.axes ~half:i.half amber);
  if t.com
  then
    List.iter s.S.com ~f:(fun (c : S.p3) ->
      RM.draw_sphere (v3 c) 0.012 violet;
      List.iter
        [ S.{ x = 0.03; y = 0.; z = 0. }
        ; { x = -0.03; y = 0.; z = 0. }
        ; { x = 0.; y = 0.03; z = 0. }
        ; { x = 0.; y = -0.03; z = 0. }
        ; { x = 0.; y = 0.; z = 0.03 }
        ; { x = 0.; y = 0.; z = -0.03 }
        ]
        ~f:(fun d -> RM.draw_line_3d (v3 c) (v3 (at c d 1.)) violet));
  if t.joint_axes
  then
    List.iter s.S.joints ~f:(fun (j : S.joint) ->
      match j.kind with
      | S.Hinge | S.Slide ->
        let col =
          match j.kind with
          | S.Hinge -> hinge_col
          | _ -> slide_col
        in
        RM.draw_sphere (v3 j.anchor) 0.01 col;
        RM.draw_line_3d
          (v3 (at j.anchor j.axis (-0.12)))
          (v3 (at j.anchor j.axis 0.12))
          col
      | S.Ball | S.Free -> ());
  if t.velocity
  then
    List.iter s.S.velocity ~f:(fun (a : S.arrow) ->
      RM.draw_line_3d
        (v3 a.from)
        (v3
           S.
             { x = a.from.x +. a.vector.x
             ; y = a.from.y +. a.vector.y
             ; z = a.from.z +. a.vector.z
             })
        (RC.Color.create 40 100 220 255))

let draw_geom (g : S.geom) =
  let col = colour g.colour in
  match g.shape with
  | S.Sphere { centre; radius; _ } -> RM.draw_sphere (v3 centre) radius col
  | S.Capsule { a; b; radius; _ } -> RM.draw_capsule (v3 a) (v3 b) radius 8 8 col
  | S.Box { centre; axes; half; _ } -> box_edges ~centre ~axes ~half col
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

let draw ?(fps = true) (t : t) =
  let t0 = Unix.gettimeofday () in
  let scene =
    match t.follow with
    | Some f -> Mjc_viz.Log.scene t.model f.log f.index
    | None ->
      S.of_state
        ~frames:t.frames
        ~contacts:t.contacts
        ~forces:t.forces
        ~inertia:t.inertia
        ~com:t.com
        ~joints:t.joint_axes
        ~velocity:t.velocity
        ~energy:t.energy
        t.model
        t.data
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
  (* Contacts: the point coloured by penetration, the normal arrow by its
     force, the friction force in red, and the pyramid's four edges as the cone
     it is. *)
  let force_scale = 0.0006 in
  List.iter scene.S.contacts ~f:(fun (c : S.contact) ->
    RM.draw_sphere (v3 c.point) 0.02 (colour (S.depth_colour c.depth));
    let arrow (v : S.p3) col =
      let tip =
        S.
          { x = c.point.x +. (v.x *. force_scale)
          ; y = c.point.y +. (v.y *. force_scale)
          ; z = c.point.z +. (v.z *. force_scale)
          }
      in
      RM.draw_line_3d (v3 c.point) (v3 tip) col
    in
    arrow (S.scale c.normal c.force) (Color.create 30 130 60 255);
    arrow c.tangent_force (Color.create 200 60 50 255);
    if c.dim > 1
    then (
      let t1, t2 = c.tangents in
      let mu = c.friction *. c.force in
      List.iter
        [ t1, 1.; t1, -1.; t2, 1.; t2, -1. ]
        ~f:(fun (d, sgn) ->
          let s = sgn *. mu in
          let edge =
            S.
              { x = c.normal.x +. (s *. d.x)
              ; y = c.normal.y +. (s *. d.y)
              ; z = c.normal.z +. (s *. d.z)
              }
          in
          arrow edge (Color.create 214 138 26 200))));
  draw_overlays t scene;
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
    let line y size text = RT.draw_text text 10 y size black in
    (* A replayed frame's time and counts are the record's, not the idle
       state's: the state has never been stepped in this process. *)
    let time, ncon, nefc, state =
      match t.follow with
      | Some f ->
        let fr = Mjc_viz.Log.frame f.log f.index in
        fr.time, fr.ncon, fr.nefc, if f.playing then "playing" else "paused"
      | None ->
        ( Nx.item [] t.data.Mjc.Data.time
        , Stdlib.Int32.to_int (Nx.item [] t.data.Mjc.Data.ncon)
        , Stdlib.Int32.to_int (Nx.item [] t.data.Mjc.Data.nefc)
        , if t.paused then "PAUSED" else "running" )
    in
    line
      10
      20
      (Stdlib.Printf.sprintf "t %.2f s   ncon %d   nefc %d   %s" time ncon nefc state);
    (match t.follow with
     | Some f ->
       line
         34
         18
         (Stdlib.Printf.sprintf
            "replay %d / %d  %s%s"
            f.index
            (Mjc_viz.Log.length f.log - 1)
            (if f.playing then "playing" else "paused")
            (match f.divergence with
             | Some i -> Stdlib.Printf.sprintf "  diverged at %d (d)" i
             | None -> ""))
     | None -> line 34 18 (joint_line t));
    line
      56
      16
      "space pause | . step | r reset | c camera | g grid | f frames | n contacts";
    line 74 16 "i inertia | m com | j joints | v velocity | e energy | h hud | esc quit";
    (match scene.S.energy with
     | Some e ->
       line
         94
         16
         (Stdlib.Printf.sprintf
            "E %.3f J  (kinetic %.3f, potential %.3f)"
            (e.kinetic +. e.potential)
            e.kinetic
            e.potential)
     | None -> ());
    if fps then RT.draw_fps 10 (if Option.is_some scene.S.energy then 114 else 96))

(* [c] puts the camera back where the run started: the framed view, which is
   in [t] because the model may have moved since. *)
let camera_reset (t : t) = look t t.eye t.target

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
    if RC.is_key_pressed RC.Key.Space
    then (
      match t.follow with
      | Some f -> f.playing <- not f.playing
      | None -> t.paused <- not t.paused);
    if RC.is_key_pressed RC.Key.Period
    then (
      match t.follow with
      | Some f ->
        f.playing <- false;
        f.index <- Int.min (f.index + 1) (Mjc_viz.Log.length f.log - 1)
      | None ->
        t.paused <- true;
        t.data <- Mjc.Step.step t.model t.data);
    if RC.is_key_pressed RC.Key.R then t.data <- fresh t.model;
    if RC.is_key_pressed RC.Key.G then t.grid <- not t.grid;
    if RC.is_key_pressed RC.Key.F then t.frames <- not t.frames;
    if RC.is_key_pressed RC.Key.N then t.contacts <- not t.contacts;
    if RC.is_key_pressed RC.Key.X then t.forces <- not t.forces;
    if RC.is_key_pressed RC.Key.H then t.hud <- not t.hud;
    if RC.is_key_pressed RC.Key.I then t.inertia <- not t.inertia;
    if RC.is_key_pressed RC.Key.M then t.com <- not t.com;
    if RC.is_key_pressed RC.Key.J then t.joint_axes <- not t.joint_axes;
    if RC.is_key_pressed RC.Key.V then t.velocity <- not t.velocity;
    if RC.is_key_pressed RC.Key.E then t.energy <- not t.energy;
    if RC.is_key_pressed RC.Key.C then camera_reset t;
    (match t.follow with
     | None -> ()
     | Some f ->
       if RC.is_key_pressed RC.Key.Left
       then (
         f.playing <- false;
         f.index <- Int.max (f.index - 1) 0);
       if RC.is_key_pressed RC.Key.Right
       then (
         f.playing <- false;
         f.index <- Int.min (f.index + 1) (Mjc_viz.Log.length f.log - 1));
       if RC.is_key_pressed RC.Key.Home then f.index <- 0;
       if RC.is_key_pressed RC.Key.D
       then (
         f.playing <- false;
         match f.divergence with
         | Some i -> f.index <- i
         | None -> ());
       if RC.is_key_pressed RC.Key.End then f.index <- Mjc_viz.Log.length f.log - 1);
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
    (* Physics, or the replay cursor. *)
    (match t.follow with
     | Some f ->
       if f.playing then f.index <- Int.min (f.index + 1) (Mjc_viz.Log.length f.log - 1)
     | None ->
       if not t.paused
       then
         for _ = 1 to steps_per_frame do
           t.data <- Mjc.Step.step t.model t.data;
           match t.log with
           | Some w -> Mjc_viz.Log.append w ~step:(Stdlib.Int64.of_int !drawn) t.data
           | None -> ()
         done);
    (* Draw *)
    RC.begin_drawing ();
    RC.clear_background (RC.Color.create 247 247 250 255);
    (* A still has no frame rate to report. *)
    draw ~fps:(Option.is_none shot) t;
    (* The screenshot reads the frame out of the buffer, so it has to happen
       between the draw calls and the swap. *)
    (match shot with
     | Some path when !drawn = last - 1 ->
       (* raylib batches the 2D pass until EndDrawing, so a screenshot taken
          here would miss the HUD and every other 2D call. Flush first. *)
       RL.draw_render_batch_active ();
       RC.take_screenshot path
     | _ -> ());
    RC.end_drawing ();
    Int.incr drawn
  done;
  let elapsed = Unix.gettimeofday () -. start in
  RC.close_window ();
  (match t.log with
   | Some w -> Mjc_viz.Log.close w
   | None -> ());
  (match t.follow with
   | Some f -> Mjc_viz.Log.close_reader f.log
   | None -> ());
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
