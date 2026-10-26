(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Ten starts, one target: the IK landscape as a picture.

   Every arm solves the same pose task from a different initial configuration,
   and the movie is the fan closing on it. The solver is the shipped one
   ([Geom_robot.Ik]) with a small step limit, so the iterates are the keyframes
   and [Geom_robot_viz.resample] spreads them over the film. The picture shows
   what a success-rate table hides: which starts travel far, which barely move,
   and that the solutions differ while the pose is the same.

   Writes [robot/viz/demo/out/swarm.{png,svg,mp4}]. Run from [geom/]:
   [dune exec robot/viz/demo/swarm.exe]. *)

open Nx
module R = Geom_robot.Robot
module T = Geom_robot.Task
module Ik = Geom_robot.Ik
module V = Geom_robot_viz
module P = Geom_viz.Primitive
module Draw = Geom_viz.Draw
module Frame = Geom_viz.Frame
module Movie = Geom_viz.Movie
module C = Hugin_gg.Color

let f32 = float32
let out = "robot/viz/demo/out"
let fps = 24
let duration = 2.
let nframes = int_of_float (duration *. Float.of_int fps)
let model = R.six_dof ()
let tool = R.frame model "ee"

(* The target: a pose the arm reaches at a configuration away from every start,
   so the fan has somewhere to travel. *)
let q_target = Nx.create f32 [| 6 |] [| 0.70; -0.45; 0.85; -0.60; 0.40; 0.95 |]
let target = R.pose model ~frame:tool q_target
let task = T.pose model ~frame:tool ~target

(* {1 The paths}

   One trajectory per start, indexed [start][frame] — the order the entity
   motions want. Each start is solved and then walked from where it began to
   where it ended at a constant rate; animating the iterates themselves would
   put the whole approach in the first frames, since Gauss-Newton converges
   geometrically, and leave the rest of the film motionless.

   The starts are drawn within about a radian of the goal. Further out — around
   1.7 here — three of the ten settle into a local minimum of the pose error and
   stay there, which is a true property of pose IK and not something a damping
   schedule fixes; the demo keeps its starts in one basin so that the picture is
   about multiplicity rather than about failures. *)

let starts = 10

let paths =
  Rng.with_key (Rng.key 7) (fun () ->
    let q0 = Nx.mul_s (Nx.randn f32 [| starts; 6 |]) 1.0 in
    Array.init starts (fun i ->
      let start = Nx.slice [ I i ] q0 in
      let q_end = (Ik.solve ~max_iter:150 ~step_limit:0.3 task ~q:start).q in
      Array.init nframes (fun k -> V.linear_path start q_end ~n:nframes k)))

(* {1 The fans} *)

(* One drawable robot per start, with a colour and joints small enough not to
   swamp the links: the styles of a [V.t] are meant to be replaced like this. *)
let arm i =
  let tone = 0.10 +. (0.045 *. Float.of_int i) in
  let colour = C.v ~alpha:0.55 tone 0.28 0.62 in
  let style =
    { V.default_style with
      link = { Draw.default with fill = None; stroke = Some colour; width = 3.5 }
    ; joint =
        { Draw.default with
          fill = Some (C.v ~alpha:0.55 0.95 0.55 0.15)
        ; stroke = Some colour
        ; width = 0.6
        }
    ; trail =
        { Draw.default with
          fill = None
        ; stroke = Some (C.v ~alpha:0.35 0.2 0.3 0.6)
        ; width = 1.2
        }
    }
  in
  V.make ~style ~joint_radius:0.035 model

let entities =
  List.concat
    (List.init starts (fun i ->
       V.animated (arm i) ~q:(V.sample ~fps paths.(i)) ~frame:tool ~trail:paths.(i) ()))

let markers =
  List.map (Draw.static ~layer:5 ~style:V.default_style.target) (V.target target)

let scene =
  Draw.create
    (V.camera ~eye:(P.p3 3.4 (-3.7) 2.2) ())
    ((V.stage (arm 0) :: entities) @ markers)

(* {1 Output} *)

let () =
  if not (Sys.file_exists out) then ignore (Sys.command ("mkdir -p " ^ out));
  let reached =
    Array.fold_left
      (fun n traj ->
         if Nx.item [] (T.norm task traj.(nframes - 1)) < 1e-2 then n + 1 else n)
      0
      paths
  in
  Format.printf
    "%d of %d starts reached the target to 1e-2; the still is the fan of distinct \
     configurations that all satisfy it@."
    reached
    starts;
  let spec = { Movie.duration; fps; scene; density = 1.5 } in
  let picture = Draw.picture_at scene ~time:(duration -. (1. /. Float.of_int fps)) in
  let page = Frame.create ~density:2. ~width:800. ~height:620. picture in
  Frame.save_png page (out ^ "/swarm.png");
  Frame.save_svg page (out ^ "/swarm.svg");
  Movie.to_mp4 spec ~path:(out ^ "/swarm.mp4");
  Format.printf "wrote %s/swarm.{png,svg,mp4}@." out
