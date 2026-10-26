(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The six-joint arm, drawn with [geom_robot_viz]: a still of the pose it
   reaches and a movie of the reach. Writes

     robot/viz/demo/out/arm.{png,svg,mp4}

   Run from [geom/]: [dune exec robot/viz/demo/arm.exe]. *)

open Nx
module R = Geom_robot.Robot
module M = Geom.Motor
module V = Geom_robot_viz
module Draw = Geom_viz.Draw
module Frame = Geom_viz.Frame
module Movie = Geom_viz.Movie

let f32 = float32
let out = "robot/viz/demo/out"
let fps = 24
let duration = 2.
let nframes = int_of_float (duration *. Float.of_int fps)
let model = R.six_dof ()
let tool = R.frame model "ee"
let drawn = V.make ~joint_radius:0.065 model

(* {1 The target and the reach} *)

(* A target a few degrees away in joint space: an arbitrary pose in the
   workspace has solutions that wind a joint through several turns, which is a
   correct answer and a terrible demo. *)
let q_start = Nx.create f32 [| 6 |] [| 0.35; -0.75; 0.45; -0.95; 0.30; -0.50 |]

let q_reach =
  Nx.add q_start (Nx.create f32 [| 6 |] [| 0.55; 0.50; -0.45; 0.40; 0.55; -0.70 |])

let target = R.pose model ~frame:tool q_reach

(* Damped Gauss-Newton on the body error, with a trust region on the step: §4.5
   of the design note, which now ships as [Geom_robot.Ik] — this file keeps its
   own step so that the picture does not depend on the solver being finished. *)
let solve q0 =
  let q = ref q0 in
  for _ = 1 to 60 do
    let e = M.log (M.mul (M.inverse target) (R.pose model ~frame:tool !q)) in
    let j = R.jacobian ~space:`Body model ~frame:tool !q in
    let jt = Nx.transpose j in
    let normal = Nx.add (Nx.matmul jt j) (Nx.mul_s (Nx.eye f32 6) 1e-5) in
    let step =
      Nx.reshape
        [| 6 |]
        (Nx.matmul (Nx.inv normal) (Nx.reshape [| 6; 1 |] (Nx.matmul jt e)))
    in
    let largest = Nx.max (Nx.abs step) in
    let scale = Nx.minimum_s (Nx.div (Nx.scalar f32 0.25) largest) 1. in
    q := Nx.sub !q (Nx.mul scale step)
  done;
  let e = M.log (M.mul (M.inverse target) (R.pose model ~frame:tool !q)) in
  Format.printf
    "reached the target to %.2e with |q| <= %.2f rad@."
    (Float.sqrt (Nx.item [] (Nx.sum (Nx.mul e e))))
    (Nx.item [] (Nx.max (Nx.abs !q)));
  !q

(* {1 The picture} *)

let qs =
  let q_end = solve q_start in
  Array.init nframes (fun k -> V.linear_path q_start q_end ~n:nframes k)

let scene =
  Draw.create
    (V.camera ())
    (V.stage drawn
     :: V.animated drawn ~q:(V.sample ~fps qs) ~frame:tool ~trail:qs ~target ())

let () =
  let () =
    if not (Sys.file_exists out)
    then (
      ignore (Sys.command ("mkdir -p " ^ out));
      ())
  in
  let spec = { Movie.duration; fps; scene; density = 1.5 } in
  (* The still is the last frame: the arm at the pose it reached, the path it
     took and the target marker. *)
  let picture = Draw.picture_at scene ~time:(duration -. (1. /. Float.of_int fps)) in
  let still = Frame.create ~density:2. ~width:800. ~height:620. picture in
  Frame.save_png still (out ^ "/arm.png");
  Frame.save_svg still (out ^ "/arm.svg");
  Frame.save_png (Movie.frame_at spec (Movie.frame_count spec / 2)) (out ^ "/reach.png");
  Movie.to_mp4 spec ~path:(out ^ "/arm.mp4");
  Format.printf "wrote %s/arm.{png,svg,mp4} (%d frames)@." out (Movie.frame_count spec)
