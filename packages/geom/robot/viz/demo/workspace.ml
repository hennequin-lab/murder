(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A workspace cloud coloured by conditioning.

   Sample configurations, ask the model where the tool is and how well
   conditioned the Jacobian is there, and draw a dot per sample — cold where the
   arm is far from singular, hot where sigma_min collapses. It is the cheapest
   useful picture of a manipulator, and it is one batched [pose] and one batched
   [jacobian] call: five lines of model, the rest is drawing.

   Writes [robot/viz/demo/out/workspace.png]. Run from [geom/]:
   [dune exec robot/viz/demo/workspace.exe]. *)

open Nx
module R = Geom_robot.Robot
module M = Geom.Motor
module V = Geom_robot_viz
module P = Geom_viz.Primitive
module Draw = Geom_viz.Draw
module Frame = Geom_viz.Frame
module C = Hugin_gg.Color

let f32 = float32
let out = "robot/viz/demo/out"
let model = R.six_dof ()
let tool = R.frame model "ee"
let drawn = V.make model

(* {1 The samples} *)

let samples = 600

let cloud =
  (* Seeded, so that the picture and the fraction it reports do not move between
     runs of an unchanged program. *)
  let q =
    Rng.with_key (Rng.key 3) (fun () -> Nx.mul_s (Nx.randn f32 [| samples; 6 |]) 2.4)
  in
  let positions = Nx.to_array (M.translation (R.pose model ~frame:tool q)) in
  (* sigma_min of the geometric Jacobian: how far this configuration is from
     losing a degree of freedom. *)
  let sigma_min =
    Nx.to_array
      (Nx.cast f32 (Nx.slice [ A; I 5 ] (Nx.svdvals (R.jacobian model ~frame:tool q))))
  in
  Array.init samples (fun i ->
    let p =
      P.p3 positions.((3 * i) + 0) positions.((3 * i) + 1) positions.((3 * i) + 2)
    in
    p, sigma_min.(i))

(* Four conditioning bands, cold to hot. *)
let bands =
  [ 0.3, Float.infinity, C.v ~alpha:0.55 0.16 0.24 0.52
  ; 0.1, 0.3, C.v ~alpha:0.55 0.22 0.45 0.62
  ; 0.02, 0.1, C.v ~alpha:0.7 0.85 0.55 0.15
  ; 0., 0.02, C.v ~alpha:0.9 0.75 0.12 0.10
  ]

let dot_style color = { Draw.default with fill = Some color; stroke = None; width = 0. }

let dots =
  List.concat_map
    (fun (cold, hot, color) ->
       let points =
         List.filter_map
           (fun (p, s) -> if s >= cold && s < hot then Some p else None)
           (Array.to_list cloud)
       in
       List.map
         (Draw.static ~layer:1 ~style:(dot_style color))
         (List.map (fun p -> P.Sphere { center = p; radius = 0.016 }) points))
    bands

(* The arm that generated them, at one configuration, for scale. *)
let arm =
  V.entities
    ~layer:2
    drawn
    (Nx.create f32 [| 6 |] [| 0.35; -0.75; 0.45; -0.95; 0.30; -0.50 |])

let worst_band = List.length (List.filter (fun (_, s) -> s < 0.02) (Array.to_list cloud))

let () =
  if not (Sys.file_exists out) then ignore (Sys.command ("mkdir -p " ^ out));
  let scene = Draw.create (V.camera ~eye:(P.p3 3.6 (-3.9) 2.4) ()) (dots @ arm) in
  let picture = Draw.picture_at scene ~time:0. in
  let page = Frame.create ~density:2. ~width:800. ~height:620. picture in
  Frame.save_png page (out ^ "/workspace.png");
  Frame.save_svg page (out ^ "/workspace.svg");
  Format.printf
    "wrote %s/workspace.{png,svg}: %d samples, %d of them with sigma_min < 0.02@."
    out
    samples
    worst_band
