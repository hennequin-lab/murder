(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A geometric task: put the tool point on a circle.

   The target is a conformal circle, built from three points on it; the task is
   one contraction, `X_dot dot (M X M~)`, compacted to its five blades; the
   solver is `Geom_robot.Ik` with one iteration per frame. Nothing here knows
   what a circle is beyond the object it is given — the same file would drive the
   point onto a sphere, a line or a plane by changing one line of construction.

   Writes [robot/viz/demo/out/tasks.{png,svg,mp4}]. Run from [geom/]:
   [dune exec robot/viz/demo/tasks.exe]. *)

open Nx
module R = Geom_robot.Robot
module T = Geom_robot.Task
module Ik = Geom_robot.Ik
module M = Geom.Motor
module Cga = Geom.Cga
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
let drawn = V.make model

(* {1 The target: a conformal circle} *)

let centre = [| 0.85; -0.55; 1.35 |]
let radius = 0.32
let normal = [| 0.30; 0.25; 0.92 |]

(* An orthonormal basis of the circle's plane, from its normal. *)
let plane_basis =
  let n =
    let length = Float.sqrt (Array.fold_left (fun s x -> s +. (x *. x)) 0. normal) in
    Array.map (fun x -> x /. length) normal
  in
  let a = [| 0.; 0.; 1. |] in
  let d = (a.(0) *. n.(0)) +. (a.(1) *. n.(1)) +. (a.(2) *. n.(2)) in
  let u = Array.map2 (fun x y -> x -. (d *. y)) a n in
  let length = Float.sqrt (Array.fold_left (fun s x -> s +. (x *. x)) 0. u) in
  let u = Array.map (fun x -> x /. length) u in
  let v =
    [| (n.(1) *. u.(2)) -. (n.(2) *. u.(1))
     ; (n.(2) *. u.(0)) -. (n.(0) *. u.(2))
     ; (n.(0) *. u.(1)) -. (n.(1) *. u.(0))
    |]
  in
  u, v

let point_at angle =
  let u, v = plane_basis in
  let c = Float.cos angle
  and s = Float.sin angle in
  Cga.point3
    (centre.(0) +. (radius *. ((c *. u.(0)) +. (s *. v.(0)))))
    (centre.(1) +. (radius *. ((c *. u.(1)) +. (s *. v.(1)))))
    (centre.(2) +. (radius *. ((c *. u.(2)) +. (s *. v.(2)))))

(* A square of the circle's plane, just larger than the circle: the reader needs
   to know which way it tilts, not that it is infinite. *)
let plane_patch =
  let u, v = plane_basis in
  let corner su sv =
    let h = 1.5 *. radius in
    P.p3
      (centre.(0) +. ((su *. h *. u.(0)) +. (sv *. h *. v.(0))))
      (centre.(1) +. ((su *. h *. u.(1)) +. (sv *. h *. v.(1))))
      (centre.(2) +. ((su *. h *. u.(2)) +. (sv *. h *. v.(2))))
  in
  [ corner (-1.) (-1.); corner 1. (-1.); corner 1. 1.; corner (-1.) 1. ]

let circle =
  Cga.circle
    (point_at 0.)
    (point_at (2. *. Float.pi /. 3.))
    (point_at (4. *. Float.pi /. 3.))

let task = T.primitive model ~frame:tool ~reference:(Cga.point3 0. 0. 0.) ~target:circle

(* {1 The solve, and the path the film follows} *)

let q_start = Nx.create f32 [| 6 |] [| 0.55; 0.35; -0.55; 0.75; -0.35; 0.45 |]

(* The solver converges geometrically, so animating its iterates would spend the
   first tenth of the film on the whole approach and the rest on a still. Solve
   first, then walk the joint-space path from the start to the answer at a
   constant rate: the tool slides onto the circle for the whole two seconds. *)
let q_end = (Ik.solve ~max_iter:100 task ~q:q_start).q
let qs = Array.init nframes (fun k -> V.linear_path q_start q_end ~n:nframes k)

(* A sanity check that the demo is worth watching: the distance the tool covers
   along that path. *)
let travelled =
  let points =
    Array.map (fun q -> Nx.to_array (M.translation (R.pose model ~frame:tool q))) qs
  in
  let total = ref 0. in
  for i = 1 to Array.length points - 1 do
    let a = points.(i - 1)
    and b = points.(i) in
    total
    := !total
       +. Float.sqrt
            (((b.(0) -. a.(0)) ** 2.)
             +. ((b.(1) -. a.(1)) ** 2.)
             +. ((b.(2) -. a.(2)) ** 2.))
  done;
  !total

(* {1 The picture} *)

let circle_style =
  { Draw.default with
    fill = None
  ; stroke = Some (C.v ~alpha:0.95 0.10 0.45 0.65)
  ; width = 3.
  }

let scene =
  Draw.create
    (V.camera ~eye:(P.p3 3.1 (-3.5) 2.1) ())
    ((V.stage drawn
      :: Draw.static ~layer:4 ~style:circle_style (P.of_cga_circle circle)
      :: Draw.static
           ~layer:1
           ~style:
             { Draw.default with
               fill = Some (C.v ~alpha:0.07 0.10 0.45 0.65)
             ; stroke = None
             }
           (P.Plane
              { normal = P.p3 normal.(0) normal.(1) normal.(2)
              ; offset =
                  (normal.(0) *. centre.(0))
                  +. (normal.(1) *. centre.(1))
                  +. (normal.(2) *. centre.(2))
              ; patch = plane_patch
              })
      :: V.animated drawn ~q:(V.sample ~fps qs) ~frame:tool ~trail:qs ())
     @ [ Draw.static
           ~layer:4
           ~style:{ Draw.default with fill = Some (C.v 0.10 0.45 0.65); stroke = None }
           (P.Sphere { center = P.p3 centre.(0) centre.(1) centre.(2); radius = 0.02 })
       ])

let () =
  if not (Sys.file_exists out) then ignore (Sys.command ("mkdir -p " ^ out));
  let residual = T.norm task qs.(nframes - 1) in
  (* An independent check that the picture is not lying: the tool point against
     the circle's own centre and radius, as the library extracts them. *)
  let circle_centre = Nx.to_array (Cga.coords (Cga.circle_center circle)) in
  let circle_radius = Float.sqrt (Nx.item [] (Cga.circle_radius2 circle)) in
  let p = Nx.to_array (M.translation (R.pose model ~frame:tool qs.(nframes - 1))) in
  let distance =
    Float.sqrt
      (((p.(0) -. circle_centre.(0)) ** 2.)
       +. ((p.(1) -. circle_centre.(1)) ** 2.)
       +. ((p.(2) -. circle_centre.(2)) ** 2.))
    -. circle_radius
  in
  (* The distance to the circle at five frames of the film: monotone and even is
     what "the movie moves" means, numerically. *)
  let distance_at k =
    let q = qs.(k) in
    let p = Nx.to_array (M.translation (R.pose model ~frame:tool q)) in
    Float.sqrt
      (((p.(0) -. circle_centre.(0)) ** 2.)
       +. ((p.(1) -. circle_centre.(1)) ** 2.)
       +. ((p.(2) -. circle_centre.(2)) ** 2.))
    -. circle_radius
  in
  Format.printf
    "@,distance to the circle at frames 0/12/24/36/47: %s@."
    (String.concat
       " "
       (List.map (fun k -> Printf.sprintf "%.2f" (distance_at k)) [ 0; 12; 24; 36; 47 ]));
  Format.printf
    "task %a: residual %.3e, final distance %.3e, tool travelled %.3f@."
    T.pp
    task
    (Nx.item [] residual)
    distance
    travelled;
  let spec = { Movie.duration; fps; scene; density = 1.5 } in
  let picture = Draw.picture_at scene ~time:(duration -. (1. /. Float.of_int fps)) in
  let page = Frame.create ~density:2. ~width:800. ~height:620. picture in
  Frame.save_png page (out ^ "/tasks.png");
  Frame.save_svg page (out ^ "/tasks.svg");
  Movie.to_mp4 spec ~path:(out ^ "/tasks.mp4");
  Format.printf "wrote %s/tasks.{png,svg,mp4}@." out
