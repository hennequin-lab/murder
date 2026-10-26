(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Geom_robot.Task and Geom_robot.Ik.

   Two things are checked everywhere: that a residual means what it says — a
   signed distance, a power, an incidence — and that its analytic Jacobian is the
   derivative of that residual, by central differences. A task whose Jacobian
   disagrees with its own value is the failure mode this layer has to make
   impossible to ship. *)

open Nx
open Windtrap
module R = Geom_robot.Robot
module T = Geom_robot.Task
module Ik = Geom_robot.Ik
module Cga = Geom.Cga
module M = Geom.Motor

let f32 = float32
let q_of l = Nx.create f32 [| List.length l |] (Array.of_list l)
let arr x = Nx.to_array x
let item i x = Nx.item i x

let close_tensor ?(tol = 2e-3) a b =
  let a = arr (Nx.cast Nx.float64 a)
  and b = arr (Nx.cast Nx.float64 b) in
  Array.length a = Array.length b
  && Array.for_all2
       (fun x y ->
          Float.abs (x -. y)
          <= tol *. Float.max 1. (Float.max (Float.abs x) (Float.abs y)))
       a
       b

let equal_tensor ?msg ?(tol = 2e-3) expected actual =
  equal
    ?msg
    (Testable.make ~pp:Nx.pp ~equal:(fun a b -> close_tensor ~tol a b))
    expected
    actual

let equal_floats ?msg ?(eps = 1e-4) expected actual =
  equal ?msg (array (float eps)) expected (arr actual)

let bump n i h =
  let a = Array.make n 0. in
  a.(i) <- h;
  Nx.create f32 [| n |] a

(* The derivative of a residual, by central differences on its own value. *)
let fd_jacobian task q =
  let dof = (Nx.shape q).(0) in
  let dim = T.dim task in
  let h = 1e-3 in
  Array.init dof (fun i ->
    let plus = T.residual task (Nx.add q (bump dof i h))
    and minus = T.residual task (Nx.sub q (bump dof i h)) in
    Nx.div_s (Nx.sub plus minus) (2. *. h))
  |> Array.to_list
  |> List.map (fun column -> Nx.reshape [| dim; 1 |] column)
  |> fun columns -> Nx.concatenate ~axis:1 columns

let check_jacobian ?(tol = 2e-2) label task q =
  equal_tensor
    ~msg:(label ^ ": the Jacobian is the derivative")
    ~tol
    (T.jacobian task q)
    (fd_jacobian task q)

let arm = R.six_dof ()
let tool = R.frame arm "ee"

(* {1 Pose tasks} *)

let test_pose_task () =
  let q_goal = q_of [ 0.5; -0.4; 0.6; -0.8; 0.3; 0.7 ] in
  let target = R.pose arm ~frame:tool q_goal in
  let task = T.pose arm ~frame:tool ~target in
  equal ~msg:"six residuals" int 6 (T.dim task);
  (* The residual is the body-frame error, so it vanishes at the goal and equals
     log of the relative motion elsewhere. *)
  equal_tensor ~msg:"zero at the goal" (Nx.zeros f32 [| 6 |]) (T.residual task q_goal);
  let q = q_of [ 0.1; 0.2; -0.3; 0.4; 0.1; -0.2 ] in
  equal_tensor
    ~msg:"the residual is the relative log"
    (M.log (M.mul (M.inverse target) (R.pose arm ~frame:tool q)))
    (T.residual task q);
  (* The pose Jacobian is the body Jacobian, not the derivative of the
     logarithm: it is exact where the error vanishes and first-order accurate
     near it, which is what a Gauss-Newton step on a pose needs. Check it where
     that claim is testable, and check the exact statement — [J dq] is the
     right-trivialised derivative of the pose — in the model's own tests. *)
  let near = Nx.add q_goal (Nx.mul_s (q_of [ 1.; -1.; 0.7; 1.; -0.8; 0.5 ]) 0.01) in
  equal
    ~msg:"a near solution has a small error"
    bool
    true
    (item [] (T.norm task near) < 5e-2);
  check_jacobian ~tol:0.15 "pose near the solution" task near;
  (* And the solver drives it home. *)
  let result = Ik.solve ~max_iter:60 task ~q in
  equal ~msg:"converged" bool true (item [] result.converged);
  equal ~msg:"iterations" bool true (result.iterations <= 20);
  equal ~msg:"residual" bool true (Float.abs (item [] result.residual) < 1e-5)

(* {1 Geometric tasks: one formula, six object pairs} *)

let plane z =
  Cga.plane ~normal:(Nx.create f32 [| 3 |] [| 0.; 0.; 1. |]) ~offset:(Nx.scalar f32 z)

let tool_point q = arr (M.translation (R.pose arm ~frame:tool q))

let test_point_on_plane () =
  let task =
    T.primitive arm ~frame:tool ~reference:(Cga.point3 0. 0. 0.) ~target:(plane 1.2)
  in
  equal ~msg:"a point on a plane is one residual" int 1 (T.dim task);
  let q = q_of [ 0.2; -0.5; 0.4; -0.7; 0.2; 0.5 ] in
  let height = (tool_point q).(2) in
  equal_tensor
    ~msg:"the residual is the signed distance"
    (Nx.create f32 [| 1 |] [| height -. 1.2 |])
    (T.residual task q);
  check_jacobian "point/plane" task q;
  let result = Ik.solve ~max_iter:100 task ~q in
  equal ~msg:"reached the plane" bool true (item [] result.converged);
  equal ~msg:"on the plane" (float 1e-4) 1.2 (tool_point result.q).(2)

let test_point_on_sphere () =
  (* The sphere is built through the tool point at this configuration, so the
     residual there is zero; the power is its value elsewhere. *)
  let q = q_of [ 0.3; -0.6; 0.5; -0.4; 0.2; 0.8 ] in
  let p = tool_point q in
  let centre = Nx.create f32 [| 3 |] [| 0.; 0.; 1. |] in
  let radius =
    Nx.scalar
      f32
      (Float.sqrt
         (((p.(0) -. 0.) ** 2.) +. ((p.(1) -. 0.) ** 2.) +. ((p.(2) -. 1.) ** 2.)))
  in
  let sphere = Cga.sphere ~center:centre ~radius in
  let task =
    T.primitive arm ~frame:tool ~reference:(Cga.point3 0. 0. 0.) ~target:sphere
  in
  equal ~msg:"a point on a sphere is one residual" int 1 (T.dim task);
  equal_tensor ~msg:"zero on the sphere" (Nx.zeros f32 [| 1 |]) (T.residual task q);
  check_jacobian "point/sphere" task (q_of [ 0.1; -0.2; 0.3; -0.1; 0.4; 0.2 ])

let test_point_on_line () =
  let q = q_of [ 0.25; -0.4; 0.35; -0.55; 0.15; 0.6 ] in
  let p = tool_point q in
  (* A line through the tool point, along a direction that is not axis-aligned. *)
  let dir = [| 0.3; -0.5; 0.8 |] in
  let at t =
    Cga.point3
      (p.(0) +. (t *. dir.(0)))
      (p.(1) +. (t *. dir.(1)))
      (p.(2) +. (t *. dir.(2)))
  in
  let line = Cga.line (at 0.) (at 1.) in
  let task = T.primitive arm ~frame:tool ~reference:(Cga.point3 0. 0. 0.) ~target:line in
  equal ~msg:"a point on a line is a grade-one residual" int 5 (T.dim task);
  equal_tensor ~msg:"zero on the line" (Nx.zeros f32 [| 5 |]) (T.residual task q);
  check_jacobian "point/line" task (q_of [ 0.4; -0.3; 0.2; -0.5; 0.3; 0.4 ])

let test_line_through_point () =
  (* The dual task: the tool's own line must pass through a fixed point. The
     reference is a line through the tool origin, so it is the moving object. *)
  let q = q_of [ 0.2; -0.3; 0.4; -0.5; 0.1; 0.6 ] in
  let line = Cga.line (Cga.point3 0. 0. 0.) (Cga.point3 0. 0. 1.) in
  let target = Cga.point3 0. 0. 2. in
  let task = T.primitive arm ~frame:tool ~reference:line ~target in
  equal ~msg:"a line through a point is a grade-one residual" int 5 (T.dim task);
  check_jacobian "line/point" task q

let test_weights_and_stack () =
  let q = q_of [ 0.2; -0.5; 0.4; -0.7; 0.2; 0.5 ] in
  let task =
    T.primitive arm ~frame:tool ~reference:(Cga.point3 0. 0. 0.) ~target:(plane 1.2)
  in
  let heavy = T.weight task [| 3. |] in
  equal_tensor
    ~msg:"weights scale the residual"
    (Nx.mul_s (T.residual task q) 3.)
    (T.residual heavy q);
  equal_tensor
    ~msg:"weights scale the Jacobian"
    (Nx.mul_s (T.jacobian task q) 3.)
    (T.jacobian heavy q);
  raises_match
    ~msg:"a wrong number of weights is rejected"
    (Exn.invalid_arg ~substring:"weight")
    (fun () -> ignore (T.weight task [| 1.; 2. |]));
  (* A pose task and a geometric one stack into one residual. *)
  let pose = T.pose arm ~frame:tool ~target:(R.pose arm ~frame:tool q) in
  let stacked = T.stack [ pose; task ] in
  equal ~msg:"dimensions add" int 7 (T.dim stacked);
  let expected = Nx.concatenate ~axis:0 [ T.residual pose q; T.residual task q ] in
  equal_tensor ~msg:"residuals concatenate" expected (T.residual stacked q);
  check_jacobian "stacked" stacked q;
  raises_match
    ~msg:"stacking different models is rejected"
    (Exn.invalid_arg ~substring:"different models")
    (fun () ->
       let other = R.planar2 () in
       let q2 = Nx.create f32 [| 2 |] [| 0.3; 0.4 |] in
       let other_task =
         T.pose
           other
           ~frame:(R.frame other "ee")
           ~target:(R.pose other ~frame:(R.frame other "ee") q2)
       in
       ignore (T.stack [ pose; other_task ]))

(* {1 Batching} *)

let test_batching () =
  let task =
    T.primitive arm ~frame:tool ~reference:(Cga.point3 0. 0. 0.) ~target:(plane 1.2)
  in
  let rows = [ [ 0.2; -0.5; 0.4; -0.7; 0.2; 0.5 ]; [ -0.3; 0.1; 0.6; 0.2; -0.4; 0.8 ] ] in
  let flat = List.concat rows in
  let batch = Nx.reshape [| 2; 6 |] (Nx.create f32 [| 12 |] (Array.of_list flat)) in
  let r = T.residual task batch
  and j = T.jacobian task batch in
  equal ~msg:"residual shape" (array int) [| 2; 1 |] (Nx.shape r);
  equal ~msg:"Jacobian shape" (array int) [| 2; 1; 6 |] (Nx.shape j);
  List.iteri
    (fun i row ->
       let q = q_of row in
       equal_tensor
         ~msg:(Printf.sprintf "row %d of the residual" i)
         (T.residual task q)
         (Nx.slice [ I i ] r);
       equal_tensor
         ~msg:(Printf.sprintf "row %d of the Jacobian" i)
         (T.jacobian task q)
         (Nx.slice [ I i ] j))
    rows

(* {1 The solver} *)

let test_solver () =
  let q_goal = q_of [ 0.5; -0.4; 0.6; -0.8; 0.3; 0.7 ] in
  let target = R.pose arm ~frame:tool q_goal in
  let task = T.pose arm ~frame:tool ~target in
  (* Limits hold the solution where they say. *)
  let lo = Nx.full f32 [| 6 |] (-0.1)
  and hi = Nx.full f32 [| 6 |] 0.1 in
  let result =
    Ik.solve ~max_iter:30 ~lower:lo ~upper:hi task ~q:(q_of [ 0.; 0.; 0.; 0.; 0.; 0. ])
  in
  let bounded = arr result.q in
  equal
    ~msg:"every coordinate stays within the limits"
    bool
    true
    (Array.for_all (fun x -> x >= -0.1001 && x <= 0.1001) bounded);
  equal
    ~msg:"and it does not converge, since the goal is outside them"
    bool
    false
    (item [] result.converged);
  (* The step limit bounds the per-iteration travel, so a long reach is walked,
     not jumped: the trust region is part of the policy. *)
  let small =
    Ik.solve ~max_iter:1 ~step_limit:0.05 task ~q:(q_of [ 0.; 0.; 0.; 0.; 0.; 0. ])
  in
  let travel = arr small.q in
  equal
    ~msg:"one step moves at most the step limit"
    bool
    true
    (Array.for_all (fun x -> Float.abs x <= 0.0501) travel);
  (* A batch of starts solves as a batch: shapes out, convergence per sample. *)
  let flat =
    List.concat [ [ 0.; 0.; 0.; 0.; 0.; 0. ]; [ 0.4; -0.3; 0.5; -0.6; 0.2; 0.6 ] ]
  in
  let starts = Nx.reshape [| 2; 6 |] (Nx.create f32 [| 12 |] (Array.of_list flat)) in
  let batched = Ik.solve ~max_iter:60 task ~q:starts in
  equal ~msg:"a batch keeps its shape" (array int) [| 2; 6 |] (Nx.shape batched.q);
  equal
    ~msg:"a batch converges per sample"
    (array bool)
    [| true; true |]
    (arr batched.converged)

let () =
  exit
    (run
       "geom robot tasks"
       [ group "pose" [ test "the residual and its derivative" test_pose_task ]
       ; group
           "geometry"
           [ test "a point on a plane" test_point_on_plane
           ; test "a point on a sphere" test_point_on_sphere
           ; test "a point on a line" test_point_on_line
           ; test "a line through a point" test_line_through_point
           ; test "weights and stacking" test_weights_and_stack
           ]
       ; group "batching" [ test "a batch of configurations" test_batching ]
       ; group "solver" [ test "limits, the step limit, and batches" test_solver ]
       ])
