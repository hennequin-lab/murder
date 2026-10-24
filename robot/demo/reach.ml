(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reaching a pose with the shipped solver.

   The smallest end-to-end use of the layer: build a model, state a pose target
   as a task, and let [Geom_robot.Ik] drive the configuration there. Everything
   it needs is public — [Task.pose], [Ik.iterate] and [Ik.solve] — which is the
   property the design note asks of the solver: a policy over residuals, not a
   black box around them. *)

open Nx
module R = Geom_robot.Robot
module M = Geom.Motor
module T = Geom_robot.Task
module Ik = Geom_robot.Ik

let f32 = float32
let q_of a = Nx.create f32 [| Array.length a |] a

let () =
  let model = R.six_dof () in
  let frame = R.frame model "ee" in
  (* A target a few degrees from the start, so the picture is a reach and not a
     contortion: an arbitrary pose also has solutions that wind a joint through
     several turns. *)
  let q0 = q_of [| 0.35; -0.75; 0.45; -0.95; 0.30; -0.50 |] in
  let q_goal = Nx.add q0 (q_of [| 0.55; 0.50; -0.45; 0.40; 0.55; -0.70 |]) in
  let target = R.pose model ~frame q_goal in
  let task = T.pose model ~frame ~target in
  Format.printf "@[<v 0>%a@]@." R.pp model;
  Format.printf "@,reaching %a@." T.pp task;
  (* The eager loop, one step at a time, so the convergence is visible. *)
  let state = ref (Ik.init ~q:q0 ()) in
  for i = 1 to 10 do
    state := Ik.iterate task !state;
    Format.printf
      "@,iteration %2d: residual %.3e (damping %.1e)@."
      i
      (Nx.item [] (T.norm task (Ik.q !state)))
      (Nx.item [] (Ik.lambda !state))
  done;
  (* And the batch driver, which stops on its own. *)
  let result = Ik.solve ~max_iter:50 task ~q:q0 in
  let p = Nx.to_array (M.translation (R.pose model ~frame result.q)) in
  let pt = Nx.to_array (M.translation target) in
  Format.printf
    "@,\
     Ik.solve: %d iterations, residual %.2e, converged %b@.tip at (%6.3f, %6.3f, %6.3f), \
     target (%6.3f, %6.3f, %6.3f)@."
    result.iterations
    (Nx.item [] result.residual)
    (Nx.item [] result.converged)
    p.(0)
    p.(1)
    p.(2)
    pt.(0)
    pt.(1)
    pt.(2)
