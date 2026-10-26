(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A tour of what the robot layer can do today.

   None of this is library code: every section is a few lines of [Nx] on top of
   [Robot.pose], [Robot.jacobian] and their relatives, which is the point — the
   layer exposes values and curves, and the algorithms around them are the
   caller's. What it does not have yet is listed at the end.

   Run with [dune exec robot/demo/tour.exe]. *)

open Nx
module R = Geom_robot.Robot
module M = Geom.Motor

let f32 = float32
let q_of a = Nx.create f32 [| Array.length a |] a
let section title = Format.printf "@,== %s ==@," title
let tip m ~frame q = M.translation (R.pose m ~frame q)
let report label x = Format.printf "  %-28s %a@." label Nx.pp x
let shape_of x = String.concat "x" (Array.map string_of_int (Nx.shape x) |> Array.to_list)

(* 1. Forward kinematics is a batch map: one call for a cloud of
   configurations, one call for the time axis of a trajectory. *)
let workspace () =
  section "batched forward kinematics: a 4096-configuration workspace cloud";
  let m = R.six_dof () in
  let ee = R.frame m "ee" in
  let q = Nx.mul_s (Nx.randn f32 [| 4096; 6 |]) 2.5 in
  let cloud = tip m ~frame:ee q in
  Format.printf "  cloud shape                  %s@." (shape_of cloud);
  report "smallest x" (Nx.min (Nx.slice [ A; I 0 ] cloud));
  report "largest x" (Nx.max (Nx.slice [ A; I 0 ] cloud));
  report "smallest z" (Nx.min (Nx.slice [ A; I 2 ] cloud));
  report "largest z" (Nx.max (Nx.slice [ A; I 2 ] cloud));
  report "mean reach" (Nx.mean (Nx.sqrt (Nx.sum ~axes:[ 1 ] (Nx.mul cloud cloud))))

(* 2. Frames are values, so any two of them can be compared: where is the tool
   seen from a camera bolted to the third link? *)
let arm_with_camera () =
  let link i = if i = 0 then "base" else Printf.sprintf "l%d" i in
  let axis i = if i mod 2 = 0 then 0., 0., 1. else 0., 1., 0. in
  let joint i =
    { R.name = Printf.sprintf "j%d" i
    ; kind = R.Revolute (axis i)
    ; parent = link i
    ; child = link (i + 1)
    ; origin = (if i = 0 then M.one else M.translate3 0. 0. 0.4)
    ; limits = None
    }
  in
  let tilted =
    M.mul
      (M.translate3 0. 0. 0.15)
      (M.rotate ~axis:(Nx.create f32 [| 3 |] [| 1.; 0.; 0. |]) ~angle:(Nx.scalar f32 1.2))
  in
  match
    R.create
      ~links:(List.init 7 link)
      ~joints:(List.init 6 joint)
      ~frames:
        [ { R.name = "camera"; link = "l3"; pose = tilted }
        ; { R.name = "tool"; link = "l6"; pose = M.translate3 0. 0. 0.3 }
        ]
      ()
  with
  | Ok m -> m
  | Error msg -> failwith msg

let camera () =
  section "frames: the tool as seen from a camera mounted on link3";
  let m = arm_with_camera () in
  let seen =
    R.relative
      m
      ~from:(R.frame m "camera")
      ~to_:(R.frame m "tool")
      (q_of [| 0.3; -0.2; 0.5; 0.1; -0.4; 0.2 |])
  in
  report "tool position in camera" (M.translation seen);
  report
    "distance to the tool"
    (Nx.sqrt (Nx.sum (Nx.mul (M.translation seen) (M.translation seen))))

(* 3. The geometric Jacobian is a velocity map: [J qdot] is the tool's twist,
   whose linear part is the tool point's velocity. The finite difference is
   what pins the convention, and it agrees. *)
let velocity () =
  section "differential kinematics: J qdot is the tool's velocity";
  let m = R.six_dof () in
  let ee = R.frame m "ee" in
  let q = q_of [| 0.4; -0.3; 0.8; 0.2; -0.5; 0.9 |] in
  let qd = q_of [| 0.1; 0.2; -0.3; 0.4; -0.5; 0.6 |] in
  let twist =
    Nx.reshape [| 6 |] (Nx.matmul (R.jacobian m ~frame:ee q) (Nx.reshape [| 6; 1 |] qd))
  in
  let h = 1e-3 in
  let difference =
    Nx.div_s
      (Nx.sub
         (tip m ~frame:ee (Nx.add q (Nx.mul_s qd h)))
         (tip m ~frame:ee (Nx.sub q (Nx.mul_s qd h))))
      (2. *. h)
  in
  report "J qdot, angular part" (Nx.slice [ R (0, 3) ] twist);
  report "J qdot, linear part" (Nx.slice [ R (3, 6) ] twist);
  report "dp/dt by finite difference" difference;
  Format.printf
    "  the derivative of the motor itself is separate: dM/dq is %s@."
    (shape_of (R.motion_jacobian m ~frame:ee q))

(* 4. Transposed, the same Jacobian is statics: a wrench at the tool maps to
   joint torques through virtual work, tau = J^T w. *)
let statics () =
  section "statics: a 10 N push at the tool, tau = J^T w";
  let m = R.six_dof () in
  let ee = R.frame m "ee" in
  let q = q_of [| 0.4; -0.3; 0.8; 0.2; -0.5; 0.9 |] in
  let wrench = Nx.create f32 [| 6 |] [| 0.; 0.; 0.; 0.; 0.; -10. |] in
  let tau =
    Nx.matmul (Nx.transpose (R.jacobian m ~frame:ee q)) (Nx.reshape [| 6; 1 |] wrench)
  in
  report "joint torques" (Nx.reshape [| 6 |] tau)

(* 5. Diagnostics over a batch: the smallest singular value of [J] says how far
   a configuration is from a singularity, and a batch gives its distribution. *)
let singularity () =
  section "diagnostics: sigma_min of J over 512 configurations";
  let m = R.six_dof () in
  let ee = R.frame m "ee" in
  let q = Nx.mul_s (Nx.randn f32 [| 512; 6 |]) 2.5 in
  let sv = Nx.svdvals (R.jacobian m ~frame:ee q) in
  let sigma_min = Nx.cast f32 (Nx.slice [ A; I 5 ] sv) in
  Format.printf "  singular values              %s@." (shape_of sv);
  report "mean sigma_min" (Nx.mean sigma_min);
  report "worst sigma_min" (Nx.min sigma_min);
  report "fraction below 0.02" (Nx.mean (Nx.cast f32 (Nx.less_s sigma_min 0.02)))

(* 6. An inverse-kinematics solver is a dozen lines of damped Gauss-Newton on
   [R.pose], [R.jacobian] and [Motor.log]. The library deliberately ships none
   yet: the solver policy — limits, nullspace, restarts, convergence tests — is
   research. Targets come from forward kinematics, so they are reachable.

   The pairing is the whole trick: [log (T^-1 M)] is a body-frame error, so it
   goes with [~space:`Body`]. Targets, Jacobians and steps are all batches, so
   the loop solves eight problems at once, and joint limits are a projection. *)
let ik () =
  section "inverse kinematics: 8 reachable pose targets, batched Gauss-Newton";
  let m = R.six_dof () in
  let ee = R.frame m "ee" in
  let limit = 1.2 in
  let clamp = Nx.clamp ~min:(-.limit) ~max:limit in
  (* Targets strictly inside the limits: one drawn exactly on a limit leaves the
     solution on the boundary, where the projection and the step fight. *)
  let truth = Nx.tanh (Nx.mul_s (Nx.randn f32 [| 8; 6 |]) 1.5) in
  let targets = R.pose m ~frame:ee truth in
  let q = ref (Nx.tanh (Nx.add truth (Nx.mul_s (Nx.randn f32 [| 8; 6 |]) 0.4))) in
  let lambda = ref (Nx.full f32 [| 8 |] 1e-2) in
  let error_at q = M.log (M.mul (M.inverse targets) (R.pose m ~frame:ee q)) in
  let norm e = Nx.sum ~axes:[ 1 ] (Nx.mul e e) in
  for _ = 1 to 30 do
    let e = error_at !q in
    let j = R.jacobian ~space:`Body m ~frame:ee !q in
    let jt = Nx.transpose ~axes:[ 0; 2; 1 ] j in
    let normals =
      Nx.add (Nx.matmul jt j) (Nx.mul (Nx.reshape [| 8; 1; 1 |] !lambda) (Nx.eye f32 6))
    in
    let grads = Nx.matmul jt (Nx.reshape [| 8; 6; 1 |] e) in
    let step = Nx.reshape [| 8; 6 |] (Nx.matmul (Nx.inv normals) grads) in
    let candidate = clamp (Nx.sub !q step) in
    (* Levenberg-Marquardt per sample: accept a step that reduces the error and
       relax the damping, otherwise keep the configuration and stiffen it. *)
    let better = Nx.less (norm (error_at candidate)) (norm e) in
    q := Nx.where (Nx.reshape [| 8; 1 |] better) candidate !q;
    lambda := Nx.where better (Nx.mul_s !lambda 0.5) (Nx.mul_s !lambda 4.)
  done;
  let final = error_at !q in
  let residual = Nx.reshape [| 8 |] (Nx.sqrt (Nx.sum ~axes:[ 1 ] (Nx.mul final final))) in
  report "final residuals" residual;
  report "converged (< 1e-5)" (Nx.sum (Nx.cast f32 (Nx.less_s residual 1e-5)))

(* 7. Traceability: the model is pure [Nx], so it compiles and differentiates as
   it stands, and [pose_diff] swaps the branches for the closed-form derivative
   without changing the signature. *)
let traceable () =
  section "traceability: jit, jacfwd, and the custom rule";
  let m = R.six_dof () in
  let ee = R.frame m "ee" in
  let q = q_of [| 0.4; -0.3; 0.8; 0.2; -0.5; 0.9 |] in
  let compiled = Rune.jit' (fun q -> tip m ~frame:ee q) in
  let _ = compiled q in
  report "jit - eager" (Nx.sub (compiled q) (tip m ~frame:ee q));
  let analytic = R.motion_jacobian m ~frame:ee q in
  report
    "jacfwd(exp) - analytic"
    (Nx.sub (Rune.jacfwd' (fun q -> R.pose m ~frame:ee q) q) analytic);
  report
    "jacfwd(rule) - analytic"
    (Nx.sub (Rune.jacfwd' (R.pose_diff m ~frame:ee) q) analytic)

let () =
  Rng.with_key (Rng.key 7) (fun () ->
    workspace ();
    camera ();
    velocity ();
    statics ();
    singularity ();
    ik ();
    traceable ();
    Format.printf
      "@,\
       Not here yet: task residuals (point, line, plane, circle, sphere, point pair), a \
       shipped IK solver with limits and nullspace, Jdot, manipulability, dynamics, \
       model I/O, floating bases, and a robot renderer.@.")
