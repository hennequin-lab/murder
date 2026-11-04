(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* V2's overlays: inertia boxes, centres of mass, joint axes, velocities and
   the energy HUD. Each is a pure function of a model and a state, so each is
   checked against arithmetic done by hand rather than against a picture. *)

open Base
open Nx
open Windtrap
module S = Mjc_viz.Scene

let fixtures = "../../test/fixtures"

let load name =
  match Mjc.Mjcb.read (fixtures ^ "/" ^ name ^ ".mjcb") with
  | Ok m -> m
  | Error e -> fail e

let state name =
  let m = load name in
  m, Mjc.Forward.forward m (Mjc.Data.create m)

let close ?(tol = 1e-12) a b = Float.( <= ) (Float.abs (a -. b)) tol
let yes msg b = equal ~msg bool true b

(* A body's inertia box is the box with the body's diagonal inertia, so a box
   geom's box comes back as itself and a sphere's as [r sqrt(3/5)]. *)
let test_inertia_of_a_box_and_a_sphere () =
  let m, d = state "blocks" in
  let boxes = S.inertia_boxes m d in
  equal ~msg:"one per body" int 2 (List.length boxes);
  let crate = List.nth_exn boxes 0
  and ball = List.nth_exn boxes 1 in
  let hx, hy, hz = crate.half in
  yes "crate hx" (close hx 0.2);
  yes "crate hy" (close hy 0.2);
  yes "crate hz" (close hz 0.1);
  let ux, uy, uz = crate.axes in
  yes "crate x axis" (close ux.S.x 1.);
  yes "crate y axis" (close uy.S.y 1.);
  yes "crate z axis" (close uz.S.z 1.);
  let expected = 0.09 *. Float.sqrt (3. /. 5.) in
  let ax, ay, az = ball.half in
  yes "ball hx" (close ~tol:1e-9 ax expected);
  yes "ball hy" (close ~tol:1e-9 ay expected);
  yes "ball hz" (close ~tol:1e-9 az expected)

(* The subtree centre of mass of the world is the mass-weighted mean of the
   bodies; a leaf's is its own. *)
let test_com () =
  let m, d = state "blocks" in
  let com = S.subtree_com m d in
  equal ~msg:"world and both bodies" int 3 (List.length com);
  let mass = Nx.to_array m.Mjc.Model.params.Mjc.Model.Params.body_mass
  and xipos = Nx.to_array d.Mjc.Data.xipos in
  let total = mass.(1) +. mass.(2) in
  let expect_x = ((mass.(1) *. xipos.(3)) +. (mass.(2) *. xipos.(6))) /. total in
  let expect_z = ((mass.(1) *. xipos.(5)) +. (mass.(2) *. xipos.(8))) /. total in
  yes "world com x" (close ~tol:1e-12 (List.nth_exn com 0).S.x expect_x);
  yes "world com z" (close ~tol:1e-12 (List.nth_exn com 0).S.z expect_z);
  (* the crate carries nothing: its subtree is itself *)
  yes "crate com x" (close (List.nth_exn com 1).x xipos.(3));
  yes "crate com z" (close (List.nth_exn com 1).z xipos.(5))

(* A hinge on an unrotated body turns about the axis the model gave it, and a
   slide moves along its own. *)
let test_joint_axes () =
  let m, d = state "limited" in
  let js = S.joint_axes m d in
  equal ~msg:"two joints" int 2 (List.length js);
  let shoulder = List.nth_exn js 0
  and extend = List.nth_exn js 1 in
  equal ~msg:"name" string "shoulder" shoulder.name;
  equal
    ~msg:"kind"
    string
    "hinge"
    (match shoulder.kind with
     | S.Hinge -> "hinge"
     | _ -> "?");
  yes "anchor z" (close shoulder.anchor.S.z 1.);
  yes "axis y" (close shoulder.axis.S.y 1.);
  yes "axis x" (close shoulder.axis.S.x 0.);
  yes "slide anchor z" (close extend.anchor.S.z 0.4);
  yes "slide axis x" (close extend.axis.S.x 1.)

(* A velocity arrow is the body's linear velocity at its centre of mass. With
   the shoulder spinning at 1 rad/s about y, the arm's centre of mass 0.3 below
   the anchor moves at 0.3 m/s: along -x, since omega x r with omega = +y and
   r = -z gives -x. *)
let test_velocity () =
  let m, d = state "limited" in
  let d = Mjc.Forward.forward m d in
  let dof = m.Mjc.Model.topo.jnt_dofadr.(0) in
  let d =
    { d with Mjc.Data.qvel = Nx.set [ I dof ] (Nx.scalar Nx.float64 1.) d.Mjc.Data.qvel }
  in
  let d = Mjc.Forward.forward m d in
  let v = S.velocities m d in
  equal ~msg:"one per body with mass" int 2 (List.length v);
  let arm = List.nth_exn v 0 in
  yes "arm vx" (close ~tol:1e-9 arm.vector.S.x (-0.3));
  yes "arm vy" (close ~tol:1e-9 arm.vector.S.y 0.);
  yes "arm vz" (close ~tol:1e-9 arm.vector.S.z 0.)

(* At rest the kinetic energy is zero and the potential is minus the mass
   moment of gravity; a free body moving at v has [1/2 m v^2]. *)
let test_energy () =
  let m, d = state "floating" in
  let e = S.mechanical_energy m d in
  let mass = Nx.item [ 1 ] m.Mjc.Model.params.Mjc.Model.Params.body_mass
  and xipos = Nx.to_array d.Mjc.Data.xipos
  and g = Nx.to_array m.Mjc.Model.params.Mjc.Model.Params.gravity in
  let expect =
    -1. *. (mass *. ((g.(0) *. xipos.(3)) +. (g.(1) *. xipos.(4)) +. (g.(2) *. xipos.(5))))
  in
  yes "potential" (close ~tol:1e-9 e.potential expect);
  yes "kinetic at rest" (close e.kinetic 0.);
  (* a free joint's first three velocities are linear *)
  let d =
    { d with Mjc.Data.qvel = Nx.set [ I 2 ] (Nx.scalar Nx.float64 1.) d.Mjc.Data.qvel }
  in
  let d = Mjc.Forward.forward m d in
  let e = S.mechanical_energy m d in
  yes "kinetic at 1 m/s" (close ~tol:1e-9 e.kinetic (0.5 *. mass))

(* The per-body sum must be the quadratic form it stands in for, or the HUD is
   a plausible-looking number. Both are read from the same state, with the mass
   matrix built once here to compare against. A free body first — no subtree is
   not its body — then chains, where [cvel]'s subtree reference shows up. *)
let kinetic_by_quadratic (_ : Mjc.Model.host) d =
  0.5
  *. Nx.item [] (Nx.dot (Nx.dot d.Mjc.Data.mass_matrix d.Mjc.Data.qvel) d.Mjc.Data.qvel)

let test_energy_is_the_quadratic_form () =
  List.iter [ "floating"; "limited"; "chain6"; "looped" ] ~f:(fun name ->
    let m = load name in
    let nv = m.Mjc.Model.topo.nv in
    let d = Mjc.Data.create m in
    let d =
      { d with
        Mjc.Data.qvel =
          Nx.init Nx.float64 [| nv |] (fun i -> 0.1 *. Float.of_int (i.(0) + 1))
      }
    in
    let d = Mjc.Forward.forward m d in
    let e = S.mechanical_energy m d in
    equal ~msg:(name ^ " kinetic") (float 1e-9) e.kinetic (kinetic_by_quadratic m d))

(* The layers are off unless asked for, which is what keeps a frame cheap when
   nothing needs them. *)
let test_layers_are_optional () =
  let m, d = state "limited" in
  let bare = S.of_state m d in
  equal ~msg:"no inertia" int 0 (List.length bare.inertia);
  equal ~msg:"no com" int 0 (List.length bare.com);
  equal ~msg:"no joints" int 0 (List.length bare.joints);
  equal ~msg:"no velocity" int 0 (List.length bare.velocity);
  equal ~msg:"no energy" bool true (Option.is_none bare.energy);
  let full =
    S.of_state ~inertia:true ~com:true ~joints:true ~velocity:true ~energy:true m d
  in
  equal ~msg:"inertia" int 2 (List.length full.inertia);
  equal ~msg:"com" int 3 (List.length full.com);
  equal ~msg:"joints" int 2 (List.length full.joints);
  equal ~msg:"velocity" int 2 (List.length full.velocity);
  equal ~msg:"energy" bool true (Option.is_some full.energy)

let () =
  Stdlib.exit
    (run
       "mjc viz: overlays"
       [ test "inertia of a box and a sphere" test_inertia_of_a_box_and_a_sphere
       ; test "centre of mass" test_com
       ; test "joint axes" test_joint_axes
       ; test "velocity" test_velocity
       ; test "energy" test_energy
       ; test "energy is the quadratic form" test_energy_is_the_quadratic_form
       ; test "layers are optional" test_layers_are_optional
       ])
