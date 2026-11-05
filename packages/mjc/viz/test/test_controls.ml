(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Joint control, which is the FK debugging tool: picking a joint, writing one
   coordinate, and the forward pass that makes the links follow. No display is
   involved, which is the point of keeping it out of the window. *)

open Base
open Windtrap
module C = Mjc_viz.Controls
module S = Mjc_viz.Scene

let fixtures = "../../test/fixtures"

let load name =
  match Mjc.Mjcb.read (fixtures ^ "/" ^ name ^ ".mjcb") with
  | Ok m -> m
  | Error e -> fail e

let state name =
  let m = load name in
  m, Mjc.Forward.forward m (Mjc.Data.create m)

let close a b = Float.( <= ) (Float.abs (a -. b)) 1e-12

let joint m name =
  let js = C.joints m in
  match
    Array.find js ~f:(fun (j : C.joint) ->
      String.equal j.name name
      || String.equal (Stdlib.Printf.sprintf "joint %d" j.index) name)
  with
  | Some j -> j
  | None -> fail (Stdlib.Printf.sprintf "no joint %s" name)

(* The model's joints are what the model says they are: kinds, addresses and
   ranges, in index order. *)
let test_joints () =
  let m = load "limited" in
  let js = C.joints m in
  equal ~msg:"count" int 2 (Array.length js);
  equal ~msg:"name 0" string "shoulder" js.(0).name;
  equal ~msg:"kind 0" bool true (C.driven js.(0));
  equal
    ~msg:"range 0"
    (option (pair (float 1e-12) (float 1e-12)))
    (Some (-0.4, 0.4))
    js.(0).limits;
  equal ~msg:"name 1" string "extend" js.(1).name;
  equal
    ~msg:"range 1"
    (option (pair (float 1e-12) (float 1e-12)))
    (Some (-0.1, 0.15))
    js.(1).limits;
  (* the addresses are the model's *)
  equal ~msg:"qposadr" int m.Mjc.Model.topo.jnt_qposadr.(1) js.(1).qposadr

(* A hinge writes radians, a slide writes length, and the clamp is the model's
   range rather than the caller's wish. *)
let test_set_and_clamp () =
  let m, d = state "limited" in
  let shoulder = joint m "shoulder" in
  let d, written = C.set m d shoulder 0.25 in
  equal ~msg:"written" bool true (close written 0.25);
  equal ~msg:"qpos" bool true (close (C.position d shoulder) 0.25);
  let d, written = C.set m d shoulder 5.0 in
  equal ~msg:"clamped to hi" bool true (close written 0.4);
  equal ~msg:"qpos clamped" bool true (close (C.position d shoulder) 0.4);
  let d, written = C.set m d shoulder (-5.0) in
  equal ~msg:"clamped to lo" bool true (close written (-0.4));
  equal ~msg:"qpos clamped" bool true (close (C.position d shoulder) (-0.4))

(* The point of the tool: the links follow. The arm's geom sits 0.3 below the
   shoulder anchor at (0, 0, 1), so after the hinge turns by [t] about +y, the
   capsule's midpoint is at (-0.3 sin t, 0, 1 - 0.3 cos t). The range is
   ±0.4 rad, so [t] is chosen inside it; that the value was clamped outside is
   the previous test's business. *)
let test_kinematics_follows () =
  let m, d = state "limited" in
  let shoulder = joint m "shoulder" in
  let t = 0.35 in
  let d, written = C.set m d shoulder t in
  equal ~msg:"written" bool true (close written t);
  let scene = S.of_state m d in
  match (List.nth_exn scene.geoms 0).shape with
  | Capsule { a; b; _ } ->
    let mx = (a.S.x +. b.S.x) /. 2.
    and my = (a.S.y +. b.S.y) /. 2.
    and mz = (a.S.z +. b.S.z) /. 2. in
    equal
      ~msg:"midpoint x"
      bool
      true
      (Float.( <= ) (Float.abs (mx +. (0.3 *. Float.sin t))) 1e-9);
    equal ~msg:"midpoint y" bool true (Float.( <= ) (Float.abs my) 1e-9);
    equal
      ~msg:"midpoint z"
      bool
      true
      (Float.( <= ) (Float.abs (mz -. (1. -. (0.3 *. Float.cos t)))) 1e-9)
  | _ -> fail "geom 0 is not a capsule"

(* One keystroke, and the reference position. *)
let test_nudge_and_zero () =
  let m, d = state "limited" in
  let extend = joint m "extend" in
  let d, _ = C.nudge m d extend 3.0 in
  equal ~msg:"three steps" bool true (close (C.position d extend) 0.03);
  let d, _ = C.zero m d extend in
  equal ~msg:"zero is qpos0" bool true (close (C.position d extend) 0.0)

(* A free joint is not driven: the write is refused and the state is
   untouched, rather than moving whichever of its seven coordinates came
   first. *)
let test_free_not_driven () =
  let m, d = state "floating" in
  let root = joint m "root" in
  equal
    ~msg:"kind"
    string
    "free"
    (match root.kind with
     | Free -> "free"
     | _ -> "other");
  equal ~msg:"driven" bool false (C.driven root);
  let before = Mjc.Data.create m in
  let d, written = C.nudge m d root 1.0 in
  equal ~msg:"written is unchanged" bool true (close written (C.position d root));
  equal
    ~msg:"qpos untouched"
    bool
    true
    (Array.equal
       Float.( = )
       (Nx.to_array d.Mjc.Data.qpos)
       (Nx.to_array (Mjc.Kinematics.forward m before).Mjc.Data.qpos))

(* The HUD line says what the joint is and why it cannot move, which is how a
   user learns that a ball joint is not drivable rather than watching a
   keystroke do nothing. *)
let test_describe () =
  let m, d = state "limited" in
  equal
    ~msg:"driven"
    string
    "shoulder = 0.000  (-0.400 .. 0.400)  [ ]"
    (C.describe d (joint m "shoulder"));
  let mf, df = state "floating" in
  equal
    ~msg:"free"
    string
    "root: free joint, not driven"
    (C.describe df (joint mf "root"))

let () =
  Stdlib.exit
    (run
       "mjc viz: joint control"
       [ test "joints" test_joints
       ; test "set and clamp" test_set_and_clamp
       ; test "kinematics follows" test_kinematics_follows
       ; test "nudge and zero" test_nudge_and_zero
       ; test "free joint is not driven" test_free_not_driven
       ; test "describe" test_describe
       ])
