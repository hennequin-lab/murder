(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rigid-body math. The laws are properties; the specific values pin the
   conventions MuJoCo uses, which the FK fixtures then confirm end to end. *)

open Base
open Nx
open Windtrap
open Support
module R = Mjc.Rigid

let q a = quat_of_array a
let v a = vec_of_array a

let quat_close
      ?(tol = 1e-14)
      ~name
      (expected : float array)
      (actual : Nx.host Nx.float64_t)
  : unit
  =
  parity ~tol ~name expected (Nx.to_array actual)

let test_known_products () =
  let i = q [| 0.; 1.; 0.; 0. |]
  and j = q [| 0.; 0.; 1.; 0. |] in
  quat_close ~name:"i * j = k" [| 0.; 0.; 0.; 1. |] (R.quat_mul i j);
  let a = q [| 0.5; 0.5; 0.5; 0.5 |]
  and b = q [| 0.5; -0.5; 0.5; -0.5 |] in
  quat_close ~name:"half-angle product" [| 0.5; -0.5; 0.5; 0.5 |] (R.quat_mul a b)

let test_conventions () =
  (* A quarter turn about z takes x to y: the matrix is the active rotation
     from the rotated frame to the parent, as MuJoCo's [mju_quat2Mat]. *)
  let half = Float.pi /. 4. in
  let rot = q [| Float.cos half; 0.; 0.; Float.sin half |] in
  let m = R.quat_to_mat rot in
  let rotated = R.mat_vec m (v [| 1.; 0.; 0. |]) in
  parity ~tol:1e-15 ~name:"z quarter turn of x" [| 0.; 1.; 0. |] (to_array rotated);
  let direct = R.quat_rot_vec rot (v [| 1.; 0.; 0. |]) in
  parity ~tol:1e-15 ~name:"rot_vec agrees" (to_array rotated) (to_array direct);
  (* A hinge about an axis that is not a basis vector. *)
  let axis = v [| 0.; 1.; 0. |] in
  let q90 =
    R.quat_of_axis_angle ~axis ~angle:(Nx.create f64 [| 1 |] [| Float.pi /. 2. |])
  in
  let z = R.quat_rot_vec q90 (v [| 0.; 0.; 1. |]) in
  parity ~tol:1e-15 ~name:"y quarter turn of z" [| 1.; 0.; 0. |] (to_array z)

let test_rotation_matrix () =
  (* R^T R = I and det R = 1, checked on a random unit quaternion. *)
  let rot = q [| 0.6; 0.0; 0.8; 0.0 |] in
  let m = Nx.to_array (R.quat_to_mat rot) in
  let e i j = m.((3 * i) + j) in
  for i = 0 to 2 do
    for j = 0 to 2 do
      let dot = ref 0.0 in
      for k = 0 to 2 do
        dot := !dot +. (e i k *. e j k)
      done;
      let expected = if Int.equal i j then 1.0 else 0.0 in
      equal ~msg:(Printf.sprintf "R^T R at %d,%d" i j) (float 1e-15) expected !dot
    done
  done;
  let det =
    (e 0 0 *. ((e 1 1 *. e 2 2) -. (e 1 2 *. e 2 1)))
    -. (e 0 1 *. ((e 1 0 *. e 2 2) -. (e 1 2 *. e 2 0)))
    +. (e 0 2 *. ((e 1 0 *. e 2 1) -. (e 1 1 *. e 2 0)))
  in
  equal ~msg:"det R" (float 1e-14) 1.0 det

let test_small_rotvec () =
  (* The series guard: at 1e-12 the rotation is the identity to machine
     precision, not 0/0. *)
  (* A rotation of 1.7e-12 rad has vector part 8.7e-13: the series gives the
     analytic value, where the naive sin(x)/x would give 0/0. *)
  let tiny = R.quat_of_rotvec (v [| 1e-12; 1e-12; 1e-12 |]) in
  parity ~tol:1e-16 ~name:"tiny rotvec" [| 1.; 5e-13; 5e-13; 5e-13 |] (to_array tiny);
  let zero = R.quat_of_rotvec (v [| 0.; 0.; 0. |]) in
  parity ~tol:1e-24 ~name:"zero rotvec" [| 1.; 0.; 0.; 0. |] (to_array zero);
  (* At a normal angle it agrees with the axis-angle construction. *)
  let r = v [| 0.3; -0.4; 0.5 |] in
  let n = Float.sqrt ((0.3 *. 0.3) +. (0.4 *. 0.4) +. (0.5 *. 0.5)) in
  let axis = v [| 0.3 /. n; -0.4 /. n; 0.5 /. n |] in
  let by_axis = R.quat_of_axis_angle ~axis ~angle:(Nx.create f64 [| 1 |] [| n |]) in
  parity
    ~tol:1e-15
    ~name:"rotvec agrees with axis-angle"
    (to_array by_axis)
    (to_array (R.quat_of_rotvec r))

let test_integrate () =
  (* Integrating a zero velocity normalizes and changes nothing else. *)
  let drifted = q [| 1.7; 0.0; 0.0; 0.0 |] in
  let stopped = R.quat_integrate drifted (v [| 0.; 0.; 0. |]) ~scale:1e-3 in
  parity ~tol:1e-15 ~name:"zero velocity" [| 1.; 0.; 0.; 0. |] (to_array stopped);
  (* A full turn about z in [1] second, integrated in 100 steps, returns to
     the identity. *)
  let w = v [| 0.; 0.; 2. *. Float.pi |] in
  let step = R.quat_integrate (q [| 1.; 0.; 0.; 0. |]) w ~scale:0.01 in
  let state = ref step in
  for _ = 1 to 99 do
    state := R.quat_integrate !state w ~scale:0.01
  done;
  (* A full turn is the identity rotation, but the quaternion is [-1;0;0;0]:
     quaternions double-cover SO(3), so the claim is about the matrix. *)
  let identity = R.quat_to_mat (q [| 1.; 0.; 0.; 0. |]) in
  let turned = R.quat_to_mat !state in
  parity
    ~tol:1e-14
    ~name:"a full turn is the identity rotation"
    (to_array identity)
    (to_array turned)

let test_properties =
  group
    "properties"
    [ prop
        "mul is associative"
        Gen.(triple unit_quat_gen unit_quat_gen unit_quat_gen)
        (fun (a, b, c) ->
           let left = R.quat_mul (R.quat_mul (q a) (q b)) (q c) in
           let right = R.quat_mul (q a) (R.quat_mul (q b) (q c)) in
           parity ~tol:1e-13 ~name:"associativity" (to_array left) (to_array right))
    ; prop "conjugate is the inverse" unit_quat_gen (fun a ->
        let one = R.quat_mul (q a) (R.quat_conj (q a)) in
        parity ~tol:1e-14 ~name:"q q* = 1" [| 1.; 0.; 0.; 0. |] (to_array one))
    ; prop "the matrix and the quaternion rotate alike" quat_and_vec_gen (fun (a, x) ->
        let by_quat = R.quat_rot_vec (q a) (v x) in
        let by_mat = R.mat_vec (R.quat_to_mat (q a)) (v x) in
        parity ~tol:1e-13 ~name:"rot_vec = R v" (to_array by_mat) (to_array by_quat))
    ]

(* [quat_of_mat] must invert [quat_to_mat], including on the rotations that
   fall outside the trace branch: the branch is a representative choice, and the
   matrix must come back whatever the representative. *)
let test_mat_roundtrip () =
  let n = 64 in
  let q =
    Nx.init Nx.float64 [| n; 4 |] (fun i ->
      let k = i.(0)
      and j = i.(1) in
      let v = Float.of_int ((k * 4) + j + 1) in
      Float.sin (v *. 0.7))
  in
  let q = R.quat_normalize q in
  let m = R.quat_to_mat q in
  let q' = R.quat_of_mat m in
  let m' = R.quat_to_mat q' in
  parity ~tol:1e-12 ~name:"quat_of_mat round trip" (Nx.to_array m) (Nx.to_array m')

let () =
  Stdlib.exit
    (run
       "mjc rigid"
       [ test "known products" test_known_products
       ; test "conventions" test_conventions
       ; test "rotation matrix" test_rotation_matrix
       ; test "small rotations" test_small_rotvec
       ; test "integration" test_integrate
       ; test "matrix to quaternion" test_mat_roundtrip
       ; test_properties
       ])
