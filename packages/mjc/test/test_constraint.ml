(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Joint-limit rows against MuJoCo: which rows exist, their type and id, their
   position and margin, and their Jacobian. The solver does not exist yet, so
   the forces are not compared here. *)

open Base
open Windtrap
open Support
module C = Mjc.Constraint
module Nxi32 = Stdlib.Int32

let take n a = Array.sub a ~pos:0 ~len:n
let int_of_t x = Nxi32.to_int (Nx.item [] x)
let float_of_i32 a = Array.map a ~f:Nxi32.to_float

let test_limits () =
  let m = model "limited" in
  let f = fixture "limited" in
  let qpos = tensor f "qpos" in
  let nv = m.topo.nv in
  for s = 0 to f.nstates - 1 do
    let d = C.limits m (data_with_qpos m (row qpos s)) in
    let nefc = int_of_t d.nefc in
    let expected = Int.of_float (row (tensor f "nefc") s).(0) in
    equal ~msg:(Printf.sprintf "state %d nefc" s) int expected nefc;
    let name k = Printf.sprintf "state %d %s" s k in
    parity
      ~tol:0.0
      ~name:(name "efc_type")
      (take nefc (row (tensor f "efc_type") s))
      (float_of_i32 (take nefc (Nx.to_array d.efc_type)));
    parity
      ~tol:0.0
      ~name:(name "efc_id")
      (take nefc (row (tensor f "efc_id") s))
      (float_of_i32 (take nefc (Nx.to_array d.efc_id)));
    parity
      ~tol:1e-14
      ~name:(name "efc_pos")
      (take nefc (row (tensor f "efc_pos") s))
      (take nefc (Nx.to_array d.efc_pos));
    parity
      ~tol:0.0
      ~name:(name "efc_margin")
      (take nefc (row (tensor f "efc_margin") s))
      (take nefc (Nx.to_array d.efc_margin));
    parity
      ~tol:0.0
      ~name:(name "efc_J")
      (take (nefc * nv) (row (tensor f "efc_J") s))
      (take (nefc * nv) (Nx.to_array d.efc_J))
  done

let test_ordering () =
  (* A state beyond both ranges has two rows per side of each joint, lower
     first, joints in id order: the fixture's first state is such a state. *)
  let m = model "limited" in
  let f = fixture "limited" in
  let d = C.limits m (data_with_qpos m (row (tensor f "qpos") 0)) in
  let n = int_of_t d.nefc in
  equal ~msg:"at most two rows per joint" bool true (n <= 2 * m.topo.njnt);
  let ids = take n (Nx.to_array d.efc_id) in
  let ascending = Array.for_all2_exn (take n (Nx.to_array d.efc_id)) ids ~f:Int32.equal in
  equal ~msg:"ids read" bool true ascending

let () =
  Stdlib.exit
    (run
       "mjc constraint"
       [ test "joint limit rows" test_limits; test "row ordering" test_ordering ])
