(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Joint coordinates and position integration, against MuJoCo's own
   [mj_integratePos] and [mj_normalizeQuat] on the fixture states. *)

open Base
open Windtrap
open Support
module P = Mjc.Pose

let models = [ "pendulum"; "chain6"; "floating"; "ball" ]

let test_widths () =
  List.iter models ~f:(fun name ->
    let t = (model name).topo in
    let qpos_width =
      List.sum (module Int) (List.init t.njnt ~f:(P.qpos_width t)) ~f:Fn.id
    in
    let dof_width =
      List.sum (module Int) (List.init t.njnt ~f:(P.dof_width t)) ~f:Fn.id
    in
    equal ~msg:(name ^ " qpos width") int t.nq qpos_width;
    equal ~msg:(name ^ " dof width") int t.nv dof_width)

let test_integration () =
  List.iter models ~f:(fun name ->
    let m = model name in
    let f = fixture name in
    let qpos = tensor f "integ_qpos"
    and qvel = tensor f "integ_qvel"
    and expected = tensor f "integ_qpos_next" in
    let dt = Nx.item [] m.params.timestep in
    for s = 0 to f.nstates - 1 do
      let got =
        P.integrate_pos m.topo ~qpos:(vec1 (row qpos s)) ~qvel:(vec1 (row qvel s)) ~dt
      in
      parity
        ~tol:1e-15
        ~name:(Printf.sprintf "%s integrate state %d" name s)
        (row expected s)
        (Nx.to_array got)
    done)

let test_normalize () =
  List.iter [ "floating"; "ball" ] ~f:(fun name ->
    let m = model name in
    let t = m.topo in
    let f = fixture name in
    let qpos = tensor f "qpos" in
    (* Drift every quaternion by 1.3 and normalize it back. *)
    let drifted = row qpos 0 in
    for j = 0 to t.njnt - 1 do
      let adr = t.jnt_qposadr.(j) in
      let off =
        match t.jnt_type.(j) with
        | Mjc.Topo.Ball -> 0
        | Mjc.Topo.Free -> 3
        | Mjc.Topo.Hinge | Mjc.Topo.Slide -> -1
      in
      if off >= 0
      then
        for k = 0 to 3 do
          drifted.(adr + off + k) <- drifted.(adr + off + k) *. 1.3
        done
    done;
    let got = P.normalize_quat t (vec1 drifted) in
    parity ~tol:1e-15 ~name:(name ^ " normalize") (row qpos 0) (Nx.to_array got))

let () =
  Stdlib.exit
    (run
       "mjc pose"
       [ test "widths" test_widths
       ; test "position integration" test_integration
       ; test "quaternion normalization" test_normalize
       ])
