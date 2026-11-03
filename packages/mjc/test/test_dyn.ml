(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The velocity stage and the smooth acceleration, then ten steps of the whole
   engine against MuJoCo's own mj_step. The trajectory is the end-to-end test:
   it fails if any stage, or the order of the integrator, is wrong. *)

open Base
open Windtrap
open Support
module F = Mjc.Forward
module S = Mjc.Step

let models = [ "pendulum"; "chain6"; "floating"; "ball" ]

let staged m s =
  let f = fixture m in
  let qpos = tensor f "qpos"
  and qvel = tensor f "qvel" in
  let d = data_with_qpos (model m) (row qpos s) in
  F.forward (model m) { d with qvel = vec1 (row qvel s) }

let test_velocity () =
  List.iter models ~f:(fun name ->
    let f = fixture name in
    for s = 0 to f.nstates - 1 do
      let d = staged name s in
      List.iter
        [ "cvel", d.cvel
        ; "cdof_dot", d.cdof_dot
        ; "qfrc_bias", d.qfrc_bias
        ; "qfrc_passive", d.qfrc_passive
        ; "qfrc_smooth", d.qfrc_smooth
        ]
        ~f:(fun (k, got) ->
          parity
            ~tol:1e-12
            ~name:(Printf.sprintf "%s %s state %d" name k s)
            (row (tensor f k) s)
            (Nx.to_array got))
    done)

let test_trajectory () =
  List.iter models ~f:(fun name ->
    let m = model name in
    let f = fixture_traj name in
    let qpos = tensor f "traj_qpos"
    and qvel = tensor f "traj_qvel" in
    let qacc = tensor f "traj_qacc" in
    let states = f.nstates in
    let d = ref { (data_with_qpos m (row qpos 0)) with qvel = vec1 (row qvel 0) } in
    (* A composed step accumulates the stages' rounding, so its tolerance is
       the design note's for a composed stage, not the 1e-12 of one stage. *)
    for k = 0 to states - 2 do
      let fwd = F.forward m !d in
      parity
        ~tol:1e-10
        ~name:(Printf.sprintf "%s qacc %d" name k)
        (row qacc k)
        (Nx.to_array fwd.qacc);
      d := S.advance m fwd;
      parity
        ~tol:1e-10
        ~name:(Printf.sprintf "%s qpos %d" name (k + 1))
        (row qpos (k + 1))
        (Nx.to_array !d.qpos);
      parity
        ~tol:1e-10
        ~name:(Printf.sprintf "%s qvel %d" name (k + 1))
        (row qvel (k + 1))
        (Nx.to_array !d.qvel)
    done)

let test_step () =
  (* [step] is [advance] of [forward], and time advances by dt. *)
  let m = model "pendulum" in
  let f = fixture_traj "pendulum" in
  let d =
    { (data_with_qpos m (row (tensor f "traj_qpos") 0)) with
      qvel = vec1 (row (tensor f "traj_qvel") 0)
    }
  in
  let once = S.step m d in
  let twice = S.step m once in
  let dt = Nx.item [] m.params.timestep in
  equal ~msg:"time after one step" (float 1e-15) dt (Nx.item [] once.time);
  equal ~msg:"time after two steps" (float 1e-15) (2.0 *. dt) (Nx.item [] twice.time)

let () =
  Stdlib.exit
    (run
       "mjc dynamics"
       [ test "velocity stage and bias" test_velocity
       ; test "ten steps against mj_step" test_trajectory
       ; test "step composes and counts time" test_step
       ])
