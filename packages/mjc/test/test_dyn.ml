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

let models =
  [ "pendulum"
  ; "chain6"
  ; "floating"
  ; "ball"
  ; "actuators"
  ; "stiff"
  ; "arm2"
  ; "tendonlim"
  ; "exclude"
  ]

(* A fixture carries a tensor only for the stages its model has: a model with
   no actuators has no control to compare. The reader's only failure is a
   missing name, which is exactly the question being asked. *)
let optional f name =
  try Some (tensor f name) with
  | Support.Bad _ -> None

let staged m s =
  let f = fixture m in
  let qpos = tensor f "qpos"
  and qvel = tensor f "qvel" in
  let d = data_with_qpos (model m) (row qpos s) in
  let d = { d with Mjc.Data.qvel = vec1 (row qvel s) } in
  (* A model with actuators is asked about the state it was recorded at, which
     includes the control and the activation. *)
  let d =
    match optional f "ctrl" with
    | None -> d
    | Some c -> { d with Mjc.Data.ctrl = vec1 (row c s) }
  in
  let d =
    match optional f "act" with
    | None -> d
    | Some a -> { d with Mjc.Data.act = vec1 (row a s) }
  in
  F.forward (model m) d

let test_velocity () =
  List.iter models ~f:(fun name ->
    let f = fixture name in
    for s = 0 to f.nstates - 1 do
      let d = staged name s in
      List.iter
        [ "cvel", d.cvel
        ; "cdof_dot", d.cdof_dot
        ; "subtree_linvel", d.subtree_linvel
        ; "subtree_angmom", d.subtree_angmom
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
    let optional name = optional f name in
    (* An actuator model's step reads the control and the activation, and the
       recorded trajectory was driven with them. *)
    let ctrls = optional "traj_ctrl"
    and acts = optional "traj_act" in
    let set_ctrl d k =
      match ctrls with
      | None -> d
      | Some c -> { d with Mjc.Data.ctrl = vec1 (row c k) }
    in
    let d = ref { (data_with_qpos m (row qpos 0)) with qvel = vec1 (row qvel 0) } in
    d := set_ctrl !d 0;
    (match acts with
     | None -> ()
     | Some a -> d := { !d with Mjc.Data.act = vec1 (row a 0) });
    for k = 0 to states - 2 do
      d := set_ctrl !d k;
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

(* implicitfast: the velocity update always goes through [M + h diag(B)], so
   the trajectory is the integrator's even where Euler would have skipped the
   solve. *)
let test_implicitfast () =
  let m = model "fast" in
  let topo : Mjc.Topo.t = m.topo in
  equal
    ~msg:"integrator read"
    bool
    true
    (Poly.( = ) topo.integrator Mjc.Topo.Implicit_fast);
  let f = fixture_traj "fast" in
  let qpos = tensor f "traj_qpos"
  and qvel = tensor f "traj_qvel" in
  let d = ref { (data_with_qpos m (row qpos 0)) with qvel = vec1 (row qvel 0) } in
  for k = 0 to f.nstates - 2 do
    d := S.step m !d;
    parity
      ~tol:1e-10
      ~name:(Printf.sprintf "fast qpos %d" (k + 1))
      (row qpos (k + 1))
      (Nx.to_array !d.qpos);
    parity
      ~tol:1e-10
      ~name:(Printf.sprintf "fast qvel %d" (k + 1))
      (row qvel (k + 1))
      (Nx.to_array !d.qvel)
  done

let () =
  Stdlib.exit
    (run
       "mjc dynamics"
       [ test "velocity stage and bias" test_velocity
       ; test "ten steps against mj_step" test_trajectory
       ; test "step composes and counts time" test_step
       ; test "implicitfast trajectory" test_implicitfast
       ])
