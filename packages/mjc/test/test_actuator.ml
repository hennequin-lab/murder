(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Actuators against MuJoCo: the transmission's length and velocity, the scalar
   force each actuator makes, the generalized force it contributes, and the
   activation trajectory of a stateful one.

   The corpus model carries one joint per actuator kind v1 has, so a single
   fixture decides all of them: a fixed-gain motor, a position servo, a
   velocity servo, an integrated setpoint, a damper written as a general
   actuator with zero gain, and a general gain/bias polynomial. *)

open Base
open Windtrap
open Support

let name = "actuators"
let m = model name
let f = fixture name
let ctrl = tensor f "ctrl"
let act = tensor f "act"

(* One recorded state: the pose, the velocity, the control and the activation,
   with everything the actuator stage computes left at zero. *)
let state s =
  let d = data_with_qpos m (row (tensor f "qpos") s) in
  { d with
    Mjc.Data.qvel = vec1 (row (tensor f "qvel") s)
  ; ctrl = vec1 (row ctrl s)
  ; act = vec1 (row act s)
  }

let parity_of ~tol s field got =
  parity
    ~tol
    ~name:(Printf.sprintf "%s state %d %s" name s field)
    (row (tensor f field) s)
    (Nx.to_array got)

let test_forces () =
  List.iter (List.range 0 f.nstates) ~f:(fun s ->
    let d = Mjc.Actuator.forward m (state s) in
    (* The engine's own scalars first: a wrong transmission length would show
       up in the force, but not where it comes from. *)
    parity_of ~tol:1e-12 s "actuator_length" d.Mjc.Data.actuator_length;
    parity_of ~tol:1e-12 s "actuator_velocity" d.Mjc.Data.actuator_velocity;
    parity_of ~tol:1e-12 s "actuator_force" d.Mjc.Data.actuator_force;
    parity_of ~tol:1e-12 s "qfrc_actuator" d.Mjc.Data.qfrc_actuator)

let test_act_dynamics () =
  (* A stateful actuator's job is what it does over time: [act' = ctrl] for
     the integrated setpoint, and the force follows the integrated state, not
     the control. *)
  List.iter (List.range 0 f.nstates) ~f:(fun s ->
    let d = Mjc.Actuator.forward m (state s) in
    parity_of ~tol:1e-12 s "act_dot" d.Mjc.Data.act_dot)

let () =
  (* Temporary: the trajectory's first state, by hand, so a wrong force at one
     state can be read off instead of inferred from a trajectory. *)
  (match Sys.getenv "MJC_PROBE_ACTUATOR" with
   | None -> ()
   | Some _ ->
     let t = fixture_traj name in
     let d =
       { (data_with_qpos m (row (tensor t "traj_qpos") 0)) with
         Mjc.Data.qvel = vec1 (row (tensor t "traj_qvel") 0)
       ; ctrl = vec1 (row (tensor t "traj_ctrl") 0)
       ; act = vec1 (row (tensor t "traj_act") 0)
       }
     in
     let tr = Mjc.Transmission.forward m d in
     let d = Mjc.Actuator.forward m d in
     let pp label v =
       Stdlib.Printf.printf
         "%s %s\n%!"
         label
         (String.concat
            ~sep:" "
            (Array.to_list (Array.map (Nx.to_array v) ~f:(Stdlib.Printf.sprintf "%.6f"))))
     in
     pp "probe ctrl    " d.Mjc.Data.ctrl;
     pp "probe length  " d.Mjc.Data.actuator_length;
     pp "probe velocity" d.Mjc.Data.actuator_velocity;
     pp "probe force   " d.Mjc.Data.actuator_force;
     pp "probe qfrc    " d.Mjc.Data.qfrc_actuator;
     pp "probe moment0 " (Nx.slice [ I 5; A ] tr.Mjc.Transmission.moment));
  Stdlib.exit
  @@ run
       "mjc actuator"
       [ test "transmission and force" test_forces
       ; test "activation dynamics" test_act_dynamics
       ; test "activation trajectory" (fun () ->
           let t = fixture_traj name in
           let qpos = tensor t "traj_qpos"
           and qvel = tensor t "traj_qvel"
           and tact = tensor t "traj_act"
           and tctrl = tensor t "traj_ctrl" in
           let d =
             ref
               { (data_with_qpos m (row qpos 0)) with
                 Mjc.Data.qvel = vec1 (row qvel 0)
               ; ctrl = vec1 (row tctrl 0)
               ; act = vec1 (row tact 0)
               }
           in
           for k = 0 to t.nstates - 2 do
             d := Mjc.Step.step m !d;
             parity
               ~tol:1e-10
               ~name:(Printf.sprintf "%s act %d" name (k + 1))
               (row tact (k + 1))
               (Nx.to_array !d.Mjc.Data.act);
             parity
               ~tol:1e-8
               ~name:(Printf.sprintf "%s qpos %d" name (k + 1))
               (row qpos (k + 1))
               (Nx.to_array !d.Mjc.Data.qpos);
             parity
               ~tol:1e-8
               ~name:(Printf.sprintf "%s qvel %d" name (k + 1))
               (row qvel (k + 1))
               (Nx.to_array !d.Mjc.Data.qvel);
             d := { !d with Mjc.Data.ctrl = vec1 (row tctrl (k + 1)) }
           done)
       ]
