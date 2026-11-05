(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fixed tendons against MuJoCo: the length a tendon measures, the Jacobian
   that says which degrees of freedom it measures it with, and the velocity the
   two imply. Then a tendon-driven finger, because a tendon actuator's moment
   arm is its gear times that Jacobian and a wrong arm is invisible in the
   geometry alone. *)

open Base
open Windtrap
open Support

let name = "finger"
let m = model name
let f = fixture name

let () =
  Stdlib.exit
  @@ run
       "mjc tendon"
       [ test "length, Jacobian and velocity" (fun () ->
           for s = 0 to f.nstates - 1 do
             let d = data_with_qpos m (row (tensor f "qpos") s) in
             let d = { d with Mjc.Data.qvel = vec1 (row (tensor f "qvel") s) } in
             let t = Mjc.Tendon.forward m d in
             let check what got =
               parity
                 ~tol:1e-12
                 ~name:(Printf.sprintf "%s state %d %s" name s what)
                 (row (tensor f what) s)
                 (Nx.to_array got)
             in
             check "ten_length" t.Mjc.Tendon.length;
             check "ten_velocity" t.Mjc.Tendon.velocity;
             check "ten_J" t.Mjc.Tendon.jac
           done)
       ; test "tendon actuator trajectory" (fun () ->
           (* One motor drives both joints through the tendon, so this is the
              test that the moment arm is the gear times the Jacobian and not
              just the gear. *)
           let t = fixture_traj name in
           let qpos = tensor t "traj_qpos"
           and qvel = tensor t "traj_qvel"
           and ctrl = tensor t "traj_ctrl" in
           let d =
             ref
               { (data_with_qpos m (row qpos 0)) with
                 Mjc.Data.qvel = vec1 (row qvel 0)
               ; ctrl = vec1 (row ctrl 0)
               }
           in
           for k = 0 to t.nstates - 2 do
             d := { !d with Mjc.Data.ctrl = vec1 (row ctrl k) };
             d := Mjc.Step.step m !d;
             parity
               ~tol:1e-9
               ~name:(Printf.sprintf "%s qpos %d" name (k + 1))
               (row qpos (k + 1))
               (Nx.to_array !d.Mjc.Data.qpos);
             parity
               ~tol:1e-9
               ~name:(Printf.sprintf "%s qvel %d" name (k + 1))
               (row qvel (k + 1))
               (Nx.to_array !d.Mjc.Data.qvel)
           done)
       ]
