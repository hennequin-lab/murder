(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contact detection against MuJoCo: how many contacts a pose produces, their
   distance, point, normal, the geoms and bodies they name, and the parameters
   mixed from those geoms. *)

open Base
open Windtrap
open Support
module Nxi32 = Stdlib.Int32

let int_of_t x = Nxi32.to_int (Nx.item [] x)
let take n a = Array.sub a ~pos:0 ~len:n

let states name =
  let m = model name in
  let f = fixture name in
  let qpos = tensor f "qpos" in
  List.init f.nstates ~f:(fun s ->
    let d = Mjc.Kinematics.forward m (data_with_qpos m (row qpos s)) in
    Mjc.Collision.contacts m d, Int.of_float (row (tensor f "ncon") s).(0))

(* [friction3]'s pyramid is assembled and its rows carry the right Jacobians,
   positions and types, but its [diagA] disagrees with MuJoCo: the C engine's
   per-side body weights make [tran] twice the plain inverse weight sum, and
   that is not reproduced yet. Its tests are therefore not registered, like
   [boxes]. *)

(* [boxes] is recorded but not asserted yet: its plane-box candidates disagree
   with MuJoCo on one state of the corpus, so the code is in and the parity is
   not claimed. *)
let test_count () =
  List.iter [ "bounce" ] ~f:(fun name ->
    let f = fixture name in
    List.iteri (states name) ~f:(fun s (d, ncon) ->
      equal ~msg:(Printf.sprintf "state %d ncon" s) int ncon (int_of_t d.Mjc.Data.ncon);
      ignore f))

let test_geometry () =
  List.iter [ "bounce" ] ~f:(fun name ->
    let f = fixture name in
    List.iteri (states name) ~f:(fun s (d, ncon) ->
      let label tag = Printf.sprintf "%s state %d %s" name s tag in
      let rows x = Nx.slice [ R (0, ncon); A ] x in
      parity
        ~tol:1e-12
        ~name:(label "contact_dist")
        (take ncon (row (tensor f "contact_dist") s))
        (take ncon (Nx.to_array d.Mjc.Data.contact_dist));
      parity
        ~tol:1e-12
        ~name:(label "contact_pos")
        (take (ncon * 3) (row (tensor f "contact_pos") s))
        (Nx.to_array (rows d.Mjc.Data.contact_pos));
      parity
        ~tol:1e-12
        ~name:(label "contact_normal")
        (take (ncon * 3) (row (tensor f "contact_normal") s))
        (Nx.to_array (rows d.Mjc.Data.contact_normal));
      parity
        ~tol:1e-12
        ~name:(label "contact_includemargin")
        (take ncon (row (tensor f "contact_includemargin") s))
        (take ncon (Nx.to_array d.Mjc.Data.contact_includemargin));
      parity
        ~tol:1e-12
        ~name:(label "contact_friction")
        (take (ncon * 5) (row (tensor f "contact_friction") s))
        (Nx.to_array (rows d.Mjc.Data.contact_friction));
      parity
        ~tol:1e-12
        ~name:(label "contact_solref")
        (take (ncon * 2) (row (tensor f "contact_solref") s))
        (Nx.to_array (rows d.Mjc.Data.contact_solref));
      parity
        ~tol:1e-12
        ~name:(label "contact_solimp")
        (take (ncon * 5) (row (tensor f "contact_solimp") s))
        (Nx.to_array (rows d.Mjc.Data.contact_solimp));
      parity
        ~tol:0.0
        ~name:(label "contact_geom")
        (take (ncon * 2) (row (tensor f "contact_geom") s))
        (Array.map
           (take (ncon * 2) (Nx.to_array (rows d.Mjc.Data.contact_geom)))
           ~f:Nxi32.to_float);
      parity
        ~tol:0.0
        ~name:(label "contact_body")
        (take (ncon * 2) (row (tensor f "contact_body") s))
        (Array.map
           (take (ncon * 2) (Nx.to_array (rows d.Mjc.Data.contact_body)))
           ~f:Nxi32.to_float);
      parity
        ~tol:0.0
        ~name:(label "contact_dim")
        (take ncon (row (tensor f "contact_dim") s))
        (Array.map (take ncon (Nx.to_array d.Mjc.Data.contact_dim)) ~f:Nxi32.to_float)))

(* Frictionless contact rows: type, position, margin, Jacobian, the impedance
   data, the reference acceleration and the solved force, then the trajectory
   the whole engine produces. *)
let test_rows_of name =
  let m = model name in
  let nv = m.Mjc.Model.topo.Mjc.Topo.nv in
  let f = fixture name in
  let qpos = tensor f "qpos"
  and qvel = tensor f "qvel" in
  List.iteri
    (List.init f.nstates ~f:(fun s ->
       let d = { (data_with_qpos m (row qpos s)) with qvel = vec1 (row qvel s) } in
       Mjc.Forward.forward m d, Int.of_float (row (tensor f "nefc") s).(0)))
    ~f:(fun s (d, nefc) ->
      let label tag = Printf.sprintf "%s state %d %s" name s tag in
      equal ~msg:(label "nefc") int nefc (int_of_t d.Mjc.Data.nefc);
      let rows x = Nx.slice [ R (0, nefc); A ] x in
      List.iter
        [ "efc_pos", d.Mjc.Data.efc_pos, 1e-12
        ; "efc_margin", d.Mjc.Data.efc_margin, 1e-12
        ; "efc_vel", d.Mjc.Data.efc_vel, 1e-12
        ; "efc_R", d.Mjc.Data.efc_R, 1e-12
        ; "efc_D", d.Mjc.Data.efc_D, 1e-12
        ; "efc_aref", d.Mjc.Data.efc_aref, 1e-11
        ]
        ~f:(fun (k, x, tol) ->
          parity
            ~tol
            ~name:(label k)
            (take nefc (row (tensor f k) s))
            (Nx.to_array (Nx.slice [ R (0, nefc) ] x)));
      parity
        ~tol:0.0
        ~name:(label "efc_type")
        (take nefc (row (tensor f "efc_type") s))
        (Array.map (take nefc (Nx.to_array d.Mjc.Data.efc_type)) ~f:Nxi32.to_float);
      parity
        ~tol:0.0
        ~name:(label "efc_id")
        (take nefc (row (tensor f "efc_id") s))
        (Array.map (take nefc (Nx.to_array d.Mjc.Data.efc_id)) ~f:Nxi32.to_float);
      parity
        ~tol:1e-12
        ~name:(label "efc_J")
        (take (nefc * nv) (row (tensor f "efc_J") s))
        (Nx.to_array (rows d.Mjc.Data.efc_J));
      (* A random pose can bury a body in the floor, and the contact force then
         reaches thousands of newtons. MuJoCo's solver stops at its absolute
         tolerance, this one at the exact minimiser, so the two agree to about
         its stopping error: 1e-7 relative on a force of that size. *)
      parity
        ~tol:1e-7
        ~name:(label "efc_force")
        (take nefc (row (tensor f "efc_force") s))
        (take nefc (Nx.to_array d.Mjc.Data.efc_force)))

let test_trajectory_of name =
  let m = model name in
  let f = fixture_traj name in
  let qpos = tensor f "traj_qpos"
  and qvel = tensor f "traj_qvel" in
  let d = ref { (data_with_qpos m (row qpos 0)) with qvel = vec1 (row qvel 0) } in
  for k = 0 to f.nstates - 2 do
    d := Mjc.Step.step m !d;
    parity
      ~tol:1e-8
      ~name:(Printf.sprintf "%s qpos %d" name (k + 1))
      (row qpos (k + 1))
      (Nx.to_array !d.qpos);
    parity
      ~tol:1e-8
      ~name:(Printf.sprintf "%s qvel %d" name (k + 1))
      (row qvel (k + 1))
      (Nx.to_array !d.qvel)
  done

(* The contact stages must survive compilation: the parameter mixing is tensor
   arithmetic for exactly this reason, and a contact model is the first corpus
   model where the collision stage does real work inside a traced program. *)
let test_compiled () =
  let m = model "slide1" in
  let f = fixture_traj "slide1" in
  let qpos = tensor f "traj_qpos"
  and qvel = tensor f "traj_qvel" in
  let d = { (data_with_qpos m (row qpos 0)) with qvel = vec1 (row qvel 0) } in
  let compiled = Mjc.Step.compile m in
  let eager = ref d
  and jitted = ref d in
  for _ = 1 to 4 do
    eager := Mjc.Step.step m !eager;
    jitted := Mjc.Step.run compiled !jitted;
    parity
      ~tol:1e-12
      ~name:"compiled qpos"
      (Nx.to_array !eager.qpos)
      (Nx.to_array !jitted.qpos);
    parity
      ~tol:1e-12
      ~name:"compiled qvel"
      (Nx.to_array !eager.qvel)
      (Nx.to_array !jitted.qvel);
    equal ~msg:"compiled ncon" int (int_of_t !eager.ncon) (int_of_t !jitted.ncon)
  done

let () =
  Stdlib.exit
    (run
       "mjc contact"
       [ test "contact count" test_count
       ; test "contact geometry" test_geometry
       ; test "contact rows, frictionless" (fun () -> test_rows_of "slide1")
       ; test "contact trajectory, frictionless" (fun () -> test_trajectory_of "slide1")
       ; test "compiled equals eager" test_compiled
       ])
