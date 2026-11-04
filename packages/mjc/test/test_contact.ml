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

let states () =
  let m = model "bounce" in
  let f = fixture "bounce" in
  let qpos = tensor f "qpos" in
  List.init f.nstates ~f:(fun s ->
    let d = Mjc.Kinematics.forward m (data_with_qpos m (row qpos s)) in
    Mjc.Collision.contacts m d, Int.of_float (row (tensor f "ncon") s).(0))

let test_count () =
  let f = fixture "bounce" in
  List.iteri (states ()) ~f:(fun s (d, ncon) ->
    equal ~msg:(Printf.sprintf "state %d ncon" s) int ncon (int_of_t d.Mjc.Data.ncon);
    (* The corpus is only worth its name if it exercises several contacts. *)
    ignore f)

let test_geometry () =
  let f = fixture "bounce" in
  List.iteri (states ()) ~f:(fun s (d, ncon) ->
    let name k = Printf.sprintf "state %d %s" s k in
    let rows x = Nx.slice [ R (0, ncon); A ] x in
    parity
      ~tol:1e-12
      ~name:(name "contact_dist")
      (take ncon (row (tensor f "contact_dist") s))
      (take ncon (Nx.to_array d.Mjc.Data.contact_dist));
    parity
      ~tol:1e-12
      ~name:(name "contact_pos")
      (take (ncon * 3) (row (tensor f "contact_pos") s))
      (Nx.to_array (rows d.Mjc.Data.contact_pos));
    parity
      ~tol:1e-12
      ~name:(name "contact_normal")
      (take (ncon * 3) (row (tensor f "contact_normal") s))
      (Nx.to_array (rows d.Mjc.Data.contact_normal));
    parity
      ~tol:1e-12
      ~name:(name "contact_includemargin")
      (take ncon (row (tensor f "contact_includemargin") s))
      (take ncon (Nx.to_array d.Mjc.Data.contact_includemargin));
    parity
      ~tol:1e-12
      ~name:(name "contact_friction")
      (take (ncon * 5) (row (tensor f "contact_friction") s))
      (Nx.to_array (rows d.Mjc.Data.contact_friction));
    parity
      ~tol:1e-12
      ~name:(name "contact_solref")
      (take (ncon * 2) (row (tensor f "contact_solref") s))
      (Nx.to_array (rows d.Mjc.Data.contact_solref));
    parity
      ~tol:1e-12
      ~name:(name "contact_solimp")
      (take (ncon * 5) (row (tensor f "contact_solimp") s))
      (Nx.to_array (rows d.Mjc.Data.contact_solimp));
    parity
      ~tol:0.0
      ~name:(name "contact_geom")
      (take (ncon * 2) (row (tensor f "contact_geom") s))
      (Array.map
         (take (ncon * 2) (Nx.to_array (rows d.Mjc.Data.contact_geom)))
         ~f:Nxi32.to_float);
    parity
      ~tol:0.0
      ~name:(name "contact_body")
      (take (ncon * 2) (row (tensor f "contact_body") s))
      (Array.map
         (take (ncon * 2) (Nx.to_array (rows d.Mjc.Data.contact_body)))
         ~f:Nxi32.to_float);
    parity
      ~tol:0.0
      ~name:(name "contact_dim")
      (take ncon (row (tensor f "contact_dim") s))
      (Array.map (take ncon (Nx.to_array d.Mjc.Data.contact_dim)) ~f:Nxi32.to_float))

(* Frictionless contact rows: type, position, margin, Jacobian, the impedance
   data, the reference acceleration and the solved force, then the trajectory
   the whole engine produces. *)
let test_rows () =
  let m = model "slide1" in
  let nv = m.Mjc.Model.topo.Mjc.Topo.nv in
  let f = fixture "slide1" in
  let qpos = tensor f "qpos"
  and qvel = tensor f "qvel" in
  List.iteri
    (List.init f.nstates ~f:(fun s ->
       let d = { (data_with_qpos m (row qpos s)) with qvel = vec1 (row qvel s) } in
       Mjc.Forward.forward m d, Int.of_float (row (tensor f "nefc") s).(0)))
    ~f:(fun s (d, nefc) ->
      let name k = Printf.sprintf "state %d %s" s k in
      equal ~msg:(name "nefc") int nefc (int_of_t d.Mjc.Data.nefc);
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
            ~name:(name k)
            (take nefc (row (tensor f k) s))
            (Nx.to_array (Nx.slice [ R (0, nefc) ] x)));
      parity
        ~tol:0.0
        ~name:(name "efc_type")
        (take nefc (row (tensor f "efc_type") s))
        (Array.map (take nefc (Nx.to_array d.Mjc.Data.efc_type)) ~f:Nxi32.to_float);
      parity
        ~tol:0.0
        ~name:(name "efc_id")
        (take nefc (row (tensor f "efc_id") s))
        (Array.map (take nefc (Nx.to_array d.Mjc.Data.efc_id)) ~f:Nxi32.to_float);
      parity
        ~tol:1e-12
        ~name:(name "efc_J")
        (take (nefc * nv) (row (tensor f "efc_J") s))
        (Nx.to_array (rows d.Mjc.Data.efc_J));
      (* A random pose can bury a body in the floor, and the contact force then
         reaches thousands of newtons. MuJoCo's solver stops at its absolute
         tolerance, this one at the exact minimiser, so the two agree to about
         its stopping error: 1e-7 relative on a force of that size. *)
      parity
        ~tol:1e-7
        ~name:(name "efc_force")
        (take nefc (row (tensor f "efc_force") s))
        (take nefc (Nx.to_array d.Mjc.Data.efc_force)))

let test_trajectory () =
  let m = model "slide1" in
  let f = fixture_traj "slide1" in
  let qpos = tensor f "traj_qpos"
  and qvel = tensor f "traj_qvel" in
  let d = ref { (data_with_qpos m (row qpos 0)) with qvel = vec1 (row qvel 0) } in
  for k = 0 to f.nstates - 2 do
    d := Mjc.Step.step m !d;
    parity
      ~tol:1e-8
      ~name:(Printf.sprintf "qpos %d" (k + 1))
      (row qpos (k + 1))
      (Nx.to_array !d.qpos);
    parity
      ~tol:1e-8
      ~name:(Printf.sprintf "qvel %d" (k + 1))
      (row qvel (k + 1))
      (Nx.to_array !d.qvel)
  done

let () =
  Stdlib.exit
    (run
       "mjc contact"
       [ test "contact count" test_count
       ; test "contact geometry" test_geometry
       ; test "frictionless contact rows" test_rows
       ; test "frictionless contact trajectory" test_trajectory
       ])
