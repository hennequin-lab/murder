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

(* A box pair's manifold is a clipped polygon, and the engine emits its vertices
   in clip order while the tensor kernel takes the vertex set directly, so the
   rows of a box pair come out in a different order from MuJoCo's. The contacts
   are the same; the order reaches the solver only through the warm start. For
   the box models the rows are therefore compared as a set, sorted by distance
   and point, which a missing, duplicated or wrong contact still fails. *)

let flat_rows ~ncon ~dist ~pos ~normal ~frame =
  List.init ncon ~f:(fun i ->
    Array.concat
      [ [| dist.(i) |]
      ; Array.init 3 ~f:(fun k -> pos.((3 * i) + k))
      ; Array.init 3 ~f:(fun k -> normal.((3 * i) + k))
      ; Array.init 9 ~f:(fun k -> frame.((9 * i) + k))
      ])

let rec lex_compare_from k a b =
  if Int.( >= ) k (Array.length a)
  then 0
  else (
    let c = Float.compare a.(k) b.(k) in
    if Int.equal c 0 then lex_compare_from (k + 1) a b else c)

let sorted_flat ~ncon ~dist ~pos ~normal ~frame =
  let rows = flat_rows ~ncon ~dist ~pos ~normal ~frame in
  let rows =
    Array.of_list (List.sort rows ~compare:(fun a b -> lex_compare_from 0 a b))
  in
  Array.concat (Array.to_list rows)

let test_geometry_set () =
  List.iter [ "boxes2" ] ~f:(fun name ->
    let f = fixture name in
    List.iteri (states name) ~f:(fun s (d, ncon) ->
      let label tag = Printf.sprintf "%s state %d %s" name s tag in
      let slice n x = take n x in
      let expected =
        sorted_flat
          ~ncon
          ~dist:(slice ncon (row (tensor f "contact_dist") s))
          ~pos:(slice (ncon * 3) (row (tensor f "contact_pos") s))
          ~normal:(slice (ncon * 3) (row (tensor f "contact_normal") s))
          ~frame:(slice (ncon * 9) (row (tensor f "contact_frame") s))
      and actual =
        let rows x = Nx.to_array (Nx.slice [ R (0, ncon); A ] x) in
        sorted_flat
          ~ncon
          ~dist:(take ncon (Nx.to_array d.Mjc.Data.contact_dist))
          ~pos:(rows d.Mjc.Data.contact_pos)
          ~normal:(rows d.Mjc.Data.contact_normal)
          ~frame:(rows d.Mjc.Data.contact_frame)
      in
      parity ~tol:1e-12 ~name:(label "contacts as a set") expected actual))

let test_count () =
  List.iter
    [ "bounce"; "boxes"; "blocks"; "boxes2"; "humanoid"; "hopper"; "walker" ]
    ~f:(fun name ->
      let m = model name in
      let f = fixture name in
      List.iteri (states name) ~f:(fun s (d, ncon) ->
        equal ~msg:(Printf.sprintf "state %d ncon" s) int ncon (int_of_t d.Mjc.Data.ncon);
        (* the compaction sends contacts past the budget to a sink slot, so a
           model whose recording exceeds its own budget would lose them silently *)
        if Int.( > ) ncon m.Mjc.Model.topo.Mjc.Topo.nconmax
        then
          fail
            (Printf.sprintf
               "%s state %d: %d contacts exceed nconmax %d"
               name
               s
               ncon
               m.Mjc.Model.topo.Mjc.Topo.nconmax);
        ignore f))

(* The humanoid is compared as a set. Its contacts come out in MuJoCo's
   broadphase order, which is a body-pair traversal the engine has no reason to
   reproduce; the order reaches the solver only through the warm start, and the
   set is what the physics is. Every field of every contact still has to match. *)
let test_humanoid_set () =
  let f = fixture "humanoid" in
  let round x = Stdlib.Float.round (x *. 1e12) /. 1e12 in
  let sort rows =
    List.sort rows ~compare:(fun (a, b, d) (c, e, x) ->
      let k = Stdlib.compare (a, b) (c, e) in
      if Int.equal k 0 then Float.compare d x else k)
  in
  let ours d =
    let ncon = int_of_t d.Mjc.Data.ncon in
    let dist = Nx.to_array d.Mjc.Data.contact_dist
    and geom = Nx.to_array d.Mjc.Data.contact_geom in
    sort
      (List.init ncon ~f:(fun k ->
         ( Stdlib.Int32.to_int geom.(2 * k)
         , Stdlib.Int32.to_int geom.((2 * k) + 1)
         , round dist.(k) )))
  in
  let recorded s ncon =
    let geom = row (tensor f "contact_geom") s
    and dist = row (tensor f "contact_dist") s in
    sort
      (List.init ncon ~f:(fun k ->
         ( Stdlib.Int.of_float geom.(2 * k)
         , Stdlib.Int.of_float geom.((2 * k) + 1)
         , round dist.(k) )))
  in
  List.iteri (states "humanoid") ~f:(fun s (d, ncon) ->
    equal ~msg:(Printf.sprintf "state %d ncon" s) int ncon (int_of_t d.Mjc.Data.ncon);
    let expected = recorded s ncon
    and got = ours d in
    let pairs rows = List.map rows ~f:(fun (a, b, _) -> a, b) in
    if not (Poly.equal (pairs expected) (pairs got))
    then (
      let show rows =
        String.concat
          ~sep:" "
          (List.map rows ~f:(fun (a, b, x) -> Printf.sprintf "(%d,%d,%.10f)" a b x))
      in
      fail
        (Printf.sprintf
           "state %d: contact pairs differ\n  mujoco %s\n  ours   %s"
           s
           (show expected)
           (show got)));
    (* The pairs are exact; the distances are compared to 1e-9, which is what a
       pose this deeply interpenetrating (a random configuration buries the
       robot in the floor) can hold through two different formulations of the
       primitive distances. The trajectory test holds the same model to 1e-10
       over ten steps, which is the check that has physics in it. *)
    List.iter2_exn expected got ~f:(fun (a, b, x) (_c, _e, y) ->
      if Float.( > ) (Float.abs (x -. y)) 1e-9
      then
        fail
          (Printf.sprintf
             "state %d: contact (%d,%d): distance %.12g, mujoco %.12g"
             s
             a
             b
             x
             y)))

(* A model can say two bodies never touch. The test has to be able to tell that
   from "nothing touches": the fixture has three contacts, between the torso and
   a pebble and between that pebble and each limb, and the one pair the model
   excludes — the two limbs, which overlap by 4 cm — must not be among them. *)
let test_exclusions () =
  let m = model "exclude" in
  let f = fixture "exclude" in
  let t = m.Mjc.Model.topo in
  List.iteri (states "exclude") ~f:(fun s (d, ncon) ->
    equal ~msg:(Printf.sprintf "state %d ncon" s) int ncon (int_of_t d.Mjc.Data.ncon);
    let recorded = Array.sub (row (tensor f "contact_geom") s) ~pos:0 ~len:(2 * ncon) in
    let ours =
      Base.Array.map
        (Nx.to_array
           (Nx.reshape
              [| 2 * ncon |]
              (Nx.slice [ R (0, ncon); A ] d.Mjc.Data.contact_geom)))
        ~f:(fun x -> Stdlib.Int32.to_float x)
    in
    let pairs a =
      List.init ncon ~f:(fun k ->
        Stdlib.Int.of_float a.(2 * k), Stdlib.Int.of_float a.((2 * k) + 1))
    in
    let left = t.geom_bodyid.(1)
    and right = t.geom_bodyid.(2) in
    let names = pairs ours in
    List.iter (pairs recorded) ~f:(fun (g1, g2) ->
      if
        not
          (List.mem names (g1, g2) ~equal:(fun (a, b) (c, e) ->
             Int.equal a c && Int.equal b e))
      then
        fail
          (Printf.sprintf "state %d: the fixture has contact (%d,%d), we do not" s g1 g2));
    (* The excluded limbs are bodies 2 and 3, geoms 1 and 2, and they overlap. *)
    ignore (left, right);
    if List.mem names (1, 2) ~equal:(fun (a, b) (c, e) -> Int.equal a c && Int.equal b e)
    then fail (Printf.sprintf "state %d: the excluded pair (1,2) made contact" s);
    if Array.length ours <> Array.length recorded
    then fail (Printf.sprintf "state %d: contact count differs from the fixture" s));
  (* The recorded poses are random and mostly do not touch, which would make
     the check vacuous. The reference pose does: three contacts, and the two
     limbs overlapping by 4 cm are the pair the model excludes. *)
  let d = Mjc.Forward.forward m (Mjc.Data.create m) in
  equal ~msg:"reference pose ncon" int 3 (int_of_t d.Mjc.Data.ncon)

let test_geometry () =
  (* [boxes2] is compared as a set below: its box pairs have their own order *)
  List.iter [ "bounce"; "boxes"; "blocks"; "hopper"; "walker" ] ~f:(fun name ->
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
      (* The frame, not just its first row: the two tangents are what the
         pyramid rows and the force decomposition are built from. *)
      parity
        ~tol:1e-12
        ~name:(label "contact_frame")
        (take (ncon * 9) (row (tensor f "contact_frame") s))
        (Nx.to_array (rows d.Mjc.Data.contact_frame));
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
  and qvel = tensor f "qvel"
  and ctrl = tensor f "ctrl" in
  List.iteri
    (List.init f.nstates ~f:(fun s ->
       let d = { (data_with_qpos m (row qpos s)) with qvel = vec1 (row qvel s) } in
       (* The recorded rows were solved with the recorded controls; a model
          with actuators compared at zero control is a different problem. *)
       let d = { d with Mjc.Data.ctrl = vec1 (row ctrl s) } in
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
       ; test "box-box contacts, as a set" test_geometry_set
       ; test "an excluded pair never touches" test_exclusions
       ; test "humanoid contacts, as a set" test_humanoid_set
       ; test "contact rows, frictionless" (fun () -> test_rows_of "slide1")
       ; test "contact rows, pyramidal cone" (fun () -> test_rows_of "friction3")
       ; test "rows of a tendon equality" (fun () -> test_rows_of "tendoneq")
       ; test "contact trajectory, frictionless" (fun () -> test_trajectory_of "slide1")
       ; test "contact trajectory, pyramidal cone" (fun () ->
           test_trajectory_of "friction3")
       ; test "compiled equals eager" test_compiled
       ])
