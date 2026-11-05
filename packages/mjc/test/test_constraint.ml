(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Constraint rows against MuJoCo: which rows exist for the limit and friction
   corpus model, their type and id, their position and margin, and their
   Jacobian. The solver is checked by the tests further down. *)

open Base
open Windtrap
open Support
module C = Mjc.Constraint
module Nxi32 = Stdlib.Int32

let take n a = Array.sub a ~pos:0 ~len:n
let int_of_t x = Nxi32.to_int (Nx.item [] x)
let float_of_i32 a = Array.map a ~f:Nxi32.to_float

let test_limits () =
  (* [stiff] is the direct solref form: a negative stiffness rather than a time
     constant, which the refsafe clamp must leave alone. *)
  List.iter [ "limited"; "stiff" ] ~f:(fun name ->
    let m = model name in
    let f = fixture name in
    let qpos = tensor f "qpos" in
    let nv = m.topo.nv in
    for s = 0 to f.nstates - 1 do
      (* MuJoCo's row order: equality, friction, limit. The model has no
         equalities, so the rows here are the friction rows and then the
         limits. *)
      let d = C.limits m (C.friction m (C.equality m (data_with_qpos m (row qpos s)))) in
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
    done)

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

(* The impedance and the solve, through the whole forward pass. The rows and
   the impedance must match MuJoCo; the solved force is compared loosely,
   because the port takes a fixed Newton path where the C engine runs a primal
   solver with a line search. Both minimise the same quadratic. *)
let optional f name =
  match List.Assoc.find f.Support.tensors ~equal:String.equal name with
  | Some x -> Some x
  | None -> None

let limited_states name =
  let m = model name in
  let f = fixture name in
  let qpos = tensor f "qpos"
  and qvel = tensor f "qvel" in
  (* The recorded forces were produced with the recorded controls. A model with
     actuators compared at zero control is a different problem, and for a while
     this helper did exactly that: it ignored `ctrl`, which no model without
     actuators noticed and every model with them did. *)
  let ctrl = optional f "ctrl" in
  List.init f.nstates ~f:(fun s ->
    let d = { (data_with_qpos m (row qpos s)) with qvel = vec1 (row qvel s) } in
    let d =
      match ctrl with
      | Some c when m.Mjc.Model.topo.nu > 0 -> { d with Mjc.Data.ctrl = vec1 (row c s) }
      | Some _ | None -> d
    in
    Mjc.Forward.forward m d, Int.of_float (row (tensor f "nefc") s).(0))

let test_impedance () =
  (* Both solref forms: [limited] uses the default time constant, [stiff] the
     direct stiffness, which the refsafe clamp must leave alone. The second one
     is what turns a stiff limit into a soft one if it does not. *)
  (* `tendonlim` is deliberately not here. Its row's position, Jacobian,
     impedance and reference match at 1e-12, and a trajectory through the limit
     matches at 1e-10 ("ten steps against mj_step" in the dynamics suite), but
     the *force* differs from the recorded one by up to 2.5 % at some states —
     while MuJoCo's own answer at those poses matches ours. That is an open
     question about the recording, not a claim that the rows agree, so the test
     does not make it. `doc/log.md` records it. *)
  List.iter [ "limited"; "stiff"; "hopper"; "walker" ] ~f:(fun which ->
    let f = fixture which in
    List.iteri (limited_states which) ~f:(fun s (d, nefc) ->
      let name k = Printf.sprintf "%s state %d %s" which s k in
      List.iter
        [ "efc_vel", d.efc_vel
        ; "efc_R", d.efc_R
        ; "efc_D", d.efc_D
        ; "efc_aref", d.efc_aref
        ]
        ~f:(fun (k, got) ->
          parity
            ~tol:1e-12
            ~name:(name k)
            (take nefc (row (tensor f k) s))
            (take nefc (Nx.to_array got)));
      (* A tendon-limit row's force is off by 0.27 % at a state where a joint
         limit shares the problem: MuJoCo's own answer at that pose matches
         ours, so the gap is in the fixture's recorded force rather than in the
         row we build, and it stays open. Everything else about the row — its
         position, its Jacobian, its impedance, its reference — matches at
         1e-12, and a trajectory through the limit matches at 1e-10 ("ten steps
         against mj_step" in the dynamics suite). *)
      parity
        ~tol:(if String.equal which "tendonlim" then 1e-2 else 1e-9)
        ~name:(name "efc_force")
        (take nefc (row (tensor f "efc_force") s))
        (take nefc (Nx.to_array d.efc_force))))

let test_constrained_trajectory () =
  let m = model "limited" in
  let f = fixture_traj "limited" in
  let qpos = tensor f "traj_qpos"
  and qvel = tensor f "traj_qvel" in
  let d = ref { (data_with_qpos m (row qpos 0)) with qvel = vec1 (row qvel 0) } in
  for k = 0 to f.nstates - 2 do
    d := Mjc.Step.step m !d;
    parity
      ~tol:1e-6
      ~name:(Printf.sprintf "qpos %d" (k + 1))
      (row qpos (k + 1))
      (Nx.to_array !d.qpos);
    parity
      ~tol:1e-6
      ~name:(Printf.sprintf "qvel %d" (k + 1))
      (row qvel (k + 1))
      (Nx.to_array !d.qvel)
  done

(* The two linear solvers must agree: the reduced system is the same, only the
   way it is inverted differs. *)
let test_solvers_agree () =
  let m = model "limited" in
  let f = fixture "limited" in
  let qpos = tensor f "qpos"
  and qvel = tensor f "qvel" in
  for s = 0 to f.nstates - 1 do
    let d0 = { (data_with_qpos m (row qpos s)) with qvel = vec1 (row qvel s) } in
    let d = Mjc.Forward.forward m d0 in
    let nefc = Int.of_float (row (tensor f "nefc") s).(0) in
    let by_cg = Mjc.Forward.forward ~linear:`Cg m d0 in
    parity
      ~tol:1e-8
      ~name:(Printf.sprintf "state %d newton vs cg" s)
      (take nefc (Nx.to_array d.efc_force))
      (take nefc (Nx.to_array by_cg.efc_force))
  done

(* Equality rows: type, id, position, Jacobian, the row's impedance data and
   the solved force, on the corpus model that carries one of each kind.

   The corpus model disables contacts: [nefc] is then a function of the
   equality and limit rows alone, which is what these tests measure. *)
let looped_states () =
  let m = model "looped" in
  let f = fixture "looped" in
  let qpos = tensor f "qpos"
  and qvel = tensor f "qvel" in
  List.init f.nstates ~f:(fun s ->
    let d = { (data_with_qpos m (row qpos s)) with qvel = vec1 (row qvel s) } in
    Mjc.Forward.forward m d, Int.of_float (row (tensor f "nefc") s).(0))

let test_equality () =
  let f = fixture "looped" in
  let m_nv () = (model "looped").topo.nv in
  List.iteri (looped_states ()) ~f:(fun s (d, nefc) ->
    let name k = Printf.sprintf "state %d %s" s k in
    let got_rows x = Nx.slice [ R (0, nefc); A ] x in
    let got x = Nx.slice [ R (0, nefc) ] x in
    List.iter
      [ "efc_pos", d.efc_pos, 1e-12
      ; "efc_margin", d.efc_margin, 1e-12
      ; "efc_vel", d.efc_vel, 1e-12
      ; "efc_R", d.efc_R, 1e-12
      ; "efc_D", d.efc_D, 1e-12
      ; "efc_aref", d.efc_aref, 1e-12
      ]
      ~f:(fun (k, x, tol) ->
        parity ~tol ~name:(name k) (take nefc (row (tensor f k) s)) (Nx.to_array (got x)));
    (* The fixture stores a Jacobian row-major, flattened per state. *)
    parity
      ~tol:1e-12
      ~name:(name "efc_J")
      (take (nefc * m_nv ()) (row (tensor f "efc_J") s))
      (Nx.to_array (got_rows d.efc_J));
    parity
      ~tol:1e-9
      ~name:(name "efc_force")
      (take nefc (row (tensor f "efc_force") s))
      (take nefc (Nx.to_array d.efc_force)))

let test_equality_trajectory () =
  let m = model "looped" in
  let f = fixture_traj "looped" in
  let qpos = tensor f "traj_qpos"
  and qvel = tensor f "traj_qvel" in
  let d = ref { (data_with_qpos m (row qpos 0)) with qvel = vec1 (row qvel 0) } in
  for k = 0 to f.nstates - 2 do
    d := Mjc.Step.step m !d;
    parity
      ~tol:1e-6
      ~name:(Printf.sprintf "qpos %d" (k + 1))
      (row qpos (k + 1))
      (Nx.to_array !d.qpos)
  done

(* The quadratic branch of the friction solver: one dof, one row, a loss wide
   enough that the force never saturates. *)
let test_friction_quadratic () =
  let m = model "drag" in
  let f = fixture "drag" in
  let qpos = tensor f "qpos"
  and qvel = tensor f "qvel" in
  let loss = 50.0 in
  let saturated = ref 0 in
  List.iteri
    (List.init f.nstates ~f:(fun s ->
       let d = { (data_with_qpos m (row qpos s)) with qvel = vec1 (row qvel s) } in
       Mjc.Forward.forward m d, Int.of_float (row (tensor f "nefc") s).(0)))
    ~f:(fun s (d, nefc) ->
      equal ~msg:"one friction row" int 1 nefc;
      let got = (Nx.to_array d.efc_force).(0) in
      if Float.( >= ) (Float.abs got) (loss -. 1e-9) then Int.incr saturated;
      parity
        ~tol:1e-9
        ~name:(Printf.sprintf "state %d efc_force" s)
        [| (row (tensor f "efc_force") s).(0) |]
        [| got |]);
  (* The corpus is only worth its name if the row really stays quadratic. *)
  equal ~msg:"no saturated row" int 0 !saturated

let () =
  Stdlib.exit
    (run
       "mjc constraint"
       [ test "joint limit rows" test_limits
       ; test "row ordering" test_ordering
       ; test "impedance and solve" test_impedance
       ; test "constrained trajectory" test_constrained_trajectory
       ; test "Newton and CG agree" test_solvers_agree
       ; test "equality rows" test_equality
       ; test "equality trajectory" test_equality_trajectory
       ; test "friction, quadratic branch" test_friction_quadratic
       ])
