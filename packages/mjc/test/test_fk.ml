(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Forward kinematics against MuJoCo's own [mj_kinematics], from the states
   the fixture recorded, plus the batched and compiled agreement the design
   note asks for: a vmapped forward pass is a loop, and a jitted one is the
   eager one. *)

open Base
open Windtrap
open Support
module K = Mjc.Kinematics

let models = [ "pendulum"; "chain6"; "floating"; "ball" ]

let pose_fields =
  [ "xpos"
  ; "xquat"
  ; "xanchor"
  ; "xaxis"
  ; "xipos"
  ; "ximat"
  ; "geom_xpos"
  ; "geom_xmat"
  ; "site_xpos"
  ; "site_xmat"
  ]

let pose (d : Mjc.Data.host) field =
  match field with
  | "xpos" -> d.xpos
  | "xquat" -> d.xquat
  | "xanchor" -> d.xanchor
  | "xaxis" -> d.xaxis
  | "xipos" -> d.xipos
  | "ximat" -> d.ximat
  | "geom_xpos" -> d.geom_xpos
  | "geom_xmat" -> d.geom_xmat
  | "site_xpos" -> d.site_xpos
  | "site_xmat" -> d.site_xmat
  | field -> failwith ("unknown pose field " ^ field)

let test_parity () =
  List.iter models ~f:(fun name ->
    let m = model name
    and f = fixture name in
    let qpos = tensor f "qpos" in
    for s = 0 to f.nstates - 1 do
      let d = K.forward m (data_with_qpos m (row qpos s)) in
      List.iter pose_fields ~f:(fun field ->
        parity
          ~tol:1e-12
          ~name:(Printf.sprintf "%s %s state %d" name field s)
          (row (tensor f field) s)
          (Nx.to_array (pose d field)))
    done)

(* The whole forward pass, batched. Every leaf of the argument carries the
   state axis, so the constants [forward] derives from the argument carry it
   too. *)
let test_vmap () =
  List.iter models ~f:(fun name ->
    let m = model name
    and f = fixture name in
    let qpos = tensor f "qpos" in
    let states = List.init f.nstates ~f:(fun s -> data_with_qpos m (row qpos s)) in
    let out =
      Rune.vmap
        (Nx.Ptree.( @-> ) (Mjc.Data.ptree ()) (Nx.Ptree.returns (Mjc.Data.ptree ())))
        (K.forward m)
        (batch_data states)
    in
    let slice flat s width = Array.sub flat ~pos:(s * width) ~len:width in
    for s = 0 to f.nstates - 1 do
      parity
        ~tol:1e-12
        ~name:(Printf.sprintf "%s vmap xpos state %d" name s)
        (row (tensor f "xpos") s)
        (slice (Nx.to_array out.xpos) s (Array.length (row (tensor f "xpos") s)));
      parity
        ~tol:1e-12
        ~name:(Printf.sprintf "%s vmap xquat state %d" name s)
        (row (tensor f "xquat") s)
        (slice (Nx.to_array out.xquat) s (Array.length (row (tensor f "xquat") s)))
    done)

let test_jit () =
  List.iter models ~f:(fun name ->
    let m = model name
    and f = fixture name in
    let qpos = tensor f "qpos" in
    let data = data_with_qpos m (row qpos 0) in
    let compiled =
      Rune.jit
        (Nx.Ptree.( @-> ) (Mjc.Data.ptree ()) (Nx.Ptree.returns (Mjc.Data.ptree ())))
        (K.forward m)
    in
    let eager = K.forward m data in
    let jitted = compiled data in
    List.iter pose_fields ~f:(fun field ->
      parity
        ~tol:1e-14
        ~name:(Printf.sprintf "%s jit %s" name field)
        (Nx.to_array (pose eager field))
        (Nx.to_array (pose jitted field))))

let test_check () =
  List.iter models ~f:(fun name ->
    match K.check (model name) with
    | Ok () -> ()
    | Error e -> fail e)

let test_shape_key () =
  List.iter models ~f:(fun name ->
    let m = model name in
    let k = Mjc.Model.shape_key m in
    equal ~msg:"nbody" int m.topo.nbody k.nbody;
    equal ~msg:"nv" int m.topo.nv k.nv;
    equal ~msg:"budget" bool true (k.nconmax > 0 && k.nefcmax > 0);
    equal
      ~msg:"equal to itself"
      bool
      true
      (Mjc.Model.equal_shape_key k (Mjc.Model.shape_key m)))

let () =
  Stdlib.exit
    (run
       "mjc kinematics"
       [ test "matches the C engine" test_parity
       ; test "batched equals the loop" test_vmap
       ; test "compiled equals eager" test_jit
       ; test "the model is in scope" test_check
       ; test "shape keys" test_shape_key
       ])
