(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The [.mjcb] interchange: the corpus reads and validates, a round trip is
   bit-exact, and each way of damaging a file is refused with the field named. *)

open Base
module Bytes = Stdlib.Bytes
module Format = Stdlib.Format
open Windtrap
open Support
module Mjcb = Mjc.Mjcb
module Model = Mjc.Model

let models = [ "pendulum"; "chain6"; "floating"; "ball" ]

let params (m : Model.host) : (string * Nx.host Nx.float64_t) list =
  let p = m.params in
  [ "body_pos", p.body_pos
  ; "body_quat", p.body_quat
  ; "body_ipos", p.body_ipos
  ; "body_iquat", p.body_iquat
  ; "body_mass", p.body_mass
  ; "body_inertia", p.body_inertia
  ; "body_subtreemass", p.body_subtreemass
  ; "jnt_axis", p.jnt_axis
  ; "jnt_pos", p.jnt_pos
  ; "jnt_range", p.jnt_range
  ; "jnt_limited", p.jnt_limited
  ; "jnt_stiffness", p.jnt_stiffness
  ; "jnt_margin", p.jnt_margin
  ; "jnt_solref", p.jnt_solref
  ; "jnt_solimp", p.jnt_solimp
  ; "dof_armature", p.dof_armature
  ; "dof_damping", p.dof_damping
  ; "dof_frictionloss", p.dof_frictionloss
  ; "dof_invweight0", p.dof_invweight0
  ; "qpos0", p.qpos0
  ; "qpos_spring", p.qpos_spring
  ; "geom_pos", p.geom_pos
  ; "geom_quat", p.geom_quat
  ; "geom_size", p.geom_size
  ; "geom_rgba", p.geom_rgba
  ; "geom_friction", p.geom_friction
  ; "geom_margin", p.geom_margin
  ; "geom_gap", p.geom_gap
  ; "geom_solmix", p.geom_solmix
  ; "geom_solref", p.geom_solref
  ; "geom_solimp", p.geom_solimp
  ; "site_pos", p.site_pos
  ; "site_quat", p.site_quat
  ; "site_size", p.site_size
  ; "gravity", p.gravity
  ; "timestep", p.timestep
  ]

let test_read () =
  List.iter models ~f:(fun name ->
    let m = model name in
    (match Model.validate m with
     | Ok () -> ()
     | Error e -> fail e);
    equal
      ~msg:(name ^ " records its MuJoCo version")
      bool
      true
      (not (String.equal m.topo.mujoco_version "unknown"));
    equal
      ~msg:(name ^ " topology validates")
      bool
      true
      (Result.is_ok (Mjc.Topo.validate m.topo)))

let test_roundtrip () =
  List.iter models ~f:(fun name ->
    let m = model name in
    let bytes =
      match Mjcb.to_string m with
      | Ok s -> s
      | Error e -> fail e
    in
    let m' =
      match Mjcb.of_string ~source:name bytes with
      | Ok m -> m
      | Error e -> fail e
    in
    equal ~msg:(name ^ " name") string m.topo.name m'.topo.name;
    equal
      ~msg:(name ^ " mujoco version")
      string
      m.topo.mujoco_version
      m'.topo.mujoco_version;
    equal ~msg:(name ^ " topology") bool true (Mjc.Topo.equal m.topo m'.topo);
    equal
      ~msg:(name ^ " shape key")
      string
      (Format.asprintf "%a" Model.pp_shape_key (Model.shape_key m))
      (Format.asprintf "%a" Model.pp_shape_key (Model.shape_key m'));
    List.iter2_exn (params m) (params m') ~f:(fun (field, a) (field', b) ->
      equal ~msg:(Printf.sprintf "%s %s" name field) string field field';
      parity ~tol:0.0 ~name:(name ^ " " ^ field) (Nx.to_array a) (Nx.to_array b)))

(* Each damage must be refused by name, not by a crash somewhere later. *)
let expect_refused label ~mentions s =
  match Mjcb.of_string ~source:"test" s with
  | Ok _ -> fail (label ^ ": accepted a damaged file")
  | Error e ->
    equal
      ~msg:(Printf.sprintf "%s: %S" label e)
      bool
      true
      (String.is_substring e ~substring:mentions)

let test_damage () =
  let m = model "chain6" in
  let bytes =
    match Mjcb.to_string m with
    | Ok s -> s
    | Error e -> fail e
  in
  let mutate i f =
    let b = Bytes.of_string bytes in
    Bytes.set b i (Char.of_int_exn (f (Char.to_int (Bytes.get b i))));
    Bytes.to_string b
  in
  expect_refused "bad magic" ~mentions:"magic" (mutate 0 (fun c -> c + 1));
  let wrong_version =
    let b = Bytes.of_string bytes in
    Bytes.set b 4 (Char.of_int_exn 99);
    Bytes.to_string b
  in
  expect_refused "wrong version" ~mentions:"version" wrong_version;
  expect_refused
    "truncated"
    ~mentions:"truncated"
    (String.sub bytes ~pos:0 ~len:(String.length bytes / 2));
  (* Rename a tensor: the reader must say which one is missing. *)
  let renamed =
    let needle = "body_pos" in
    let rec find i =
      if i + String.length needle > String.length bytes
      then fail "body_pos not found"
      else if String.equal (String.sub bytes ~pos:i ~len:(String.length needle)) needle
      then i
      else find (i + 1)
    in
    let b = Bytes.of_string bytes in
    Bytes.set b (find 0) 'x';
    Bytes.to_string b
  in
  expect_refused "renamed field" ~mentions:"body_pos" renamed

let test_file () =
  let m = model "pendulum" in
  let path = Stdlib.Filename.temp_file "mjc" ".mjcb" in
  (match Mjcb.write path m with
   | Ok () -> ()
   | Error e -> fail e);
  let m' =
    match Mjcb.read path with
    | Ok m -> m
    | Error e -> fail e
  in
  equal ~msg:"written and read" bool true (Mjc.Topo.equal m.topo m'.topo);
  Stdlib.Sys.remove path

let () =
  Stdlib.exit
    (run
       "mjc mjcb"
       [ test "reads the corpus" test_read
       ; test "round trip" test_roundtrip
       ; test "refuses damaged files" test_damage
       ; test "writes and reads a file" test_file
       ])
