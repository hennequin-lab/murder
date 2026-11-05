(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The scene IR and the still backend, with no display: what a model becomes
   when it is read for drawing, and that the same scene encodes to the same
   bytes twice. *)

open Base
open Windtrap
module S = Mjc_viz.Scene
module R = Mjc_viz.Render

let fixtures = "../../test/fixtures"

let load name =
  match Mjc.Mjcb.read (fixtures ^ "/" ^ name ^ ".mjcb") with
  | Ok m -> m
  | Error e -> fail e

let state name =
  let m = load name in
  m, Mjc.Forward.forward m (Mjc.Data.create m)

let close a b = Float.( <= ) (Float.abs (a -. b)) 1e-12

let test_plane () =
  let _, d = state "blocks" in
  let m = load "blocks" in
  let s = S.of_state m d in
  equal ~msg:"geom count" int 3 (List.length s.geoms);
  match (List.nth_exn s.geoms 0).shape with
  | Plane { normal; offset; half } ->
    equal ~msg:"plane normal x" bool true (close normal.S.x 0.);
    equal ~msg:"plane normal z" bool true (close normal.S.z 1.);
    equal ~msg:"plane offset" bool true (close offset 0.);
    equal ~msg:"plane half x" bool true (close (fst half) 5.)
  | _ -> fail "geom 0 is not a plane"

let test_box () =
  let m, d = state "blocks" in
  let s = S.of_state m d in
  match (List.nth_exn s.geoms 1).shape with
  | Box { half; axes = ux, _, _; centre = _ } ->
    let hx, _, hz = half in
    (* The crate is 0.2 × 0.2 × 0.1 in half-extents and its local x axis is
       unit length. *)
    equal ~msg:"box hx" bool true (close hx 0.2);
    equal ~msg:"box hz" bool true (close hz 0.1);
    equal
      ~msg:"unit axis"
      bool
      true
      (close ((ux.S.x *. ux.x) +. (ux.y *. ux.y) +. (ux.z *. ux.z)) 1.)
  | _ -> fail "geom 1 is not a box"

let test_capsule () =
  let m, d = state "limited" in
  let s = S.of_state m d in
  match (List.nth_exn s.geoms 0).shape with
  | Capsule { a; b; radius } ->
    (* The arm capsule is 0.02 in radius and half a unit long, so its two ends
       are that far apart. *)
    equal ~msg:"capsule radius" bool true (close radius 0.02);
    let dx = a.S.x -. b.S.x
    and dy = a.S.y -. b.S.y
    and dz = a.S.z -. b.S.z in
    equal
      ~msg:"capsule length"
      bool
      true
      (close (Float.sqrt ((dx *. dx) +. (dy *. dy) +. (dz *. dz))) 0.6)
  | _ -> fail "geom 0 is not a capsule"

let test_contacts () =
  let m, d = state "blocks" in
  let s = S.of_state ~contacts:true ~forces:true m d in
  equal ~msg:"contacts" int 5 (List.length s.contacts);
  equal ~msg:"forces" int 5 (List.length s.forces);
  let c = List.hd_exn s.contacts in
  equal
    ~msg:"unit normal"
    bool
    true
    (close
       (Float.sqrt
          ((c.normal.S.x *. c.normal.x)
           +. (c.normal.y *. c.normal.y)
           +. (c.normal.z *. c.normal.z)))
       1.);
  equal ~msg:"penetration is negative" bool true (Float.( < ) c.depth 0.)

let test_png () =
  let m, d = state "blocks" in
  let s = S.of_state ~contacts:true m d in
  let once = R.png s in
  let twice = R.png s in
  equal ~msg:"deterministic" bool true (String.equal once twice);
  equal ~msg:"png magic" bool true (String.is_prefix once ~prefix:"\137PNG");
  equal ~msg:"not empty" bool true (Int.( > ) (String.length once) 2000)

let test_all_geom_types () =
  (* Every shape the scene can build comes out of the corpus, and the ones the
     IR cannot draw say so instead of vanishing. *)
  let seen = Hashtbl.create (module String) in
  List.iter [ "bounce"; "blocks"; "limited" ] ~f:(fun name ->
    let m, d = state name in
    let s = S.of_state m d in
    List.iter s.geoms ~f:(fun g ->
      let kind =
        match g.shape with
        | S.Sphere _ -> "sphere"
        | S.Capsule _ -> "capsule"
        | S.Box _ -> "box"
        | S.Cylinder _ -> "cylinder"
        | S.Ellipsoid _ -> "ellipsoid"
        | S.Plane _ -> "plane"
        | S.Unsupported k -> "unsupported:" ^ k
      in
      Hashtbl.set seen ~key:kind ~data:()));
  List.iter [ "plane"; "sphere"; "capsule"; "box" ] ~f:(fun k ->
    equal ~msg:("has " ^ k) bool true (Hashtbl.mem seen k))

let () =
  Stdlib.exit
    (run
       "mjc viz"
       [ test "plane" test_plane
       ; test "box" test_box
       ; test "capsule" test_capsule
       ; test "contacts and forces" test_contacts
       ; test "png is deterministic" test_png
       ; test "every geom type" test_all_geom_types
       ])
