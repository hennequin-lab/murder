(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What a model needs and the engine does not have.

   The geometry a model must cover is not the list of geom types it contains but
   the set of *candidate* pairs -- after the body filter, the parent filter and
   the contype/conaffinity mask -- and `Collision.unsupported_pairs` reports it,
   which is the list M7 is written against.

   dune exec packages/mjc/bench/pairs.exe test/fixtures/quadruped.mjcb *)

open Base
module Printf = Stdlib.Printf

let name = function
  | Mjc.Topo.Plane -> "plane"
  | Hfield -> "hfield"
  | Sphere -> "sphere"
  | Capsule -> "capsule"
  | Ellipsoid -> "ellipsoid"
  | Cylinder -> "cylinder"
  | Box -> "box"
  | Mesh -> "mesh"

let () =
  let path = Stdlib.Sys.argv.(1) in
  let m =
    match Mjc.Mjcb.read path with
    | Ok m -> m
    | Error e -> failwith e
  in
  let gaps = Mjc.Collision.unsupported_pairs m in
  let total = List.fold gaps ~init:0 ~f:(fun acc (_, n) -> acc + n) in
  Printf.printf
    "%s: %d candidate pairs the engine cannot solve, in %d type pairs\n\n"
    path
    total
    (List.length gaps);
  Printf.printf "| pair | candidate pairs |\n| --- | ---: |\n%!";
  List.iter gaps ~f:(fun ((a, b), n) ->
    Printf.printf "| %s vs %s | %d |\n%!" (name a) (name b) n)
