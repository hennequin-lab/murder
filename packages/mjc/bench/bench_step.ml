(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The composed step: microseconds per step, eager, compiled, and batched for
   the per-environment cost. Run:

   dune exec packages/mjc/bench/bench_step.exe *)

open Base
open Mjc
module Printf = Stdlib.Printf

let now () = Unix.gettimeofday ()

let load name =
  match Mjcb.read (Printf.sprintf "packages/mjc/test/fixtures/%s.mjcb" name) with
  | Ok m -> m
  | Error e ->
    Printf.eprintf "%s\n" e;
    Stdlib.exit 1

let time ~steps f x =
  let x = ref x in
  for _ = 1 to 20 do
    x := f !x
  done;
  let t0 = now () in
  for _ = 1 to steps do
    x := f !x
  done;
  (now () -. t0) /. Float.of_int steps

(* Every leaf stacked along a new leading axis: the axis [Rune.vmap] maps
   over. *)
let batch ds =
  match ds with
  | [] -> invalid_arg "batch"
  | [ d ] -> d
  | d0 :: d1 :: rest ->
    let both = Nx.Ptree.map2 Data.ptree (fun _ a b -> Nx.stack ~axis:0 [ a; b ]) d0 d1 in
    List.fold rest ~init:both ~f:(fun acc x ->
      Nx.Ptree.map2
        Data.ptree
        (fun _ a b -> Nx.concatenate ~axis:0 [ a; Nx.unsqueeze ~axes:[ 0 ] b ])
        acc
        x)

let lanes = 256
let us s = Float.round (s *. 1e7) /. 10.0

(* NOTE: a model whose geoms can touch traces for minutes until the contact
   Jacobians are batched -- each candidate walks its dof chain with one traced
   [set] per dof. chain6 is one such model (six capsules), so this bench does
   not finish today; the numbers below are for models without contacts. The
   step itself is unaffected. *)
let () =
  Printf.printf
    "| model | nv | nbody | eager | compiled | vmap %d, per env |\n\
     | --- | ---: | ---: | ---: | ---: | ---: |\n"
    lanes;
  List.iter
    [ "pendulum"; "chain6"; "floating"; "ball"; "limited"; "looped"; "fast" ]
    ~f:(fun name ->
      let m = load name in
      let d = Data.create m in
      let eager = time ~steps:2000 (fun d -> Step.step m d) d in
      let compiled = Step.compile m in
      let one = Step.run compiled in
      let _ = one d in
      let compiled_time = time ~steps:2000 one d in
      let batched = batch (List.init lanes ~f:(fun _ -> Data.create m)) in
      let vec = Rune.vmap Nx.Ptree.(Data.ptree @-> returns Data.ptree) one in
      let _ = vec batched in
      let per_env = time ~steps:10 vec batched /. Float.of_int lanes in
      Printf.printf
        "| %s | %d | %d | %.1f us | %.1f us | %.1f us |\n"
        name
        m.topo.nv
        m.topo.nbody
        (us eager)
        (us compiled_time)
        (us per_env))
