(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The composed step: microseconds per step, eager and compiled, for the
   corpus models. Run: dune exec packages/mjc/bench/bench_step.exe *)

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
  (* warm up, then time *)
  let x = ref x in
  for _ = 1 to 50 do
    x := f !x
  done;
  let t0 = now () in
  for _ = 1 to steps do
    x := f !x
  done;
  (now () -. t0) /. Float.of_int steps

let () =
  Printf.printf
    "| model | nv | nbody | eager step | compiled step |\n\
     | --- | ---: | ---: | ---: | ---: |\n";
  List.iter [ "pendulum"; "chain6"; "floating"; "ball" ] ~f:(fun name ->
    let m = load name in
    let d = Data.create m in
    let eager = time ~steps:2000 (fun d -> Step.step m d) d in
    let compiled_time =
      try
        let f = Rune.jit Nx.Ptree.(Data.ptree @-> returns Data.ptree) (Step.step m) in
        let _ = f d in
        Some (time ~steps:2000 f d)
      with
      | e ->
        Printf.eprintf "%s: %s\n" name (Exn.to_string e);
        None
    in
    let us s = Float.round (s *. 1e7) /. 10.0 in
    Printf.printf
      "| %s | %d | %d | %.1f us | %s |\n"
      name
      m.topo.nv
      m.topo.nbody
      (us eager)
      (match compiled_time with
       | None -> "-"
       | Some s -> Printf.sprintf "%.1f us" (us s)))
