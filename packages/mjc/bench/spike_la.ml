(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The M0 linear-algebra spike: what a factorization costs, eagerly, compiled,
   and per environment under [vmap]. The numbers this prints go into
   doc/spike-la.md, and they answer gate G0.

   Run: dune exec packages/mjc/bench/spike_la.exe

   Everything is float64 on the host, one process. A "solve" is a Cholesky
   factor followed by three triangular solves, which is what a step does: one
   factor for [M], one right-hand side for [qacc_smooth], and two more for the
   impulse solve. *)

open Base
module Printf = Stdlib.Printf
open Nx

let now () = Unix.gettimeofday ()

(* Seconds per call, after [warmup] calls and over [reps] calls. *)
let time ?(warmup = 3) ?(reps = 20) f x =
  for _ = 1 to warmup do
    ignore (f x)
  done;
  let t0 = now () in
  for _ = 1 to reps do
    ignore (f x)
  done;
  (now () -. t0) /. Float.of_int reps

(* A symmetric positive definite matrix of size [n], well conditioned enough
   that the factor does not clamp. *)
let spd n =
  let a =
    Nx.init Nx.float64 [| n; n |] (fun i ->
      let r = Float.of_int i.(0)
      and c = Float.of_int i.(1) in
      if Float.equal r c
      then 2.0 +. (n |> Float.of_int)
      else 1.0 /. (1.0 +. Float.abs (r -. c)))
  in
  let b = Nx.init Nx.float64 [| n; 1 |] (fun i -> 1.0 +. Float.of_int i.(0)) in
  ignore b;
  a

let solve_lower_upper a b =
  let l = Nx.cholesky a in
  let y = Nx.solve_triangular ~triangle:Lower l b in
  let z = Nx.solve_triangular ~triangle:Lower ~transpose:Transpose l y in
  let w = Nx.solve_triangular ~triangle:Lower l z in
  Nx.solve_triangular ~triangle:Lower ~transpose:Transpose l w

let lu_solve a b = Nx.solve a b
let factor a = Nx.cholesky a
let matvec a x = Nx.matmul a x
let us s = Float.round (s *. 1e7) /. 10.0 (* microseconds, 0.1 us resolution *)
let ms s = Float.round (s *. 1e4) /. 10.0 (* milliseconds, 0.1 ms resolution *)

let () =
  let sizes = [ 8; 27; 39; 60 ] in
  let batch = 256 in
  Printf.printf "linear algebra spike, float64, host\n\n";
  Printf.printf
    "| n | eager factor | eager solve | jit factor | jit solve | vmap %d solve, per env \
     | compile, factor | compile, solve |\n"
    batch;
  Printf.printf "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
  List.iter sizes ~f:(fun n ->
    let a = spd n in
    let b = Nx.ones Nx.float64 [| n |] in
    let eager_factor = time factor a in
    let eager_solve = time (fun a -> solve_lower_upper a b) a in
    (* Compiled: build a fresh function and time its first call, which
       includes the compile, then its steady state. *)
    let jit_factor = Rune.jit' factor in
    let t0 = now () in
    let _ = jit_factor a in
    let compile_factor = now () -. t0 in
    let jit_factor_time = time jit_factor a in
    let jit_solve = Rune.jit' (fun a -> solve_lower_upper a b) in
    let t0 = now () in
    let _ = jit_solve a in
    let compile_solve = now () -. t0 in
    let jit_solve_time = time jit_solve a in
    (* Batched: one factor and three solves per lane, matrices as arguments. *)
    let ab =
      Nx.init Nx.float64 [| batch; n; n |] (fun i ->
        let r = Float.of_int i.(1)
        and c = Float.of_int i.(2) in
        if Float.equal r c
        then 2.0 +. Float.of_int n
        else 1.0 /. (1.0 +. Float.abs (r -. c)))
    in
    let bb = Nx.ones Nx.float64 [| batch; n |] in
    let vmapped =
      Rune.vmap
        Nx.Ptree.(pair tensor tensor @-> returns tensor)
        (fun (a, b) -> solve_lower_upper a b)
    in
    let _ = vmapped (ab, bb) in
    let batched = time vmapped (ab, bb) in
    Printf.printf
      "| %d | %.1f us | %.1f us | %.1f us | %.1f us | %.1f us | %.2f s | %.2f s |\n"
      n
      (us eager_factor)
      (us eager_solve)
      (us jit_factor_time)
      (us jit_solve_time)
      (us batched /. Float.of_int batch)
      compile_factor
      compile_solve);
  Printf.printf "\n";
  (* The LU comparison the design note warns about, at the sizes that matter. *)
  Printf.printf "| n | eager LU | jit LU |\n| ---: | ---: | ---: |\n";
  List.iter [ 27; 39 ] ~f:(fun n ->
    let a = spd n in
    let b = Nx.ones Nx.float64 [| n |] in
    let eager = time (fun a -> lu_solve a b) a in
    let jit = Rune.jit' (fun a -> lu_solve a b) in
    let _ = jit a in
    let compiled = time jit a in
    Printf.printf "| %d | %.1f us | %.1f us |\n" n (us eager) (us compiled));
  Printf.printf "\n";
  (* Dispatch floor: what a loop iteration costs, for reference. *)
  let xs = Nx.arange Nx.int32 0 1000 1 in
  let scan_fn =
    Rune.jit' (fun q ->
      Rune.scan' (fun q _ -> Nx.add_s q 1.0, Nx.scalar Nx.float64 0.0) ~init:q xs |> fst)
  in
  let q = Nx.ones Nx.float64 [| 1000 |] in
  let t0 = now () in
  let _ = scan_fn q in
  let compile_scan = now () -. t0 in
  let per_iter = time ~warmup:1 ~reps:5 scan_fn q in
  Printf.printf
    "scan of 1000 elementwise steps: %.2f ms (%.0f ns per iteration), compiled in %.2f s\n"
    (ms per_iter)
    (per_iter *. 1e9 /. 1000.0)
    compile_scan
