(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Is the Cholesky kernel the problem, or the flops?

   A step spends 86% of its time in one [nb x nb] Cholesky factor per round, and
   at [rows=24] and 512 lanes that is 24 rounds of 512 factors of a 24x24 matrix
   in 34.9 us/env-step -- about 3.7 GFLOP/s, which is a small fraction of what
   the device does on large matrices. Either the arithmetic is not the cost, or
   the kernel is not efficient at this size.

   So: the same eight-round-equivalent solve, four ways, same inputs, same
   device. [Nx.cholesky] plus two triangular solves is what the engine does.
   {!Linalg.cholesky} is the blocked version, which is composed of matmuls and a
   small unblocked panel factor, so it exercises the kernels that are efficient
   at this size. If the composed version wins, the engine's default linear solver
   is leaving time on the table for a reason that has nothing to do with
   algorithms.

   dune exec packages/mjc/bench/la.exe *)

open Base
module Printf = Stdlib.Printf

let now () = Unix.gettimeofday ()

(* What the engine does today. *)
let kernel ((a, b) : ('v, 's, 'd) Nx.t * ('v, 's, 'd) Nx.t) =
  let l = Nx.cholesky a in
  Nx.solve_triangular
    ~triangle:Lower
    ~transpose:Transpose
    l
    (Nx.solve_triangular ~triangle:Lower l b)

(* The blocked one: same answer to 1e-14 n, different arithmetic. *)
let blocked ((a, b) : ('v, 's, 'd) Nx.t * ('v, 's, 'd) Nx.t) = Mjc.Linalg.solve_spd a b

let time name f place (a_host, b_host) reps =
  let a = Nx.place place a_host
  and b = Nx.place place b_host in
  let g = Rune.jit Nx.Ptree.(pair tensor tensor @-> returns tensor) f in
  let out = ref (g (a, b)) in
  let sync () = Nx.sync (Nx.slice [ I 0 ] (Nx.reshape [| -1 |] !out)) in
  sync ();
  let t0 = now () in
  for _ = 1 to reps do
    out := g (a, b)
  done;
  sync ();
  let ms = (now () -. t0) /. Float.of_int reps *. 1e3 in
  let _ = name in
  ms, Nx.to_array (Nx.reshape [| -1 |] (Nx.place Nx.Host.on !out))

let problem nb lanes =
  let lead = [| lanes; nb; nb |] in
  let b =
    Nx.create
      Nx.float64
      lead
      (Array.init (lanes * nb * nb) ~f:(fun i -> Float.sin (Float.of_int i)))
  in
  let a =
    Nx.add
      (Nx.matmul b (Nx.matrix_transpose b))
      (Nx.mul_s (Nx.broadcast_to lead (Nx.eye Nx.float64 nb)) (Float.of_int nb))
  in
  let rhs =
    Nx.create
      Nx.float64
      [| lanes; nb; 1 |]
      (Array.init (lanes * nb) ~f:(fun i -> Float.cos (Float.of_int i)))
  in
  a, rhs

let worst_rel (got : float array) (want : float array) =
  let n = Stdlib.min (Array.length got) (Array.length want) in
  let acc = ref 0.0 in
  for i = 0 to n - 1 do
    let y = want.(i) in
    acc := Float.max !acc (Float.abs (got.(i) -. y) /. Float.max 1.0 (Float.abs y))
  done;
  !acc

let amd_gpus () =
  match Rig_amd_amdgpu.count () with
  | Error _ -> []
  | Ok n ->
    List.filter_map (List.init n ~f:Fn.id) ~f:(fun i ->
      match Rig.open_ (Rig_amd_amdgpu.gpu i) with
      | Ok dev when Rig.computes dev -> Some dev
      | _ -> None)

let () =
  let gpus = amd_gpus () in
  Printf.printf
    "| rows | lanes | device | kernel ms | blocked ms | kernel/blocked | blocked vs \
     kernel |\n";
  Printf.printf "| ---: | ---: | --- | ---: | ---: | ---: | ---: |\n%!";
  List.iter [ 24; 40; 76 ] ~f:(fun nb ->
    let lanes = 512
    and reps = 5 in
    let a, b = problem nb lanes in
    let row device k_ms b_ms rel =
      Printf.printf
        "| %d | %d | %s | %.2f | %.2f | %.2fx | %.1e |\n%!"
        nb
        lanes
        device
        k_ms
        b_ms
        (k_ms /. b_ms)
        rel
    in
    let hk, outk = time "kernel" kernel Nx.Host.on (a, b) reps in
    let hb, outb = time "blocked" blocked Nx.Host.on (a, b) reps in
    row "host" hk hb (worst_rel outb outk);
    match gpus with
    | [] -> ()
    | dev :: _ ->
      let module D = (val Nx.devices [ dev ] : Nx.Devices) in
      let gk, goutk = time "kernel" kernel D.on (a, b) reps in
      let gb, goutb = time "blocked" blocked D.on (a, b) reps in
      row "gpu" gk gb (worst_rel goutb goutk))
