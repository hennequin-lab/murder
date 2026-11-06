(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Does batching a loop pay?

   `Rune.scan` lowers to a real loop with carried buffers, so a `vmap` around
   it has two possible shapes: one loop whose buffers carry a lane axis, or one
   loop per lane. The first costs a lane's work per lane; the second pays every
   loop and operation overhead once per lane. Same question for the linear
   algebra a solver round is made of.

   Numbers per lane, so that a batched run is comparable with a single one.

   dune exec packages/mjc/bench/spike_batch.exe *)

open Base
module Printf = Stdlib.Printf

let now () = Unix.gettimeofday ()
let steps = 64

(* Median of [reps] timings, each after one warm-up, microseconds. *)
let time ~reps f x =
  ignore (f x);
  let ts =
    List.init reps ~f:(fun _ ->
      let t0 = now () in
      ignore (f x);
      now () -. t0)
  in
  let sorted = List.sort ts ~compare:Float.compare in
  List.nth_exn sorted (List.length sorted / 2) *. 1e6

(* A loop of [steps] cheap steps: [carry <- carry + 1] on [nb] elements. *)
let loop_add () =
  let phases = Nx.zeros Nx.float64 [| steps; 1 |] in
  fun (x : Nx.host Nx.float64_t) ->
    let final, _ =
      Rune.scan
        Nx.Ptree.tensor
        Nx.Ptree.tensor
        Nx.Ptree.tensor
        (fun carry (_ : Nx.host Nx.float64_t) ->
           Nx.mul_s (Nx.add_s carry 1.0) 1.0, Nx.sum carry)
        ~init:x
        phases
    in
    final

(* A loop whose step is a matmul and a Cholesky solve: the shape of one
   active-set round, without any of the row bookkeeping. *)
let loop_solve ~nb =
  let phases = Nx.zeros Nx.float64 [| steps; 1 |] in
  let eye = Nx.eye Nx.float64 nb in
  fun (x : Nx.host Nx.float64_t) ->
    let final, _ =
      Rune.scan
        Nx.Ptree.tensor
        Nx.Ptree.tensor
        Nx.Ptree.tensor
        (fun carry (_ : Nx.host Nx.float64_t) ->
           let row = Nx.reshape [| 1; nb |] carry in
           let m = Nx.add (Nx.matmul (Nx.matrix_transpose row) row) (Nx.mul_s eye 4.0) in
           let next =
             Nx.reshape
               [| nb |]
               (Nx.solve_triangular
                  ~triangle:Lower
                  (Nx.cholesky m)
                  (Nx.reshape [| nb; 1 |] (Nx.cos carry)))
           in
           Nx.mul_s next 0.1, Nx.sum next)
        ~init:x
        phases
    in
    final

let report label ~lanes ~single ~batched =
  let per_lane = batched /. Float.of_int lanes in
  Printf.printf
    "| %s | %.1f us | %.1f us | %.1f us | %.2fx |\n"
    label
    single
    batched
    per_lane
    (per_lane /. single)

(* The composition that matters for a real step: a *compiled* program under a
   map. `vmap (jit f)` and `jit (vmap f)` are different programs, and a step
   that is already compiled is the first. *)
let compositions ~nb =
  let x =
    Nx.create Nx.float64 [| nb |] (Array.init nb ~f:(fun i -> 0.1 *. Float.of_int i))
  in
  let batched_input =
    Nx.reshape
      [| 8; nb |]
      (Nx.concatenate ~axis:0 (List.init 8 ~f:(fun _ -> Nx.reshape [| 1; nb |] x)))
  in
  let f = loop_solve ~nb in
  let ptree = Nx.Ptree.(tensor @-> returns tensor) in
  let single = time ~reps:5 f x in
  (* (a) map over the compiled program *)
  let compiled = Rune.jit ptree f in
  let a = time ~reps:5 (Rune.vmap ptree compiled) batched_input in
  (* (b) compile the mapped program *)
  let b = time ~reps:5 (Rune.jit ptree (Rune.vmap ptree f)) batched_input in
  (* (c) map over an uncompiled program *)
  let c = time ~reps:5 (Rune.vmap ptree f) batched_input in
  ignore c;
  Printf.printf
    "| %d | %.0f us | %.0f us | %.0f us | %.0f us | %.2fx | %.2fx |\n"
    nb
    single
    (a /. 8.0)
    (b /. 8.0)
    (c /. 8.0)
    (a /. 8.0 /. single)
    (b /. 8.0 /. single)

(* Which stage of a real step is the one that does not batch: the stages of
   the engine's own pipeline, single and under a map, per lane. *)
let stages name =
  let m =
    match Mjc.Mjcb.read (Printf.sprintf "packages/mjc/test/fixtures/%s.mjcb" name) with
    | Ok m -> m
    | Error e -> failwith e
  in
  let d = { (Mjc.Data.create m) with qpos = m.Mjc.Model.params.Mjc.Model.Params.qpos0 } in
  let d = Mjc.Forward.forward m d in
  let ptree = Nx.Ptree.(Mjc.Data.ptree @-> returns Mjc.Data.ptree) in
  let lanes = 8 in
  let batched_input =
    Nx.Ptree.map
      Mjc.Data.ptree
      (fun _ x ->
         Nx.concatenate ~axis:0 (List.init lanes ~f:(fun _ -> Nx.unsqueeze ~axes:[ 0 ] x)))
      d
  in
  let row label f =
    let single = time ~reps:3 f d in
    let mapped = Rune.vmap ptree f in
    let batched = time ~reps:3 mapped batched_input /. Float.of_int lanes in
    Printf.printf
      "| %s | %.0f us | %.0f us | %.0f us | %.2fx |\n"
      label
      single
      batched
      batched
      (batched /. single)
  in
  (* Where the batched factorisation breaks even with the kernel's: per lane,
     for a few lane counts. *)
  Printf.printf
    "| %s, solve stage | single | lanes | per lane | per lane / single |\n"
    name;
  Printf.printf "| --- | ---: | ---: | ---: | ---: |\n";
  List.iter [ 8; 32 ] ~f:(fun lanes ->
    let batched_input =
      Nx.Ptree.map
        Mjc.Data.ptree
        (fun _ x ->
           Nx.concatenate
             ~axis:0
             (List.init lanes ~f:(fun _ -> Nx.unsqueeze ~axes:[ 0 ] x)))
        d
    in
    List.iter
      [ "kernel", `Cholesky; "batched", `Batched ]
      ~f:(fun (label, linear) ->
        let f d = Mjc.Solver.forward ~linear ~iters:8 m d in
        let single = time ~reps:2 f d in
        let per_lane =
          time ~reps:2 (Rune.vmap ptree f) batched_input /. Float.of_int lanes
        in
        Printf.printf
          "| %s | %.0f us | %d | %.0f us | %.2fx |\n"
          label
          single
          lanes
          per_lane
          (per_lane /. single));
    Printf.printf "\n");
  Printf.printf "| %s, stage | single | batched per lane | per lane / single |\n" name;
  Printf.printf "| --- | ---: | ---: | ---: |\n";
  row "kinematics" (fun d -> Mjc.Kinematics.forward m d);
  row "mass" (fun d -> Mjc.Mass.forward m d);
  row "contacts" (fun d -> Mjc.Collision.contacts m d);
  row "passive + bias" (fun d -> Mjc.Bias.forward m (Mjc.Passive.forward m d));
  row "solve, kernel" (fun d -> Mjc.Solver.forward ~iters:8 m d);
  row "solve, batched" (fun d -> Mjc.Solver.forward ~linear:`Batched ~iters:8 m d);
  (* The whole step, interpreted and compiled: the composition a training loop
     actually picks, and the one the perf table measures. *)
  row "step, interpreted" (fun d -> Mjc.Step.step m d);
  let compiled = Mjc.Step.compile m in
  row "step, compiled" (fun d -> Mjc.Step.run compiled d);
  Printf.printf "\n"

let () =
  Printf.printf "steps per loop: %d, median of 5 runs, microseconds\n\n" steps;
  Printf.printf "Composition: a program of %d Cholesky-solve steps, per lane.\n\n" steps;
  Printf.printf
    "| nb | single | vmap(jit f) | jit(vmap f) | vmap f | vmap(jit)/single | \
     jit(vmap)/single |\n\
     | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
  List.iter [ 8; 64 ] ~f:(fun nb -> compositions ~nb);
  Printf.printf "\n";
  List.iter [ "pendulum"; "chain6"; "walker" ] ~f:stages;
  Printf.printf "Where the batched win goes: one loop body at a time, per lane.\n\n";
  Printf.printf
    "| nb | matmul single | matmul batched | chol+solve single | chol+solve batched |\n\
     | ---: | ---: | ---: | ---: | ---: |\n";
  List.iter [ 8; 16; 24; 32; 48; 64 ] ~f:(fun nb ->
    let x =
      Nx.create Nx.float64 [| nb |] (Array.init nb ~f:(fun i -> 0.05 *. Float.of_int i))
    in
    let batched_input =
      Nx.reshape
        [| 8; nb |]
        (Nx.concatenate ~axis:0 (List.init 8 ~f:(fun _ -> Nx.reshape [| 1; nb |] x)))
    in
    let ptree = Nx.Ptree.(tensor @-> returns tensor) in
    let phases = Nx.zeros Nx.float64 [| steps; 1 |] in
    let eye = Nx.eye Nx.float64 nb in
    let matmul_only (x : Nx.host Nx.float64_t) =
      let final, _ =
        Rune.scan
          Nx.Ptree.tensor
          Nx.Ptree.tensor
          Nx.Ptree.tensor
          (fun carry (_ : Nx.host Nx.float64_t) ->
             let row = Nx.reshape [| 1; nb |] carry in
             let m =
               Nx.add (Nx.matmul (Nx.matrix_transpose row) row) (Nx.mul_s eye 4.0)
             in
             ( Nx.reshape [| nb |] (Nx.matmul m (Nx.reshape [| nb; 1 |] carry))
             , Nx.sum carry ))
          ~init:x
          phases
      in
      final
    in
    let chol_only (x : Nx.host Nx.float64_t) =
      let final, _ =
        Rune.scan
          Nx.Ptree.tensor
          Nx.Ptree.tensor
          Nx.Ptree.tensor
          (fun carry (_ : Nx.host Nx.float64_t) ->
             let row = Nx.reshape [| 1; nb |] carry in
             let m =
               Nx.add (Nx.matmul (Nx.matrix_transpose row) row) (Nx.mul_s eye 4.0)
             in
             let next =
               Nx.reshape
                 [| nb |]
                 (Nx.solve_triangular
                    ~triangle:Lower
                    (Nx.cholesky m)
                    (Nx.reshape [| nb; 1 |] (Nx.cos carry)))
             in
             Nx.mul_s next 0.1, Nx.sum next)
          ~init:x
          phases
      in
      final
    in
    let mm_s = time ~reps:3 matmul_only x in
    let mm_b = time ~reps:3 (Rune.vmap ptree matmul_only) batched_input /. 8.0 in
    let ch_s = time ~reps:3 chol_only x in
    let ch_b = time ~reps:3 (Rune.vmap ptree chol_only) batched_input /. 8.0 in
    Printf.printf
      "| %d | %.0f us | %.0f us (%.2fx) | %.0f us | %.0f us (%.2fx) |\n"
      nb
      mm_s
      mm_b
      (mm_b /. mm_s)
      ch_s
      ch_b
      (ch_b /. ch_s));
  Printf.printf
    "| loop body | single lane | %d lanes | per lane | per lane / single |\n\
     | --- | ---: | ---: | ---: | ---: |\n"
    8;
  List.iter [ 8; 64 ] ~f:(fun nb ->
    let n = Float.of_int nb in
    let x =
      Nx.create
        Nx.float64
        [| nb |]
        (Array.init nb ~f:(fun i -> 0.1 *. (Float.of_int i /. n)))
    in
    let single = time ~reps:5 (loop_add ()) x in
    let batched_input =
      Nx.reshape
        [| 8; nb |]
        (Nx.concatenate ~axis:0 (List.init 8 ~f:(fun _ -> Nx.reshape [| 1; nb |] x)))
    in
    let map = Rune.vmap Nx.Ptree.(tensor @-> returns tensor) (loop_add ()) in
    let batched = time ~reps:5 map batched_input in
    report (Printf.sprintf "add, nb = %d" nb) ~lanes:8 ~single ~batched;
    let single = time ~reps:5 (loop_solve ~nb) x in
    let map = Rune.vmap Nx.Ptree.(tensor @-> returns tensor) (loop_solve ~nb) in
    let batched = time ~reps:5 map batched_input in
    report (Printf.sprintf "matmul + Cholesky, nb = %d" nb) ~lanes:8 ~single ~batched)
