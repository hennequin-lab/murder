(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Where a batched step's time goes on a device.

   `doc/perf.md` §11 shows the step is 453 us per environment at 512 lanes on
   the iGPU and that this is not throughput-bound, so the question is which
   stage spends it. Each stage is compiled for the device, run over a batch,
   and timed with a sync; the shares are what the next optimisation should
   follow rather than the stage names that look expensive.

   dune exec packages/mjc/bench/gpuprof.exe -- hopper 512 *)

open Base
module Printf = Stdlib.Printf
module Step = Mjc.Step
module Data = Mjc.Data
module Model = Mjc.Model
module Kin = Mjc.Kinematics
module Mass = Mjc.Mass
module Coll = Mjc.Collision
module Passive = Mjc.Passive
module Bias = Mjc.Bias
module Act = Mjc.Actuator
module Tendon = Mjc.Tendon
module Constraint = Mjc.Constraint
module Solver = Mjc.Solver

let load name =
  let path = Stdlib.Filename.concat "test/fixtures" (name ^ ".mjcb") in
  match Mjc.Mjcb.read path with
  | Ok m -> m
  | Error e -> failwith e

let now () = Unix.gettimeofday ()

(* A compiled stage over a batch, timed with a sync: without one the clock
   measures submission, which is how a 38x claim once became 1.6x. *)
module Prof (Set : Nx.Devices) = struct
  let place = Set.on
  let ptree = Nx.Ptree.( @-> ) (Data.ptree ()) (Nx.Ptree.returns (Data.ptree ()))

  let batch (d : Data.host) lanes =
    Nx.Ptree.map
      (Data.ptree ())
      (fun _ x ->
         Nx.concatenate ~axis:0 (List.init lanes ~f:(fun _ -> Nx.unsqueeze ~axes:[ 0 ] x)))
      d
    |> Data.place place

  let sync_data (d : Set.d Data.data) = Nx.sync (Nx.slice [ I 0 ] d.time)

  (* [time_with name f] times a stage over a batch. The batch goes through
     `vmap` of a compiled step rather than through tensor code with a leading
     lane axis, because several stages still reshape at fixed ranks -- which is
     itself worth knowing. *)
  let time_with ~reps ~lanes name f m0 d0 =
    let m = Model.place place m0 in
    let b = batch d0 lanes in
    let g = Rune.vmap ptree (Rune.jit ptree (f m)) in
    let out = ref (g b) in
    sync_data !out;
    let t0 = now () in
    for _ = 1 to reps do
      out := g b
    done;
    sync_data !out;
    let ms = (now () -. t0) /. Float.of_int reps *. 1e3 in
    Printf.printf "| %s | %.1f | %.0f us |\n%!" name ms (ms *. 1e3 /. Float.of_int lanes);
    ms
end

let () =
  let argv = Sys.get_argv () in
  let name = if Array.length argv > 1 then argv.(1) else "hopper" in
  let lanes = if Array.length argv > 2 then Stdlib.int_of_string argv.(2) else 512 in
  let reps = if Array.length argv > 3 then Stdlib.int_of_string argv.(3) else 3 in
  let m = load name in
  let d0 = Data.create m in
  let c = Step.config m in
  let iters = 4 in
  let gpus =
    match Rig_amd_amdgpu.count () with
    | Ok n when n > 0 ->
      List.filter_map (List.init n ~f:Fn.id) ~f:(fun i ->
        match Rig.open_ (Rig_amd_amdgpu.gpu i) with
        | Ok dev -> Some dev
        | Error _ -> None)
    | _ -> []
  in
  match gpus with
  | [] -> Printf.printf "no GPU to profile\n%!"
  | dev :: _ ->
    Printf.printf "%s, %d lanes, %d reps\n\n%!" name lanes reps;
    Printf.printf "| stage | ms/step | us/env-step |\n| --- | ---: | ---: |\n%!";
    let module D = (val Nx.devices [ dev ] : Nx.Devices) in
    let module P = Prof (D) in
    let row = P.time_with ~reps ~lanes in
    (* The pipeline in the order `Forward.forward` runs it. *)
    let total = ref 0.0 in
    let add x = total := !total +. x in
    let stage name f = add (row name f m d0) in
    stage "kinematics" Kin.forward;
    stage "mass" Mass.forward;
    stage "contacts" Coll.contacts;
    stage "passive" Passive.forward;
    stage "bias" Bias.forward;
    stage "actuator" Act.forward;
    stage "tendon" (fun m d ->
      ignore (Tendon.forward m d);
      d);
    Printf.printf "| (those stages) | %.1f | |\n%!" !total;
    (* The solver has rows to work on only after a forward pass. *)
    let df = Mjc.Forward.forward m d0 in
    let solver_ms = row "solver, 4 passes" (fun m d -> Solver.forward ~iters m d) m df in
    let step_ms = row "whole step" (fun m d -> Step.step_with c m d) m d0 in
    Printf.printf
      "| **everything else** | %.1f | %.0f us |\n%!"
      (step_ms -. !total -. solver_ms)
      ((step_ms -. !total -. solver_ms) *. 1e3 /. Float.of_int lanes);
    Printf.printf
      "\nper env-step: step %.0f us, solver %.0f us, other stages %.0f us\n%!"
      (step_ms *. 1e3 /. Float.of_int lanes)
      (solver_ms *. 1e3 /. Float.of_int lanes)
      (!total *. 1e3 /. Float.of_int lanes);
    (* The solver factors an `nefcmax x nefcmax` dense matrix every round, and
       `nefcmax` is a per-model bound rather than this step's row count: hopper
       has one row and a 64-row buffer. What the buffer alone costs, at the row
       counts the corpus uses. *)
    Printf.printf "\n| row buffer | factor + 2 solves, ms | x 32 rounds, us/env-step |\n";
    Printf.printf "| ---: | ---: | ---: |\n%!";
    let at = Nx.placement (Nx.zeros Nx.float64 [| 1 |]) in
    List.iter [ 4; 8; 16; 32; 48; 64; 76 ] ~f:(fun nb ->
      let a =
        let b = Nx.ones_at at Nx.float64 [| lanes; nb; nb |] in
        Nx.add
          (Nx.matmul b (Nx.matrix_transpose b))
          (Nx.mul_s (Nx.eye_at at Nx.float64 nb) (Float.of_int nb))
      in
      let x = Nx.ones_at at Nx.float64 [| lanes; nb; 1 |] in
      let f =
        Rune.jit
          Nx.Ptree.(pair tensor tensor @-> returns tensor)
          (fun (a, x) ->
             let l = Nx.cholesky a in
             Nx.solve_triangular
               ~triangle:Lower
               ~transpose:Transpose
               l
               (Nx.solve_triangular ~triangle:Lower l x))
      in
      let out = ref (f (a, x)) in
      Nx.sync (Nx.slice [ I 0 ] (Nx.reshape [| -1 |] !out));
      let t0 = now () in
      for _ = 1 to 3 do
        out := f (a, x)
      done;
      Nx.sync (Nx.slice [ I 0 ] (Nx.reshape [| -1 |] !out));
      let ms = (now () -. t0) /. 3.0 *. 1e3 in
      Printf.printf
        "| %d | %.3f | %.0f |\n%!"
        nb
        ms
        (ms *. 32.0 *. 1e3 /. Float.of_int lanes))
