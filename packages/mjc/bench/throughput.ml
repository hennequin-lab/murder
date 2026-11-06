(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Steps per second at batch size, on the host and on the GPU.

   `doc/perf.md` measures one step's latency at one environment, which is a fair
   comparison against `mj_step` and the wrong one for training: what decides
   whether an engine is usable for RL is `env-steps/s` at a batch, and how that
   compares to the cost of the network being trained. This measures the first,
   and prints a small policy's forward+backward cost beside it as the second.

   Each row is a whole step over `lanes` environments, warm once, then timed;
   `us/env-step` is what to compare with `mj_step` on one core (5 us for
   hopper/walker, from `tools/perf_mujoco.py`).

   dune exec packages/mjc/bench/throughput.exe
   dune exec packages/mjc/bench/throughput.exe -- hopper walker
   dune exec packages/mjc/bench/throughput.exe -- --lanes 1,64,512,4096 hopper *)

open Base
module Printf = Stdlib.Printf
module Step = Mjc.Step
module Data = Mjc.Data
module Model = Mjc.Model

(* One step per environment, at whatever device the data lives on. *)
module Run (Set : Nx.Devices) = struct
  let place = Set.on

  (* [steps] steps of a batch of [lanes] environments; returns the per-step
     time in milliseconds. The batch is built and placed once, outside the
     timing, and the result is never read back. *)
  let run (c : Step.config) m0 d0 ~lanes ~steps ~budget_ms =
    let m = Model.place place m0 in
    (* The batch is assembled on the host and moved in one walk: a device set
       without kernels cannot concatenate, and a training loop would build its
       batch once anyway. *)
    let d =
      Nx.Ptree.map
        (Data.ptree ())
        (fun _ x ->
           Nx.concatenate
             ~axis:0
             (List.init lanes ~f:(fun _ -> Nx.unsqueeze ~axes:[ 0 ] x)))
        d0
    in
    let d = Data.place place d in
    let ptree = Nx.Ptree.( @-> ) (Data.ptree ()) (Nx.Ptree.returns (Data.ptree ())) in
    (* `vmap` of a *compiled* step: a device set has no eager kernels, so the
       lane axis has to live inside the compiled program rather than around an
       interpreted one. *)
    let one = Rune.jit ptree (fun d -> Step.step_with c m d) in
    let step = Rune.vmap ptree one in
    (* The first call compiles; the second tells us whether this
       configuration is worth timing at all. *)
    (* Timing a device program means waiting for it: without a sync the clock
       measures how long the host takes to *submit* the work, which is off by
       the whole depth of the queue. *)
    let sync (d : Set.d Data.data) = Nx.sync (Nx.slice [ I 0 ] d.time) in
    let d = step d in
    sync d;
    let t0 = Unix.gettimeofday () in
    let d = step d in
    sync d;
    let first_ms = (Unix.gettimeofday () -. t0) *. 1e3 in
    if Float.( > ) first_ms budget_ms
    then Stdlib.Error first_ms
    else (
      let t0 = Unix.gettimeofday () in
      let last = ref d in
      for _ = 1 to steps - 1 do
        last := step !last
      done;
      sync !last;
      Stdlib.Ok ((Unix.gettimeofday () -. t0) *. 1e3 /. Float.of_int (steps - 1)))
end

(* A policy network's cost, for the comparison the number is for: a 2x256
   tanh MLP, forward and backward, over the same batch, on the same device. *)
module Policy (Set : Nx.Devices) = struct
  let place = Set.on

  let mlp ~lanes ~steps ~obs ~act =
    let w1 = Nx.place place (Nx.mul_s (Nx.ones Nx.float32 [| obs; 256 |]) 0.01) in
    let w2 = Nx.place place (Nx.ones Nx.float32 [| 256; 256 |]) in
    let w3 = Nx.place place (Nx.ones Nx.float32 [| 256; act |]) in
    let x = Nx.place place (Nx.ones Nx.float32 [| lanes; obs |]) in
    let params = Nx.Ptree.(pair tensor (pair tensor tensor)) in
    let loss pom =
      let w1, (w2, w3) = pom in
      let h = Nx.tanh (Nx.matmul x w1) in
      let h = Nx.tanh (Nx.matmul h w2) in
      Nx.sum (Nx.tanh (Nx.matmul h w3))
    in
    let g =
      Rune.jit (Nx.Ptree.( @-> ) params (Nx.Ptree.returns params)) (Rune.grad params loss)
    in
    let p = w1, (w2, w3) in
    ignore (g p);
    let t0 = Unix.gettimeofday () in
    for _ = 2 to steps do
      ignore (g p)
    done;
    (Unix.gettimeofday () -. t0) *. 1e3 /. Float.of_int (steps - 1)
end

let load name =
  let path = Stdlib.Filename.concat "test/fixtures" (name ^ ".mjcb") in
  match Mjc.Mjcb.read path with
  | Ok m -> m
  | Error e -> failwith e

let split_lanes v =
  match String.split v ~on:',' with
  | xs -> List.filter_map xs ~f:Int.of_string_opt

let () =
  let argv = Array.to_list (Sys.get_argv ()) in
  let argv =
    match argv with
    | _ :: tl -> tl
    | [] -> []
  in
  let rec parse lanes models with_policy = function
    | [] -> lanes, List.rev models, with_policy
    | "--lanes" :: v :: tl -> parse (split_lanes v) models with_policy tl
    | "--no-policy" :: tl -> parse lanes models false tl
    | a :: tl when String.is_prefix a ~prefix:"--" -> parse lanes models with_policy tl
    | a :: tl -> parse lanes (a :: models) with_policy tl
  in
  let lanes, models, with_policy = parse [ 1; 64; 512; 4096 ] [] true argv in
  let models =
    match models with
    | [] -> [ "pendulum"; "chain6"; "hopper"; "walker" ]
    | xs -> xs
  in
  let budgets =
    match
      List.find_map argv ~f:(fun a ->
        String.chop_prefix ~prefix:"--iters=" a
        |> Option.bind ~f:(fun v -> List.hd (split_lanes v)))
    with
    | Some n -> [ 0; n ]
    | None -> [ 0 ]
  in
  let budget_ms = 25000.0 in
  Printf.printf
    "| model | lanes | device | ms/step | us/env-step | env-steps/s | x mj_step, 1 core |\n";
  Printf.printf "| --- | ---: | --- | ---: | ---: | ---: | ---: |\n%!";
  (* `mj_step` at rest per model, one core, from tools/perf_mujoco.py. *)
  let mj = [ "pendulum", 1.0; "chain6", 3.0; "hopper", 5.0; "walker", 5.0 ] in
  let reference name =
    Option.value (List.Assoc.find mj ~equal:String.equal name) ~default:5.0
  in
  let gpus =
    match Rig_amd_amdgpu.count () with
    | Ok n when n > 0 ->
      List.filter_map (List.init n ~f:Fn.id) ~f:(fun i ->
        match Rig.open_ (Rig_amd_amdgpu.gpu i) with
        | Ok dev -> Some dev
        | Error _ -> None)
    | _ -> []
  in
  List.iter models ~f:(fun name ->
    let m = load name in
    let d0 = Data.create m in
    List.iter budgets ~f:(fun iters ->
      let c = if Int.equal iters 0 then Step.config m else Step.config ~iters m in
      List.iter lanes ~f:(fun n ->
        let steps = if n <= 64 then 6 else if n <= 512 then 3 else 2 in
        let label =
          if Int.equal iters 0 then name else Printf.sprintf "%s, iters=%d" name iters
        in
        let row device ms =
          let per = ms *. 1e3 /. Float.of_int n in
          Printf.printf
            "| %s | %d | %s | %.2f | %.2f | %.0f | %.0fx |\n%!"
            label
            n
            device
            ms
            per
            (1e6 /. per)
            (per /. reference name)
        in
        let module H = Run (Nx.Host) in
        (match H.run c m d0 ~lanes:n ~steps ~budget_ms with
         | Stdlib.Ok ms -> row "host" ms
         | Stdlib.Error _ ->
           Printf.printf "| %s | %d | host | skipped | | | |\n%!" label n);
        match gpus with
        | [] -> ()
        | dev :: _ ->
          let module D = (val Nx.devices [ dev ] : Nx.Devices) in
          let module G = Run (D) in
          (match G.run c m d0 ~lanes:n ~steps ~budget_ms with
           | Stdlib.Ok ms -> row "gpu" ms
           | Stdlib.Error _ ->
             Printf.printf "| %s | %d | gpu | skipped | | | |\n%!" name n))));
  if with_policy
  then (
    Printf.printf "\n| device | lanes | policy fwd+bwd, ms | policy us/sample |\n%!";
    Printf.printf "| --- | ---: | ---: | ---: |\n%!";
    let module H = Policy (Nx.Host) in
    List.iter [ 512; 4096 ] ~f:(fun n ->
      let ms = H.mlp ~lanes:n ~steps:5 ~obs:50 ~act:10 in
      Printf.printf "| host | %d | %.2f | %.2f |\n%!" n ms (ms *. 1e3 /. Float.of_int n));
    match gpus with
    | [] -> ()
    | dev :: _ ->
      let module D = (val Nx.devices [ dev ] : Nx.Devices) in
      let module G = Policy (D) in
      List.iter [ 512; 4096 ] ~f:(fun n ->
        let ms = G.mlp ~lanes:n ~steps:5 ~obs:50 ~act:10 in
        Printf.printf "| gpu | %d | %.2f | %.2f |\n%!" n ms (ms *. 1e3 /. Float.of_int n)))
