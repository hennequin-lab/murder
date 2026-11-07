(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The number an RL loop is judged by: env-steps per second end to end, split
   into the parts it is made of.

   Every throughput figure quoted until now was microseconds per simulator step,
   which is a component, not a result: PPO spends its time in three places --
   rolling the environment out, evaluating the policy, and taking the gradient
   the policy is updated with -- and a simulator that is fast while the other
   two dominate has not made anything faster. This measures all three on the
   same batch and device and prints their shares.

   The policy is the usual 2x256 tanh MLP. The gradient segment is the one
   model-based RL needs: the derivative of the state after one step with respect
   to the model's own parameters, which is the cost the differentiable engine
   exists to make affordable.

   Run: dune exec packages/mjc/bench/rl.ml -- --lanes 512 hopper *)

open Base
module Printf = Stdlib.Printf
module Step = Mjc.Step
module Data = Mjc.Data
module Model = Mjc.Model

let load name =
  let path = Stdlib.Filename.concat "test/fixtures" (name ^ ".mjcb") in
  match Mjc.Mjcb.read path with
  | Ok m -> m
  | Error e -> failwith e

(* One environment's state, replicated into [lanes] lanes on the host: a device
   set has no kernels to concatenate with, and a training loop builds its batch
   once anyway. *)
let batch (d : Nx.host Data.data) ~lanes =
  Nx.Ptree.map
    (Data.ptree ())
    (fun _ x ->
       Nx.concatenate ~axis:0 (List.init lanes ~f:(fun _ -> Nx.unsqueeze ~axes:[ 0 ] x)))
    d

(* A 2x256 tanh MLP, forward and backward over the batch: what PPO's actor costs
   at this batch size, on the device the simulation runs on. *)
module Policy (Set : Nx.Devices) = struct
  let place = Set.on

  let run ~lanes ~obs ~act ~steps =
    let w1 = Nx.place place (Nx.mul_s (Nx.ones Nx.float32 [| obs; 256 |]) 0.01) in
    let w2 = Nx.place place (Nx.mul_s (Nx.ones Nx.float32 [| 256; 256 |]) 0.01) in
    let w3 = Nx.place place (Nx.mul_s (Nx.ones Nx.float32 [| 256; act |]) 0.01) in
    let x = Nx.place place (Nx.ones Nx.float32 [| lanes; obs |]) in
    let ptree = Nx.Ptree.(pair tensor (pair tensor tensor)) in
    let loss (w1, (w2, w3)) =
      let h = Nx.tanh (Nx.matmul x w1) in
      let h = Nx.tanh (Nx.matmul h w2) in
      Nx.sum (Nx.tanh (Nx.matmul h w3))
    in
    let f =
      Rune.jit
        Nx.Ptree.(ptree @-> returns (pair tensor ptree))
        (fun p -> Rune.value_and_grad ptree loss p)
    in
    let p = w1, (w2, w3) in
    let _ = f p in
    let sync () = Nx.sync (fst (f p)) in
    let t0 = Unix.gettimeofday () in
    for _ = 1 to steps do
      sync ()
    done;
    (Unix.gettimeofday () -. t0) *. 1e3 /. Float.of_int steps
end

(* The simulator's side. Same harness as [throughput.ml]: a compiled step,
   vmapped over lanes, timed with a sync. *)
module Sim (Set : Nx.Devices) = struct
  let place = Set.on

  let prepare (c : Step.config) m0 d0 ~lanes =
    let m = Model.place place m0 in
    let d = Data.place place (batch d0 ~lanes) in
    let ptree = Nx.Ptree.( @-> ) (Data.ptree ()) (Nx.Ptree.returns (Data.ptree ())) in
    let one = Rune.jit ptree (fun d -> Step.step_with c m d) in
    m, d, Rune.vmap ptree one

  let sync (d : Set.d Data.data) = Nx.sync (Nx.slice [ I 0 ] d.time)

  let rollout step d ~steps =
    let d = step d in
    sync d;
    let t0 = Unix.gettimeofday () in
    let last = ref d in
    for _ = 1 to steps do
      last := step !last
    done;
    sync !last;
    (Unix.gettimeofday () -. t0) *. 1e3 /. Float.of_int steps

  (* The gradient model-based RL needs: d(state after one step)/d(model), taken
     through the whole step with the KKT rule. Reported separately because it is
     the segment a differentiable engine is supposed to win, and the one whose
     absence makes a loop impossible rather than slow. *)
  let gradient (c : Step.config) m d ~steps =
    let ptree = Nx.Ptree.( @-> ) (Model.ptree ()) (Nx.Ptree.returns (Data.ptree ())) in
    let f = Rune.jit ptree (fun m -> Step.step_with c (Model.place place m) d) in
    let scalar m =
      let d = f m in
      Nx.sum (Nx.slice [ A; I 0 ] d.Mjc.Data.qpos)
    in
    let grad =
      Rune.jit
        (Nx.Ptree.( @-> ) (Model.ptree ()) (Nx.Ptree.returns (Model.ptree ())))
        (fun m -> Rune.value_and_grad (Model.ptree ()) scalar m |> snd)
    in
    let _ = grad m in
    let t0 = Unix.gettimeofday () in
    for _ = 1 to steps do
      let _ : Set.d Model.model = grad m in
      ()
    done;
    (Unix.gettimeofday () -. t0) *. 1e3 /. Float.of_int steps
end

module Host = Sim (Nx.Host)
module PHost = Policy (Nx.Host)

let split_lanes v =
  match String.split v ~on:',' with
  | xs -> List.filter_map xs ~f:Int.of_string_opt

let () =
  let argv = Array.to_list (Sys.get_argv ()) in
  let argv =
    match argv with
    | _ :: rest -> rest
    | [] -> []
  in
  let lanes = ref [ 512 ] in
  let rows = ref None in
  let steps = ref 20 in
  let models = ref [] in
  let rec parse = function
    | [] -> ()
    | "--lanes" :: v :: rest ->
      lanes := split_lanes v;
      parse rest
    | "--rows" :: v :: rest ->
      rows := Int.of_string_opt v;
      parse rest
    | "--steps" :: v :: rest ->
      steps := Int.of_string v;
      parse rest
    | m :: rest when String.is_prefix m ~prefix:"--" ->
      (match String.chop_prefix m ~prefix:"--rows=" with
       | Some v -> rows := Int.of_string_opt v
       | None ->
         (match String.chop_prefix m ~prefix:"--steps=" with
          | Some v -> steps := Int.of_string v
          | None -> ()));
      parse rest
    | m :: rest ->
      models := m :: !models;
      parse rest
  in
  parse argv;
  let models = List.rev !models in
  let models = if List.is_empty models then [ "hopper" ] else models in
  List.iter models ~f:(fun name ->
    let m = load name in
    let d = Mjc.Forward.forward m (Data.create m) in
    let lanes = List.hd_exn !lanes in
    let obs = (Nx.shape d.Data.qpos).(0) + (Nx.shape d.Data.qvel).(0) in
    let act = Nx.shape d.Data.ctrl |> fun s -> if Array.is_empty s then 1 else s.(0) in
    Printf.printf "\n%s: %d lanes, obs %d, act %d, %d steps\n%!" name lanes obs act !steps;
    Printf.printf
      "| segment | ms/iter | us/env-step | share |\n| --- | ---: | ---: | ---: |\n%!";
    let c = Step.config ?rows:!rows m in
    let _, d, step = Host.prepare c m d ~lanes in
    let rollout = Host.rollout step d ~steps:!steps in
    let policy = PHost.run ~lanes ~obs ~act ~steps:!steps in
    (* The gradient segment is the one part that does not run yet: tracing a
       compiled program inside the gradient's own trace fails in [Rune], and
       that is the next thing to fix rather than something to report as a
       number. *)
    let grad =
      try Host.gradient c m d ~steps:!steps with
      | _ -> Float.nan
    in
    let total = rollout +. policy +. if Float.is_nan grad then 0. else grad in
    let pct x = 100. *. x /. total in
    Printf.printf
      "| rollout (sim step) | %.3f | %.2f | %.0f%% |\n\
       | policy fwd+bwd | %.3f | %.2f | %.0f%% |\n\
       | sim gradient, 1 step | %s | %s | %s |\n\
       | **total** | **%.3f** | **%.2f** | |\n\
       %!"
      rollout
      (rollout *. 1e3 /. Float.of_int lanes)
      (pct rollout)
      policy
      (policy *. 1e3 /. Float.of_int lanes)
      (pct policy)
      (if Float.is_nan grad then "unavailable" else Printf.sprintf "%.3f" grad)
      (if Float.is_nan grad
       then "unavailable"
       else Printf.sprintf "%.2f" (grad *. 1e3 /. Float.of_int lanes))
      (if Float.is_nan grad then "unavailable" else Printf.sprintf "%.0f%%" (pct grad))
      total
      (total *. 1e3 /. Float.of_int lanes);
    Printf.printf
      "env-steps/s: rollout %.0f, whole loop %.0f\n%!"
      (Float.of_int lanes *. 1e3 /. rollout)
      (Float.of_int lanes *. 1e3 /. total))
