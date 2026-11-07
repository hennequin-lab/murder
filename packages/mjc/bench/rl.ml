(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The number an RL loop is judged by: env-steps per second end to end, split
   into the parts it is made of.

   Every throughput figure quoted before this was microseconds per simulator
   step, which is a component, not a result. An agent loop spends its time in
   three places -- rolling the environment out, evaluating the policy, and the
   gradient the policy is updated with -- and a simulator that is fast while the
   other two dominate has not made anything faster. This measures all three on
   one batch and one device and prints their shares.

   The policy is the usual 2x256 tanh MLP, forward and backward. The gradient is
   the one model-based RL needs: the derivative of the state after one step with
   respect to the model's own parameters, which is what a differentiable engine
   exists to make affordable.

   Run: dune exec packages/mjc/bench/rl.exe -- --lanes 512 --rows=24 hopper *)

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

(* One environment's state replicated into [lanes] lanes on the host: a set
   without kernels cannot concatenate, and a training loop builds its batch once
   anyway. *)
let batch (d : Nx.host Data.data) ~lanes =
  Nx.Ptree.map
    (Data.ptree ())
    (fun _ x ->
       Nx.concatenate ~axis:0 (List.init lanes ~f:(fun _ -> Nx.unsqueeze ~axes:[ 0 ] x)))
    d

(* The three segments, in milliseconds per iteration of that segment. A
   segment that cannot run on this device is [None], which is a different
   statement from a fast one: [Rune.value_and_grad] runs its program eagerly, and
   a device set minted without kernels cannot execute an eager op. *)
type split =
  { rollout : float
  ; policy : float option
  ; grad : float option
  }

module Bench (Set : Nx.Devices) = struct
  let place = Set.on

  (* A device program is only timed once it has finished: without a sync the
     clock measures how long the host takes to submit the work. *)
  let sync (d : Set.d Data.data) = Nx.sync (Nx.slice [ I 0 ] d.time)

  let sim (c : Step.config) m0 d0 ~lanes =
    let m = Model.place place m0 in
    let d = Data.place place (batch d0 ~lanes) in
    let ptree = Nx.Ptree.( @-> ) (Data.ptree ()) (Nx.Ptree.returns (Data.ptree ())) in
    let one = Rune.jit ptree (fun d -> Step.step_with c m d) in
    m, d, Rune.vmap ptree one

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

  (* The gradient the loop needs, taken through one step with the KKT rule. The
     step is written out inside the differentiated function rather than called
     as a compiled program: a compiled call inside a trace sends the tracer into
     the compiled program's own tracing, which is not what a [jit] boundary
     means. *)
  let gradient (c : Step.config) m d ~steps =
    let scalar m =
      let d = Step.step_with c m d in
      Nx.sum d.Data.qpos
    in
    let ptree = Model.ptree () in
    (* [Rune.jit] around this would nest two program builders -- the gradient
       brings its own -- so it is called the way [spike_root.ml] calls it. *)
    let sync () =
      let v, _ = Rune.value_and_grad ptree scalar m in
      Nx.sync v
    in
    sync ();
    let t0 = Unix.gettimeofday () in
    for _ = 1 to steps do
      sync ()
    done;
    (Unix.gettimeofday () -. t0) *. 1e3 /. Float.of_int steps

  let policy ~lanes ~obs ~act ~steps =
    let w1 = Nx.mul_s (Nx.ones Nx.float32 [| obs; 256 |]) 0.01 |> Nx.place place in
    let w2 = Nx.mul_s (Nx.ones Nx.float32 [| 256; 256 |]) 0.01 |> Nx.place place in
    let w3 = Nx.ones Nx.float32 [| 256; act |] |> Nx.place place in
    let x = Nx.ones Nx.float32 [| lanes; obs |] |> Nx.place place in
    let ptree = Nx.Ptree.(pair tensor (pair tensor tensor)) in
    let loss (w1, (w2, w3)) =
      let h = Nx.tanh (Nx.matmul x w1) in
      let h = Nx.tanh (Nx.matmul h w2) in
      Nx.sum (Nx.tanh (Nx.matmul h w3))
    in
    let p = w1, (w2, w3) in
    let sync () =
      let v, _ = Rune.value_and_grad ptree loss p in
      Nx.sync v
    in
    sync ();
    let t0 = Unix.gettimeofday () in
    for _ = 1 to steps do
      sync ()
    done;
    (Unix.gettimeofday () -. t0) *. 1e3 /. Float.of_int steps

  (* The gradient is taken on one environment's data: [Step.step_with] is a
     single-environment program, and [vmap] is what turns it into a batch. Its
     cost therefore does not amortise the way the rollout's does, which is why
     it is reported as a multiple of one step rather than folded into the loop's
     total. *)
  let run (c : Step.config) m d ~lanes ~steps ~obs ~act ~gradients =
    let m, batch, step = sim c m d ~lanes in
    let rollout = rollout step batch ~steps in
    let policy = if gradients then Some (policy ~lanes ~obs ~act ~steps) else None in
    let grad =
      if gradients then Some (gradient c m (Data.place place d) ~steps) else None
    in
    { rollout; policy; grad }
end

let split_lanes v =
  match String.split v ~on:',' with
  | xs -> List.filter_map xs ~f:Int.of_string_opt

let () =
  let argv =
    match Array.to_list (Sys.get_argv ()) with
    | _ :: tl -> tl
    | [] -> []
  in
  let lanes = ref 512 in
  let rows = ref None in
  let steps = ref 10 in
  let models = ref [] in
  let rec parse = function
    | [] -> ()
    | a :: tl when String.is_prefix a ~prefix:"--lanes=" ->
      lanes := List.hd_exn (split_lanes (String.chop_prefix_exn a ~prefix:"--lanes="));
      parse tl
    | a :: tl when String.is_prefix a ~prefix:"--rows=" ->
      rows := Int.of_string_opt (String.chop_prefix_exn a ~prefix:"--rows=");
      parse tl
    | a :: tl when String.is_prefix a ~prefix:"--steps=" ->
      steps := Int.of_string (String.chop_prefix_exn a ~prefix:"--steps=");
      parse tl
    | a :: tl when String.is_prefix a ~prefix:"--" -> parse tl
    | a :: tl ->
      models := a :: !models;
      parse tl
  in
  parse argv;
  let models =
    match List.rev !models with
    | [] -> [ "hopper" ]
    | xs -> xs
  in
  let lanes = !lanes
  and steps = !steps in
  let gpus =
    match Rig_amd_amdgpu.count () with
    | Ok n when n > 0 ->
      List.filter_map (List.init n ~f:Fn.id) ~f:(fun i ->
        match Rig.open_ (Rig_amd_amdgpu.gpu i) with
        | Ok dev -> Some dev
        | Error _ -> None)
    | _ -> []
  in
  Printf.printf
    "| model | device | lanes | rollout us/env-step | policy us/env-step | sim gradient \
     ms/env-step | gradient / step | rollout share | env-steps/s rollout | env-steps/s \
     loop |\n";
  Printf.printf
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n%!";
  let report name device split =
    (* Model-based RL replaces the policy gradient with a model gradient, so
       the loop total is the rollout plus the policy. *)
    let total = split.rollout +. Option.value split.policy ~default:0.0 in
    let per ms = ms *. 1e3 /. Float.of_int lanes in
    let rate ms = Float.of_int lanes *. 1e3 /. ms in
    Printf.printf
      "| %s | %s | %d | %.1f | %s | %s | %s | %.0f%% | %.0f | %.0f |\n%!"
      name
      device
      lanes
      (per split.rollout)
      (Option.value_map split.policy ~default:"-" ~f:(fun v ->
         Printf.sprintf "%.1f" (per v)))
      (Option.value_map split.grad ~default:"-" ~f:(Printf.sprintf "%.1f"))
      (Option.value_map split.grad ~default:"-" ~f:(fun v ->
         Printf.sprintf "%.1fx" (v *. 1e3 /. per split.rollout)))
      (100. *. split.rollout /. total)
      (rate split.rollout)
      (rate total)
  in
  List.iter models ~f:(fun name ->
    let m = load name in
    let d = Mjc.Forward.forward m (Data.create m) in
    let obs = (Nx.shape d.Data.qpos).(0) + (Nx.shape d.Data.qvel).(0) in
    let act =
      match Nx.shape d.Data.ctrl with
      | [||] -> 1
      | s -> s.(0)
    in
    let c = Step.config ?rows:!rows m in
    let module H = Bench (Nx.Host) in
    report name "host" (H.run c m d ~lanes ~steps ~obs ~act ~gradients:true);
    match gpus with
    | [] -> ()
    | dev :: _ ->
      let module D = (val Nx.devices [ dev ] : Nx.Devices) in
      let module G = Bench (D) in
      (* A gradient program cannot be compiled on a set without kernels: Rune
         runs [value_and_grad] eagerly, and its nesting inside a [jit] is what
         the earlier failure was about. Only the rollout is timed there. *)
      report name "gpu" (G.run c m d ~lanes ~steps ~obs ~act ~gradients:false))
