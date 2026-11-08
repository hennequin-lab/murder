(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The table of M7's acceptance: microseconds per environment-step at a sweep
   of batch sizes, on the host and the GPU, beside MuJoCo's `mj_step` on one
   core of the same machine.

   Three quantities per model and lane count: the time of one whole step over
   the batch, the same divided by the lanes, and the same as a multiple of
   `mj_step`. Compile time and the eager single-environment step are reported
   per model, because a training loop pays one of those once and the other
   never.

   dune exec packages/mjc/bench/bench_step.exe -- --json > table.json
   dune exec packages/mjc/bench/bench_step.exe -- --lanes 1,64,256,1024 hopper
   dune exec packages/mjc/bench/bench_step.exe -- --rows=24 humanoid quadruped *)

open Base
module Printf = Stdlib.Printf
module Step = Mjc.Step
module Data = Mjc.Data
module Model = Mjc.Model

(* One step per environment, at whatever device the data lives on. *)
module Run (Set : Nx.Devices) = struct
  let place = Set.on

  (* [steps] steps of a batch of [lanes] environments; returns the per-step
     time in milliseconds. The batch is assembled once, outside the timing. *)
  let run (c : Step.config) m0 d0 ~lanes ~steps ~budget_ms =
    let m = Model.place place m0 in
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
    let one = Rune.jit ptree (fun d -> Step.step_with c m d) in
    let step = Rune.vmap ptree one in
    (* A device program has to be waited for, or the clock measures how long the
       host spent submitting work. *)
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

let load name =
  let path = Stdlib.Filename.concat "test/fixtures" (name ^ ".mjcb") in
  match Mjc.Mjcb.read path with
  | Ok m -> m
  | Error e -> failwith e

let split_ints v = List.filter_map (String.split v ~on:',') ~f:Int.of_string_opt
let now () = Unix.gettimeofday ()

let timed ~steps f =
  let t0 = now () in
  let x = ref (f ()) in
  for _ = 2 to steps do
    x := f ()
  done;
  (now () -. t0) /. Float.of_int steps

(* `mj_step` on one core of this machine, from `tools/perf_mujoco.py`. A model
   missing from this table prints as blank rather than as a guess. *)
let mj_step_us =
  [ "pendulum", 0.6
  ; "hopper", 5.0
  ; "walker", 5.0
  ; "humanoid", 18.7
  ; "quadruped", 9.0
  ; "quad", 9.0
  ; "cyl", 2.0
  ; "ball", 1.0
  ; "chain6", 3.0
  ]

(* The row budget the model needs, found by trying the ones a model plausibly
   needs rather than by guessing: the budget raises when it is short, so the
   smallest that does not raise is the model's worst case over the run. *)
let candidate_budgets = [ 2; 4; 8; 12; 16; 20; 24; 32; 40; 48; 64; 80; 96; 128; 160; 202 ]
let reference name = List.Assoc.find mj_step_us ~equal:String.equal name

type row =
  { model : string
  ; lanes : int
  ; device : string
  ; ms : float
  ; per : float
  }

let rows : row list ref = ref []
let skips : string list ref = ref []
let compiles : (string * float) list ref = ref []
let single : (string * int * float) list ref = ref []

let () =
  let argv = Array.to_list (Sys.get_argv ()) in
  let argv =
    match argv with
    | _ :: tl -> tl
    | [] -> []
  in
  let json = ref false in
  let lanes = ref [ 1; 64; 256; 1024 ]
  and rows_budget = ref None
  and iters = ref None
  and models = ref [] in
  let rec parse = function
    | [] -> ()
    | "--json" :: tl ->
      json := true;
      parse tl
    | "--no-json" :: tl ->
      json := false;
      parse tl
    | "--lanes" :: v :: tl ->
      lanes := split_ints v;
      parse tl
    | "--rows" :: v :: tl ->
      rows_budget := List.hd (split_ints v);
      parse tl
    | "--iters" :: v :: tl ->
      iters := List.hd (split_ints v);
      parse tl
    | a :: tl when String.is_prefix a ~prefix:"--lanes=" ->
      lanes := split_ints (String.chop_prefix_exn a ~prefix:"--lanes=");
      parse tl
    | a :: tl when String.is_prefix a ~prefix:"--rows=" ->
      rows_budget := List.hd (split_ints (String.chop_prefix_exn a ~prefix:"--rows="));
      parse tl
    | a :: tl when String.is_prefix a ~prefix:"--iters=" ->
      iters := List.hd (split_ints (String.chop_prefix_exn a ~prefix:"--iters="));
      parse tl
    | a :: tl when String.is_prefix a ~prefix:"--" -> parse tl
    | a :: tl ->
      models := a :: !models;
      parse tl
  in
  parse argv;
  let models =
    match List.rev !models with
    | [] -> [ "pendulum"; "hopper"; "walker"; "humanoid"; "quad"; "cyl" ]
    | xs -> xs
  in
  let lanes = !lanes in
  let budget_ms =
    if Float.( > ) (Float.of_int (List.last_exn lanes)) 2048. then 60000. else 25000.
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
    (* The budget the model needs: the smallest that does not raise. Per
       model, because it is a property of the model and not of the sweep. *)
    let budget =
      match !rows_budget with
      | Some r -> r
      | None ->
        let module H = Run (Nx.Host) in
        let try_rows rows =
          match H.run (Step.config ~rows m) m d0 ~lanes:1 ~steps:1 ~budget_ms:60000. with
          | _ -> Some rows
          | exception Invalid_argument _ -> None
        in
        List.find_map candidate_budgets ~f:try_rows
        |> Option.value ~default:m.topo.nefcmax
    in
    let c = Step.config ?iters:!iters ~rows:budget m in
    (* Eager and compiled at one environment, both at the model's own budget so
       that the two numbers describe the same configuration. *)
    let ptree = Nx.Ptree.( @-> ) (Data.ptree ()) (Nx.Ptree.returns (Data.ptree ())) in
    let one = Rune.jit ptree (fun d -> Step.step_with c m d) in
    let t0 = now () in
    let _ = one d0 in
    compiles := (name, now () -. t0) :: !compiles;
    let compiled = timed ~steps:100 (fun () -> one d0) in
    single := (name, budget, compiled *. 1e6) :: !single;
    List.iter lanes ~f:(fun n ->
      let steps = if n <= 64 then 6 else if n <= 512 then 3 else 2 in
      let add device ms =
        rows
        := { model = name; lanes = n; device; ms; per = ms *. 1e3 /. Float.of_int n }
           :: !rows
      in
      (* A configuration the row budget cannot serve is a skip with its reason,
         not a crash and not a blank cell: the budget's whole point is to say
         what it wanted. *)
      let guard device f =
        match f () with
        | ms -> add device ms
        | exception Invalid_argument why ->
          skips := Printf.sprintf "%s %d %s: %s" name n device why :: !skips
        | exception Stdlib.Exit ->
          skips := Printf.sprintf "%s %d %s: over the time budget" name n device :: !skips
      in
      let module H = Run (Nx.Host) in
      guard "host" (fun () ->
        match H.run c m d0 ~lanes:n ~steps ~budget_ms with
        | Stdlib.Ok ms -> ms
        | Stdlib.Error _ -> raise Stdlib.Exit);
      match gpus with
      | [] -> ()
      | dev :: _ ->
        let module D = (val Nx.devices [ dev ] : Nx.Devices) in
        let module G = Run (D) in
        guard "gpu" (fun () ->
          match G.run c m d0 ~lanes:n ~steps ~budget_ms with
          | Stdlib.Ok ms -> ms
          | Stdlib.Error _ -> raise Stdlib.Exit)));
  let rows = List.rev !rows in
  let f x = Printf.sprintf "%.4f" x in
  let opt v =
    match v with
    | None -> "null"
    | Some x -> Int.to_string x
  in
  let single_json =
    List.rev_map !single ~f:(fun (m, rows, c) ->
      Printf.sprintf
        "    {\"model\": \"%s\", \"rows\": %d, \"compiled_us\": %s}"
        m
        rows
        (f c))
  in
  let compile_json =
    List.rev_map !compiles ~f:(fun (m, s) -> Printf.sprintf "    \"%s\": %s" m (f s))
  in
  let row_json r =
    let ratio =
      match reference r.model with
      | None -> "null"
      | Some us -> f (r.per /. us)
    in
    Printf.sprintf
      "    {\"model\": \"%s\", \"lanes\": %d, \"device\": \"%s\", \"ms\": %s, \"us\": \
       %s, \"x_mj\": %s}"
      r.model
      r.lanes
      r.device
      (f r.ms)
      (f r.per)
      ratio
  in
  let mj_json =
    List.map mj_step_us ~f:(fun (m, us) -> Printf.sprintf "\"%s\": %.2f" m us)
  in
  if !json
  then (
    let p fmt = Printf.printf (Stdlib.( ^^ ) fmt "\n") in
    p "{";
    p "  \"config\": {\"rows\": %s, \"iters\": %s}," (opt !rows_budget) (opt !iters);
    p "  \"single\": [";
    p "%s" (String.concat ~sep:",\n" single_json);
    p "  ],";
    p "  \"compile_s\": {";
    p "%s" (String.concat ~sep:",\n" compile_json);
    p "  },";
    p "  \"steps\": [";
    p "%s" (String.concat ~sep:",\n" (List.map rows ~f:row_json));
    p "  ],";
    p "  \"mj_step_us\": {%s}" (String.concat ~sep:", " mj_json);
    p "}")
  else (
    Printf.printf
      "| model | lanes | device | ms/step | us/env-step | env-steps/s | x mj_step, 1 \
       core |\n";
    Printf.printf "| --- | ---: | --- | ---: | ---: | ---: | ---: |\n";
    List.iter rows ~f:(fun r ->
      let ratio =
        match reference r.model with
        | None -> "-"
        | Some us -> Printf.sprintf "%.0fx" (r.per /. us)
      in
      Printf.printf
        "| %s | %d | %s | %.2f | %.2f | %.0f | %s |\n"
        r.model
        r.lanes
        r.device
        r.ms
        r.per
        (1e6 /. r.per)
        ratio);
    Printf.printf "\n| model | rows | compiled, us | compile + first call, s |\n";
    Printf.printf "| --- | ---: | ---: | ---: |\n";
    List.iter (List.rev !single) ~f:(fun (m, rows, c) ->
      let s =
        List.Assoc.find !compiles ~equal:String.equal m |> Option.value ~default:0.
      in
      Printf.printf "| %s | %d | %.2f | %.2f |\n" m rows c s);
    match !skips with
    | [] -> ()
    | xs ->
      Printf.printf "\nskipped:\n";
      List.iter (List.rev xs) ~f:(fun s -> Printf.printf "- %s\n" s))
