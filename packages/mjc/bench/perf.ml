(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Wall time per step, eager and compiled, for the engines' shared models.

   The MuJoCo counterpart is `tools/perf_mujoco.py`, which times `mj_step` on
   the same fixture XMLs from the same state (the model's reference pose, no
   velocity, no control). Run both from the repository root:

   dune exec packages/mjc/bench/perf.exe hopper walker humanoid ...
   /tmp/mjcenv/bin/python packages/mjc/tools/perf_mujoco.py \
   packages/mjc/test/fixtures hopper walker humanoid ...

   The two tables document each other: same models, same state, same machine,
   one thread. What they do not share is the solver's schedule, and `perf.md`
   says so. *)

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

(* Two thousand steps after fifty warm-up ones, wall clock around the loop. A
   compiled step is one call into the traced program; an eager step is the
   library's own graph evaluated per call. *)
let time ~steps f x =
  let x = ref x in
  for _ = 1 to 50 do
    x := f !x
  done;
  let t0 = now () in
  for _ = 1 to steps do
    x := f !x
  done;
  (now () -. t0) /. Float.of_int steps

let us s = Float.round (s *. 1e7) /. 10.0

let () =
  let models =
    match Array.to_list Stdlib.Sys.argv with
    | _ :: names -> names
    | _ -> [ "pendulum"; "chain6"; "bounce"; "boxes2"; "hopper"; "walker"; "humanoid" ]
  in
  Printf.printf
    "| model | nv | rows | nefcmax | dt | eager | compiled | per_env, vmap 8 | first \
     call |\n\
     | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
  List.iter models ~f:(fun name ->
    let m = load name in
    (* The reference pose at rest: the same state the MuJoCo script starts
       from, so the two tables compare like with like. *)
    let d = { (Data.create m) with qpos = m.params.qpos0 } in
    (* The contact count at the rest pose, for the table: a model standing
       still asks less of the solver than one mid-flight. *)
    let d0 = Step.step m d in
    let nefc = Stdlib.Int32.to_int (Nx.item [] d0.nefc) in
    let eager = time ~steps:500 (fun d -> Step.step m d) d in
    let compiled = Step.compile m in
    let one = Step.run compiled in
    (* The first call traces, compiles, and — with BEAM set — searches kernel
       optimisations and times the candidates. It is the compile cost and it is
       reported separately because a beam search makes it minutes. *)
    let t0 = now () in
    let _ = one d in
    let compile_time = now () -. t0 in
    let compiled_time = time ~steps:500 one d in
    let lanes = 8 in
    (* Stack every leaf along a new leading axis, 0-d ones included; the vmap
       maps over that axis. *)
    let batched =
      match List.init lanes ~f:(fun _ -> Data.create m) with
      | [] -> invalid_arg "perf: no lanes"
      | d0 :: d1 :: rest ->
        let both =
          Nx.Ptree.map2 (Data.ptree ()) (fun _ a b -> Nx.stack ~axis:0 [ a; b ]) d0 d1
        in
        List.fold rest ~init:both ~f:(fun acc x ->
          Nx.Ptree.map2
            (Data.ptree ())
            (fun _ a b -> Nx.concatenate ~axis:0 [ a; Nx.unsqueeze ~axes:[ 0 ] b ])
            acc
            x)
      | [ d ] -> d
    in
    let vec =
      Rune.vmap (Nx.Ptree.( @-> ) (Data.ptree ()) (Nx.Ptree.returns (Data.ptree ()))) one
    in
    let _ = vec batched in
    let per_env = time ~steps:50 vec batched /. Float.of_int lanes in
    Printf.printf
      "| %s | %d | %d | %d | %.3g | %.0f us | %.0f us | %.0f us | %.2f s |\n"
      name
      m.topo.nv
      nefc
      m.topo.nefcmax
      (Nx.item [] m.params.timestep)
      (us eager)
      (us compiled_time)
      (us per_env)
      compile_time);
  (* Where the time goes: the same forward pass with the solver's schedule at
     one outer iteration and at the budget the step uses. The difference is the
     active-set schedule's share, and it is what the M7 recommendations in
     `doc/perf.md` are sized against. *)
  Printf.printf
    "\n\
     | model | forward, iters 1 | forward, iters 8 | schedule share |\n\
     | --- | ---: | ---: | ---: |\n";
  List.iter models ~f:(fun name ->
    let m = load name in
    let d = { (Data.create m) with qpos = m.params.qpos0 } in
    let fwd iters = time ~steps:200 (fun d -> Forward.forward ~iters m d) d in
    let one = fwd 1
    and eight = fwd 8 in
    Printf.printf
      "| %s | %.0f us | %.0f us | %.0f%% |\n"
      name
      (us one)
      (us eight)
      (100.0 *. (eight -. one) /. eight))
