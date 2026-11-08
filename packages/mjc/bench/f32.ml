(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What float32 costs in accuracy, and where it costs a *decision* instead.

   The first table steps each model from its recorded mid-trajectory state in
   both dtypes and reports the worst relative difference over 20 steps. The
   second steps from each model's *resting* state, which is where a contact
   distance can sit within a rounding of the margin: there the two dtypes can
   disagree about which contacts exist, and that is a different statement from a
   tolerance miss — it is a hard threshold, and float32 moves the state to the
   other side of it.

   dune exec packages/mjc/bench/f32.exe *)

open Base
module Mjc = Mjc
module Printf = Stdlib.Printf
module Format = Stdlib.Format

let load name =
  match Mjc.Mjcb.read ("test/fixtures/" ^ name ^ ".mjcb") with
  | Ok m -> m
  | Error e -> failwith e

let max_abs a b =
  let x = Nx.to_array a
  and y = Nx.to_array b in
  Array.fold2_exn x y ~init:0. ~f:(fun acc p q -> Float.max acc (Float.abs (p -. q)))

let scale t =
  Array.fold (Nx.to_array t) ~init:0. ~f:(fun acc x -> Float.max acc (Float.abs x))

let rel a b = max_abs a b /. Float.max 1e-12 (Float.max (scale a) (scale b))
let int_of x = Stdlib.Int32.to_int (Nx.item [] x)

(* Both dtypes from the same state, stepped together. Reports the worst
   relative difference and the first step at which the active sets disagree. *)
let compare_from ?(steps = 20) name =
  let m = load name in
  let m32 = Mjc.Model.cast Nx.float32 m in
  let c = Mjc.Step.config m in
  let d64 = ref (Mjc.Data.create m)
  and d32 = ref (Mjc.Data.create m32) in
  let worst = ref 0.
  and absolute = ref 0.
  and flipped = ref 0 in
  for k = 1 to steps do
    d64 := Mjc.Step.step_with c m !d64;
    d32 := Mjc.Step.step_with c m32 !d32;
    let disagrees =
      int_of !d64.Mjc.Data.nefc <> int_of !d32.Mjc.Data.nefc
      || int_of !d64.Mjc.Data.ncon <> int_of !d32.Mjc.Data.ncon
    in
    if disagrees && Int.equal !flipped 0 then flipped := k;
    worst := Float.max !worst (rel !d64.Mjc.Data.qpos !d32.Mjc.Data.qpos);
    worst := Float.max !worst (rel !d64.Mjc.Data.qvel !d32.Mjc.Data.qvel);
    absolute := Float.max !absolute (max_abs !d64.Mjc.Data.qpos !d32.Mjc.Data.qpos)
  done;
  Printf.printf
    "| %s | %d | %d/%d | %d/%d | %.2e | %.2e | %s |\n%!"
    name
    steps
    (int_of !d64.Mjc.Data.nefc)
    (int_of !d32.Mjc.Data.nefc)
    (int_of !d64.Mjc.Data.ncon)
    (int_of !d32.Mjc.Data.ncon)
    !worst
    !absolute
    (if Int.equal !flipped 0 then "never" else Printf.sprintf "step %d" !flipped)

let () =
  let models = [ "hopper"; "walker"; "humanoid"; "boxes2"; "bounce" ] in
  Printf.printf
    "| model | steps | nefc 64/32 | ncon 64/32 | worst relative | worst absolute | \
     active set |\n";
  Printf.printf "| --- | ---: | --- | --- | ---: | ---: | --- |\n%!";
  List.iter models ~f:(fun name ->
    (* A resting start, where thresholds are ambiguous. *)
    compare_from name);
  Printf.printf
    "\n\
     The first five rows start every model at its resting pose, which is where the \
     interesting disagreement happens: a contact distance within one rounding of the \
     margin. Seeded from recorded mid-trajectory states the two dtypes agree on the \
     active set at every step and differ by 1.5e-7 to 6.1e-5 -- test/test_dtype.ml is \
     that comparison, and it is the gate.\n\n\
    \     Read the walker's 0.95 next to its absolute difference: that model starts\n\
    \     with every joint coordinate within a rounding of zero, so a relative measure\n\
    \     divides by nothing. The absolute column is the one to read, and the active\n\
    \     set column is the one that decides whether the two runs are comparable at\n\
    \     all.\n"
