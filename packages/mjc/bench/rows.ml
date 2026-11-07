(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* How much of the row buffer a step actually uses, and how much of that is
   doing work.

   The solver factors an [nb x nb] matrix where [nb] is the *budget* the step was
   compiled for, not the number of rows the step produced, and the measurement in
   [doc/perf.md] says that cubic term is 86% of a step. So the arithmetic of any
   future win is [(budget / working set)^3], and this is the denominator: for
   every state of a recorded trajectory, the row count and the number of rows
   with a non-zero force.

   The second number is an upper bound on the working set, not the working set:
   the engine pins a row by comparing its force against its bounds, and a boxed
   friction row sitting at a bound has a non-zero force, so it counts here as
   free when it is pinned. If the bound is small enough to be interesting, the
   real working set is smaller still.

   dune exec packages/mjc/bench/rows.exe *)

open Base
module Printf = Stdlib.Printf

let load name =
  match Mjc.Mjcb.read (Printf.sprintf "test/fixtures/%s.mjcb" name) with
  | Ok m -> m
  | Error e -> failwith e

let median xs =
  match List.sort xs ~compare:Int.compare with
  | [] -> 0
  | ys ->
    let n = List.length ys in
    List.nth_exn ys (n / 2)

let () =
  Printf.printf
    "| model | states | rows median | rows max | non-zero force median | non-zero max | \
     (budget/set)^3 |\n";
  Printf.printf "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n%!";
  List.iter [ "hopper"; "walker"; "humanoid" ] ~f:(fun name ->
    let m = load name in
    let path = Printf.sprintf "test/fixtures/%s.g4.mjbf" name in
    let qpos, qvel, _, act, ctrl, _, _, (steps, _) = G4lib.fixture path in
    let budget = (m.Mjc.Model.topo : Mjc.Topo.t).Mjc.Topo.nefcmax in
    let rows = ref []
    and nfs = ref [] in
    for k = 0 to steps do
      let d = Mjc.Data.create m in
      let d =
        { d with
          Mjc.Data.qpos = Nx.slice [ I k; A ] qpos
        ; qvel = Nx.slice [ I k; A ] qvel
        ; act = Nx.slice [ I k; A ] act
        ; ctrl = Nx.slice [ I k; A ] ctrl
        }
      in
      let d = Mjc.Forward.forward m d in
      let n = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.nefc) in
      let lam = Nx.to_array d.Mjc.Data.efc_force in
      let scale =
        Array.fold (Array.sub lam ~pos:0 ~len:n) ~init:0.0 ~f:(fun acc x ->
          Float.max acc (Float.abs x))
      in
      let tol = Float.max 1e-300 (scale *. 1e-9) in
      let live =
        Array.count (Array.sub lam ~pos:0 ~len:n) ~f:(fun x ->
          Float.( > ) (Float.abs x) tol)
      in
      rows := n :: !rows;
      nfs := live :: !nfs
    done;
    let rows = List.rev !rows
    and nfs = List.rev !nfs in
    let rows_max = List.fold rows ~init:0 ~f:Int.max
    and nf_max = List.fold nfs ~init:0 ~f:Int.max in
    Printf.printf
      "| %s | %d | %d | %d | %d | %d | %.1fx |\n%!"
      name
      (steps + 1)
      (median rows)
      rows_max
      (median nfs)
      nf_max
      (let r = Float.of_int budget /. Float.of_int (Int.max 1 nf_max) in
       r *. r *. r))
