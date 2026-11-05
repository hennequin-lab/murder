(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The M6 cost sheet: what a reverse pass costs against a forward step, by
   model, with the implicit rule in place and without it.

   The number the gate cares about is reverse / forward. An unrolled solver's
   reverse pass pays for every iteration it unrolled; the implicit rule pays for
   one linear solve at the fixed active set, whatever the budget. The column
   that shows this is `implicit` against `unrolled` on a model whose rows need
   more than one iteration.

   Run: dune exec packages/mjc/bench/bench_grad.exe *)

open Base
module Printf = Stdlib.Printf

let f64 = Nx.float64

let load name =
  let candidate d =
    let path = d ^ "/" ^ name ^ ".mjcb" in
    Option.some_if (Stdlib.Sys.file_exists path) path
  in
  match List.find_map [ "test/fixtures"; "packages/mjc/test/fixtures" ] ~f:candidate with
  | Some p ->
    (match Mjc.Mjcb.read p with
     | Ok m -> m
     | Error e -> failwith e)
  | None -> failwith (Printf.sprintf "no fixture %s" name)

let time ?(warmup = 3) ?(reps = 30) f x =
  for _ = 1 to warmup do
    ignore (f x)
  done;
  let t0 = Unix.gettimeofday () in
  for _ = 1 to reps do
    ignore (f x)
  done;
  (Unix.gettimeofday () -. t0) /. Float.of_int reps

(* A state the engine reaches: step from rest with a velocity, so the rows are
   the ones a step produces rather than ones a benchmark invented. *)
let reached m =
  let d0 = Mjc.Data.create m in
  let nv = Array.fold (Nx.shape d0.qvel) ~init:1 ~f:( * ) in
  let qvel =
    Nx.create
      f64
      (Nx.shape d0.qvel)
      (Array.init nv ~f:(fun i -> 0.6 *. Float.sin (Float.of_int (3 * (i + 1)))))
  in
  let d = ref { d0 with qvel } in
  for _ = 1 to 12 do
    d := Mjc.Step.step m !d
  done;
  !d

let loss m ~n d =
  let d = ref d in
  for _ = 1 to n do
    d := Mjc.Step.step m !d
  done;
  Nx.sum (Nx.mul !d.qpos !d.qpos)

let ms x = x *. 1e3

(* Peak resident memory of the process, from the kernel's high-water mark. The
   only honest way to compare two memory behaviours is to run each in its own
   process, so this program does one mode per invocation:
   bench_grad.exe memory plain|remat [steps] *)
let peak_kb () =
  let ic = Stdlib.open_in "/proc/self/status" in
  let rec loop () =
    match Stdlib.input_line ic with
    | line ->
      if String.is_prefix line ~prefix:"VmHWM:"
      then (
        Stdlib.close_in ic;
        let digits = String.filter line ~f:(fun c -> Char.is_digit c) in
        Stdlib.int_of_string digits)
      else loop ()
    | exception End_of_file ->
      Stdlib.close_in ic;
      0
  in
  loop ()

let memory mode name steps =
  let m = load name in
  let d = reached m in
  let dyn = Mjc.Dynamics.compile m in
  let t0 = Unix.gettimeofday () in
  ignore
    (Rune.value_and_grad
       Mjc.Model.ptree
       (fun (params : Mjc.Model.host) ->
          let d =
            match mode with
            | "remat" -> Mjc.Dynamics.horizon_remat dyn params d steps
            | "scan" -> Mjc.Dynamics.horizon_scan dyn params d steps
            | _ -> Mjc.Dynamics.horizon dyn params d steps
          in
          Nx.sum (Nx.mul d.Mjc.Data.qpos d.Mjc.Data.qpos))
       m);
  Unix.gettimeofday () -. t0

let table () =
  Printf.printf
    "%-9s %5s %8s %8s %7s %8s | %9s %9s %7s\n%!"
    "model"
    "rows"
    "fwd/ms"
    "grad/ms"
    "g/f"
    "max|g|"
    "solve_i/ms"
    "solve_u/ms"
    "u/i";
  List.iter [ "chain6"; "looped"; "finger"; "blocks"; "boxes2" ] ~f:(fun name ->
    let m = load name in
    let d = reached m in
    let nefc = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.nefc) in
    (* Compiled forward, and compiled value-and-gradient: the two things a
       training loop actually runs. Comparing an interpreted reverse pass with
       a compiled forward would measure the interpreter. *)
    let dyn = Mjc.Dynamics.compile m in
    let one d = Mjc.Dynamics.run dyn m d in
    let step = Rune.jit Nx.Ptree.(Mjc.Data.ptree @-> returns Mjc.Data.ptree) one in
    let grad =
      Rune.jit
        Nx.Ptree.(Mjc.Data.ptree @-> returns Mjc.Data.ptree)
        (fun d ->
           let v, g =
             Rune.value_and_grad
               Mjc.Data.ptree
               (fun d -> Nx.sum (Nx.mul (one d).Mjc.Data.qpos (one d).Mjc.Data.qpos))
               d
           in
           ignore v;
           g)
    in
    let fwd = time step d in
    let back = time grad d in
    let g_worst =
      Array.fold (Nx.to_array (Nx.abs (grad d).Mjc.Data.qpos)) ~init:0.0 ~f:Float.max
    in
    (* The rule's saving, measured on the solver alone: the same rows solved with
       the implicit rule and with the unrolled derivative, both eager, both
       differentiated. The forward values are identical; only the backward pass
       differs. *)
    let prepared = Mjc.Forward.forward m d in
    let solve_grad ?implicit () =
      let f d =
        let v, g =
          Rune.value_and_grad
            Mjc.Data.ptree
            (fun d -> Nx.sum (Mjc.Solver.forward ?implicit ~iters:8 m d).qacc)
            d
        in
        ignore v;
        g
      in
      time f prepared
    in
    let s_imp = solve_grad ()
    and s_unr = solve_grad ~implicit:false () in
    Printf.printf
      "%-9s %5d %8.3f %8.3f %7.2f %8.2g | %9.3f %9.3f %7.2f\n%!"
      name
      nefc
      (ms fwd)
      (ms back)
      (back /. fwd)
      g_worst
      (ms s_imp)
      (ms s_unr)
      (s_unr /. s_imp))

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: "memory" :: mode :: name :: steps :: _ ->
    let seconds = memory mode name (Stdlib.int_of_string steps) in
    Printf.printf "%s %s %s %d kb %.2f s\n%!" mode name steps (peak_kb ()) seconds
  | _ -> table ()
