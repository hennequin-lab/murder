(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The M6 spike: can Rune differentiate the step as it stands, and if it can,
   how wrong is the answer?

   The step's solver is an unrolled active-set iteration. Differentiating
   through the unroll differentiates the *last iterate*, not the solution: the
   derivative depends on how many iterations were run, and it is wrong by
   however much the iteration has left to converge. Before writing an implicit
   rule, measure the baseline.

   Run: dune exec packages/mjc/bench/spike_grad.exe *)

open Base
module Printf = Stdlib.Printf

let f64 = Nx.float64

let load name =
  let candidate d =
    let path = d ^ "/" ^ name ^ ".mjcb" in
    Option.some_if (Stdlib.Sys.file_exists path) path
  in
  let path =
    match
      List.find_map [ "test/fixtures"; "packages/mjc/test/fixtures" ] ~f:candidate
    with
    | Some p -> p
    | None -> failwith (Printf.sprintf "no fixture %s" name)
  in
  match Mjc.Mjcb.read path with
  | Ok m -> m
  | Error e -> failwith e

(* Sum of squares of the position after [n] steps, from a given initial
   position. A scalar, as a gradient needs. *)
let make_loss m ~n =
  fun qpos0 ->
  let d = { (Mjc.Data.create m) with qpos = qpos0 } in
  let d = ref d in
  for _ = 1 to n do
    d := Mjc.Step.step m !d
  done;
  Nx.sum (Nx.mul !d.qpos !d.qpos)

let central ~h f x =
  let shape = Nx.shape x in
  let n = Array.fold shape ~init:1 ~f:( * ) in
  let get i = Nx.get [ i ] x in
  let perturb i v =
    let a = Nx.to_array x in
    a.(i) <- v;
    Nx.create Nx.float64 shape a
  in
  Array.init n ~f:(fun i ->
    let x0 = Nx.item [] (get i) in
    let fp = f (perturb i (x0 +. h))
    and fm = f (perturb i (x0 -. h)) in
    (Nx.item [] fp -. Nx.item [] fm) /. (2.0 *. h))

let () =
  let name = if Array.length Stdlib.Sys.argv > 1 then Stdlib.Sys.argv.(1) else "chain6" in
  let budget =
    if Array.length Stdlib.Sys.argv > 2
    then Stdlib.float_of_string Stdlib.Sys.argv.(2)
    else 8.0
  in
  let m0 = load name in
  let m =
    { m0 with
      Mjc.Model.params = { m0.params with iterations = Nx.scalar Nx.float64 budget }
    }
  in
  let n = 20 in
  let loss = make_loss m ~n in
  let d0 = Mjc.Data.create m in
  (* Away from the reference configuration, or nothing moves. *)
  let nq = Array.fold (Nx.shape d0.qpos) ~init:1 ~f:( * ) in
  let q0 =
    Nx.add
      d0.qpos
      (Nx.create
         Nx.float64
         (Nx.shape d0.qpos)
         (Array.init nq ~f:(fun i -> 0.3 *. Float.sin (Float.of_int (i + 1)))))
  in
  let value = Nx.item [] (loss q0) in
  Printf.printf "%s, %d steps, loss = %.12g\n%!" name n value;
  (* 1. Does reverse mode work at all? *)
  let _, pullback = Rune.vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun q -> loss q) q0 in
  let grad = pullback (Nx.scalar f64 1.0) in
  let fd = central ~h:1e-6 loss q0 in
  let g = Nx.to_array grad in
  Printf.printf "%-10s %-22s %-22s %s\n" "index" "reverse mode" "central diff" "rel err";
  Array.iteri g ~f:(fun i gi ->
    let err =
      if Float.equal fd.(i) 0.0 then Float.abs gi else Float.abs ((gi -. fd.(i)) /. fd.(i))
    in
    if Float.( > ) (Float.abs gi) 1e-9 || Float.( > ) (Float.abs fd.(i)) 1e-9
    then Printf.printf "%-10d %-22.12g %-22.12g %.3e\n" i gi fd.(i) err)
