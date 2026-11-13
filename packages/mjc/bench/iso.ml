(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The isolation behind [mjc_ilqr]'s open question: the same dilqr problem —
   same shapes, same horizon, same losses — with a trivial analytic dynamics
   instead of mjc's step. If the implicit gradient matches finite differences
   here and not there, the plumbing is not what is at fault.

   Run: dune exec packages/mjc/bench/iso.exe *)

open Base
module Printf = Stdlib.Printf

let f64 = Nx.float64
let n = 4
let m = 2
let horizon = 20
let vec a = Nx.create f64 [| Array.length a |] a
let target = [| 1.4; -1.2 |]

(* A deliberately trivial smooth dynamics with the same shapes. *)
let dt = 0.01

let dynamics _ ~k:_ ~x ~u =
  let qpos = Nx.slice [ R (0, 2) ] x
  and qvel = Nx.slice [ R (2, 4) ] x in
  Nx.concatenate ~axis:0 [ Nx.add qpos (Nx.mul_s qvel dt); Nx.add qvel (Nx.mul_s u dt) ]

let running_loss _ ~k:_ ~x ~u =
  let err = Nx.sub (Nx.slice [ R (0, 2) ] x) (vec target) in
  Nx.add (Nx.sum (Nx.mul err err)) (Nx.mul_s (Nx.sum (Nx.mul u u)) 0.01)

let final_loss _ ~x =
  let err = Nx.sub (Nx.slice [ R (0, 2) ] x) (vec target) in
  Nx.mul_s (Nx.sum (Nx.mul err err)) 40.0

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
  | None -> failwith "no fixture"

let problem : (Nx.float64_elt, Mjc.Model.host, Nx.host) Dilqr.Problem.t =
  { dtype = f64
  ; n
  ; m
  ; horizon
  ; free_initial = false
  ; verbose = false
  ; params = Mjc.Model.ptree
  ; dynamics
  ; running_loss
  ; final_loss
  }

let objective taus =
  let xs, us = Dilqr.split_taus problem taus in
  let xf = xs.(Array.length xs - 1) in
  let err = Nx.sub (Nx.slice [ R (0, 2) ] xf) (vec target) in
  let effort =
    Array.fold us ~init:(Nx.scalar f64 0.0) ~f:(fun a u -> Nx.add a (Nx.sum (Nx.mul u u)))
  in
  Nx.add (Nx.mul_s (Nx.sum (Nx.mul err err)) 10.0) (Nx.mul_s effort 0.001)

let theta = load "arm2"
let x0 = vec [| 0.0; 0.0; 0.0; 0.0 |]

let ins =
  { Dilqr.Inputs.theta; x0; us0 = Array.init horizon ~f:(fun _ -> Nx.zeros f64 [| m |]) }

let solve = Dilqr.solve_adaptive problem
let si i = solve ~max_iter:100 ~conv_threshold:1e-10 i

let () =
  let taus = si ins in
  let g =
    Nx.to_array
      (Rune.grad (Dilqr.inputs problem) (fun i -> objective (si i)) ins).Dilqr.Inputs.x0
  in
  let h = 1e-5 in
  let fd =
    Array.init n ~f:(fun j ->
      let at d =
        let a = Nx.to_array x0 in
        a.(j) <- a.(j) +. d;
        Nx.item [] (objective (si { ins with Dilqr.Inputs.x0 = vec a }))
      in
      (at h -. at (-.h)) /. (2.0 *. h))
  in
  Printf.printf "obj %.8f\n" (Nx.item [] (objective taus));
  Printf.printf
    "implicit %s\n"
    (String.concat ~sep:", " (Array.to_list (Array.map g ~f:(Printf.sprintf "%.6f"))));
  Printf.printf
    "fd       %s\n"
    (String.concat ~sep:", " (Array.to_list (Array.map fd ~f:(Printf.sprintf "%.6f"))))
