(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Batched planning: a batch of double-integrator targets over one shared
   dynamics, solved in one call, with the compiled gradient of the total cost
   with respect to every member's target.

   The dynamics and the losses are written for a single member; the batched
   solve maps them over the leading batch axis. *)

open Base
open Stdio
module Nx = Nx

let dt = 0.1

module Integrator = struct
  module Params = struct
    type 'a t =
      { mass : 'a
      ; target : 'a
      }
    [@@deriving ptree]
  end

  let n = 2
  let m = 1
  let horizon = 20
  let verbose = false

  let dynamics { Params.mass; _ } ~k:_ ~x ~u =
    let pos = Nx.slice [ R (0, 1) ] x
    and vel = Nx.slice [ R (1, 2) ] x in
    let force = Nx.div u mass in
    Nx.concatenate
      ~axis:0
      [ Nx.add pos (Nx.mul_s vel dt); Nx.add vel (Nx.mul_s force dt) ]

  let running_loss { Params.target; _ } ~k:_ ~x ~u =
    let pos = Nx.slice [ R (0, 1) ] x
    and vel = Nx.slice [ R (1, 2) ] x in
    let err = Nx.sub pos target in
    Nx.sum
      (Nx.add
         (Nx.mul_s (Nx.add (Nx.mul err err) (Nx.mul vel vel)) 0.5)
         (Nx.mul_s (Nx.mul u u) 0.05))

  let final_loss { Params.target; _ } ~x =
    let pos = Nx.slice [ R (0, 1) ] x
    and vel = Nx.slice [ R (1, 2) ] x in
    let err = Nx.sub pos target in
    Nx.mul_s (Nx.sum (Nx.add (Nx.mul err err) (Nx.mul vel vel))) 5.0

  let problem : (Nx.float64_elt, Nx.float64_t Params.t) Dilqr.Problem.t =
    { dtype = Nx.float64
    ; n
    ; m
    ; horizon
    ; verbose
    ; params = Nx.Ptree.instantiate (module Params)
    ; dynamics
    ; running_loss
    ; final_loss
    }
end

let scalar64 v = Nx.scalar Nx.float64 v
let b = 4
let targets = [| 1.0; -0.6; 0.3; 1.5 |]

let theta : Nx.float64_t Integrator.Params.t =
  { mass = Nx.broadcast_to [| b |] (scalar64 1.0)
  ; target = Nx.create Nx.float64 [| b; 1 |] targets
  }

let x0 = Nx.create Nx.float64 [| b; 2 |] [| 1.0; 0.0; 1.0; 0.0; 1.0; 0.0; 1.0; 0.0 |]
let us0 = Array.init Integrator.horizon ~f:(fun _ -> Nx.zeros Nx.float64 [| b; 1 |])

let inputs : (Nx.float64_t Integrator.Params.t, Nx.float64_t) Dilqr.Inputs.t =
  { Dilqr.Inputs.theta; x0; us0 }

let inputs_ptree = Dilqr.inputs Integrator.problem

let () =
  printf "── batched solve (b = %d) ──\n%!" b;
  let sol = Dilqr.Solver.solve_fixed Integrator.problem ~iters:8 theta x0 us0 in
  let taus = Dilqr.taus_of Integrator.problem sol.xs sol.us in
  let xs, _ = Dilqr.split_taus Integrator.problem taus in
  let xf = xs.(Integrator.horizon) in
  for i = 0 to b - 1 do
    printf
      "member %d: target %6.3f  final position %8.4f  cost %8.4f\n%!"
      i
      targets.(i)
      (Nx.item [ i; 0 ] xf)
      (Nx.item [ i ] sol.cost)
  done;
  (* the gradient of the summed costs w.r.t. every member's target, compiled *)
  let total (i : (Nx.float64_t Integrator.Params.t, Nx.float64_t) Dilqr.Inputs.t) =
    Nx.sum
      (Dilqr.loss_of_taus
         Integrator.problem
         i.Dilqr.Inputs.theta
         (Dilqr.solve_ilqr_fixed Integrator.problem ~iters:8 i))
  in
  let grads =
    Rune.jit
      Nx.Ptree.(inputs_ptree @-> returns inputs_ptree)
      (fun i -> Rune.grad inputs_ptree total i)
      inputs
  in
  printf "compiled gradient of the total cost w.r.t. the targets:\n%!";
  for i = 0 to b - 1 do
    printf
      "  target %6.3f -> %10.4f\n%!"
      targets.(i)
      (Nx.item [ i; 0 ] grads.theta.target)
  done
