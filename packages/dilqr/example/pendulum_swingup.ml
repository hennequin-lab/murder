(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Pendulum swing-up with differentiable iLQR.

   The problem: drive a torque-limited-free pendulum from below-horizontal to
   upright, minimizing control effort and upright-tracking error. The example
   demonstrates:

   1. an adaptive iLQR solve (host-side control flow, eager);
   2. a traceable fixed-iteration solve, compiled with Rune.jit;
   3. the implicit gradient of the solve, computed eagerly and compiled
   (Rune.grad inside Rune.jit compiles the forward solve and the implicit
   VJP together). *)

open Base
open Stdio
module Nx = Nx

let dt = 0.1

(* Pendulum: x = [angle; angular velocity], control = torque, damping 0.1.
   Swing UP from hanging (angle 0) to upright (angle π). *)
module Pendulum = struct
  module Params = struct
    type 'a t =
      { gravity : 'a
      ; mass : 'a
      ; length : 'a
      }
    [@@deriving ptree]
  end

  let n = 2
  let m = 1
  let horizon = 40
  let verbose = true
  let target = 3.14159265358979

  let dynamics { Params.gravity; mass; length } ~k:_ ~x ~u =
    let theta = Nx.slice [ R (0, 1) ] x
    and omega = Nx.slice [ R (1, 2) ] x in
    let ml2 = Nx.mul mass (Nx.mul length length) in
    let grav_torque = Nx.mul (Nx.mul gravity (Nx.mul mass length)) (Nx.sin theta) in
    let acc = Nx.div (Nx.sub u (Nx.add grav_torque (Nx.mul_s omega 0.1))) ml2 in
    Nx.concatenate
      ~axis:0
      [ Nx.add theta (Nx.mul_s omega dt); Nx.add omega (Nx.mul_s acc dt) ]

  let running_loss _ ~k:_ ~x ~u =
    let theta = Nx.slice [ R (0, 1) ] x
    and omega = Nx.slice [ R (1, 2) ] x in
    let dtheta = Nx.sub_s theta target in
    Nx.sum
      (Nx.add
         (Nx.mul_s (Nx.add (Nx.mul dtheta dtheta) (Nx.mul omega omega)) 0.5)
         (Nx.mul_s (Nx.mul u u) 0.01))

  let final_loss _ ~x =
    let theta = Nx.slice [ R (0, 1) ] x
    and omega = Nx.slice [ R (1, 2) ] x in
    let dtheta = Nx.sub_s theta target in
    Nx.mul_s (Nx.sum (Nx.add (Nx.mul dtheta dtheta) (Nx.mul omega omega))) 5.0

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

let vec64 a = Nx.create Nx.float64 [| Array.length a |] a
let scalar64 v = Nx.scalar Nx.float64 v
let item t = Nx.item [] t

(* The user loss on the solve output: control effort + upright-tracking of
   the final state. Differentiable in (theta, x0, us0) through the implicit
   gradient. *)
let loss ~target problem taus =
  let xs, us = Dilqr.split_taus problem taus in
  let xf = xs.(Array.length xs - 1) in
  let ctrl =
    Array.fold us ~init:(scalar64 0.0) ~f:(fun a u ->
      Nx.add a (Nx.mul_s (Nx.sum (Nx.mul u u)) 0.5))
  in
  let err = Nx.sub_s (Nx.slice [ R (0, 1) ] xf) target in
  Nx.add ctrl (Nx.mul_s (Nx.sum (Nx.mul err err)) 1.0)

let theta : Nx.float64_t Pendulum.Params.t =
  { gravity = scalar64 9.81; mass = scalar64 1.0; length = scalar64 1.0 }

let x0 = vec64 [| 0.0; 0.0 |]
let us0 = Array.init Pendulum.horizon ~f:(fun _ -> Nx.create Nx.float64 [| 1 |] [| 0.0 |])

let inputs : (Nx.float64_t Pendulum.Params.t, Nx.float64_t) Dilqr.Inputs.t =
  { Dilqr.Inputs.theta; x0; us0 }

let inputs_ptree = Dilqr.inputs Pendulum.problem

(* ──────────────────────────────────────────────────────────────────────────
   1. Adaptive solve (eager): swing the pendulum up.
   ────────────────────────────────────────────────────────────────────────── *)

let solve_adaptive () =
  printf "── adaptive solve ──\n";
  let taus =
    Dilqr.solve_ilqr Pendulum.problem ~max_iter:100 ~conv_threshold:1e-9 inputs
  in
  let xs, us = Dilqr.split_taus Pendulum.problem taus in
  let xf = xs.(Array.length xs - 1) in
  printf
    "final state: theta = %g (target %g), omega = %g\n"
    (Nx.item [ 0 ] (Nx.slice [ R (0, 1) ] xf))
    Pendulum.target
    (Nx.item [ 0 ] (Nx.slice [ R (1, 2) ] xf));
  printf
    "first few controls: %g %g %g %g\n"
    (Nx.item [ 0 ] us.(0))
    (Nx.item [ 0 ] us.(1))
    (Nx.item [ 0 ] us.(2))
    (Nx.item [ 0 ] us.(3));
  printf
    "loss on the solve output: %g\n"
    (item (loss ~target:Pendulum.target Pendulum.problem taus));
  (* the implicit gradient of the user loss w.r.t. the solve inputs *)
  let grads =
    Rune.grad
      inputs_ptree
      (fun i ->
         loss
           ~target:Pendulum.target
           Pendulum.problem
           (Dilqr.solve_ilqr Pendulum.problem ~max_iter:100 ~conv_threshold:1e-9 i))
      inputs
  in
  printf
    "implicit grad w.r.t. x0: [%.4g, %.4g]\n"
    (Nx.item [ 0 ] grads.x0)
    (Nx.item [ 1 ] grads.x0);
  printf "implicit grad w.r.t. gravity: %.4g\n" (item grads.theta.gravity);
  grads

(* ──────────────────────────────────────────────────────────────────────────
   2. Compiled solve and compiled gradient.
   ────────────────────────────────────────────────────────────────────────── *)

(* The gradient of the fixed-iteration solve, compiled as one program: the
   forward solve and its implicit VJP compile together. Every horizon is a
   loop ([Rune.scan]), so a program costs its iterations, not its iterations
   times its horizon. *)
let compiled_grad ~iters =
  Rune.jit
    Nx.Ptree.(inputs_ptree @-> returns inputs_ptree)
    (fun i ->
       Rune.grad
         inputs_ptree
         (fun i ->
            loss
              ~target:Pendulum.target
              Pendulum.problem
              (Dilqr.solve_ilqr_fixed Pendulum.problem ~iters i))
         i)
    inputs

let solve_compiled adaptive =
  printf "\n── compiled (jit) solve and gradient ──\n";
  (* A fixed-iteration solve approximates the adaptive one, and its implicit
     gradient converges to the adaptive solve's as the solve converges. *)
  let show name (g : (Nx.float64_t Pendulum.Params.t, Nx.float64_t) Dilqr.Inputs.t) =
    printf
      "%-18s x0 = [%9.4g, %9.4g]  gravity = %9.4g\n%!"
      name
      (Nx.item [ 0 ] g.x0)
      (Nx.item [ 1 ] g.x0)
      (item g.theta.gravity)
  in
  List.iter [ 2; 4; 6; 10; 20 ] ~f:(fun iters ->
    show (Printf.sprintf "compiled iters %d" iters) (compiled_grad ~iters));
  show "adaptive" adaptive;
  (* The jitted solve itself, at a converged iteration count. *)
  let taus =
    Rune.jit
      Nx.Ptree.(inputs_ptree @-> returns Nx.Ptree.tensor)
      (fun i -> Dilqr.solve_ilqr_fixed Pendulum.problem ~iters:10 i)
      inputs
  in
  let xs, _ = Dilqr.split_taus Pendulum.problem taus in
  printf
    "jitted solve (10 iterations) final theta: %g\n%!"
    (Nx.item [ 0 ] (Nx.slice [ R (0, 1) ] xs.(Array.length xs - 1)));
  (* sanity: the compiled gradient matches the eager one for the same map *)
  let eager =
    Rune.grad
      inputs_ptree
      (fun i ->
         loss
           ~target:Pendulum.target
           Pendulum.problem
           (Dilqr.solve_ilqr_fixed Pendulum.problem ~iters:10 i))
      inputs
  in
  show "eager iters 10" eager

let () =
  let adaptive = solve_adaptive () in
  solve_compiled adaptive
