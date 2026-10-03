(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Double-integrator MPC: repeatedly re-solve a short-horizon iLQR from the
   current state, compiled end to end with Rune.jit (solve + implicit
   gradient in one program, CPU device by default; pass a device name to move
   it to GPU). *)

open Base
open Stdio
module Nx = Nx

let dt = 0.1

module Integrator = struct
  module Ctx = struct
    type 'a params = { mass : 'a } [@@deriving ptree]

    module Params = struct
      type 'a t = 'a params

      let walk = walk_params
    end

    type w = Nx.float64_elt
    type value = (float, w) Nx.t

    let dtype = Nx.float64
    let n = 2
    let m = 1
    let horizon = 20
    let verbose = true

    let dynamics { mass } ~k:_ ~x ~u =
      let pos = Nx.slice [ R (0, 1) ] x
      and vel = Nx.slice [ R (1, 2) ] x in
      let force = Nx.div u mass in
      Nx.concatenate
        ~axis:0
        [ Nx.add pos (Nx.mul_s vel dt); Nx.add vel (Nx.mul_s force dt) ]

    let running_loss _ ~k:_ ~x ~u =
      let pos = Nx.slice [ R (0, 1) ] x
      and vel = Nx.slice [ R (1, 2) ] x in
      Nx.sum
        (Nx.add
           (Nx.mul_s (Nx.add (Nx.mul pos pos) (Nx.mul vel vel)) 0.5)
           (Nx.mul_s (Nx.mul u u) 0.05))

    let final_loss _ ~x =
      let pos = Nx.slice [ R (0, 1) ] x
      and vel = Nx.slice [ R (1, 2) ] x in
      Nx.mul_s (Nx.sum (Nx.add (Nx.mul pos pos) (Nx.mul vel vel))) 5.0
  end

  module D = Dilqr.Make (Ctx)
end

let vec64 a = Nx.create Nx.float64 [| Array.length a |] a
let scalar64 v = Nx.scalar Nx.float64 v
let item t = Nx.item [] t

(* The compiled solve step: given (theta, x0, warm-start controls), return
   the stacked trajectory. *)
let solve_step =
  Rune.jit
    Nx.Ptree.(Integrator.D.inputs_ptree @-> returns Nx.Ptree.tensor)
    (fun i -> Integrator.D.solve_ilqr_fixed ~iters:4 ~mu:1.0 i)

(* The compiled gradient step: solve + implicit VJP in one program. *)
let grad_step =
  Rune.jit
    Nx.Ptree.(Integrator.D.inputs_ptree @-> returns Integrator.D.inputs_ptree)
    (fun i ->
       Rune.grad
         Integrator.D.inputs_ptree
         (fun i ->
            Integrator.D.loss_of_taus
              i.theta
              (Integrator.D.solve_ilqr_fixed ~iters:4 ~mu:1.0 i)
            |> Nx.sum)
         i)

let () =
  printf "── double-integrator MPC ──\n%!";
  let theta : Integrator.Ctx.value Integrator.Ctx.Params.t = { mass = scalar64 1.0 } in
  let x0 = ref (vec64 [| 1.0; 0.0 |]) in
  let us =
    Array.init Integrator.Ctx.horizon ~f:(fun _ -> Nx.create Nx.float64 [| 1 |] [| 0.0 |])
  in
  (* warm-start the receding-horizon solves from the previous solution *)
  let rec mpc step =
    if step = 0
    then ()
    else (
      let inputs : Integrator.D.value Integrator.D.Inputs.t =
        { Integrator.D.Inputs.theta; x0 = !x0; us0 = us }
      in
      let taus = solve_step inputs in
      let xs, us_new = Integrator.D.split_taus taus in
      (* apply the first control, observe the true next state *)
      let x1 = Integrator.Ctx.dynamics theta ~k:0 ~x:!x0 ~u:us_new.(0) in
      printf
        "step %2d: pos = %8.4f  vel = %8.4f  cost = %8.4f\n%!"
        step
        (Nx.item [ 0 ] !x0)
        (Nx.item [ 1 ] !x0)
        (item (Integrator.D.loss_of_taus theta taus |> Nx.sum));
      x0 := x1;
      (* shift the warm start *)
      for k = 0 to Integrator.Ctx.horizon - 2 do
        us.(k) <- us_new.(k + 1)
      done;
      us.(Integrator.Ctx.horizon - 1) <- us_new.(Integrator.Ctx.horizon - 1);
      mpc (step - 1))
  in
  mpc 8;
  (* gradient of the accumulated cost w.r.t. the mass parameter, compiled *)
  let inputs : Integrator.D.value Integrator.D.Inputs.t =
    { Integrator.D.Inputs.theta; x0 = !x0; us0 = us }
  in
  let grads = grad_step inputs in
  printf
    "compiled implicit grad w.r.t. x0: [%.4g, %.4g]\n%!"
    (Nx.item [ 0 ] grads.x0)
    (Nx.item [ 1 ] grads.x0);
  printf "compiled implicit grad w.r.t. mass: %.4g\n%!" (item grads.theta.mass)
