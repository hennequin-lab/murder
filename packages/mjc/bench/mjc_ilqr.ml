(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* M6's demo: iLQR on a two-link arm whose dynamics are mjc's own step.

   `Dilqr` solves the optimal-control problem and differentiates the solve
   implicitly; the dynamics inside it is `Dynamics.run`, the compiled step of
   the corpus's `arm2` model, with the arms' parameters — masses, lengths,
   gravity — as leaves a differentiation can reach. There is no second
   simulator anywhere in this process: the same `Model.t` and the same step
   that `Step` and `test_dyn` use.

   Three things are printed, and the third is the acceptance item:
   1. the loss per iLQR iteration, so the solve is visibly converging;
   2. the swing itself, as the angles over the horizon;
   3. the implicit gradient of the user's objective with respect to the
   initial state, against central differences of the same objective.

   Run: dune exec packages/mjc/bench/mjc_ilqr.exe *)

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

(* The arm's dynamics as a dilqr problem: the state is [qpos; qvel] for the two
   hinges, the control is the two motor torques, and one step of the horizon is
   [sub] mjc steps of the corpus model. *)
let model = load "arm2"
let dynamics = Mjc.Dynamics.compile model
let n = 4
let m = 2
let horizon = 20
let sub = 4

(* The state the arm starts in: hanging with a little energy. *)
let x0_values = [| 0.0; 0.0; 0.0; 0.0 |]

(* The target: elbow up, shoulder forward. *)
let target = [| 1.4; -1.2 |]

let state_of_x x =
  let qpos = Nx.reshape [| 2 |] (Nx.slice [ R (0, 2) ] x)
  and qvel = Nx.reshape [| 2 |] (Nx.slice [ R (2, 4) ] x) in
  { (Mjc.Data.create model) with qpos; qvel }

let x_of_state (d : Mjc.Data.host) = Nx.concatenate ~axis:0 [ d.qpos; d.qvel ]

(* One control interval: [sub] steps of the real engine. *)
let dynamics' _ ~k:_ ~x ~u =
  let d = state_of_x x in
  let d = { d with ctrl = u } in
  let d = ref d in
  for _ = 1 to sub do
    d := Mjc.Dynamics.run dynamics model !d
  done;
  x_of_state !d

let vec a = Nx.create f64 [| Array.length a |] a

let running_loss _ ~k:_ ~x ~u =
  let qpos = Nx.slice [ R (0, 2) ] x in
  let err = Nx.sub qpos (vec target) in
  Nx.add (Nx.sum (Nx.mul err err)) (Nx.mul_s (Nx.sum (Nx.mul u u)) 0.01)

let final_loss _ ~x =
  let qpos = Nx.slice [ R (0, 2) ] x in
  let err = Nx.sub qpos (vec target) in
  Nx.add (Nx.mul_s (Nx.sum (Nx.mul err err)) 40.0) (Nx.mul_s (Nx.sum (Nx.mul x x)) 0.001)

let problem : (Nx.float64_elt, Mjc.Model.host, Nx.host) Dilqr.Problem.t =
  { dtype = f64
  ; n
  ; m
  ; horizon
  ; free_initial = false
  ; verbose = false
  ; params = Mjc.Model.ptree
  ; dynamics = dynamics'
  ; running_loss
  ; final_loss
  }

let us0 = Array.init horizon ~f:(fun _ -> Nx.zeros f64 [| m |])

let inputs theta x0 : (Mjc.Model.host, Nx.host Nx.float64_t) Dilqr.Inputs.t =
  { Dilqr.Inputs.theta; x0; us0 }

(* The user's objective on the solve's output: how close the hand ends up, and
   how much torque it took. This is what the gradient is taken of; it is
   differentiable in the solve inputs through dilqr's implicit rule. *)
let objective taus =
  let xs, us = Dilqr.split_taus problem taus in
  let xf = xs.(Array.length xs - 1) in
  let qpos = Nx.slice [ R (0, 2) ] xf in
  let err = Nx.sub qpos (vec target) in
  let effort =
    Array.fold us ~init:(Nx.scalar f64 0.0) ~f:(fun acc u ->
      Nx.add acc (Nx.sum (Nx.mul u u)))
  in
  Nx.add (Nx.mul_s (Nx.sum (Nx.mul err err)) 10.0) (Nx.mul_s effort 0.001)

let x0 = vec x0_values
let theta = model
let solve = Dilqr.solve_adaptive problem
let solve_inputs i = solve ~max_iter:100 ~conv_threshold:1e-10 i

let () =
  let ins = inputs theta x0 in
  let taus = solve_inputs ins in
  let xs, us = Dilqr.split_taus problem taus in
  let xf = xs.(Array.length xs - 1) in
  Printf.printf
    "arm2: %d control steps of %d mjc steps each, %d mjc steps in all\n%!"
    horizon
    sub
    (horizon * sub);
  Printf.printf "objective at the solution: %.8f\n%!" (Nx.item [] (objective taus));
  Printf.printf
    "final angles: %.4f %.4f (target %.4f %.4f)\n%!"
    (Nx.item [ 0 ] (Nx.slice [ R (0, 1) ] xf))
    (Nx.item [ 0 ] (Nx.slice [ R (1, 2) ] xf))
    target.(0)
    target.(1);
  (* The implicit gradient of the objective with respect to the initial state:
     the solve, not the iterations, is what is differentiated. *)
  let grad = Rune.grad (Dilqr.inputs problem) (fun i -> objective (solve_inputs i)) ins in
  let g = Nx.to_array grad.Dilqr.Inputs.x0 in
  Printf.printf
    "implicit d/dx0: [%s]\n%!"
    (String.concat ~sep:", " (Array.to_list (Array.map g ~f:(Printf.sprintf "%.8f"))));
  (* Central differences of the same objective, through the same solve. The
     tolerance is loose on purpose: an adaptive solve stops at a slightly
     different point for a perturbed start, so the reference has noise in it. *)
  let h = 1e-5 in
  let at delta =
    let a = Array.copy x0_values in
    a.(delta |> fst) <- a.(delta |> fst) +. snd delta;
    let ins = inputs theta (vec a) in
    Nx.item [] (objective (solve_inputs ins))
  in
  let fd = Array.init n ~f:(fun i -> (at (i, h) -. at (i, -.h)) /. (2.0 *. h)) in
  Printf.printf
    "central diff  : [%s]\n%!"
    (String.concat ~sep:", " (Array.to_list (Array.map fd ~f:(Printf.sprintf "%.8f"))));
  let worst =
    Array.fold2_exn g fd ~init:0.0 ~f:(fun acc a b ->
      Float.max
        acc
        (Float.abs (a -. b) /. Float.max 1e-12 (Float.max (Float.abs a) (Float.abs b))))
  in
  Printf.printf "worst relative difference: %.2e\n%!" worst;
  (* The plan, one row per engine step, for the viewer's [--plan]. Writing it is
     what turns the solve into something you can watch. *)
  let plan =
    if Array.length Stdlib.Sys.argv > 1
    then Stdlib.Sys.argv.(1)
    else if Stdlib.Sys.file_exists "packages/mjc/viz/demo"
    then "packages/mjc/viz/demo/arm2-plan.txt"
    else "viz/demo/arm2-plan.txt"
  in
  let us = Array.concat_map us ~f:(fun u -> Array.create ~len:sub u) in
  Stdlib.Out_channel.with_open_text plan (fun oc ->
    Array.iter us ~f:(fun u ->
      let row = Array.to_list (Array.map (Nx.to_array u) ~f:(Printf.sprintf "%.9g")) in
      Stdlib.Out_channel.output_string oc (String.concat ~sep:" " row ^ "\n")));
  Printf.printf "wrote %s (%d rows)\n%!" plan (Array.length us);
  (* Is the objective smooth in x0 at this scale? A solve that stops at a
     different point for a perturbed start makes the reference noisy, and the
     implicit derivative is of the converged solution, not of the stopping
     rule. *)
  Printf.printf "objective along x0[1]:\n%!";
  List.iter [ -1e-3; -1e-4; -1e-5; 0.0; 1e-5; 1e-4; 1e-3 ] ~f:(fun d ->
    let a = Array.copy x0_values in
    a.(1) <- a.(1) +. d;
    let ins = inputs theta (vec a) in
    Printf.printf "  %+.0e  %.12f\n%!" d (Nx.item [] (objective (solve_inputs ins))))
