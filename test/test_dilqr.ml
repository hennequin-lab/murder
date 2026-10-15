(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Dilqr test suite:
   - unit checks on the regularization schedule,
   - LQR correctness (double integrator: cost decreases, control
     stationarity of the solution, checked with independent autodiff),
   - nonlinear convergence (pendulum regulation to upright),
   - implicit gradients vs. central finite differences (float64),
   - the fixed (traceable) solve is competitive with the adaptive one,
   - JIT round-trips: compiled solve and compiled grad match eager results. *)

open Base
open Windtrap
open Stdio
module Nx = Nx

let dt = 0.1
let require ~msg b = if not b then failwith ("assert failed: " ^ msg)

let check_close ~msg ~tol got expected =
  let g = Nx.to_array got
  and e = Nx.to_array expected in
  require ~msg (Array.length g = Array.length e);
  Array.iteri g ~f:(fun i a ->
    let b = e.(i) in
    if not Float.(Float.abs (a -. b) <= tol +. (tol *. Float.abs b))
    then
      Stdio.printf
        "MISMATCH %s: got %.8g expected %.8g (diff %.3g, tol %g)\n%!"
        msg
        a
        b
        (a -. b)
        tol;
    require ~msg Float.(Float.abs (a -. b) <= tol +. (tol *. Float.abs b)))

(* Central finite difference of a scalar function of a [k]-dim float64
   vector. *)
let fd_grad ~eps f (x : Nx.float64_t) : Nx.float64_t =
  let n = (Nx.shape x).(0) in
  let out = Array.create ~len:n 0.0 in
  for i = 0 to n - 1 do
    let xi = Nx.item [ i ] x in
    let xp = Nx.set [ I i ] (Nx.scalar Nx.float64 (xi +. eps)) x in
    let xm = Nx.set [ I i ] (Nx.scalar Nx.float64 (xi -. eps)) x in
    out.(i) <- (f xp -. f xm) /. (2.0 *. eps)
  done;
  Nx.create Nx.float64 [| n |] out

let vec64 a = Nx.create Nx.float64 [| Array.length a |] a
let scalar64 v = Nx.scalar Nx.float64 v
let item t = Nx.item [] t

let us_of (vs : float list list) : Nx.float64_t array =
  Array.of_list (List.map ~f:(fun l -> vec64 (Array.of_list l)) vs)

(* ──────────────────────────────────────────────────────────────────────────
   Problems. Both are float64 so gradient checks against finite differences
   are meaningful.
   ────────────────────────────────────────────────────────────────────────── *)

(* Double integrator: x = [position; velocity], control = force. *)
module Integrator = struct
  module Params = struct
    type 'a t = { mass : 'a } [@@deriving ptree]
  end

  let n = 2
  let m = 1
  let horizon = 20
  let verbose = false

  let dynamics { Params.mass } ~k:_ ~x ~u =
    let pos = Nx.slice [ R (0, 1) ] x
    and vel = Nx.slice [ R (1, 2) ] x in
    let force = Nx.div u mass in
    Nx.concatenate
      ~axis:0
      [ Nx.add pos (Nx.mul_s vel dt); Nx.add vel (Nx.mul_s force dt) ]

  (* The state cost ramps with the step: [k] reaches the loss as data, so
     the scans carry time with their rows. *)
  let running_loss _ ~k ~x ~u =
    let pos = Nx.slice [ R (0, 1) ] x
    and vel = Nx.slice [ R (1, 2) ] x in
    let weight = Nx.add_s (Nx.mul_s k 0.05) 1.0 in
    Nx.sum
      (Nx.add
         (Nx.mul_s (Nx.mul (Nx.add (Nx.mul pos pos) (Nx.mul vel vel)) weight) 0.5)
         (Nx.mul_s (Nx.mul u u) 0.05))

  let final_loss _ ~x =
    let pos = Nx.slice [ R (0, 1) ] x
    and vel = Nx.slice [ R (1, 2) ] x in
    Nx.mul_s (Nx.sum (Nx.add (Nx.mul pos pos) (Nx.mul vel vel))) 5.0

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

(* Pendulum: x = [angle; angular velocity], control = torque, damping
   coefficient 0.1. Regulate the upright position (angle = π). *)
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
  let horizon = 30
  let verbose = false
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

(* Batched double integrator: one mass per member and one target per member.
   The dynamics and losses are written for a single member — the solver vmaps
   them over the leading batch axis — so the same function serves the batched
   and the unbatched solve. *)
module Batched = struct
  module Params = struct
    type 'a t =
      { mass : 'a
      ; target : 'a
      }
    [@@deriving ptree]
  end

  let n = 2
  let m = 1
  let horizon = 12
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

(* ──────────────────────────────────────────────────────────────────────────
   Tests
   ────────────────────────────────────────────────────────────────────────── *)

let test_regularisation () =
  let open Dilqr.Regularisation in
  let r = increase init in
  equal ~msg:"delta grows" (float 1e-12) 2.0 r.delta;
  equal ~msg:"mu grows" (float 1e-12) 2.0 r.mu;
  let r = decrease r in
  equal ~msg:"delta shrinks" (float 1e-12) 0.5 r.delta;
  equal ~msg:"mu decays" (float 1e-12) 1.0 r.mu;
  let rec decay r n = if n = 0 then r else decay (decrease r) (n - 1) in
  let r = decay (increase init) 40 in
  equal ~msg:"mu floors at 1e-6" (float 1e-15) 1e-6 r.mu

(* The adaptive solve detects a failed factorization by the NaNs it leaves:
   the Cholesky factor of a non-positive-definite [Quu] has every element NaN,
   and the backward sweep carries that into the expected-decrease
   accumulators the retry tests. *)
let test_a_failed_factorization_is_nan () =
  let open Dilqr.Lqr in
  let one s = Nx.reshape s (Nx.scalar Nx.float64 1.0)
  and zero s = Nx.zeros Nx.float64 s in
  let blocks =
    Rows.
      { a = one [| 1; 1; 1 |]
      ; b = one [| 1; 1; 1 |]
      ; lx = zero [| 1; 1 |]
      ; lu = zero [| 1; 1 |]
      ; lxx = zero [| 1; 1; 1 |]
      ; luu = Nx.reshape [| 1; 1; 1 |] (Nx.scalar Nx.float64 (-1.0))
      ; lux = zero [| 1; 1; 1 |]
      }
  in
  let tape = { blocks; fx = zero [| 1 |]; fxx = zero [| 1; 1 |] } in
  let finite d1 d2 =
    Nx.item [] (Nx.all (Nx.isfinite d1)) && Nx.item [] (Nx.all (Nx.isfinite d2))
  in
  let _, d1, d2 = backward ~mu:0.0 ~tape ~lin:(lin_of_tape tape) in
  require
    ~msg:"an indefinite Quu has NaN expected-decrease accumulators"
    (not (finite d1 d2));
  let _, d1, d2 = backward ~mu:2.0 ~tape ~lin:(lin_of_tape tape) in
  require ~msg:"regularization makes Quu positive definite" (finite d1 d2)

(* Stationarity of the double-integrator solution: the autodiff gradient of
   the total cost w.r.t. each control vanishes. Autodiff through the true
   rollout — independent of the Riccati code under test. *)
let test_integrator_converges () =
  let module P = Integrator in
  let theta : Nx.float64_t P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let cost_of_us us = snd (Dilqr.Solver.rollout P.problem theta x0 us) in
  let cost0 = item (cost_of_us us0) in
  let sol = Dilqr.Solver.solve P.problem theta x0 us0 in
  let cost1 = item sol.cost in
  require ~msg:"solve reduces cost" Float.(cost1 < 0.5 *. cost0);
  let worst = ref 0.0 in
  Array.iteri sol.us ~f:(fun k _ ->
    let g =
      Rune.grad'
        (fun uk ->
           let us' = Array.copy sol.us in
           us'.(k) <- uk;
           cost_of_us us')
        sol.us.(k)
    in
    worst := Float.max !worst (Float.abs (Nx.item [ 0 ] g)));
  printf "worst stationarity: %g\n" !worst;
  require ~msg:"control stationarity" Float.(!worst < 5e-4)

let test_pendulum_converges () =
  let module P = Pendulum in
  let theta : Nx.float64_t P.Params.t =
    { gravity = scalar64 9.81; mass = scalar64 1.0; length = scalar64 1.0 }
  in
  let x0 = vec64 [| P.target -. 0.3; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let cost0 = item (snd (Dilqr.Solver.rollout P.problem theta x0 us0)) in
  let sol = Dilqr.Solver.solve P.problem ~max_iter:80 theta x0 us0 in
  let cost1 = item sol.cost in
  require ~msg:"pendulum reduces cost" Float.(cost1 < 0.5 *. cost0);
  let xf = sol.xs.(Array.length sol.xs - 1) in
  let err = Float.abs (Nx.item [ 0 ] (Nx.slice [ R (0, 1) ] xf) -. P.target) in
  require ~msg:"pendulum reaches upright" Float.(err < 0.1)

(* Implicit gradient vs. central finite differences, w.r.t. x0 and theta, on
   the pendulum with a tightly converged solve. *)
let test_implicit_gradient_matches_fd () =
  let module P = Pendulum in
  let theta : Nx.float64_t P.Params.t =
    { gravity = scalar64 9.81; mass = scalar64 1.3; length = scalar64 0.9 }
  in
  let x0 = vec64 [| P.target -. 0.4; 0.1 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.02 ])) in
  let inputs : (Nx.float64_t P.Params.t, Nx.float64_t) Dilqr.Inputs.t =
    { Dilqr.Inputs.theta; x0; us0 }
  in
  (* A user loss on the solve output, different from the problem's own cost. *)
  let loss taus =
    let xs, us = Dilqr.split_taus P.problem taus in
    let xf = xs.(Array.length xs - 1) in
    let ctrl =
      Array.fold us ~init:(scalar64 0.0) ~f:(fun a u ->
        Nx.add a (Nx.mul_s (Nx.sum (Nx.mul u u)) 0.5))
    in
    let err = Nx.sub_s (Nx.slice [ R (0, 1) ] xf) P.target in
    Nx.add ctrl (Nx.mul_s (Nx.sum (Nx.mul err err)) 1.0)
  in
  let value inputs =
    item (loss (Dilqr.solve_ilqr P.problem ~max_iter:100 ~conv_threshold:1e-10 inputs))
  in
  let grads =
    Rune.grad
      (Dilqr.inputs P.problem)
      (fun i -> loss (Dilqr.solve_ilqr P.problem ~max_iter:100 ~conv_threshold:1e-10 i))
      inputs
  in
  (* x0 gradient vs FD *)
  let eps = 1e-5 in
  let fd_x0 = fd_grad ~eps (fun x0 -> value { inputs with x0 }) x0 in
  check_close ~msg:"grad x0 (pendulum)" ~tol:2e-3 grads.x0 fd_x0;
  (* theta gradients vs FD, one scalar at a time: the FD input vector starts
     at the parameter's actual value and [f] offsets it. *)
  let fd_theta base f =
    fd_grad
      ~eps
      (fun v -> value { inputs with theta = f (base +. item (Nx.slice [ I 0 ] v)) })
      (vec64 [| 0.0 |])
  in
  check_close
    ~msg:"grad gravity"
    ~tol:2e-3
    grads.theta.gravity
    (fd_theta (item theta.gravity) (fun g ->
       { gravity = scalar64 g; mass = theta.mass; length = theta.length }));
  check_close
    ~msg:"grad mass"
    ~tol:2e-3
    grads.theta.mass
    (fd_theta (item theta.mass) (fun mm ->
       { gravity = theta.gravity; mass = scalar64 mm; length = theta.length }));
  check_close
    ~msg:"grad length"
    ~tol:2e-3
    grads.theta.length
    (fd_theta (item theta.length) (fun l ->
       { gravity = theta.gravity; mass = theta.mass; length = scalar64 l }));
  (* us0 gradient is zero by construction *)
  Array.iter grads.us0 ~f:(fun u ->
    require ~msg:"grad us0 is zero" Float.(Float.abs (Nx.item [ 0 ] u) < 1e-12))

(* The fixed (traceable) solve reaches costs competitive with the adaptive
   one. *)
let test_fixed_solve_works () =
  let module P = Integrator in
  let theta : Nx.float64_t P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let sol = Dilqr.Solver.solve_fixed P.problem ~iters:8 theta x0 us0 in
  let adaptive = Dilqr.Solver.solve P.problem ~max_iter:20 theta x0 us0 in
  let c_fixed = item sol.cost
  and c_adaptive = item adaptive.cost in
  require ~msg:"fixed solve competitive" Float.(c_fixed < 1.05 *. c_adaptive)

(* When every line-search trial is rejected — here because the trial steps
   are NaN — the fixed solve must keep the current trajectory. The argmin of
   an all-infinity score vector returns 0, so the solver must not select that
   fallback trial. *)
let test_fixed_all_rejected () =
  let module P = Integrator in
  let theta : Nx.float64_t P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let sol =
    Dilqr.Solver.solve_fixed P.problem ~alphas:[ Float.nan ] ~iters:2 theta x0 us0
  in
  require ~msg:"all-rejected cost stays finite" (Float.is_finite (item sol.cost));
  Array.iter sol.us ~f:(fun u ->
    require
      ~msg:"all-rejected keeps the current controls"
      (Float.is_finite (Nx.item [ 0 ] u) && Float.( = ) (Nx.item [ 0 ] u) 0.0))

(* JIT: the compiled fixed solve equals the eager one. *)
let test_jit_solve () =
  let module P = Integrator in
  let theta : Nx.float64_t P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let inputs : (Nx.float64_t P.Params.t, Nx.float64_t) Dilqr.Inputs.t =
    { Dilqr.Inputs.theta; x0; us0 }
  in
  let eager = Dilqr.solve_ilqr_fixed P.problem ~iters:5 inputs in
  let jitted =
    Rune.jit
      Nx.Ptree.(Dilqr.inputs P.problem @-> returns Nx.Ptree.tensor)
      (fun i -> Dilqr.solve_ilqr_fixed P.problem ~iters:5 i)
      inputs
  in
  check_close ~msg:"jit solve taus" ~tol:1e-10 jitted eager

(* JIT: the compiled gradient — forward solve and implicit VJP, all traced —
   matches the eager gradient. *)
let test_jit_grad () =
  let module P = Integrator in
  let theta : Nx.float64_t P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let inputs : (Nx.float64_t P.Params.t, Nx.float64_t) Dilqr.Inputs.t =
    { Dilqr.Inputs.theta; x0; us0 }
  in
  let loss taus =
    let xs, us = Dilqr.split_taus P.problem taus in
    let xf = xs.(Array.length xs - 1) in
    Nx.add
      (Nx.sum (Array.fold us ~init:(scalar64 0.0) ~f:(fun a u -> Nx.add a (Nx.mul u u))))
      (Nx.mul_s (Nx.sum (Nx.mul xf xf)) 5.0)
  in
  let eager_grad =
    Rune.grad
      (Dilqr.inputs P.problem)
      (fun i -> loss (Dilqr.solve_ilqr_fixed P.problem ~iters:4 i))
      inputs
  in
  let compiled =
    Rune.jit
      Nx.Ptree.(Dilqr.inputs P.problem @-> returns (Dilqr.inputs P.problem))
      (fun i ->
         Rune.grad
           (Dilqr.inputs P.problem)
           (fun i -> loss (Dilqr.solve_ilqr_fixed P.problem ~iters:4 i))
           i)
      inputs
  in
  check_close ~msg:"jit grad x0" ~tol:1e-9 compiled.x0 eager_grad.x0;
  check_close ~msg:"jit grad mass" ~tol:1e-9 compiled.theta.mass eager_grad.theta.mass

(* Batched inputs: every leaf of the parameters, of the initial state and of
   every initial control carries the leading batch axis. *)
let batched_setup () =
  let module P = Batched in
  let b = 3 in
  let theta : Nx.float64_t P.Params.t =
    { mass = Nx.create Nx.float64 [| b |] [| 1.0; 1.0; 1.0 |]
    ; target = Nx.create Nx.float64 [| b; 1 |] [| 1.0; -0.5; 0.8 |]
    }
  in
  let x0 = Nx.create Nx.float64 [| b; 2 |] [| 1.0; 0.0; 1.0; 0.0; 1.0; 0.0 |] in
  let us0 = Array.init P.horizon ~f:(fun _ -> Nx.zeros Nx.float64 [| b; 1 |]) in
  b, theta, x0, us0

(* Member [i]'s slice of a batched setup. *)
let batched_member (theta : Nx.float64_t Batched.Params.t) x0 us0 i =
  let theta_i : Nx.float64_t Batched.Params.t =
    { mass = Nx.slice [ I i ] theta.Batched.Params.mass
    ; target = Nx.slice [ I i ] theta.Batched.Params.target
    }
  in
  let x0_i = Nx.slice [ I i ] x0 in
  let us0_i = Array.map us0 ~f:(fun u -> Nx.slice [ I i ] u) in
  theta_i, x0_i, us0_i

(* A batched fixed solve gives, member by member, what the unbatched solve
   gives for that member. *)
let test_batched_fixed_solve () =
  let module P = Batched in
  let b, theta, x0, us0 = batched_setup () in
  let sol = Dilqr.Solver.solve_fixed P.problem ~iters:8 theta x0 us0 in
  require ~msg:"batched cost shape" (Array.length (Nx.shape sol.cost) = 1);
  require ~msg:"batched cost size" ((Nx.shape sol.cost).(0) = b);
  (* the stacked trajectory leads with the batch and then the horizon *)
  let taus = Dilqr.taus_of P.problem sol.xs sol.us in
  require
    ~msg:"batched taus shape"
    (Array.equal Int.equal (Nx.shape taus) [| b; P.horizon + 1; P.n + P.m |]);
  let xs, us = Dilqr.split_taus P.problem taus in
  (* the trajectory helpers are batched too *)
  let xs_b, cost_b = Dilqr.Solver.rollout P.problem theta x0 us0 in
  let tape_b : (float, Nx.float64_elt) Dilqr.Lqr.tape =
    Dilqr.Solver.blocks P.problem theta sol.xs sol.us
  in
  for i = 0 to b - 1 do
    let theta_i, x0_i, us0_i = batched_member theta x0 us0 i in
    let sol_i = Dilqr.Solver.solve_fixed P.problem ~iters:8 theta_i x0_i us0_i in
    check_close
      ~msg:(Printf.sprintf "batched cost %d" i)
      ~tol:1e-10
      (Nx.slice [ I i ] sol.cost)
      sol_i.cost;
    Array.iteri sol.xs ~f:(fun k t ->
      check_close
        ~msg:(Printf.sprintf "batched xs %d.%d" i k)
        ~tol:1e-10
        (Nx.slice [ I i ] t)
        sol_i.xs.(k);
      check_close
        ~msg:(Printf.sprintf "batched taus %d.%d" i k)
        ~tol:1e-10
        (Nx.slice [ I i ] xs.(k))
        sol_i.xs.(k));
    Array.iteri sol.us ~f:(fun k t ->
      check_close
        ~msg:(Printf.sprintf "batched us %d.%d" i k)
        ~tol:1e-10
        (Nx.slice [ I i ] t)
        sol_i.us.(k);
      check_close
        ~msg:(Printf.sprintf "batched taus us %d.%d" i k)
        ~tol:1e-10
        (Nx.slice [ I i ] us.(k))
        sol_i.us.(k));
    (* batched rollout and blocks *)
    let xs_i, cost_i = Dilqr.Solver.rollout P.problem theta_i x0_i us0_i in
    let tape_i = Dilqr.Solver.blocks P.problem theta_i sol_i.xs sol_i.us in
    check_close
      ~msg:(Printf.sprintf "batched rollout cost %d" i)
      ~tol:1e-10
      (Nx.slice [ I i ] cost_b)
      cost_i;
    Array.iteri xs_i ~f:(fun k x ->
      check_close
        ~msg:(Printf.sprintf "batched rollout xs %d.%d" i k)
        ~tol:1e-10
        (Nx.slice [ I i ] xs_b.(k))
        x);
    check_close
      ~msg:(Printf.sprintf "batched blocks a %d" i)
      ~tol:1e-10
      (Nx.slice [ I i ] tape_b.Dilqr.Lqr.blocks.a)
      tape_i.Dilqr.Lqr.blocks.a
  done

(* The batched implicit gradient is the member's gradient, and the same
   compiled. *)
let test_batched_grad () =
  let module P = Batched in
  let b, theta, x0, us0 = batched_setup () in
  let inputs : (Nx.float64_t P.Params.t, Nx.float64_t) Dilqr.Inputs.t =
    { Dilqr.Inputs.theta; x0; us0 }
  in
  let objective i =
    Nx.sum
      (Dilqr.loss_of_taus
         P.problem
         i.Dilqr.Inputs.theta
         (Dilqr.solve_ilqr_fixed P.problem ~iters:5 i))
  in
  let g = Rune.grad (Dilqr.inputs P.problem) objective inputs in
  for i = 0 to b - 1 do
    let theta_i, x0_i, us0_i = batched_member theta x0 us0 i in
    let inputs_i : (Nx.float64_t P.Params.t, Nx.float64_t) Dilqr.Inputs.t =
      { Dilqr.Inputs.theta = theta_i; x0 = x0_i; us0 = us0_i }
    in
    let g_i = Rune.grad (Dilqr.inputs P.problem) objective inputs_i in
    check_close
      ~msg:(Printf.sprintf "batched grad target %d" i)
      ~tol:1e-9
      (Nx.slice [ I i ] g.theta.target)
      g_i.theta.target;
    check_close
      ~msg:(Printf.sprintf "batched grad mass %d" i)
      ~tol:1e-9
      (Nx.slice [ I i ] g.theta.mass)
      g_i.theta.mass;
    check_close
      ~msg:(Printf.sprintf "batched grad x0 %d" i)
      ~tol:1e-9
      (Nx.slice [ I i ] g.x0)
      g_i.x0
  done;
  Array.iter g.us0 ~f:(fun u ->
    require ~msg:"batched grad us0 is zero" Float.(Nx.item [] (Nx.max (Nx.abs u)) < 1e-12));
  (* compiled: the batched solve and its implicit VJP in one program *)
  let compiled =
    Rune.jit
      Nx.Ptree.(Dilqr.inputs P.problem @-> returns (Dilqr.inputs P.problem))
      (fun i -> Rune.grad (Dilqr.inputs P.problem) objective i)
      inputs
  in
  check_close ~msg:"batched jit grad x0" ~tol:1e-9 compiled.x0 g.x0;
  check_close
    ~msg:"batched jit grad target"
    ~tol:1e-9
    compiled.theta.target
    g.theta.target;
  check_close ~msg:"batched jit grad mass" ~tol:1e-9 compiled.theta.mass g.theta.mass

(* The adaptive solve has one host-side line search; batches are for the
   fixed solve. *)
let test_adaptive_rejects_batches () =
  let module P = Batched in
  let b, theta, x0, us0 = batched_setup () in
  require ~msg:"batch is not empty" (b > 0);
  match Dilqr.Solver.solve P.problem theta x0 us0 with
  | exception Failure msg ->
    require
      ~msg:"adaptive solve names batches"
      (String.is_substring msg ~substring:"batched")
  | _ -> failwith "the adaptive solve accepted a batch"

let () =
  Stdlib.exit
    (run
       "dilqr"
       [ test "regularisation schedule" test_regularisation
       ; test "a failed factorization is NaN" test_a_failed_factorization_is_nan
       ; test "double integrator converges" test_integrator_converges
       ; test "pendulum converges" test_pendulum_converges
       ; test
           "implicit gradient matches finite differences"
           test_implicit_gradient_matches_fd
       ; test "fixed solve is competitive" test_fixed_solve_works
       ; test
           "fixed solve keeps the trajectory when all trials are rejected"
           test_fixed_all_rejected
       ; test "jit solve matches eager" test_jit_solve
       ; test "jit grad matches eager" test_jit_grad
       ; test "batched fixed solve matches unbatched" test_batched_fixed_solve
       ; test "batched gradient matches unbatched" test_batched_grad
       ; test "adaptive solve rejects batches" test_adaptive_rejects_batches
       ])
