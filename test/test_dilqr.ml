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
    let verbose = false

    let dynamics { mass } ~k:_ ~x ~u =
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
  end

  module D = Dilqr.Make (Ctx)
end

(* Pendulum: x = [angle; angular velocity], control = torque, damping
   coefficient 0.1. Regulate the upright position (angle = π). *)
module Pendulum = struct
  module Ctx = struct
    type 'a params =
      { gravity : 'a
      ; mass : 'a
      ; length : 'a
      }
    [@@deriving ptree]

    module Params = struct
      type 'a t = 'a params

      let walk = walk_params
    end

    type w = Nx.float64_elt
    type value = (float, w) Nx.t

    let dtype = Nx.float64
    let n = 2
    let m = 1
    let horizon = 30
    let verbose = false
    let target = 3.14159265358979

    let dynamics { gravity; mass; length } ~k:_ ~x ~u =
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
  end

  module D = Dilqr.Make (Ctx)
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

(* Stationarity of the double-integrator solution: the autodiff gradient of
   the total cost w.r.t. each control vanishes. Autodiff through the true
   rollout — independent of the Riccati code under test. *)
let test_integrator_converges () =
  let module D = Integrator.D in
  let module Ctx = Integrator.Ctx in
  let theta : Ctx.value Ctx.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init Ctx.horizon ~f:(fun _ -> [ 0.0 ])) in
  let cost_of_us us = snd (D.Solver.rollout theta x0 us) in
  let cost0 = item (cost_of_us us0) in
  let sol = D.Solver.solve theta x0 us0 in
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
  let module D = Pendulum.D in
  let module Ctx = Pendulum.Ctx in
  let theta : Ctx.value Ctx.Params.t =
    { gravity = scalar64 9.81; mass = scalar64 1.0; length = scalar64 1.0 }
  in
  let x0 = vec64 [| Ctx.target -. 0.3; 0.0 |] in
  let us0 = us_of (List.init Ctx.horizon ~f:(fun _ -> [ 0.0 ])) in
  let cost0 = item (snd (D.Solver.rollout theta x0 us0)) in
  let sol = D.Solver.solve ~max_iter:80 theta x0 us0 in
  let cost1 = item sol.cost in
  require ~msg:"pendulum reduces cost" Float.(cost1 < 0.5 *. cost0);
  let xf = sol.xs.(Array.length sol.xs - 1) in
  let err = Float.abs (Nx.item [ 0 ] (Nx.slice [ R (0, 1) ] xf) -. Ctx.target) in
  require ~msg:"pendulum reaches upright" Float.(err < 0.1)

(* Implicit gradient vs. central finite differences, w.r.t. x0 and theta, on
   the pendulum with a tightly converged solve. *)
let test_implicit_gradient_matches_fd () =
  let module D = Pendulum.D in
  let module Ctx = Pendulum.Ctx in
  let theta : Ctx.value Ctx.Params.t =
    { gravity = scalar64 9.81; mass = scalar64 1.3; length = scalar64 0.9 }
  in
  let x0 = vec64 [| Ctx.target -. 0.4; 0.1 |] in
  let us0 = us_of (List.init Ctx.horizon ~f:(fun _ -> [ 0.02 ])) in
  let inputs : D.value D.Inputs.t = { theta; x0; us0 } in
  (* A user loss on the solve output, different from the problem's own cost. *)
  let loss taus =
    let xs, us = D.split_taus taus in
    let xf = xs.(Array.length xs - 1) in
    let ctrl =
      Array.fold us ~init:(scalar64 0.0) ~f:(fun a u ->
        Nx.add a (Nx.mul_s (Nx.sum (Nx.mul u u)) 0.5))
    in
    let err = Nx.sub_s (Nx.slice [ R (0, 1) ] xf) Ctx.target in
    Nx.add ctrl (Nx.mul_s (Nx.sum (Nx.mul err err)) 1.0)
  in
  let value inputs =
    item (loss (D.solve_ilqr ~max_iter:100 ~conv_threshold:1e-10 inputs))
  in
  let grads =
    Rune.grad
      D.inputs_ptree
      (fun i -> loss (D.solve_ilqr ~max_iter:100 ~conv_threshold:1e-10 i))
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
  let module D = Integrator.D in
  let module Ctx = Integrator.Ctx in
  let theta : Ctx.value Ctx.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init Ctx.horizon ~f:(fun _ -> [ 0.0 ])) in
  let sol = D.Solver.solve_fixed ~iters:8 theta x0 us0 in
  let adaptive = D.Solver.solve ~max_iter:20 theta x0 us0 in
  let c_fixed = item sol.cost
  and c_adaptive = item adaptive.cost in
  require ~msg:"fixed solve competitive" Float.(c_fixed < 1.05 *. c_adaptive)

(* JIT: the compiled fixed solve equals the eager one. *)
let test_jit_solve () =
  let module D = Integrator.D in
  let module Ctx = Integrator.Ctx in
  let theta : Ctx.value Ctx.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init Ctx.horizon ~f:(fun _ -> [ 0.0 ])) in
  let inputs : D.value D.Inputs.t = { theta; x0; us0 } in
  let eager = D.solve_ilqr_fixed ~iters:5 inputs in
  let jitted =
    Rune.jit
      Nx.Ptree.(D.inputs_ptree @-> returns Nx.Ptree.tensor)
      (fun i -> D.solve_ilqr_fixed ~iters:5 i)
      inputs
  in
  check_close ~msg:"jit solve taus" ~tol:1e-10 jitted eager

(* JIT: the compiled gradient — forward solve and implicit VJP, all traced —
   matches the eager gradient. *)
let test_jit_grad () =
  let module D = Integrator.D in
  let module Ctx = Integrator.Ctx in
  let theta : Ctx.value Ctx.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init Ctx.horizon ~f:(fun _ -> [ 0.0 ])) in
  let inputs : D.value D.Inputs.t = { theta; x0; us0 } in
  let loss taus =
    let xs, us = D.split_taus taus in
    let xf = xs.(Array.length xs - 1) in
    Nx.add
      (Nx.sum (Array.fold us ~init:(scalar64 0.0) ~f:(fun a u -> Nx.add a (Nx.mul u u))))
      (Nx.mul_s (Nx.sum (Nx.mul xf xf)) 5.0)
  in
  let eager_grad =
    Rune.grad D.inputs_ptree (fun i -> loss (D.solve_ilqr_fixed ~iters:4 i)) inputs
  in
  let compiled =
    Rune.jit
      Nx.Ptree.(D.inputs_ptree @-> returns D.inputs_ptree)
      (fun i ->
         Rune.grad D.inputs_ptree (fun i -> loss (D.solve_ilqr_fixed ~iters:4 i)) i)
      inputs
  in
  check_close ~msg:"jit grad x0" ~tol:1e-9 compiled.x0 eager_grad.x0;
  check_close ~msg:"jit grad mass" ~tol:1e-9 compiled.theta.mass eager_grad.theta.mass

let () =
  Stdlib.exit
    (run
       "dilqr"
       [ test "regularisation schedule" test_regularisation
       ; test "double integrator converges" test_integrator_converges
       ; test "pendulum converges" test_pendulum_converges
       ; test
           "implicit gradient matches finite differences"
           test_implicit_gradient_matches_fd
       ; test "fixed solve is competitive" test_fixed_solve_works
       ; test "jit solve matches eager" test_jit_solve
       ; test "jit grad matches eager" test_jit_grad
       ])
