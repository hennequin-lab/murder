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
open Nx
open Windtrap
open Stdio
module Nx = Nx

let dt = 0.1
let require ~msg b = if not b then failwith ("assert failed: " ^ msg)

let check_close ~msg ~tol got expected =
  let g = to_array got
  and e = to_array expected in
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
let fd_grad ~eps f (x : Nx.host Nx.float64_t) : Nx.host Nx.float64_t =
  let n = (shape x).(0) in
  let out = Array.create ~len:n 0.0 in
  for i = 0 to n - 1 do
    let xi = Nx.item [ i ] x in
    let xp = set [ I i ] (scalar float64 (xi +. eps)) x in
    let xm = set [ I i ] (scalar float64 (xi -. eps)) x in
    out.(i) <- (f xp -. f xm) /. (2.0 *. eps)
  done;
  create float64 [| n |] out

let vec64 a = create float64 [| Array.length a |] a
let scalar64 v = scalar float64 v
let item t = Nx.item [] t

let us_of (vs : float list list) : Nx.host Nx.float64_t array =
  Array.of_list (List.map ~f:(fun l -> vec64 (Array.of_list l)) vs)

(* ──────────────────────────────────────────────────────────────────────────
   Problems. Both are float64 so gradient checks against finite differences
   are meaningful.
   ────────────────────────────────────────────────────────────────────────── *)

(* Double integrator: x = [position; velocity], control = force. *)
module Integrator = struct
  module Params = struct
    type ('a, _) t = { mass : 'a } [@@deriving ptree]
  end

  let n = 2
  let m = 1
  let horizon = 20
  let verbose = false

  let dynamics { Params.mass } ~k:_ ~x ~u =
    let open Infix in
    let pos = slice [ R (0, 1) ] x
    and vel = slice [ R (1, 2) ] x in
    let force = u / mass in
    concatenate ~axis:0 [ pos + (vel *$ dt); vel + (force *$ dt) ]

  (* The state cost ramps with the step: [k] reaches the loss as data, so
     the scans carry time with their rows. *)
  let running_loss _ ~k ~x ~u =
    let open Infix in
    let pos = slice [ R (0, 1) ] x
    and vel = slice [ R (1, 2) ] x in
    let weight = (k *$ 0.05) +$ 1.0 in
    sum ((((pos * pos) + (vel * vel)) * weight *$ 0.5) + (u * u *$ 0.05))

  let final_loss _ ~x =
    let open Infix in
    let pos = slice [ R (0, 1) ] x
    and vel = slice [ R (1, 2) ] x in
    sum ((pos * pos) + (vel * vel)) *$ 5.0

  let problem
    : (Nx.float64_elt, (Nx.host Nx.float64_t, Nx.host) Params.t, Nx.host) Dilqr.Problem.t
    =
    { dtype = float64
    ; n
    ; m
    ; horizon
    ; free_initial = false
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
    type ('a, _) t =
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
    let open Infix in
    let theta = slice [ R (0, 1) ] x
    and omega = slice [ R (1, 2) ] x in
    let ml2 = mass * (length * length) in
    let grav_torque = gravity * (mass * length) * sin theta in
    let acc = (u - (grav_torque + (omega *$ 0.1))) / ml2 in
    concatenate ~axis:0 [ theta + (omega *$ dt); omega + (acc *$ dt) ]

  let running_loss _ ~k:_ ~x ~u =
    let open Infix in
    let theta = slice [ R (0, 1) ] x
    and omega = slice [ R (1, 2) ] x in
    let dtheta = theta -$ target in
    sum ((((dtheta * dtheta) + (omega * omega)) *$ 0.5) + (u * u *$ 0.01))

  let final_loss _ ~x =
    let open Infix in
    let theta = slice [ R (0, 1) ] x
    and omega = slice [ R (1, 2) ] x in
    let dtheta = theta -$ target in
    sum ((dtheta * dtheta) + (omega * omega)) *$ 5.0

  let problem
    : (Nx.float64_elt, (Nx.host Nx.float64_t, Nx.host) Params.t, Nx.host) Dilqr.Problem.t
    =
    { dtype = float64
    ; n
    ; m
    ; horizon
    ; free_initial = false
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
    type ('a, _) t =
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
    let open Infix in
    let pos = slice [ R (0, 1) ] x
    and vel = slice [ R (1, 2) ] x in
    let force = u / mass in
    concatenate ~axis:0 [ pos + (vel *$ dt); vel + (force *$ dt) ]

  let running_loss { Params.target; _ } ~k:_ ~x ~u =
    let open Infix in
    let pos = slice [ R (0, 1) ] x
    and vel = slice [ R (1, 2) ] x in
    let err = pos - target in
    sum ((((err * err) + (vel * vel)) *$ 0.5) + (u * u *$ 0.05))

  let final_loss { Params.target; _ } ~x =
    let open Infix in
    let pos = slice [ R (0, 1) ] x
    and vel = slice [ R (1, 2) ] x in
    let err = pos - target in
    sum ((err * err) + (vel * vel)) *$ 5.0

  let problem
    : (Nx.float64_elt, (Nx.host Nx.float64_t, Nx.host) Params.t, Nx.host) Dilqr.Problem.t
    =
    { dtype = float64
    ; n
    ; m
    ; horizon
    ; free_initial = false
    ; verbose
    ; params = Nx.Ptree.instantiate (module Params)
    ; dynamics
    ; running_loss
    ; final_loss
    }
end

(* A one-step problem whose control cost is concave where it starts:
   x' = x + u, l(u) = ½ (u² - 1)², no terminal cost. The local model's [Q_uu]
   is the loss Hessian 3u² - 1, which is negative at u = ½, so the undamped
   Riccati pass cannot factorize it: only damping buys a step. The loss is
   bounded below, and its minimum is u = 1. *)
module Concave = struct
  module Params = struct
    type ('a, _) t = { unit : 'a } [@@deriving ptree]
  end

  let n = 1
  let m = 1
  let horizon = 1
  let verbose = false
  let dynamics { Params.unit } ~k:_ ~x ~u = Infix.((x + u) * unit)

  let running_loss { Params.unit } ~k:_ ~x:_ ~u =
    Infix.(unit * (((u * u) -$ 1.0) * ((u * u) -$ 1.0) *$ 0.5))

  let final_loss { Params.unit } ~x:_ = Infix.(unit *$ 0.0)

  let problem
    : (Nx.float64_elt, (Nx.host Nx.float64_t, Nx.host) Params.t, Nx.host) Dilqr.Problem.t
    =
    { dtype = float64
    ; n
    ; m
    ; horizon
    ; free_initial = false
    ; verbose
    ; params = Nx.Ptree.instantiate (module Params)
    ; dynamics
    ; running_loss
    ; final_loss
    }
end

(* A batch of pendulums whose initial angles are [angles], sharing one set of
   dynamics parameters along the batch axis. *)
let pendulum_batch (angles : float array) =
  let b = Array.length angles in
  let at v = broadcast_to [| b |] (scalar64 v) in
  let theta : (Nx.host Nx.float64_t, Nx.host) Pendulum.Params.t =
    { gravity = at 9.81; mass = at 1.0; length = at 1.0 }
  in
  let x0 =
    create float64 [| b; 2 |] (Array.concat_map angles ~f:(fun a -> [| a; 0.0 |]))
  in
  let us0 = Array.init Pendulum.horizon ~f:(fun _ -> zeros float64 [| b; 1 |]) in
  theta, x0, us0

(* One pendulum from [angle], solved alone. *)
let pendulum_member ?damping ~max_iters angle =
  let theta : (Nx.host Nx.float64_t, Nx.host) Pendulum.Params.t =
    { gravity = scalar64 9.81; mass = scalar64 1.0; length = scalar64 1.0 }
  in
  let x0 = vec64 [| angle; 0.0 |] in
  let us0 = Array.init Pendulum.horizon ~f:(fun _ -> zeros float64 [| 1 |]) in
  Dilqr.Solver.solve_fixed Pendulum.problem ?damping ~max_iters theta x0 us0

(* A deterministic pseudo-random pattern of the same shape as [like]: a fixed
   float64 wave folded into the shape and cast to the leaf's dtype. Test inputs
   only — no generator, so a failing case reproduces. *)
let pattern ~seed ~(like : ('a, 'b, Nx.host) Nx.t) : ('a, 'b, Nx.host) Nx.t =
  let s = shape like in
  let numel = Array.fold s ~init:1 ~f:(fun a b -> a * b) in
  let flat = arange_f float64 0.0 (Float.of_int numel) 1.0 in
  let wave = sin Infix.((flat *$ 1.7) +$ seed) in
  Nx.cast (Nx.dtype like) (reshape s wave)

(* The same, one leaf at a time: a tangent seed, or a cotangent on the inputs. *)
let pattern_inputs ~seed (structure : ('p, Nx.host) Nx.Ptree.t) (value : 'p) : 'p =
  Nx.Ptree.map structure (fun _ leaf -> pattern ~seed ~like:leaf) value

(* The same with [k] leading lanes: a batch of directions. *)
let pattern_lanes ~seed ~k (structure : ('p, Nx.host) Nx.Ptree.t) (value : 'p) : 'p =
  Nx.Ptree.map
    structure
    (fun _ leaf ->
       concatenate
         ~axis:0
         (List.init k ~f:(fun i ->
            unsqueeze
              ~axes:[ 0 ]
              (pattern ~seed:(seed +. (1.3 *. Float.of_int i)) ~like:leaf))))
    value

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
  let one s = reshape s (scalar float64 1.0)
  and zero s = zeros float64 s in
  let blocks =
    Rows.
      { a = one [| 1; 1; 1 |]
      ; b = one [| 1; 1; 1 |]
      ; lx = zero [| 1; 1 |]
      ; lu = zero [| 1; 1 |]
      ; lxx = zero [| 1; 1; 1 |]
      ; luu = reshape [| 1; 1; 1 |] (scalar float64 (-1.0))
      ; lux = zero [| 1; 1; 1 |]
      }
  in
  let tape = { blocks; fx = zero [| 1 |]; fxx = zero [| 1; 1 |] } in
  let finite d1 d2 = Nx.item [] (all (isfinite d1)) && Nx.item [] (all (isfinite d2)) in
  let _, d1, d2, _ = backward tape ~lin:(lin_of_tape tape) in
  require
    ~msg:"an indefinite Quu has NaN expected-decrease accumulators"
    (not (finite d1 d2));
  let _, d1, d2, _ = backward ~mu:(scalar float64 2.0) tape ~lin:(lin_of_tape tape) in
  require ~msg:"regularization makes Quu positive definite" (finite d1 d2)

(* The costate is the hand recursion: on a two-step tape with a closed form,
   [λ_K = ∂φ/∂x], [λ_k = A_kᵀ λ_{k+1} + ∂l_k/∂x], and the stack holds
   [λ_1; …; λ_K]. *)
let test_costate () =
  let open Dilqr.Lqr in
  let one s v = Infix.(ones float64 s *$ v) in
  let blocks =
    Rows.
      { a = one [| 2; 1; 1 |] 0.5
      ; b = one [| 2; 1; 2 |] 1.0
      ; lx = create float64 [| 2; 1 |] [| 1.0; 2.0 |]
      ; lu = zeros float64 [| 2; 2 |]
      ; lxx = zeros float64 [| 2; 1; 1 |]
      ; luu = zeros float64 [| 2; 2; 2 |]
      ; lux = zeros float64 [| 2; 2; 1 |]
      }
  in
  let tape = { blocks; fx = one [| 1 |] 3.0; fxx = zeros float64 [| 1; 1 |] } in
  check_close
    ~msg:"costate"
    ~tol:1e-12
    (costate tape)
    (create float64 [| 2; 1 |] [| 3.5; 3.0 |])

(* Stationarity of the double-integrator solution: the autodiff gradient of
   the total cost w.r.t. each control vanishes. Autodiff through the true
   rollout — independent of the Riccati code under test. *)
let test_integrator_converges () =
  let module P = Integrator in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { mass = scalar64 1.0 } in
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
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t =
    { gravity = scalar64 9.81; mass = scalar64 1.0; length = scalar64 1.0 }
  in
  let x0 = vec64 [| P.target -. 0.3; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let cost0 = item (snd (Dilqr.Solver.rollout P.problem theta x0 us0)) in
  let sol = Dilqr.Solver.solve P.problem ~max_iter:80 theta x0 us0 in
  let cost1 = item sol.cost in
  require ~msg:"pendulum reduces cost" Float.(cost1 < 0.5 *. cost0);
  let xf = sol.xs.(Array.length sol.xs - 1) in
  let err = Float.abs (Nx.item [ 0 ] (slice [ R (0, 1) ] xf) -. P.target) in
  require ~msg:"pendulum reaches upright" Float.(err < 0.1)

(* Implicit gradient vs. central finite differences, w.r.t. x0 and theta, on
   the pendulum with a tightly converged solve. *)
let test_implicit_gradient_matches_fd () =
  let module P = Pendulum in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t =
    { gravity = scalar64 9.81; mass = scalar64 1.3; length = scalar64 0.9 }
  in
  let x0 = vec64 [| P.target -. 0.4; 0.1 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.02 ])) in
  let inputs
    : ((Nx.host Nx.float64_t, Nx.host) P.Params.t, Nx.host Nx.float64_t) Dilqr.Inputs.t
    =
    { Dilqr.Inputs.theta; x0; us0 }
  in
  (* A user loss on the solve output, different from the problem's own cost. *)
  let loss taus =
    let xs, us = Dilqr.split_taus P.problem taus in
    let xf = xs.(Array.length xs - 1) in
    let ctrl =
      Array.fold us ~init:(scalar64 0.0) ~f:(fun a u -> Infix.(a + (sum (u * u) *$ 0.5)))
    in
    let err = Infix.(slice [ R (0, 1) ] xf -$ P.target) in
    Infix.(ctrl + (sum (err * err) *$ 1.0))
  in
  let solve = Dilqr.solve_adaptive P.problem in
  let value inputs = item (loss (solve ~max_iter:100 ~conv_threshold:1e-10 inputs)) in
  let grads =
    Rune.grad
      (Dilqr.inputs P.problem)
      (fun i -> loss (solve ~max_iter:100 ~conv_threshold:1e-10 i))
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
      (fun v -> value { inputs with theta = f (base +. item (slice [ I 0 ] v)) })
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
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let sol = Dilqr.Solver.solve_fixed P.problem ~max_iters:8 theta x0 us0 in
  let adaptive = Dilqr.Solver.solve P.problem ~max_iter:20 theta x0 us0 in
  let c_fixed = item sol.cost
  and c_adaptive = item adaptive.cost in
  require ~msg:"fixed solve competitive" Float.(c_fixed < 1.05 *. c_adaptive)

(* When every line-search trial is rejected — here because the trial steps
   are NaN — the fixed solve must keep the current trajectory. It cannot buy a
   step and, with no damping to raise, it stops after the iteration rather than
   retrying what cannot improve. *)
let test_fixed_all_rejected () =
  let module P = Integrator in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let sol =
    Dilqr.Solver.solve_fixed P.problem ~alphas:[ Float.nan ] ~max_iters:2 theta x0 us0
  in
  require ~msg:"all-rejected cost stays finite" (Float.is_finite (item sol.cost));
  require ~msg:"all-rejected stops after one iteration" (Float.equal (item sol.iters) 1.0);
  Array.iter sol.us ~f:(fun u ->
    require
      ~msg:"all-rejected keeps the current controls"
      (Float.is_finite (Nx.item [ 0 ] u) && Float.( = ) (Nx.item [ 0 ] u) 0.0))

(* The iteration budget is the batch's: the solve runs the iterations its
   slowest member needs, and a member that has converged takes no step and does
   not vote, so it cannot veto the steps of the others. *)
let test_fixed_budget_and_no_veto () =
  let module P = Pendulum in
  let hanging = P.target -. 3.0 in
  let theta, x0, us0 = pendulum_batch [| P.target; hanging |] in
  let sol = Dilqr.Solver.solve_fixed P.problem ~max_iters:40 theta x0 us0 in
  let iters = item sol.iters in
  let at_target = pendulum_member ~max_iters:40 P.target
  and alone = pendulum_member ~max_iters:40 hanging in
  require
    ~msg:"a member at its optimum runs one iteration"
    (Float.equal (item at_target.iters) 1.0);
  require
    ~msg:"the batch runs the slowest member's iterations"
    (Float.equal iters (item alone.iters));
  require ~msg:"and more than the converged member's" Float.(iters > 1.0);
  require
    ~msg:"a converged member is frozen"
    Float.(Float.abs (item (slice [ I 0 ] sol.cost)) < 1e-20);
  check_close
    ~msg:"the hanging member is solved as it is alone"
    ~tol:1e-9
    (slice [ I 1 ] sol.cost)
    alone.cost

(* The batch runs its slowest member's iterations — not the mean, and not the
   sum — and a member that converges early is frozen at its converged cost. *)
let test_fixed_budget_is_the_slowest_member () =
  let module P = Pendulum in
  let near = P.target -. 0.05
  and far = P.target -. 3.0 in
  let theta, x0, us0 = pendulum_batch [| near; far |] in
  let sol = Dilqr.Solver.solve_fixed P.problem ~max_iters:40 theta x0 us0 in
  let iters = item sol.iters in
  let near_alone = pendulum_member ~max_iters:40 near
  and far_alone = pendulum_member ~max_iters:40 far in
  let fastest = Float.min (item near_alone.iters) (item far_alone.iters)
  and slowest = Float.max (item near_alone.iters) (item far_alone.iters) in
  require ~msg:"the batch spends the slowest member's iterations" Float.(iters = slowest);
  require ~msg:"and more than the quickest member's" Float.(iters > fastest);
  check_close
    ~msg:"the member that converged early keeps its converged cost"
    ~tol:1e-6
    (slice [ I 0 ] sol.cost)
    near_alone.cost

(* Damping is what makes an indefinite [Q_uu] usable. Undamped, no trial can be
   computed from the NaN factorization and the solve gives up; damped, the
   strength is raised until the model factorizes — from the schedule's own
   units, or from a seed far below what the pass needs. *)
let test_fixed_damping_recovers () =
  let module P = Concave in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { unit = scalar64 1.0 } in
  let x0 = vec64 [| 0.0 |] in
  let us0 = [| vec64 [| 0.5 |] |] in
  let plain =
    Dilqr.Solver.solve_fixed
      P.problem
      ~max_iters:20
      ~damping:Dilqr.Damping.Off
      theta
      x0
      us0
  in
  require
    ~msg:"undamped gives up after one iteration"
    (Float.equal (item plain.iters) 1.0);
  require
    ~msg:"undamped keeps the cost of its initial guess"
    Float.(Float.abs (item plain.cost -. 0.28125) < 1e-12);
  (* Damping is on by default: the solve raises the strength until the model
     factorizes, and reaches the loss's minimum. *)
  let damped = Dilqr.Solver.solve_fixed P.problem ~max_iters:20 theta x0 us0 in
  require ~msg:"adaptive damping decreases the cost" Float.(item damped.cost < 1e-5);
  require
    ~msg:"damped reaches u = 1"
    Float.(Float.abs (Nx.item [ 0 ] damped.us.(0) -. 1.0) < 1e-3);
  let climbing =
    Dilqr.Solver.solve_fixed
      P.problem
      ~max_iters:20
      ~damping:(Dilqr.Damping.Adaptive { Dilqr.Regularisation.delta = 1.0; mu = 1e-3 })
      theta
      x0
      us0
  in
  require
    ~msg:"retries climb to a strength that factorizes"
    Float.(item climbing.cost < 1e-5)

(* The adaptive solve takes the same damping choice. Pinned to no damping it
   cannot even factorize the concave model, and says so; the default buys the
   step. *)
let test_adaptive_pinned_damping_fails () =
  let module P = Concave in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { unit = scalar64 1.0 } in
  let x0 = vec64 [| 0.0 |] in
  let us0 = [| vec64 [| 0.5 |] |] in
  (match Dilqr.Solver.solve P.problem ~damping:Dilqr.Damping.Off theta x0 us0 with
   | exception Failure msg ->
     require
       ~msg:"the failure names the damping"
       (String.is_substring msg ~substring:"damping")
   | _ -> failwith "the undamped adaptive solve accepted an indefinite Quu");
  let sol = Dilqr.Solver.solve P.problem theta x0 us0 in
  require
    ~msg:"the adaptive solve's default damping buys the step"
    Float.(item sol.cost < 1e-5)

(* [max_iters = 0] runs nothing and returns the rollout of the initial guess. *)
let test_fixed_zero_budget () =
  let module P = Integrator in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let sol = Dilqr.Solver.solve_fixed P.problem ~max_iters:0 theta x0 us0 in
  let _, cost0 = Dilqr.Solver.rollout P.problem theta x0 us0 in
  require ~msg:"a zero budget runs no iteration" (Float.equal (item sol.iters) 0.0);
  check_close ~msg:"a zero budget keeps the initial guess" ~tol:1e-12 sol.cost cost0

(* The adaptive solve reports the updates it applied, which its host-side loop
   counts. *)
let test_adaptive_reports_iterations () =
  let module P = Integrator in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let sol = Dilqr.Solver.solve P.problem theta x0 us0 in
  require
    ~msg:"the adaptive solve counts its updates"
    Float.(item sol.iters > 0.0 && item sol.iters <= 20.0)

(* Damping shifts the value Hessian, and the shift expands into Tassa's three
   terms: one step's [Q_uu], [Q_ux] and [Q_xx] with [V + mu I] are what the
   explicit [mu AᵀA], [mu BᵀA] and [mu BᵀB] give. *)
let test_damping_is_a_value_hessian_shift () =
  let open Dilqr.Lqr in
  let one s v = Infix.(ones float64 s *$ v) in
  let blocks =
    Rows.
      { a = one [| 1; 1; 1 |] 0.5
      ; b = one [| 1; 1; 1 |] 1.5
      ; lx = zeros float64 [| 1; 1 |]
      ; lu = zeros float64 [| 1; 1 |]
      ; lxx = one [| 1; 1; 1 |] 1.0
      ; luu = one [| 1; 1; 1 |] 2.0
      ; lux = one [| 1; 1; 1 |] 0.5
      }
  in
  let tape = { blocks; fx = zeros float64 [| 1 |]; fxx = one [| 1; 1 |] 3.0 } in
  let mu = 0.75 in
  let a = 0.5
  and b = 1.5
  and vxx0 = 3.0 in
  (* the hand recursion, with the damping written out *)
  let quu = 2.0 +. (b *. vxx0 *. b) +. (mu *. b *. b) in
  let qux = 0.5 +. (b *. vxx0 *. a) +. (mu *. b *. a) in
  let qxx = 1.0 +. (a *. vxx0 *. a) +. (mu *. a *. a) in
  let k = -.qux /. quu in
  let vxx = qxx +. (qux *. k) in
  let common = backward_common ~mu:(scalar64 mu) tape in
  check_close
    ~msg:"damped value Hessian"
    ~tol:1e-12
    common.value_hessian
    (reshape [| 1; 1 |] (scalar64 vxx));
  check_close
    ~msg:"damped feedback gain"
    ~tol:1e-12
    common.feedback
    (reshape [| 1; 1; 1 |] (scalar64 k));
  check_close
    ~msg:"damped Q_uu is the factor's square"
    ~tol:1e-12
    Infix.(common.quu_chol * common.quu_chol)
    (reshape [| 1; 1; 1 |] (scalar64 quu))

(* JIT: the compiled fixed solve equals the eager one, for each damping mode:
   the adaptive default carries a retry loop, and the pinned modes carry none
   of it. *)
let test_jit_solve () =
  let module P = Integrator in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let inputs
    : ((Nx.host Nx.float64_t, Nx.host) P.Params.t, Nx.host Nx.float64_t) Dilqr.Inputs.t
    =
    { Dilqr.Inputs.theta; x0; us0 }
  in
  let check ~msg damping =
    let eager = Dilqr.solve_fixed P.problem ~max_iters:5 ?damping inputs in
    let jitted =
      Rune.jit
        Nx.Ptree.(Dilqr.inputs P.problem @-> returns Nx.Ptree.tensor)
        (fun i -> Dilqr.solve_fixed P.problem ~max_iters:5 ?damping i)
        inputs
    in
    check_close ~msg ~tol:1e-10 jitted eager
  in
  check ~msg:"jit solve taus (adaptive)" None;
  check ~msg:"jit solve taus (off)" (Some Dilqr.Damping.Off);
  check ~msg:"jit solve taus (fixed)" (Some (Dilqr.Damping.Fixed 1.0))

(* JIT: the compiled gradient — forward solve and implicit VJP, all traced —
   matches the eager gradient. *)
let test_jit_grad () =
  let module P = Integrator in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let inputs
    : ((Nx.host Nx.float64_t, Nx.host) P.Params.t, Nx.host Nx.float64_t) Dilqr.Inputs.t
    =
    { Dilqr.Inputs.theta; x0; us0 }
  in
  let loss taus =
    let xs, us = Dilqr.split_taus P.problem taus in
    let xf = xs.(Array.length xs - 1) in
    Infix.(
      sum (Array.fold us ~init:(scalar64 0.0) ~f:(fun a u -> a + (u * u)))
      + (sum (xf * xf) *$ 5.0))
  in
  let eager_grad =
    Rune.grad
      (Dilqr.inputs P.problem)
      (fun i -> loss (Dilqr.solve_fixed P.problem ~max_iters:4 i))
      inputs
  in
  let compiled =
    Rune.jit
      Nx.Ptree.(Dilqr.inputs P.problem @-> returns (Dilqr.inputs P.problem))
      (fun i ->
         Rune.grad
           (Dilqr.inputs P.problem)
           (fun i -> loss (Dilqr.solve_fixed P.problem ~max_iters:4 i))
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
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t =
    { mass = create float64 [| b |] [| 1.0; 1.0; 1.0 |]
    ; target = create float64 [| b; 1 |] [| 1.0; -0.5; 0.8 |]
    }
  in
  let x0 = create float64 [| b; 2 |] [| 1.0; 0.0; 1.0; 0.0; 1.0; 0.0 |] in
  let us0 = Array.init P.horizon ~f:(fun _ -> zeros float64 [| b; 1 |]) in
  b, theta, x0, us0

(* Member [i]'s slice of a batched setup. *)
let batched_member (theta : (Nx.host Nx.float64_t, Nx.host) Batched.Params.t) x0 us0 i =
  let theta_i : (Nx.host Nx.float64_t, Nx.host) Batched.Params.t =
    { mass = slice [ I i ] theta.Batched.Params.mass
    ; target = slice [ I i ] theta.Batched.Params.target
    }
  in
  let x0_i = slice [ I i ] x0 in
  let us0_i = Array.map us0 ~f:(fun u -> slice [ I i ] u) in
  theta_i, x0_i, us0_i

(* A batched fixed solve gives, member by member, what the unbatched solve
   gives for that member. The members share the batch's step, so this holds
   because every one of them takes the full step here: a batch whose members
   backtrack apart moves them together, which test_fixed_budget_and_no_veto
   pins down. *)
let test_batched_fixed_solve () =
  let module P = Batched in
  let b, theta, x0, us0 = batched_setup () in
  let sol = Dilqr.Solver.solve_fixed P.problem ~max_iters:8 theta x0 us0 in
  require ~msg:"batched cost shape" (Array.length (shape sol.cost) = 1);
  require ~msg:"batched cost size" ((shape sol.cost).(0) = b);
  (* the stacked trajectory leads with the batch and then the horizon *)
  let taus = Dilqr.taus_of P.problem sol.xs sol.us in
  require
    ~msg:"batched taus shape"
    (Array.equal Int.equal (shape taus) [| b; P.horizon + 1; P.n + P.m |]);
  let xs, us = Dilqr.split_taus P.problem taus in
  (* the trajectory helpers are batched too *)
  let xs_b, cost_b = Dilqr.Solver.rollout P.problem theta x0 us0 in
  let tape_b : (float, Nx.float64_elt, Nx.host) Dilqr.Lqr.tape =
    Dilqr.Solver.blocks P.problem theta sol.xs sol.us
  in
  for i = 0 to b - 1 do
    let theta_i, x0_i, us0_i = batched_member theta x0 us0 i in
    let sol_i = Dilqr.Solver.solve_fixed P.problem ~max_iters:8 theta_i x0_i us0_i in
    check_close
      ~msg:(Printf.sprintf "batched cost %d" i)
      ~tol:1e-10
      (slice [ I i ] sol.cost)
      sol_i.cost;
    Array.iteri sol.xs ~f:(fun k t ->
      check_close
        ~msg:(Printf.sprintf "batched xs %d.%d" i k)
        ~tol:1e-10
        (slice [ I i ] t)
        sol_i.xs.(k);
      check_close
        ~msg:(Printf.sprintf "batched taus %d.%d" i k)
        ~tol:1e-10
        (slice [ I i ] xs.(k))
        sol_i.xs.(k));
    Array.iteri sol.us ~f:(fun k t ->
      check_close
        ~msg:(Printf.sprintf "batched us %d.%d" i k)
        ~tol:1e-10
        (slice [ I i ] t)
        sol_i.us.(k);
      check_close
        ~msg:(Printf.sprintf "batched taus us %d.%d" i k)
        ~tol:1e-10
        (slice [ I i ] us.(k))
        sol_i.us.(k));
    (* batched rollout and blocks *)
    let xs_i, cost_i = Dilqr.Solver.rollout P.problem theta_i x0_i us0_i in
    let tape_i = Dilqr.Solver.blocks P.problem theta_i sol_i.xs sol_i.us in
    check_close
      ~msg:(Printf.sprintf "batched rollout cost %d" i)
      ~tol:1e-10
      (slice [ I i ] cost_b)
      cost_i;
    Array.iteri xs_i ~f:(fun k x ->
      check_close
        ~msg:(Printf.sprintf "batched rollout xs %d.%d" i k)
        ~tol:1e-10
        (slice [ I i ] xs_b.(k))
        x);
    check_close
      ~msg:(Printf.sprintf "batched blocks a %d" i)
      ~tol:1e-10
      (slice [ I i ] tape_b.Dilqr.Lqr.blocks.a)
      tape_i.Dilqr.Lqr.blocks.a
  done;
  (* The batch's solve compiles with an adaptive damping schedule too: the
     per-member strengths and the retry loop are traced like the rest. *)
  let inputs : (_, Nx.host Nx.float64_t) Dilqr.Inputs.t =
    { Dilqr.Inputs.theta; x0; us0 }
  in
  let eager = Dilqr.solve_fixed P.problem ~max_iters:8 inputs in
  let jitted =
    Rune.jit
      Nx.Ptree.(Dilqr.inputs P.problem @-> returns Nx.Ptree.tensor)
      (fun i -> Dilqr.solve_fixed P.problem ~max_iters:8 i)
      inputs
  in
  check_close ~msg:"batched jit solve with damping" ~tol:1e-10 jitted eager

(* The batched implicit gradient is the member's gradient — the derivative is
   taken at the returned trajectory, member by member, so this holds when the
   batch and the members reach the same trajectory — and the same compiled. *)
let test_batched_grad () =
  let module P = Batched in
  let b, theta, x0, us0 = batched_setup () in
  let inputs
    : ((Nx.host Nx.float64_t, Nx.host) P.Params.t, Nx.host Nx.float64_t) Dilqr.Inputs.t
    =
    { Dilqr.Inputs.theta; x0; us0 }
  in
  let objective i =
    sum
      (Dilqr.loss_of_taus
         P.problem
         i.Dilqr.Inputs.theta
         (Dilqr.solve_fixed P.problem ~max_iters:5 i))
  in
  let g = Rune.grad (Dilqr.inputs P.problem) objective inputs in
  for i = 0 to b - 1 do
    let theta_i, x0_i, us0_i = batched_member theta x0 us0 i in
    let inputs_i
      : ((Nx.host Nx.float64_t, Nx.host) P.Params.t, Nx.host Nx.float64_t) Dilqr.Inputs.t
      =
      { Dilqr.Inputs.theta = theta_i; x0 = x0_i; us0 = us0_i }
    in
    let g_i = Rune.grad (Dilqr.inputs P.problem) objective inputs_i in
    check_close
      ~msg:(Printf.sprintf "batched grad target %d" i)
      ~tol:1e-9
      (slice [ I i ] g.theta.target)
      g_i.theta.target;
    check_close
      ~msg:(Printf.sprintf "batched grad mass %d" i)
      ~tol:1e-9
      (slice [ I i ] g.theta.mass)
      g_i.theta.mass;
    check_close
      ~msg:(Printf.sprintf "batched grad x0 %d" i)
      ~tol:1e-9
      (slice [ I i ] g.x0)
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

(* [Solver.compile] compiles the tensor kernels of the adaptive solve and
   leaves the host control flow alone: same trajectory, same cost. Build the
   function once and reuse it for a second initial state of the same shapes. *)
let test_compiled_adaptive_solve () =
  let module P = Integrator in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { mass = scalar64 1.0 } in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let solve = Dilqr.Solver.compile P.problem ~beam:0 ~parallel:1 in
  let check ~msg x0 =
    let eager = Dilqr.Solver.solve P.problem ~max_iter:20 theta x0 us0 in
    let compiled = solve ~max_iter:20 theta x0 us0 in
    check_close ~msg:(msg ^ " cost") ~tol:1e-10 compiled.cost eager.cost;
    Array.iteri eager.xs ~f:(fun k x ->
      check_close ~msg:(Printf.sprintf "%s xs %d" msg k) ~tol:1e-10 compiled.xs.(k) x);
    Array.iteri eager.us ~f:(fun k u ->
      check_close ~msg:(Printf.sprintf "%s us %d" msg k) ~tol:1e-10 compiled.us.(k) u)
  in
  check ~msg:"compiled adaptive" (vec64 [| 1.0; 0.0 |]);
  check ~msg:"compiled adaptive reused" (vec64 [| -0.5; 0.2 |])

(* Nonlinear problem, so the compiled path also meets the regularization
   retries and a backtracked step. *)
let test_compiled_adaptive_pendulum () =
  let module P = Pendulum in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t =
    { gravity = scalar64 9.81; mass = scalar64 1.0; length = scalar64 1.0 }
  in
  let x0 = vec64 [| P.target -. 0.3; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let eager = Dilqr.Solver.solve P.problem ~max_iter:20 theta x0 us0 in
  let compiled =
    (Dilqr.Solver.compile P.problem ~beam:0 ~parallel:1) ~max_iter:20 theta x0 us0
  in
  check_close ~msg:"compiled pendulum cost" ~tol:1e-9 compiled.cost eager.cost;
  Array.iteri eager.us ~f:(fun k u ->
    check_close
      ~msg:(Printf.sprintf "compiled pendulum us %d" k)
      ~tol:1e-9
      compiled.us.(k)
      u)

(* The differentiable adaptive entry point compiles the same kernels when a
   JIT option is supplied, and the compiled and eager closures return the same
   trajectory. *)
let test_solve_adaptive_compiled () =
  let module P = Integrator in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let inputs
    : ((Nx.host Nx.float64_t, Nx.host) P.Params.t, Nx.host Nx.float64_t) Dilqr.Inputs.t
    =
    { Dilqr.Inputs.theta; x0; us0 }
  in
  let eager = Dilqr.solve_adaptive P.problem in
  let compiled = Dilqr.solve_adaptive P.problem ~beam:0 ~parallel:1 in
  check_close
    ~msg:"differentiable adaptive compiled vs eager"
    ~tol:1e-10
    (compiled ~max_iter:20 inputs)
    (eager ~max_iter:20 inputs)

(* The adaptive solve has one host-side line search; batches are for the
   fixed solve. *)
let test_adaptive_rejects_batches () =
  let module P = Batched in
  let b, theta, x0, us0 = batched_setup () in
  require ~msg:"batch is not empty" (b > 0);
  let () =
    match Dilqr.Solver.solve P.problem theta x0 us0 with
    | exception Failure msg ->
      require
        ~msg:"adaptive solve names batches"
        (String.is_substring msg ~substring:"batched")
    | _ -> failwith "the adaptive solve accepted a batch"
  in
  match (Dilqr.Solver.compile P.problem) theta x0 us0 with
  | exception Failure msg ->
    require
      ~msg:"compiled adaptive solve names batches"
      (String.is_substring msg ~substring:"batched")
  | _ -> failwith "the compiled adaptive solve accepted a batch"

(* ──────────────────────────────────────────────────────────────────────────
   Forward mode
   ────────────────────────────────────────────────────────────────────────── *)

let pendulum_inputs () =
  let module P = Pendulum in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t =
    { gravity = scalar64 9.81; mass = scalar64 1.3; length = scalar64 0.9 }
  in
  let x0 = vec64 [| P.target -. 0.4; 0.1 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.02 ])) in
  { Dilqr.Inputs.theta; x0; us0 }

(* The two modes are transposes: for a random cotangent on the trajectory and a
   random tangent of the solve inputs,
   [<reverse gradient, tangent> = <cotangent, JVP tangent>]. This is the law
   the forward mode is built to satisfy, and it holds far tighter than a
   finite-difference check. *)
let test_jvp_is_the_transpose_of_the_vjp () =
  let module P = Pendulum in
  let inputs = pendulum_inputs () in
  let structure = Dilqr.inputs P.problem in
  let ybar = pattern ~seed:0.3 ~like:(zeros float64 [| P.horizon + 1; P.n + P.m |]) in
  let dinputs = pattern_inputs ~seed:1.1 structure inputs in
  let grads =
    Rune.grad
      structure
      (fun i -> sum Infix.(ybar * Dilqr.solve_fixed P.problem ~max_iters:6 i))
      inputs
  in
  let dtaus =
    snd
      (Rune.jvp
         structure
         Nx.Ptree.tensor
         (fun i -> Dilqr.solve_fixed P.problem ~max_iters:6 i)
         inputs
         dinputs)
  in
  let lhs = Nx.Ptree.dot structure float64 grads dinputs in
  let rhs = sum Infix.(ybar * dtaus) in
  check_close ~msg:"<grad, v> = <cotangent, JVP v>" ~tol:1e-8 lhs rhs

(* The forward mode against central differences of the adaptive solve, one
   input at a time and on a smooth objective: the trajectory tangent contracted
   with the objective's gradient is the objective's directional derivative. Both
   modes share one idealization, so the agreement is as good as the reverse
   mode's (which the test above measures against the same kind of finite
   difference); directions are taken one at a time because a direction that
   mixes two inputs lets their contributions cancel and inflates the relative
   error. *)
let test_jvp_matches_finite_differences () =
  let module P = Pendulum in
  let inputs = pendulum_inputs () in
  let structure = Dilqr.inputs P.problem in
  let solve = Dilqr.solve_adaptive P.problem in
  let solve_jvp = solve in
  let solve i = solve ~max_iter:100 ~conv_threshold:1e-10 i in
  (* A loss on the trajectory, smooth in the trajectory. *)
  let loss taus =
    let xs, us = Dilqr.split_taus P.problem taus in
    let xf = xs.(Array.length xs - 1) in
    let ctrl =
      Array.fold us ~init:(scalar64 0.0) ~f:(fun a u -> Infix.(a + (sum (u * u) *$ 0.5)))
    in
    let err = Infix.(slice [ R (0, 1) ] xf -$ P.target) in
    Infix.(ctrl + (sum (err * err) *$ 1.0))
  in
  let taus = solve inputs in
  let ybar = Rune.grad' loss taus in
  let theta = inputs.Dilqr.Inputs.theta in
  let base_gravity = item theta.Pendulum.Params.gravity in
  let direction gravity x0 =
    { Dilqr.Inputs.theta =
        { Pendulum.Params.gravity = scalar64 gravity
        ; mass = scalar64 0.0
        ; length = scalar64 0.0
        }
    ; x0
    ; us0 = Array.map inputs.us0 ~f:zeros_like
    }
  in
  let shift d s =
    let dt = d.Dilqr.Inputs.theta in
    { inputs with
      Dilqr.Inputs.theta =
        { Pendulum.Params.gravity =
            scalar64 (base_gravity +. (s *. item dt.Pendulum.Params.gravity))
        ; mass = theta.Pendulum.Params.mass
        ; length = theta.Pendulum.Params.length
        }
    ; x0 = Infix.(inputs.x0 + (d.Dilqr.Inputs.x0 *$ s))
    }
  in
  let jvp d =
    let tangent =
      snd
        (Rune.jvp
           structure
           Nx.Ptree.tensor
           (fun i -> solve_jvp ~max_iter:100 ~conv_threshold:1e-10 i)
           inputs
           d)
    in
    item (sum Infix.(ybar * tangent))
  in
  let eps = 1e-5 in
  let check ~msg d =
    let fd =
      (item (loss (solve (shift d eps))) -. item (loss (solve (shift d (-.eps)))))
      /. (2. *. eps)
    in
    let j = jvp d in
    printf "%s: jvp %g vs finite differences %g\n" msg j fd;
    require ~msg Float.(Float.abs (j -. fd) < 5e-3 +. (5e-3 *. Float.abs fd))
  in
  check ~msg:"gravity direction" (direction 1.0 (zeros_like inputs.x0));
  check ~msg:"initial state direction" (direction 0.0 (pattern ~seed:0.9 ~like:inputs.x0))

(* The streaming tangent solve and the tape-based one are the same map: the
   factors, the costate and the per-step tangents are the same values, and only
   their storage differs. *)
let test_streaming_jvp_matches_tape_jvp () =
  let module P = Pendulum in
  let inputs = pendulum_inputs () in
  let structure = Dilqr.inputs P.problem in
  let theta = inputs.Dilqr.Inputs.theta in
  let dtheta = (pattern_inputs ~seed:1.7 structure inputs).Dilqr.Inputs.theta in
  let dx0 = (pattern_inputs ~seed:2.9 structure inputs).Dilqr.Inputs.x0 in
  let sol = Dilqr.Solver.solve_fixed P.problem ~max_iters:6 theta inputs.x0 inputs.us0 in
  let streaming = Dilqr.Solver.tangent_solve P.problem theta ~dtheta ~dx0 sol.xs sol.us in
  let tape = Dilqr.Solver.blocks P.problem theta sol.xs sol.us in
  let tangent = Dilqr.Solver.tangent P.problem theta ~dtheta sol.xs sol.us in
  let tape_based =
    Dilqr.Lqr.solve_jvp
      ~common:(Dilqr.Lqr.backward_common tape)
      ~tape
      ~costate:(Dilqr.Lqr.costate tape)
      ~tangent
      ~dx0
      ~free_initial:false
  in
  check_close
    ~msg:"streaming jvp xs"
    ~tol:1e-10
    streaming.Dilqr.Lqr.xs
    tape_based.Dilqr.Lqr.xs;
  check_close
    ~msg:"streaming jvp us"
    ~tol:1e-10
    streaming.Dilqr.Lqr.us
    tape_based.Dilqr.Lqr.us

(* A batch of directions in one pass is the directions one at a time, and the
   compiled composition is the eager one. [vmap] over the tangent seeds with
   [jvp] inside is the batched forward-mode driver. *)
let test_batched_forward_mode () =
  let module P = Pendulum in
  let inputs = pendulum_inputs () in
  let structure = Dilqr.inputs P.problem in
  let dirs = pattern_lanes ~seed:0.41 ~k:3 structure inputs in
  let lanes =
    Rune.vmap
      Nx.Ptree.(structure @-> returns tensor)
      (fun di ->
         snd
           (Rune.jvp
              structure
              Nx.Ptree.tensor
              (fun i -> Dilqr.solve_fixed P.problem ~max_iters:4 i)
              inputs
              di))
      dirs
  in
  require
    ~msg:"one lane per direction"
    (Array.equal Int.equal (shape lanes) [| 3; P.horizon + 1; P.n + P.m |]);
  for i = 0 to 2 do
    let one (value : 'p) : 'p =
      (* pick lane [i] of every tensor leaf *)
      Nx.Ptree.map structure (fun _ leaf -> slice [ I i ] leaf) value
    in
    let dtaus =
      snd
        (Rune.jvp
           structure
           Nx.Ptree.tensor
           (fun j -> Dilqr.solve_fixed P.problem ~max_iters:4 j)
           inputs
           (one dirs))
    in
    check_close ~msg:(Printf.sprintf "lane %d" i) ~tol:1e-10 (slice [ I i ] lanes) dtaus
  done;
  let compiled =
    Rune.jit
      Nx.Ptree.(structure @-> returns tensor)
      (fun (dirs : 'p) ->
         Rune.vmap
           Nx.Ptree.(structure @-> returns tensor)
           (fun di ->
              snd
                (Rune.jvp
                   structure
                   Nx.Ptree.tensor
                   (fun i -> Dilqr.solve_fixed P.problem ~max_iters:4 i)
                   inputs
                   di))
           dirs)
      dirs
  in
  check_close ~msg:"compiled batched forward mode" ~tol:1e-10 compiled lanes

(* At a converged solution the local model is a fixed point: its feedforward
   gains vanish and its own solution is the zero deviation. That is the
   precondition of the implicit derivative in either mode, and it checks the
   Riccati factors, the costate and the LQR solve against each other. The solve
   is run far past what the fixed solve's own convergence test asks for, so
   that the gains are at round-off and not at the test's third-order
   tolerance. *)
let test_local_model_is_a_fixed_point () =
  let module P = Integrator in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { mass = scalar64 1.0 } in
  let x0 = vec64 [| 1.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let sol =
    Dilqr.Solver.solve_fixed P.problem ~max_iters:40 ~conv_threshold:1e-13 theta x0 us0
  in
  let tape = Dilqr.Solver.blocks P.problem theta sol.xs sol.us in
  let common = Dilqr.Lqr.backward_common tape in
  let gains, d1, _, _ =
    Dilqr.Lqr.backward_lin ~common ~tape ~lin:(Dilqr.Lqr.lin_of_tape tape)
  in
  let worst = item (max (abs gains.Dilqr.Lqr.feedforward)) in
  printf "worst feedforward gain at the solution: %g\n" worst;
  require ~msg:"feedforward gain vanishes" Float.(worst < 1e-6);
  require ~msg:"modelled decrease vanishes" Float.(Float.abs (item d1) < 1e-6);
  let (zerosol : (float, Nx.float64_elt, Nx.host) Dilqr.Lqr.solution) =
    Dilqr.Lqr.solve tape ~x0:(zeros float64 [| P.n |])
  in
  require
    ~msg:"the local model's solution is the zero deviation"
    Float.(item (Nx.max (Nx.abs zerosol.Dilqr.Lqr.xs)) < 1e-6);
  require
    ~msg:"and the zero control"
    Float.(item (Nx.max (Nx.abs zerosol.Dilqr.Lqr.us)) < 1e-6)

(* ──────────────────────────────────────────────────────────────────────────
   The value function at the horizon's start, and the initial condition it
   implies. The reference stacks the same quadratic program densely and solves
   its KKT system: a different algorithm on the same data, so it shares no code
   with the recursion it checks.
   ────────────────────────────────────────────────────────────────────────── *)

(* A dense host matrix, row-major. The reference's systems have the horizon
   times the state and control dimensions as their size: small enough that
   dense linear algebra is the simplest correct thing. *)
module Dense = struct
  type t =
    { dim : int
    ; data : float array
    }

  let make dim = { dim; data = Array.create ~len:(dim * dim) 0.0 }

  (* [add t ~row ~col ~rows ~cols blk] adds a block at a (row, col) offset. *)
  let add t ~row ~col ~rows ~cols (blk : float array) =
    for i = 0 to rows - 1 do
      for j = 0 to cols - 1 do
        let at = ((row + i) * t.dim) + col + j in
        t.data.(at) <- t.data.(at) +. blk.((i * cols) + j)
      done
    done

  let tensor t = create float64 [| t.dim; t.dim |] t.data
end

(* [transpose ~rows ~cols a] is the [cols × rows] transpose of a row-major
   [rows × cols] block. *)
let transpose ~rows ~cols (a : float array) =
  Array.init (rows * cols) ~f:(fun i ->
    let out_row = i / rows
    and out_col = Int.rem i rows in
    a.((out_col * cols) + out_row))

let identity_array k =
  Array.init (k * k) ~f:(fun i -> if Int.equal (i / k) (Int.rem i k) then 1.0 else 0.0)

(* A deterministic tape with every block populated: positive-definite
   quadratic blocks, a small dynamics, and non-zero linear terms on both the
   state and the control, so that the value gradient is not the costate. *)
let random_tape ~n ~m ~t_steps =
  let seed = ref 0.37 in
  let draw shape =
    seed := !seed +. 0.31;
    pattern ~seed:!seed ~like:(zeros float64 shape)
  in
  let psd shape =
    let dim = Array.get shape (Array.length shape - 1) in
    let m = draw shape in
    Infix.((m *@ matrix_transpose m /$ Float.of_int dim) + eye_at (beside m) float64 dim)
  in
  let blocks =
    { Dilqr.Lqr.Rows.a = Infix.(draw [| t_steps; n; n |] *$ 0.8)
    ; b = draw [| t_steps; n; m |]
    ; lx = draw [| t_steps; n |]
    ; lu = draw [| t_steps; m |]
    ; lxx = psd [| t_steps; n; n |]
    ; luu = psd [| t_steps; m; m |]
    ; lux = Infix.(draw [| t_steps; m; n |] *$ 0.5)
    }
  in
  ({ Dilqr.Lqr.blocks; fx = draw [| n |]; fxx = psd [| n; n |] }
   : (float, Nx.float64_elt, Nx.host) Dilqr.Lqr.tape)

(* The state at step 0 of the stacked program: free, or fixed to a deviation. *)
type anchor =
  | Free
  | Fixed of float array

(* The tape's program stacked densely. Variables are
   [z = (x_0 … x_T, u_0 … u_{T−1})] in deviation coordinates; the dynamics
   enter as the equalities [x_{k+1} − A_k x_k − B_k u_k = 0]. [prior] is the
   mean and precision of a Gaussian on the absolute state [nominal + δ], in
   deviation coordinates about [nominal]. Returns the minimizer and the
   objective value. *)
let stacked_program
      ~n
      ~m
      ~(tape : (float, Nx.float64_elt, Nx.host) Dilqr.Lqr.tape)
      ~prior
      ~nominal
      ~anchor
      ()
  =
  let t_steps = (shape tape.Dilqr.Lqr.blocks.a).(0) in
  let n_states = (t_steps + 1) * n in
  let nz = n_states + (t_steps * m) in
  let n_anchor =
    match anchor with
    | Free -> 0
    | Fixed _ -> n
  in
  let n_cons = (t_steps * n) + n_anchor in
  let dim = nz + n_cons in
  let h_mat = Dense.make nz in
  let h_vec = Array.create ~len:nz 0.0 in
  let cons = Array.create ~len:(n_cons * nz) 0.0 in
  let x k = k * n
  and u k = n_states + (k * m) in
  let step t k = to_array (slice [ I k ] t) in
  let add_cons row col rows cols (blk : float array) =
    for i = 0 to rows - 1 do
      for j = 0 to cols - 1 do
        let at = ((row + i) * nz) + col + j in
        cons.(at) <- cons.(at) +. blk.((i * cols) + j)
      done
    done
  in
  for k = 0 to t_steps - 1 do
    let lxx = step tape.Dilqr.Lqr.blocks.lxx k
    and luu = step tape.Dilqr.Lqr.blocks.luu k
    and lux = step tape.Dilqr.Lqr.blocks.lux k
    and lx = step tape.Dilqr.Lqr.blocks.lx k
    and lu = step tape.Dilqr.Lqr.blocks.lu k
    and a = step tape.Dilqr.Lqr.blocks.a k
    and b = step tape.Dilqr.Lqr.blocks.b k in
    (* [½ xᵀ lxx x + ½ uᵀ luu u + uᵀ lux x], so the cross block is [luxᵀ] on
       (x, u) and [lux] on (u, x). *)
    Dense.add h_mat ~row:(x k) ~col:(x k) ~rows:n ~cols:n lxx;
    Dense.add h_mat ~row:(u k) ~col:(u k) ~rows:m ~cols:m luu;
    Dense.add h_mat ~row:(x k) ~col:(u k) ~rows:n ~cols:m (transpose ~rows:m ~cols:n lux);
    Dense.add h_mat ~row:(u k) ~col:(x k) ~rows:m ~cols:n lux;
    Array.blit ~src:lx ~src_pos:0 ~dst:h_vec ~dst_pos:(x k) ~len:n;
    Array.blit ~src:lu ~src_pos:0 ~dst:h_vec ~dst_pos:(u k) ~len:m;
    (* [x_{k+1} − A_k x_k − B_k u_k = 0] *)
    let row = k * n in
    add_cons row (x k) n n (Array.map a ~f:Float.neg);
    add_cons row (x (k + 1)) n n (identity_array n);
    add_cons row (u k) n m (Array.map b ~f:Float.neg)
  done;
  Dense.add
    h_mat
    ~row:(x t_steps)
    ~col:(x t_steps)
    ~rows:n
    ~cols:n
    (to_array tape.Dilqr.Lqr.fxx);
  Array.blit
    ~src:(to_array tape.Dilqr.Lqr.fx)
    ~src_pos:0
    ~dst:h_vec
    ~dst_pos:(x t_steps)
    ~len:n;
  (match prior with
   | None -> ()
   | Some (prior_mean, prior_precision) ->
     (* The prior is on the absolute state, so its Hessian is its precision
        and its gradient at [δ = 0] is [Λ (nominal − μ)]. *)
     Dense.add h_mat ~row:0 ~col:0 ~rows:n ~cols:n prior_precision;
     for i = 0 to n - 1 do
       for j = 0 to n - 1 do
         h_vec.(i)
         <- h_vec.(i) +. (prior_precision.((i * n) + j) *. (nominal.(j) -. prior_mean.(j)))
       done
     done);
  (match anchor with
   | Free -> ()
   | Fixed _ -> add_cons (t_steps * n) 0 n n (identity_array n));
  (* The KKT system of [min ½ zᵀ H z + hᵀ z s.t. C z + d = 0] is
     [[H Cᵀ; C 0]] [z; λ] = [−h; −d]. *)
  let rhs = Array.create ~len:dim 0.0 in
  for i = 0 to nz - 1 do
    rhs.(i) <- -.h_vec.(i)
  done;
  (match anchor with
   | Free -> ()
   | Fixed delta ->
     for i = 0 to n - 1 do
       rhs.(nz + (t_steps * n) + i) <- delta.(i)
     done);
  let kkt = { Dense.dim; data = Array.create ~len:(dim * dim) 0.0 } in
  for i = 0 to nz - 1 do
    for j = 0 to nz - 1 do
      kkt.data.((i * dim) + j) <- h_mat.data.((i * nz) + j)
    done
  done;
  for i = 0 to n_cons - 1 do
    for j = 0 to nz - 1 do
      let c = cons.((i * nz) + j) in
      kkt.data.((j * dim) + nz + i) <- c;
      kkt.data.(((nz + i) * dim) + j) <- c
    done
  done;
  let zlam = to_array (Nx.solve (Dense.tensor kkt) (create float64 [| dim |] rhs)) in
  let z = Array.sub zlam ~pos:0 ~len:nz in
  let cost = ref 0.0 in
  for i = 0 to nz - 1 do
    for j = 0 to nz - 1 do
      cost := !cost +. (0.5 *. z.(i) *. h_mat.data.((i * nz) + j) *. z.(j))
    done;
    cost := !cost +. (h_vec.(i) *. z.(i))
  done;
  z, !cost

(* The value function of the local model at the horizon's start: the gradient
   and the Hessian of [J (δ)], the optimal cost of the program with the state
   at step 0 fixed to the deviation [δ], from central differences of the dense
   reference. [δ] is a deviation, so the nominal cancels. *)
let dense_value_function ~n ~m ~(tape : (float, Nx.float64_elt, Nx.host) Dilqr.Lqr.tape) =
  let value delta =
    let _, cost =
      stacked_program
        ~n
        ~m
        ~tape
        ~prior:None
        ~nominal:(Array.create ~len:n 0.0)
        ~anchor:(Fixed delta)
        ()
    in
    cost
  in
  (* [J] is exactly quadratic, so any step is exact and the step only trades
     roundoff against the [1 / h²] amplification. *)
  let h = 1e-2 in
  let shift i j si sj =
    Array.init n ~f:(fun k ->
      (if Int.equal k i then si else 0.0) +. if Int.equal k j then sj else 0.0)
  in
  let vx =
    Array.init n ~f:(fun i ->
      (value (shift i i h 0.0) -. value (shift i i (-.h) 0.0)) /. (2.0 *. h))
  in
  let vxx =
    Array.init (n * n) ~f:(fun idx ->
      let i = idx / n
      and j = Int.rem idx n in
      (value (shift i j h h)
       -. value (shift i j h (-.h))
       -. (value (shift i j (-.h) h) -. value (shift i j (-.h) (-.h))))
      /. (4.0 *. h *. h))
  in
  vx, vxx

(* [common.value_hessian] and [backward_lin]'s value gradient are the Hessian
   and the gradient of the local model's optimal value. The tape's control
   gradients do not vanish, so a value gradient taken for a costate — which
   differs by [K_0ᵀ Q_u,0] — would not pass. *)
let test_value_function_matches_dense () =
  let n = 3
  and m = 2
  and t_steps = 4 in
  let tape = random_tape ~n ~m ~t_steps in
  let common = Dilqr.Lqr.backward_common tape in
  let _, _, _, value_grad =
    Dilqr.Lqr.backward_lin ~common ~tape ~lin:(Dilqr.Lqr.lin_of_tape tape)
  in
  let vx, vxx = dense_value_function ~n ~m ~tape in
  check_close ~msg:"value gradient" ~tol:1e-7 (create float64 [| n |] vx) value_grad;
  check_close
    ~msg:"value Hessian"
    ~tol:1e-6
    (create float64 [| n * n |] vxx)
    (reshape [| n * n |] common.Dilqr.Lqr.value_hessian)

(* [Lqr.initial_condition] minimizes the local model's value plus a Gaussian
   prior over the state at step 0: the same program with [x_0] free, solved
   densely, for the mean, and the value Hessian the reference above measures
   plus the prior's precision for the precision.

   For a linear dynamics with Gaussian losses the program is the negative log
   joint of the Gaussian model up to a constant, so its minimizer is that
   model's exact posterior mean and [Λ + V] its exact marginal precision of
   [x_0] — what the information-form (Rauch-Tung-Striebel) smoother's backward
   pass computes. *)
let test_initial_condition_matches_dense () =
  let n = 3
  and m = 2
  and t_steps = 4 in
  let tape = random_tape ~n ~m ~t_steps in
  let prior_mean = pattern ~seed:2.3 ~like:(zeros float64 [| n |]) in
  let spread = pattern ~seed:3.7 ~like:(zeros float64 [| n; n |]) in
  let prior_precision =
    Infix.(
      (spread *@ matrix_transpose spread /$ Float.of_int n)
      + (eye float64 n *$ Float.of_int n))
  in
  let nominal = pattern ~seed:5.1 ~like:(zeros float64 [| n |]) in
  let nominal_array = to_array nominal in
  let mean_of_prior, precision_of_prior = to_array prior_mean, to_array prior_precision in
  (* the program with [x_0] free: its minimizer's first block is the mean *)
  let z, _ =
    stacked_program
      ~n
      ~m
      ~tape
      ~prior:(Some (mean_of_prior, precision_of_prior))
      ~nominal:nominal_array
      ~anchor:Free
      ()
  in
  let reference_mean =
    create float64 [| n |] (Array.init n ~f:(fun i -> nominal_array.(i) +. z.(i)))
  in
  let _, vxx = dense_value_function ~n ~m ~tape in
  let reference_precision =
    create float64 [| n; n |] (Array.mapi vxx ~f:(fun i v -> v +. precision_of_prior.(i)))
  in
  let common = Dilqr.Lqr.backward_common tape in
  let _, _, _, value_grad =
    Dilqr.Lqr.backward_lin ~common ~tape ~lin:(Dilqr.Lqr.lin_of_tape tape)
  in
  let initial =
    Dilqr.Lqr.initial_condition
      ~common
      ~value_grad
      ~nominal
      ~prior:(Some (prior_mean, prior_precision))
  in
  check_close ~msg:"initial mean" ~tol:1e-9 initial.Dilqr.Lqr.mean reference_mean;
  check_close
    ~msg:"initial precision"
    ~tol:1e-9
    initial.Dilqr.Lqr.precision
    reference_precision

(* ──────────────────────────────────────────────────────────────────────────
   The local model's Gaussian: the factorization the sampler and the density
   rest on, checked against the tape's own quadratic cost. The reference is a
   different route to the same number — the cost along an arbitrary trajectory
   rather than the Riccati recursion — so a wrong factor, a wrong gain or a
   missing term fails it.
   ────────────────────────────────────────────────────────────────────────── *)

let mv = Dilqr.Lqr.matvec

(* [local_cost tape ~dx0 us] is the tape's quadratic cost along the trajectory
   that starts at [dx0] under [us]: the deviation form, so the linear terms are
   the tape's own gradients. Exactly quadratic, so nothing below depends on a
   step size. *)
let local_cost (tape : (float, Nx.float64_elt, Nx.host) Dilqr.Lqr.tape) ~dx0 us =
  let horizon = (shape tape.Dilqr.Lqr.blocks.a).(0) in
  let xs = Array.create ~len:(horizon + 1) dx0 in
  for k = 0 to horizon - 1 do
    xs.(k + 1)
    <- Infix.(
         mv (get [ k ] tape.Dilqr.Lqr.blocks.a) xs.(k)
         + mv (get [ k ] tape.Dilqr.Lqr.blocks.b) (get [ k ] us))
  done;
  let total = ref 0.0 in
  for k = 0 to horizon - 1 do
    let x = xs.(k)
    and u = get [ k ] us in
    let step =
      Infix.(
        (sum (mv (get [ k ] tape.Dilqr.Lqr.blocks.lxx) x * x) *$ 0.5)
        + (sum (mv (get [ k ] tape.Dilqr.Lqr.blocks.luu) u * u) *$ 0.5)
        + sum (mv (get [ k ] tape.Dilqr.Lqr.blocks.lux) x * u)
        + sum (get [ k ] tape.Dilqr.Lqr.blocks.lx * x)
        + sum (get [ k ] tape.Dilqr.Lqr.blocks.lu * u))
    in
    total := !total +. item step
  done;
  let xk = xs.(horizon) in
  !total
  +. item
       Infix.((sum (mv tape.Dilqr.Lqr.fxx xk * xk) *$ 0.5) + sum (tape.Dilqr.Lqr.fx * xk))

(* [stacked ~x0 us] reads a trajectory as one vector: the initial state, then
   every control, the order the joint Hessian is indexed in. *)
let stacked ~x0 us =
  create
    float64
    [| (shape x0).(0) + numel us |]
    (Array.append (to_array x0) (to_array us))

(* The joint Hessian of the local model over [(x₀, u)], by central differences
   of its own cost. The cost is exactly quadratic, so any step is exact and the
   step only trades round-off against the [1 / h²] amplification. *)
let joint_hessian ~n ~m ~horizon tape =
  let dim = n + (horizon * m) in
  let value (z : float array) =
    let x0 = create float64 [| n |] (Array.sub z ~pos:0 ~len:n) in
    let us = create float64 [| horizon; m |] (Array.sub z ~pos:n ~len:(horizon * m)) in
    local_cost tape ~dx0:x0 us
  in
  let h = 1e-2 in
  let shift i j si sj =
    Array.init dim ~f:(fun k ->
      (if Int.equal k i then si else 0.0) +. if Int.equal k j then sj else 0.0)
  in
  create
    float64
    [| dim; dim |]
    (Array.init (dim * dim) ~f:(fun idx ->
       let i = idx / dim
       and j = Int.rem idx dim in
       (value (shift i j h h)
        -. value (shift i j h (-.h))
        -. (value (shift i j (-.h) h) -. value (shift i j (-.h) (-.h))))
       /. (4.0 *. h *. h)))

(* The identity the whole Gaussian rests on: around the model's minimizer the
   cost is the value quadratic in [x₀] plus one quadratic form per step in the
   innovation [(u_k − u⋆_k) − K_k (x_k − x⋆_k)]. *)
let test_laplace_factorization () =
  let n = 3
  and m = 2
  and horizon = 4 in
  let tape = random_tape ~n ~m ~t_steps:horizon in
  let nominal = Dilqr.Lqr.solve tape ~x0:(zeros float64 [| n |]) in
  let laplace = Dilqr.Lqr.Laplace.make tape ~nominal ~prior:None in
  let mode = Dilqr.Lqr.Laplace.mean ~tape laplace in
  (* the tape's cost is in deviations from the nominal, the mode is absolute *)
  let mode_x0 = Infix.(get [ 0 ] mode.Dilqr.Lqr.xs - get [ 0 ] nominal.Dilqr.Lqr.xs) in
  let mode_us = Infix.(mode.Dilqr.Lqr.us - nominal.Dilqr.Lqr.us) in
  let star = local_cost tape ~dx0:mode_x0 mode_us in
  let precision =
    (Option.value_exn laplace.Dilqr.Lqr.Laplace.initial).Dilqr.Lqr.precision
  in
  let check ~msg dx0 us =
    let lhs = local_cost tape ~dx0 us -. star in
    let d_x0 = Infix.(dx0 - mode_x0) in
    let d_us = Infix.(us - mode_us) in
    let d_xs = Array.create ~len:(horizon + 1) d_x0 in
    for k = 0 to horizon - 1 do
      d_xs.(k + 1)
      <- Infix.(
           mv (get [ k ] tape.Dilqr.Lqr.blocks.a) d_xs.(k)
           + mv (get [ k ] tape.Dilqr.Lqr.blocks.b) (get [ k ] d_us))
    done;
    let rhs = ref (0.5 *. item Infix.(sum (mv precision d_x0 * d_x0))) in
    for k = 0 to horizon - 1 do
      let w =
        Infix.(
          get [ k ] d_us - mv (get [ k ] laplace.Dilqr.Lqr.Laplace.feedback) d_xs.(k))
      in
      let z = mv (get [ k ] laplace.Dilqr.Lqr.Laplace.quu_chol) w in
      rhs := !rhs +. (0.5 *. item Infix.(sum (z * z)))
    done;
    check_close ~msg ~tol:1e-8 (scalar64 lhs) (scalar64 !rhs)
  in
  let draw seed shape =
    Nx.Rng.normal_at Nx.Host.on ~key:(Nx.Rng.key seed) float64 shape
  in
  List.iter [ 1; 2; 3 ] ~f:(fun seed ->
    check
      ~msg:"the innovations reproduce the cost difference"
      (draw seed [| n |])
      (draw (seed + 100) [| horizon; m |]))

(* The density: the innovations' product is the dense Gaussian of the joint
   Hessian, normalisation included. The log determinant is what a wrong factor
   or a missing block would move, so this is the test that says the density
   integrates to one. *)
let test_laplace_log_prob_matches_dense () =
  let n = 2
  and m = 1
  and horizon = 3 in
  let tape = random_tape ~n ~m ~t_steps:horizon in
  let nominal = Dilqr.Lqr.solve tape ~x0:(zeros float64 [| n |]) in
  let laplace = Dilqr.Lqr.Laplace.make tape ~nominal ~prior:None in
  let mode = Dilqr.Lqr.Laplace.mean ~tape laplace in
  let hessian = joint_hessian ~n ~m ~horizon tape in
  let dim = n + (horizon * m) in
  let _, log_det = Nx.slogdet float64 hessian in
  let log_det = item log_det in
  let check ~msg x0 us =
    let delta =
      Infix.(stacked ~x0 us - stacked ~x0:(get [ 0 ] mode.Dilqr.Lqr.xs) mode.Dilqr.Lqr.us)
    in
    let quadratic = item Infix.(sum (mv hessian delta * delta) *$ 0.5) in
    let reference =
      -.quadratic
      +. (0.5 *. log_det)
      -. (0.5 *. Float.of_int dim *. Float.log (2.0 *. Float.pi))
    in
    let got =
      item (Dilqr.Lqr.Laplace.log_prob ~tape laplace ~log_scale:(scalar64 0.0) ~x0 ~us)
    in
    check_close ~msg ~tol:1e-6 (scalar64 got) (scalar64 reference)
  in
  let draw seed shape =
    Nx.Rng.normal_at Nx.Host.on ~key:(Nx.Rng.key seed) float64 shape
  in
  List.iter [ 7; 8 ] ~f:(fun seed ->
    check
      ~msg:"the density is the dense Gaussian's"
      (draw seed [| n |])
      (draw (seed + 50) [| horizon; m |]))

(* The sampler: the draws' mean is the mode and their covariance the joint
   Hessian's inverse. Monte Carlo, so the tolerances are what the sample size
   buys; the identity above is what pins the exact statement. *)
let test_laplace_sample () =
  let n = 2
  and m = 1
  and horizon = 3
  and n_samples = 20_000 in
  let tape = random_tape ~n ~m ~t_steps:horizon in
  let nominal = Dilqr.Lqr.solve tape ~x0:(zeros float64 [| n |]) in
  let laplace = Dilqr.Lqr.Laplace.make tape ~nominal ~prior:None in
  let mode = Dilqr.Lqr.Laplace.mean ~tape laplace in
  let draws =
    Dilqr.Lqr.Laplace.sample
      ~tape
      laplace
      ~log_scale:(scalar64 0.0)
      ~key:(Nx.Rng.key 11)
      ~n_samples
  in
  let dim = n + (horizon * m) in
  let sample_x0 = get [ 0 ] draws.Dilqr.Lqr.xs in
  let sample_us = Nx.transpose ~axes:[ 1; 0; 2 ] draws.Dilqr.Lqr.us in
  let rows =
    Array.init n_samples ~f:(fun i ->
      to_array (stacked ~x0:(get [ i ] sample_x0) (get [ i ] sample_us)))
  in
  let zs = create float64 [| n_samples; dim |] (Array.concat (Array.to_list rows)) in
  let centre = stacked ~x0:(get [ 0 ] mode.Dilqr.Lqr.xs) mode.Dilqr.Lqr.us in
  let mean = Nx.mean ~axes:[ 0 ] zs in
  let centred = Infix.(zs - reshape [| 1; dim |] mean) in
  let covariance =
    Infix.(matrix_transpose centred *@ centred /$ Float.of_int n_samples)
  in
  let expected = Nx.inv (joint_hessian ~n ~m ~horizon tape) in
  let spread = item (Nx.sqrt (Nx.max (Nx.abs expected))) in
  check_close
    ~msg:"the draws' mean is the mode"
    ~tol:(6.0 *. spread /. Float.sqrt (Float.of_int n_samples))
    mean
    centre;
  check_close
    ~msg:"the draws' covariance is the Hessian's inverse"
    ~tol:0.06
    covariance
    expected

(* With the initial state fixed there is no factor over [x₀]: the Gaussian is
   over the controls alone, and it is the conditional law of the joint model
   given that state. Its density is the joint one with the initial block's own
   density taken out. *)
let test_laplace_fixed_initial () =
  let n = 2
  and m = 1
  and horizon = 3 in
  let tape = random_tape ~n ~m ~t_steps:horizon in
  let nominal = Dilqr.Lqr.solve tape ~x0:(zeros float64 [| n |]) in
  let free = Dilqr.Lqr.Laplace.make tape ~nominal ~prior:None in
  let fixed = Dilqr.Lqr.Laplace.make_fixed tape ~nominal in
  require
    ~msg:"a fixed initial state has no block"
    (Option.is_none fixed.Dilqr.Lqr.Laplace.initial);
  let x0 = get [ 0 ] nominal.Dilqr.Lqr.xs in
  let us = Nx.Rng.normal_at Nx.Host.on ~key:(Nx.Rng.key 13) float64 [| horizon; m |] in
  let log_prob t =
    item (Dilqr.Lqr.Laplace.log_prob ~tape t ~log_scale:(scalar64 0.0) ~x0 ~us)
  in
  let initial = Option.value_exn free.Dilqr.Lqr.Laplace.initial in
  let delta = Infix.(x0 - initial.Dilqr.Lqr.mean) in
  let upper = cholesky ~triangle:Upper initial.Dilqr.Lqr.precision in
  let whitened = Infix.(delta *@ matrix_transpose upper) in
  let initial_log_prob =
    -0.5
    *. (item Infix.(sum (whitened * whitened))
        -. (2.0 *. item (sum (log (diag upper))))
        +. (Float.of_int n *. Float.log (2.0 *. Float.pi)))
  in
  check_close
    ~msg:"the fixed density is the joint one without the initial block"
    ~tol:1e-8
    (scalar64 (log_prob fixed))
    (scalar64 (log_prob free -. initial_log_prob));
  let draws =
    Dilqr.Lqr.Laplace.sample
      ~tape
      fixed
      ~log_scale:(scalar64 0.0)
      ~key:(Nx.Rng.key 14)
      ~n_samples:4
  in
  check_close
    ~msg:"every draw starts at the nominal's state"
    ~tol:1e-12
    (get [ 0 ] draws.Dilqr.Lqr.xs)
    (broadcast_to [| 4; n |] x0)

(* A local model whose curvature is not positive definite has no Gaussian:
   the factor is NaN. Damping is the escape hatch — the Gaussian it gives is
   the damped model's, and it exists where the undamped one does not. *)
let test_laplace_damping () =
  let n = 2
  and m = 1
  and horizon = 2 in
  let i_n = eye float64 n in
  let blocks =
    { Dilqr.Lqr.Rows.a = broadcast_to [| horizon; n; n |] i_n
    ; b = ones float64 [| horizon; n; m |]
    ; lx = zeros float64 [| horizon; n |]
    ; lu = zeros float64 [| horizon; m |]
    ; lxx = zeros float64 [| horizon; n; n |]
    ; luu = broadcast_to [| horizon; m; m |] Infix.(eye float64 m *$ 0.5)
    ; lux = zeros float64 [| horizon; m; n |]
    }
  in
  let tape =
    ({ Dilqr.Lqr.blocks; fx = zeros float64 [| n |]; fxx = Infix.(i_n *$ -1.0) }
     : (float, Nx.float64_elt, Nx.host) Dilqr.Lqr.tape)
  in
  (* the nominal is any trajectory the tape can be linearized at; this model's
     own undamped solve is NaN, which is the point *)
  let nominal : (float, Nx.float64_elt, Nx.host) Dilqr.Lqr.solution =
    { xs = zeros float64 [| horizon + 1; n |]; us = zeros float64 [| horizon; m |] }
  in
  let finite t = item (all (isfinite t)) in
  let undamped = Dilqr.Lqr.Laplace.make tape ~nominal ~prior:None in
  let initial = Option.value_exn undamped.Dilqr.Lqr.Laplace.initial in
  require
    ~msg:"a non-convex model has a NaN precision"
    (not (Bool.equal (finite initial.Dilqr.Lqr.precision) true));
  let damped = Dilqr.Lqr.Laplace.make ~mu:(scalar64 2.0) tape ~nominal ~prior:None in
  let initial = Option.value_exn damped.Dilqr.Lqr.Laplace.initial in
  require
    ~msg:"damping gives a Gaussian"
    (Bool.equal (finite initial.Dilqr.Lqr.precision) true);
  let draws =
    Dilqr.Lqr.Laplace.sample
      ~tape
      damped
      ~log_scale:(scalar64 0.0)
      ~key:(Nx.Rng.key 17)
      ~n_samples:3
  in
  require ~msg:"and finite draws" (Bool.equal (finite draws.Dilqr.Lqr.us) true)

(* ──────────────────────────────────────────────────────────────────────────
   A free initial state: the problem prices x₀ in its step-0 running loss, so
   the solve optimizes it jointly with the controls. The cost is quadratic and
   the dynamics linear, so the joint mode is exact and the derivative is the
   one the implicit function theorem gives — finite differences of a converged
   solve are the reference for both.
   ────────────────────────────────────────────────────────────────────────── *)

module Anchored = struct
  module Params = struct
    type ('a, _) t = { gain : 'a } [@@deriving ptree]
  end

  let n = 2
  let m = 1
  let horizon = 3
  let verbose = false

  (* The anchor: a Gaussian on the initial state, as the step-0 running loss.
     It is what makes the block positive definite and the mode unique. *)
  let anchor_mean = vec64 [| 1.0; -0.5 |]
  let anchor_precision = 2.0
  let target = vec64 [| 0.5; 0.0 |]

  let dynamics { Params.gain } ~k:_ ~x ~u =
    let open Infix in
    let x0 = slice [ R (0, 1) ] x
    and x1 = slice [ R (1, 2) ] x in
    concatenate ~axis:0 [ (x0 *$ 0.9) + (x1 *$ 0.1); (x1 *$ 0.8) + (u * gain) ]

  let running_loss _ ~k ~x ~u =
    let open Infix in
    let d = x - target in
    let anchor = x - anchor_mean in
    let zero = scalar_like k 0.0 in
    let anchored =
      where (Nx.equal k zero) (sum (anchor * anchor) *$ (0.5 *. anchor_precision)) zero
    in
    (sum (d * d) *$ 0.5) + (sum (u * u) *$ 0.5) + anchored

  let final_loss _ ~x =
    let open Infix in
    let d = x - target in
    sum (d * d)

  let problem
    : (Nx.float64_elt, (Nx.host Nx.float64_t, Nx.host) Params.t, Nx.host) Dilqr.Problem.t
    =
    { dtype = float64
    ; n
    ; m
    ; horizon
    ; free_initial = true
    ; verbose
    ; params = Nx.Ptree.instantiate (module Params)
    ; dynamics
    ; running_loss
    ; final_loss
    }
end

(* The problem's own cost at [(x₀, u)], stacked as one vector, along the true
   dynamics. *)
let anchored_cost theta (z : float array) =
  let x0 = create float64 [| Anchored.n |] (Array.sub z ~pos:0 ~len:Anchored.n) in
  let us =
    Array.init Anchored.horizon ~f:(fun k ->
      create float64 [| Anchored.m |] (Array.sub z ~pos:(Anchored.n + k) ~len:Anchored.m))
  in
  item (snd (Dilqr.Solver.rollout Anchored.problem theta x0 us))

(* The solve moves the initial state, and the cost is stationary where it lands:
   that is what "the solve optimizes x₀" means. A pinned solve of the same
   problem keeps the state it was given. *)
let test_free_initial_is_solved_for () =
  let module P = Anchored in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t = { gain = scalar64 1.0 } in
  let x0 = vec64 [| 0.0; 0.0 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.0 ])) in
  let solve = Dilqr.solve_fixed P.problem ~max_iters:40 ~conv_threshold:1e-14 in
  let xs, us = Dilqr.split_taus P.problem (solve { Dilqr.Inputs.theta; x0; us0 }) in
  let solved_x0 = xs.(0) in
  require
    ~msg:"the solve moved the initial state"
    Float.(item (Nx.max (Nx.abs Infix.(solved_x0 - x0))) > 1e-2);
  let z =
    Array.append (to_array solved_x0) (to_array (stack ~axis:0 (Array.to_list us)))
  in
  let value t = anchored_cost theta (to_array t) in
  let grad = fd_grad ~eps:1e-6 value (create float64 [| Array.length z |] z) in
  require
    ~msg:"the joint mode is stationary in the initial state and the controls"
    Float.(item (Nx.max (Nx.abs grad)) < 1e-5);
  let pinned = { P.problem with free_initial = false } in
  let xs, _ =
    Dilqr.split_taus pinned (Dilqr.solve_fixed pinned ~max_iters:8 { theta; x0; us0 })
  in
  check_close ~msg:"a pinned solve keeps the state it was given" ~tol:1e-12 xs.(0) x0

(* The implicit derivative of a free-initial solve, in both modes: reverse mode
   against central differences of a converged solve, and the transpose law,
   which also says that the input's initial state reaches nothing. *)
let test_free_initial_derivative () =
  let module P = Anchored in
  let theta gain : (Nx.host Nx.float64_t, Nx.host) P.Params.t =
    { gain = scalar64 gain }
  in
  let x0 = vec64 [| 0.2; 0.1 |] in
  let us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.05 ])) in
  let structure = Dilqr.inputs P.problem in
  let inputs = { Dilqr.Inputs.theta = theta 1.0; x0; us0 } in
  let solve = Dilqr.solve_fixed P.problem ~max_iters:40 ~conv_threshold:1e-14 in
  let ybar = pattern ~seed:0.4 ~like:(zeros float64 [| P.horizon + 1; P.n + P.m |]) in
  let objective i = sum Infix.(ybar * solve i) in
  let grads = Rune.grad structure objective inputs in
  let eps = 1e-6 in
  let fd =
    (item (objective { inputs with theta = theta (1.0 +. eps) })
     -. item (objective { inputs with theta = theta (1.0 -. eps) }))
    /. (2.0 *. eps)
  in
  let got = item grads.Dilqr.Inputs.theta.P.Params.gain in
  require
    ~msg:(Printf.sprintf "implicit gradient %g vs finite differences %g" got fd)
    Float.(Float.abs (got -. fd) <= 1e-5 +. (1e-4 *. Float.abs fd));
  check_close
    ~msg:"the initial state is a starting point, not an input"
    ~tol:1e-12
    grads.Dilqr.Inputs.x0
    (zeros float64 [| P.n |]);
  let dinputs = pattern_inputs ~seed:1.3 structure inputs in
  let dtaus =
    snd (Rune.jvp structure Nx.Ptree.tensor (fun i -> solve i) inputs dinputs)
  in
  check_close
    ~msg:"<grad, v> = <cotangent, JVP v>"
    ~tol:1e-8
    (Nx.Ptree.dot structure float64 grads dinputs)
    (sum Infix.(ybar * dtaus));
  (* a tangent of the input's initial state moves nothing *)
  let d0 = { dinputs with Dilqr.Inputs.x0 = pattern ~seed:2.7 ~like:x0 } in
  check_close
    ~msg:"the input's initial state tangent reaches nothing"
    ~tol:1e-10
    (snd (Rune.jvp structure Nx.Ptree.tensor (fun i -> solve i) inputs d0))
    dtaus

(* The batched free solve: one solve for two members, and each member's
   gradient the one its own solve gives — the batch's layout, the initial
   state's step and the implicit VJP included. *)
let test_free_initial_batched () =
  let module P = Anchored in
  let b = 2 in
  let solve = Dilqr.solve_fixed P.problem ~max_iters:40 ~conv_threshold:1e-14 in
  let objective i = sum (Dilqr.loss_of_taus P.problem i.Dilqr.Inputs.theta (solve i)) in
  let structure = Dilqr.inputs P.problem in
  let member i
    : ( (Nx.host Nx.float64_t, Nx.host) P.Params.t
        , (float, Nx.float64_elt, Nx.host) Nx.t )
        Dilqr.Inputs.t
    =
    { theta = { P.Params.gain = scalar64 (1.0 +. (0.2 *. Float.of_int i)) }
    ; x0 = vec64 [| 0.1 *. Float.of_int (i + 1); -0.2 |]
    ; us0 = us_of (List.init P.horizon ~f:(fun _ -> [ 0.01 *. Float.of_int (i + 1) ]))
    }
  in
  let batched
    : ( (Nx.host Nx.float64_t, Nx.host) P.Params.t
        , (float, Nx.float64_elt, Nx.host) Nx.t )
        Dilqr.Inputs.t
    =
    { theta =
        { P.Params.gain = stack ~axis:0 (List.init b ~f:(fun i -> (member i).theta.gain))
        }
    ; x0 = stack ~axis:0 (List.init b ~f:(fun i -> (member i).x0))
    ; us0 =
        Array.init P.horizon ~f:(fun k ->
          stack ~axis:0 (List.init b ~f:(fun i -> (member i).Dilqr.Inputs.us0.(k))))
    }
  in
  let g = Rune.grad structure objective batched in
  for i = 0 to b - 1 do
    let g_i = Rune.grad structure objective (member i) in
    check_close
      ~msg:(Printf.sprintf "batched free grad, member %d" i)
      ~tol:1e-9
      (slice [ I i ] g.Dilqr.Inputs.theta.P.Params.gain)
      g_i.Dilqr.Inputs.theta.P.Params.gain;
    check_close
      ~msg:(Printf.sprintf "batched free grad x0, member %d" i)
      ~tol:1e-9
      (slice [ I i ] g.Dilqr.Inputs.x0)
      g_i.Dilqr.Inputs.x0
  done

(* [Dilqr.inputs] is a hand-written walk over the problem's parameter
   structure: its visits are the parameters under [theta], the initial state,
   then the controls' count and each control at its index. *)
let test_inputs_structure () =
  let problem = { Integrator.problem with horizon = 3 } in
  let inputs
    : ( (Nx.host Nx.float64_t, Nx.host) Integrator.Params.t
        , Nx.host Nx.float64_t )
        Dilqr.Inputs.t
    =
    { theta = { mass = scalar64 1.0 }
    ; x0 = vec64 [| 1.0; 0.0 |]
    ; us0 = us_of [ [ 0.0 ]; [ 0.0 ]; [ 0.0 ] ]
    }
  in
  let visits =
    String.concat
      (List.map
         (Nx.Ptree.visits (Dilqr.inputs problem) inputs)
         ~f:(fun v -> Stdlib.Format.asprintf "%a\n" Nx.Ptree.pp_visit v))
  in
  expect visits
  @@ Stdlib.__POS_OF__
       {|
    theta.mass: a leaf
    x0: a leaf
    us0: int 3
    us0.0: a leaf
    us0.1: a leaf
    us0.2: a leaf
    |}

let () =
  Stdlib.exit
    (run
       "dilqr"
       [ test "regularisation schedule" test_regularisation
       ; test "a failed factorization is NaN" test_a_failed_factorization_is_nan
       ; test "costate is the hand recursion" test_costate
       ; test "double integrator converges" test_integrator_converges
       ; test "pendulum converges" test_pendulum_converges
       ; test
           "implicit gradient matches finite differences"
           test_implicit_gradient_matches_fd
       ; test "fixed solve is competitive" test_fixed_solve_works
       ; test
           "fixed solve keeps the trajectory when all trials are rejected"
           test_fixed_all_rejected
       ; test "the budget is the batch's, and no veto" test_fixed_budget_and_no_veto
       ; test
           "the batch spends the slowest member's iterations"
           test_fixed_budget_is_the_slowest_member
       ; test "adaptive damping recovers an indefinite Quu" test_fixed_damping_recovers
       ; test
           "pinned damping fails on an indefinite Quu"
           test_adaptive_pinned_damping_fails
       ; test "a zero budget runs nothing" test_fixed_zero_budget
       ; test "the adaptive solve reports its iterations" test_adaptive_reports_iterations
       ; test "damping is a value-Hessian shift" test_damping_is_a_value_hessian_shift
       ; test "jit solve matches eager" test_jit_solve
       ; test "jit grad matches eager" test_jit_grad
       ; test "batched fixed solve matches unbatched" test_batched_fixed_solve
       ; test "batched gradient matches unbatched" test_batched_grad
       ; test "adaptive solve rejects batches" test_adaptive_rejects_batches
       ; test "compiled adaptive solve matches eager" test_compiled_adaptive_solve
       ; test "compiled adaptive pendulum matches eager" test_compiled_adaptive_pendulum
       ; test
           "differentiable adaptive compiles the same solve"
           test_solve_adaptive_compiled
       ; test
           "forward mode is the transpose of the reverse mode"
           test_jvp_is_the_transpose_of_the_vjp
       ; test
           "forward mode matches finite differences"
           test_jvp_matches_finite_differences
       ; test
           "the streaming tangent solve is the tape one"
           test_streaming_jvp_matches_tape_jvp
       ; test "batched forward mode" test_batched_forward_mode
       ; test "the local model is a fixed point" test_local_model_is_a_fixed_point
       ; test "value function matches dense" test_value_function_matches_dense
       ; test "initial condition matches dense" test_initial_condition_matches_dense
       ; test "the local model's Gaussian factorizes" test_laplace_factorization
       ; test
           "the local model's density is the dense Gaussian's"
           test_laplace_log_prob_matches_dense
       ; test "the local model's sampler matches dense" test_laplace_sample
       ; test "a fixed initial state has no block" test_laplace_fixed_initial
       ; test "damping is the escape hatch" test_laplace_damping
       ; test "a free initial state is solved for" test_free_initial_is_solved_for
       ; test "a free initial state's derivative" test_free_initial_derivative
       ; test "a batched free initial state" test_free_initial_batched
       ; test "inputs structure" test_inputs_structure
       ])
