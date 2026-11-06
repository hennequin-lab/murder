(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base

(* MuJoCo's [mj_advance] is semi-implicit: the velocity takes the step with the
   acceleration the forward pass computed, then the position takes the step
   with the new velocity, then the clock advances.

   MuJoCo's [mj_Euler] integrates joint damping implicitly when a model has
   any: the velocity update solves [M + h diag(B)] instead of [M], so a large
   damping cannot destabilise the step. With no damping it uses the forward
   acceleration, which the model's own acceleration already is.

   [implicitfast] ([mj_implicitSkip]) always solves that same damped system,
   with or without damping, and puts the whole velocity derivative of the
   smooth forces in the matrix: the dof damping, and the actuators' own
   velocity derivative, which is what makes a velocity servo's [kv] a damping
   term rather than a force the step has to chase. v1 has no fluid and no
   muscle, so those are the two terms. *)

let advance_with ~topo ~dt ~solve ~(h : 'a Nx.float64_t) (d : 'a Data.data) : 'a Data.data
  =
  let qacc =
    if not solve
    then d.qacc
    else Mass.solve (Mass.factor h) (Nx.add d.qfrc_smooth d.qfrc_constraint)
  in
  let qvel = Nx.add d.qvel (Nx.mul_s qacc dt) in
  let qpos = Pose.integrate_pos topo ~qpos:d.qpos ~qvel ~dt in
  (* MuJoCo advances the activation states with the same explicit Euler step,
     whatever the integrator: [act += h act'] ([mj_Euler]). *)
  let act = Nx.add d.act (Nx.mul_s d.act_dot dt) in
  { d with time = Nx.add_s d.time dt; qvel; qpos; act }

(* Everything the step decides about a model, decided once and as host values:
   the timestep, whether the velocity update goes through the damped matrix
   (Euler only when some dof is damped, implicitfast always), the damping
   diagonal, the solver's iteration budget, whether any actuator is force
   limited, and whether any actuator puts a velocity term in the matrix.

   These are read from parameter tensors, which a *differentiated* model passes
   in as tensors: reading one inside the traced program raises, and it would be
   wrong even if it did not, because a budget and an integrator scheme are
   structure, not numbers to differentiate. Reading them here, once, is what
   lets {!dynamics} differentiate the step with respect to a mass. *)
type config =
  { dt : float
  ; solve : bool
  ; iters : int
  ; force_limited : bool
  ; actuator_damped : bool
  }

let config ?iters (m : Nx.host Model.model) : config =
  let p = m.params in
  let dt = Nx.item [] p.timestep in
  let damping = Nx.to_array p.dof_damping in
  let damped = Stdlib.Array.exists (fun b -> not (Float.equal b 0.0)) damping in
  let solve = damped || Poly.( = ) m.topo.integrator Topo.Implicit_fast in
  let iters =
    match iters with
    | Some n -> Stdlib.max 1 (Stdlib.min n Solver.max_iterations)
    | None ->
      Stdlib.min (Stdlib.int_of_float (Nx.item [] p.iterations)) Solver.max_iterations
  in
  let force_limited =
    Stdlib.Array.exists
      (fun x -> not (Float.equal x 0.0))
      (Nx.to_array p.actuator_forcelimited)
  in
  (* Conservative on purpose: a coefficient that is not needed is a rank-one
     update of zero, which costs a little and changes nothing, while a
     coefficient that is needed and skipped is a velocity servo that rings. *)
  let actuator_damped =
    force_limited
    || Stdlib.Array.exists
         (fun x -> not (Float.equal x 0.0))
         (Nx.to_array (Actuator.damping ~force_limited:false m (Data.create m)))
  in
  { dt; solve; iters; force_limited; actuator_damped }

(* [M - dt * (d qfrc_smooth / d qvel)]: the matrix the implicit step solves.
   The actuator term is a sum of rank-one updates, one per actuator, and it is
   skipped entirely when no actuator has a velocity derivative — which is most
   models, and keeps their step on the same graph. *)
let step_matrix (c : config) (m : 'a Model.model) (d : 'a Data.data) =
  let dt = c.dt in
  let h = Nx.add d.mass_matrix (Nx.mul_s (Nx.diag m.params.dof_damping) dt) in
  if not c.actuator_damped
  then h
  else (
    let k = Actuator.damping ~force_limited:c.force_limited m d in
    let tr = Transmission.forward m d in
    let weighted = Nx.mul (Nx.reshape [| m.topo.nu; 1 |] k) tr.Transmission.moment in
    let update = Nx.matmul (Nx.matrix_transpose tr.Transmission.moment) weighted in
    Nx.sub h (Nx.mul_s update dt))

(* [step_with c m d] is one step for a model whose *structure* is [c] and whose
   *parameters* are [m]'s, read as tensors. Nothing here reads a parameter with
   [Nx.item] or [Nx.to_array]: a parameter a differentiation tracks cannot be
   read, and this is the function a parameter gradient goes through. [m] may
   differ between calls — that is what differentiating a mass means — while [c]
   stays. *)
let config_dt (c : config) = c.dt

let step_with (c : config) (m : 'a Model.model) (d : 'a Data.data) : 'a Data.data =
  let d = Forward.forward ~iters:c.iters ~dt:c.dt m d in
  let h = if c.solve then step_matrix c m d else d.mass_matrix in
  advance_with ~topo:m.topo ~dt:c.dt ~solve:c.solve ~h d

(* [advance m d] advances [d] from a forward pass that has already run, which is
   what a caller comparing accelerations before and positions after wants. *)
let advance (m : Nx.host Model.model) (d : Nx.host Data.data) =
  let c = config m in
  let h = if c.solve then step_matrix c m d else d.mass_matrix in
  advance_with ~topo:m.topo ~dt:c.dt ~solve:c.solve ~h d

(* The eager step: the structural decisions are read here, per call, which is
   fine because an eager caller's parameters are values. *)
let step (m : Nx.host Model.model) (d : Nx.host Data.data) =
  advance m (Forward.forward m d)

type t =
  { dt : float
  ; run : Nx.host Data.data -> Nx.host Data.data
  }

let compile (m : 'a Model.model) : t =
  let c = config m in
  { dt = c.dt
  ; run =
      Rune.jit
        (Nx.Ptree.( @-> ) (Data.ptree ()) (Nx.Ptree.returns (Data.ptree ())))
        (step_with c m)
  }

let run (t : t) d = t.run d
let dt t = t.dt
