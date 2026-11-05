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

let advance_with ~topo ~dt ~solve ~(h : Nx.host Nx.float64_t) (d : Data.host) : Data.host =
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

(* The model's host values, read once per model. [solve] is whether the
   velocity update goes through the damped matrix: Euler only when some dof is
   damped, implicitfast always. *)
let config (m : Model.host) =
  let dt = Nx.item [] m.params.timestep in
  let damped =
    Stdlib.Array.exists
      (fun b -> not (Float.equal b 0.0))
      (Nx.to_array m.params.dof_damping)
  in
  let solve = damped || Poly.( = ) m.topo.integrator Topo.Implicit_fast in
  let diag = Nx.diag m.params.dof_damping in
  dt, solve, diag

(* [M - dt * (d qfrc_smooth / d qvel)]: the matrix the implicit step solves.
   The actuator term is a sum of rank-one updates, one per actuator, and it is
   skipped entirely when no actuator has a velocity derivative — which is most
   models, and keeps their step on the same graph. *)
let step_matrix (m : Model.host) (d : Data.host) ~dt ~diag =
  let h = Nx.add d.mass_matrix (Nx.mul_s diag dt) in
  let k = Actuator.damping m d in
  let any = Stdlib.Array.exists (fun x -> not (Float.equal x 0.0)) (Nx.to_array k) in
  if not any
  then h
  else (
    let tr = Transmission.forward m d in
    let weighted = Nx.mul (Nx.reshape [| m.topo.nu; 1 |] k) tr.Transmission.moment in
    let update = Nx.matmul (Nx.matrix_transpose tr.Transmission.moment) weighted in
    Nx.sub h (Nx.mul_s update dt))

let advance (m : Model.host) (d : Data.host) : Data.host =
  let dt, solve, diag = config m in
  let h = if solve then step_matrix m d ~dt ~diag else d.mass_matrix in
  advance_with ~topo:m.topo ~dt ~solve ~h d

let step m d = advance m (Forward.forward m d)

type t =
  { dt : float
  ; run : Data.host -> Data.host
  }

let compile (m : Model.host) : t =
  let dt, solve, diag = config m in
  let f d =
    let d = Forward.forward m d in
    let h = if solve then step_matrix m d ~dt ~diag else d.mass_matrix in
    advance_with ~topo:m.topo ~dt ~solve ~h d
  in
  { dt; run = Rune.jit Nx.Ptree.(Data.ptree @-> returns Data.ptree) f }

let run (t : t) d = t.run d
let dt t = t.dt
