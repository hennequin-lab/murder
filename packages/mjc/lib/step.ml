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
   with or without damping, and adds the velocity derivative of the passive
   forces to the matrix. v1 has no fluid and no actuator damping, so for the
   models it supports the two schemes solve the same system and differ only
   in whether the solve is skipped. *)

let advance_with ~topo ~dt ~solve ~diag (d : Data.host) : Data.host =
  let qacc =
    if not solve
    then d.qacc
    else (
      let h = Nx.add d.mass_matrix (Nx.mul_s diag dt) in
      Mass.solve (Mass.factor h) (Nx.add d.qfrc_smooth d.qfrc_constraint))
  in
  let qvel = Nx.add d.qvel (Nx.mul_s qacc dt) in
  let qpos = Pose.integrate_pos topo ~qpos:d.qpos ~qvel ~dt in
  { d with time = Nx.add_s d.time dt; qvel; qpos }

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

let advance (m : Model.host) (d : Data.host) : Data.host =
  let dt, solve, diag = config m in
  advance_with ~topo:m.topo ~dt ~solve ~diag d

let step m d = advance m (Forward.forward m d)

type t =
  { dt : float
  ; run : Data.host -> Data.host
  }

let compile (m : Model.host) : t =
  let dt, solve, diag = config m in
  let f d = advance_with ~topo:m.topo ~dt ~solve ~diag (Forward.forward m d) in
  { dt; run = Rune.jit Nx.Ptree.(Data.ptree @-> returns Data.ptree) f }

let run (t : t) d = t.run d
let dt t = t.dt
