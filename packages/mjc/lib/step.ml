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
   acceleration, which the model's own acceleration already is. *)

let advance_with ~topo ~dt ~damped ~diag (d : Data.host) : Data.host =
  let qacc =
    if not damped
    then d.qacc
    else (
      let h = Nx.add d.mass_matrix (Nx.mul_s diag dt) in
      Mass.solve (Mass.factor h) (Nx.add d.qfrc_smooth d.qfrc_constraint))
  in
  let qvel = Nx.add d.qvel (Nx.mul_s qacc dt) in
  let qpos = Pose.integrate_pos topo ~qpos:d.qpos ~qvel ~dt in
  { d with time = Nx.add_s d.time dt; qvel; qpos }

(* The model's two host values, read once per model. *)
let config (m : Model.host) =
  let dt = Nx.item [] m.params.timestep in
  let damped =
    Stdlib.Array.exists
      (fun b -> not (Float.equal b 0.0))
      (Nx.to_array m.params.dof_damping)
  in
  let diag = Nx.diag m.params.dof_damping in
  dt, damped, diag

let advance (m : Model.host) (d : Data.host) : Data.host =
  let dt, damped, diag = config m in
  advance_with ~topo:m.topo ~dt ~damped ~diag d

let step m d = advance m (Forward.forward m d)

type t =
  { dt : float
  ; run : Data.host -> Data.host
  }

let compile (m : Model.host) : t =
  let dt, damped, diag = config m in
  let f d = advance_with ~topo:m.topo ~dt ~damped ~diag (Forward.forward m d) in
  { dt; run = Rune.jit Nx.Ptree.(Data.ptree @-> returns Data.ptree) f }

let run (t : t) d = t.run d
let dt t = t.dt
