(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The time step is a model constant, read on the host. [M3]'s [Step.compile]
   reads it once, which is also what makes the step traceable. *)

(* MuJoCo's [mj_Euler] integrates joint damping implicitly when a model has
   any: the velocity update solves [M + h diag(B)] instead of [M], so a large
   damping cannot destabilise the step. With no damping it uses the forward
   acceleration, which is what the model's own acceleration already is. *)
let advance (m : Model.host) (d : Data.host) : Data.host =
  let dt = Nx.item [] m.params.timestep in
  let damped =
    Stdlib.Array.exists (fun b -> not (b = 0.0)) (Nx.to_array m.params.dof_damping)
  in
  let qacc =
    if not damped
    then d.qacc
    else (
      let h = Nx.add d.mass_matrix (Nx.mul_s (Nx.diag m.params.dof_damping) dt) in
      Mass.solve (Mass.factor h) (Nx.add d.qfrc_smooth d.qfrc_constraint))
  in
  let qvel = Nx.add d.qvel (Nx.mul_s qacc dt) in
  let qpos = Pose.integrate_pos m.topo ~qpos:d.qpos ~qvel ~dt in
  { d with time = Nx.add_s d.time dt; qvel; qpos }

let step m d = advance m (Forward.forward m d)
