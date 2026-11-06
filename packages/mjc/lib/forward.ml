(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let forward ?linear ?iters ?dt (m : Model.host) (d : Data.host) : Data.host =
  let d = Kinematics.forward m d in
  let d = Mass.forward m d in
  (* The contact row's Jacobian is a dof-chain walk, so contact detection needs
     the motion subspaces the mass stage computes. *)
  let d = Collision.contacts m d in
  let d = Passive.forward m d in
  let d = Bias.forward m d in
  let d = Actuator.forward m d in
  let tendons = Tendon.forward m d in
  let d =
    { d with
      Data.ten_length = tendons.Tendon.length
    ; ten_velocity = tendons.Tendon.velocity
    }
  in
  let qfrc_smooth =
    Nx.add (Nx.sub (Nx.add d.qfrc_passive d.qfrc_applied) d.qfrc_bias) d.qfrc_actuator
  in
  (* The smooth acceleration, then the constraint rows and their solve. With
     no constraints the solver leaves qacc and qfrc_constraint as they are. *)
  let qacc = Mass.solve (Mass.factor d.mass_matrix) qfrc_smooth in
  let d = { d with qfrc_smooth; qacc } in
  (* A model with no equality, no limit, no friction loss and no pair of geoms
     can never produce a row: skipping the constraint stages keeps its step on
     the smooth path alone, which is most of the corpus. *)
  let t = m.topo in
  let limits_possible = ref false in
  Array.iteri
    (fun j ty ->
       match ty with
       | Topo.Hinge | Topo.Slide -> if t.jnt_limited.(j) then limits_possible := true
       | Topo.Ball | Topo.Free -> ())
    t.jnt_type;
  let friction_possible = ref false in
  Array.iter (fun x -> if x > 0.0 then friction_possible := true) t.dof_frictionloss;
  let limits_possible = !limits_possible
  and friction_possible = !friction_possible in
  let tendon_limits_possible = ref false in
  Array.iter (fun x -> if x then tendon_limits_possible := true) t.ten_limited;
  let tendon_limits_possible = !tendon_limits_possible in
  let constrained =
    t.neq > 0
    || limits_possible
    || tendon_limits_possible
    || friction_possible
    || (t.contacts_enabled && t.ngeom > 1)
  in
  let d =
    if not constrained
    then d
    else (
      let cfg = Constraint.config m in
      let d = Constraint.equality cfg m d in
      let d = Constraint.friction cfg m d in
      let d = Constraint.limits m d in
      let d = Constraint.tendon_limits m d in
      let d = Constraint.contact cfg m d in
      let d = Constraint.impedance ?dt cfg m d in
      Solver.forward ?linear ?iters m d)
  in
  (* The sensor block reads the solved state, as MuJoCo's does. *)
  Sensor.forward m d
