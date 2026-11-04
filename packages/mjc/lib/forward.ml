(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let forward ?linear (m : Model.host) (d : Data.host) : Data.host =
  let d = Kinematics.forward m d in
  let d = Mass.forward m d in
  let d = Passive.forward m d in
  let d = Bias.forward m d in
  let qfrc_smooth =
    Nx.add (Nx.sub (Nx.add d.qfrc_passive d.qfrc_applied) d.qfrc_bias) d.qfrc_actuator
  in
  (* The smooth acceleration, then the constraint rows and their solve. With
     no constraints the solver leaves qacc and qfrc_constraint as they are. *)
  let qacc = Mass.solve (Mass.factor d.mass_matrix) qfrc_smooth in
  let d = { d with qfrc_smooth; qacc } in
  (* A model with no equality, no limit and no friction loss can never produce
     a row: skipping the constraint stages keeps its step on the smooth path
     alone, which is most of the corpus. *)
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
  let constrained = t.neq > 0 || limits_possible || friction_possible in
  if not constrained
  then d
  else (
    let d = Constraint.equality m d in
    let d = Constraint.friction m d in
    let d = Constraint.limits m d in
    let d = Constraint.impedance m d in
    Solver.forward ?linear m d)
