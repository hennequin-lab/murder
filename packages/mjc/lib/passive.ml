(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Topo

let forward (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  let spring = ref (Nx.zeros_like d.qvel) in
  for j = 0 to t.njnt - 1 do
    match t.jnt_type.(j) with
    | Hinge | Slide ->
      let qa = t.jnt_qposadr.(j)
      and va = t.jnt_dofadr.(j) in
      let x = Nx.sub (Nx.get [ qa ] d.qpos) (Nx.get [ qa ] p.qpos_spring) in
      let k = Nx.get [ j ] p.jnt_stiffness in
      (* No host branch on [k]: a zero stiffness must stay a traced zero. *)
      let force = Nx.mul_s (Nx.mul x k) (-1.0) in
      spring := Nx.set [ I va ] (Nx.add (Nx.get [ va ] !spring) force) !spring
    | Ball | Free -> ()
  done;
  { d with qfrc_passive = Nx.sub !spring (Nx.mul d.qvel p.dof_damping) }
