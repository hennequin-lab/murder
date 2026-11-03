(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Topo

(* A 6-vector times the scalar [q], broadcast along the vector. *)
let scale6 c q = Nx.mul c (Nx.reshape [| 1 |] q)

let body_dofadr t b =
  if t.body_jntnum.(b) = 0 then -1 else t.jnt_dofadr.(t.body_jntadr.(b))

let body_dofnum t b =
  let num = ref 0 in
  for k = t.body_jntadr.(b) to t.body_jntadr.(b) + t.body_jntnum.(b) - 1 do
    num
    := !num
       +
       match t.jnt_type.(k) with
       | Hinge | Slide -> 1
       | Ball -> 3
       | Free -> 6
  done;
  !num

let com_vel (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo in
  let cvel = ref (Nx.zeros_like d.cvel) in
  let cdof_dot = ref (Nx.zeros_like d.cdof_dot) in
  for b = 1 to t.nbody - 1 do
    let parent = t.body_parentid.(b) in
    let v = ref (Nx.get [ parent ] !cvel) in
    let jntadr = t.body_jntadr.(b)
    and jntnum = t.body_jntnum.(b) in
    let set_dot j =
      cdof_dot := Nx.set [ I j ] (Rigid.cross_motion !v (Nx.get [ j ] d.cdof)) !cdof_dot
    in
    let add j = v := Nx.add !v (scale6 (Nx.get [ j ] d.cdof) (Nx.get [ j ] d.qvel)) in
    for k = jntadr to jntadr + jntnum - 1 do
      let adr = t.jnt_dofadr.(k) in
      match t.jnt_type.(k) with
      (* The three axes of a ball joint share one velocity, so all their
         derivatives are taken before any of them advances it. *)
      | Free ->
        for j = adr to adr + 2 do
          add j
        done;
        for j = adr + 3 to adr + 5 do
          set_dot j
        done;
        for j = adr + 3 to adr + 5 do
          add j
        done
      | Ball ->
        for j = adr to adr + 2 do
          set_dot j
        done;
        for j = adr to adr + 2 do
          add j
        done
      | Hinge | Slide ->
        set_dot adr;
        add adr
    done;
    cvel := Nx.set [ I b ] !v !cvel
  done;
  { d with cvel = !cvel; cdof_dot = !cdof_dot }

let rne (m : Model.host) (d : Data.host) =
  let t = m.topo
  and p = m.params in
  let cacc = ref (Nx.zeros_like d.cvel) in
  (* The world body accelerates against gravity. *)
  let gravity = p.gravity in
  cacc
  := Nx.set
       [ I 0 ]
       (Nx.concatenate ~axis:0 [ Nx.zeros_like gravity; Nx.neg gravity ])
       !cacc;
  let cfrc = ref (Nx.zeros_like d.cvel) in
  for b = 1 to t.nbody - 1 do
    let parent = t.body_parentid.(b) in
    let acc = ref (Nx.get [ parent ] !cacc) in
    let first = body_dofadr t b
    and num = body_dofnum t b in
    for k = 0 to num - 1 do
      let j = first + k in
      acc := Nx.add !acc (scale6 (Nx.get [ j ] d.cdof_dot) (Nx.get [ j ] d.qvel))
    done;
    let inertia = Nx.get [ b ] d.cinert in
    let f = Rigid.mul_inert_vec inertia !acc in
    let momentum = Rigid.mul_inert_vec inertia (Nx.get [ b ] d.cvel) in
    cfrc
    := Nx.set [ I b ] (Nx.add f (Rigid.cross_force (Nx.get [ b ] d.cvel) momentum)) !cfrc;
    cacc := Nx.set [ I b ] !acc !cacc
  done;
  for b = t.nbody - 1 downto 1 do
    let parent = t.body_parentid.(b) in
    if parent > 0
    then
      cfrc
      := Nx.set [ I parent ] (Nx.add (Nx.get [ parent ] !cfrc) (Nx.get [ b ] !cfrc)) !cfrc
  done;
  let result = ref (Nx.zeros_like d.qfrc_bias) in
  for v = 0 to t.nv - 1 do
    let f = Nx.get [ t.dof_bodyid.(v) ] !cfrc in
    result := Nx.set [ I v ] (Rigid.dot6 (Nx.get [ v ] d.cdof) f) !result
  done;
  !result

let forward (m : Model.host) (d : Data.host) : Data.host =
  let d = com_vel m d in
  { d with qfrc_bias = rne m d }
