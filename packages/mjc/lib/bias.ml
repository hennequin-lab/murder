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

let com_vel (m : 'a Model.model) (d : 'a Data.data) : 'a Data.data =
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

let rne (m : 'a Model.model) (d : 'a Data.data) =
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

(* Subtree linear velocity and angular momentum, MuJoCo's [mj_subtreeVel].

   A body's velocity at its own frame origin is its com-based [cvel] moved
   there: [v = cvel_tran - (com_root - xpos) x cvel_rot], which is
   [mju_transformSpatial] without the rotation. The subtree quantities then
   accumulate up the tree — linear momentum first, converted to a velocity by
   the subtree mass at the end, and angular momentum through the offset from
   the subtree's centre of mass. *)
let subtree_vel (m : 'a Model.model) (d : 'a Data.data) : 'a Data.data =
  let t = m.topo in
  let n = t.nbody in
  let at = Nx.placement d.qpos in
  let body_lin = ref (Nx.zeros_at at Nx.float64 [| n; 3 |])
  and body_rot = ref (Nx.zeros_at at Nx.float64 [| n; 3 |]) in
  for b = 0 to n - 1 do
    let cv = Nx.get [ b ] d.cvel in
    let rot = Nx.reshape [| 3 |] (Nx.slice [ R (0, 3) ] cv)
    and tran = Nx.reshape [| 3 |] (Nx.slice [ R (3, 6) ] cv) in
    (* [mj_objectVelocity] measures a body's velocity at its *inertial*
       position, from the subtree's centre of mass. *)
    let com = Nx.get [ t.body_rootid.(b) ] d.subtree_com in
    let dif = Nx.sub (Nx.get [ b ] d.xipos) com in
    let lin = Nx.sub tran (Nx.cross dif rot) in
    body_rot := Nx.set [ I b ] rot !body_rot;
    body_lin := Nx.set [ I b ] lin !body_lin
  done;
  let body_lin = !body_lin
  and body_rot = !body_rot in
  (* linear momentum, then down the tree into the parent *)
  let lin = ref (Nx.mul body_lin (Nx.reshape [| n; 1 |] m.params.body_mass)) in
  for b = n - 1 downto 1 do
    let parent = t.body_parentid.(b) in
    lin := Nx.set [ I parent ] (Nx.add (Nx.get [ parent ] !lin) (Nx.get [ b ] !lin)) !lin
  done;
  let mass = Nx.maximum_s m.params.body_subtreemass 1e-15 in
  let linvel = Nx.div !lin (Nx.reshape [| n; 1 |] mass) in
  (* angular momentum about each subtree's centre of mass *)
  let ang = ref (Nx.zeros_at (Nx.placement d.qpos) Nx.float64 [| n; 3 |]) in
  for b = 0 to n - 1 do
    let rot = Nx.get [ b ] body_rot in
    let ximat = Nx.reshape [| 3; 3 |] (Nx.get [ b ] d.ximat) in
    let local = Nx.matmul (Nx.matrix_transpose ximat) rot in
    let scaled = Nx.mul local (Nx.get [ b ] m.params.body_inertia) in
    ang := Nx.set [ I b ] (Nx.matmul ximat scaled) !ang
  done;
  for b = n - 1 downto 1 do
    let parent = t.body_parentid.(b) in
    let dx = Nx.sub (Nx.get [ b ] d.xipos) (Nx.get [ b ] d.subtree_com) in
    let dv = Nx.sub (Nx.get [ b ] body_lin) (Nx.get [ b ] linvel) in
    let dp = Nx.mul dv (Nx.get [ b ] m.params.body_mass) in
    ang := Nx.set [ I b ] (Nx.add (Nx.get [ b ] !ang) (Nx.cross dx dp)) !ang;
    ang := Nx.set [ I parent ] (Nx.add (Nx.get [ parent ] !ang) (Nx.get [ b ] !ang)) !ang;
    (* and the whole subtree's orbital momentum about the parent's centre *)
    let dx = Nx.sub (Nx.get [ b ] d.subtree_com) (Nx.get [ parent ] d.subtree_com) in
    let dv =
      Nx.mul
        (Nx.sub (Nx.get [ b ] linvel) (Nx.get [ parent ] linvel))
        (Nx.get [ b ] m.params.body_subtreemass)
    in
    ang := Nx.set [ I parent ] (Nx.add (Nx.get [ parent ] !ang) (Nx.cross dx dv)) !ang
  done;
  { d with subtree_linvel = linvel; subtree_angmom = !ang }

let forward (m : 'a Model.model) (d : 'a Data.data) : 'a Data.data =
  let d = com_vel m d in
  let d = subtree_vel m d in
  { d with qfrc_bias = rne m d }
