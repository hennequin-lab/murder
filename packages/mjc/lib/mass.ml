(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Topo

let f64 = Nx.float64
let cols xs = Nx.stack ~axis:1 xs

(* mju_inertCom: the inertia of one body about the point [dif] from its
   inertial frame, in world orientation. Ten entries: six symmetric rotational
   terms, three first moments, the mass. *)
let inert_com ~inertia ~mat ~dif ~mass =
  let i k = Nx.slice [ A; I k ] inertia in
  let r k = Nx.slice [ A; I k ] mat in
  let d k = Nx.slice [ A; I k ] dif in
  let open Infix in
  (* Column [k] of the rotational inertia times the diagonal, as
     [mju_inertCom] builds it: [rd.(3k+i) = mat.(3i+k) * inert.(k)]. *)
  let rd =
    [ r 0 * i 0
    ; r 3 * i 0
    ; r 6 * i 0
    ; r 1 * i 1
    ; r 4 * i 1
    ; r 7 * i 1
    ; r 2 * i 2
    ; r 5 * i 2
    ; r 8 * i 2
    ]
  in
  let rd k = List.nth_exn rd k in
  (* [res.(i,j) = sum_k mat.(3i+k) * tmp.(3k+j)]. *)
  let rot_ij i j =
    let i0 = Int.(3 * i)
    and i1 = Int.((3 * i) + 1)
    and i2 = Int.((3 * i) + 2) in
    let j0 = j
    and j1 = Int.(3 + j)
    and j2 = Int.(6 + j) in
    (r i0 * rd j0) + (r i1 * rd j1) + (r i2 * rd j2)
  in
  let d2 = (d 0 * d 0) + (d 1 * d 1) + (d 2 * d 2) in
  let rxx = rot_ij 0 0 + (mass * (d2 - (d 0 * d 0))) in
  let ryy = rot_ij 1 1 + (mass * (d2 - (d 1 * d 1))) in
  let rzz = rot_ij 2 2 + (mass * (d2 - (d 2 * d 2))) in
  let rxy = rot_ij 0 1 - (mass * d 0 * d 1) in
  let rxz = rot_ij 0 2 - (mass * d 0 * d 2) in
  let ryz = rot_ij 1 2 - (mass * d 1 * d 2) in
  cols [ rxx; ryy; rzz; rxy; rxz; ryz; mass * d 0; mass * d 1; mass * d 2; mass ]

let zeros3 like =
  Nx.zeros_like (Nx.reshape [| 3 |] (Nx.slice [ R (0, 3) ] (Nx.get [ 0 ] like)))

let forward (m : 'a Model.model) (d : 'a Data.data) : 'a Data.data =
  let t = m.topo
  and p = m.params in
  let nbody = t.nbody
  and nv = t.nv in
  (* subtree_com: moments up the tree, then normalize. *)
  let moment = ref (Nx.mul d.xipos (Nx.unsqueeze ~axes:[ 1 ] p.body_mass)) in
  for b = nbody - 1 downto 1 do
    let parent = t.body_parentid.(b) in
    let child = Nx.get [ b ] !moment in
    moment := Nx.set [ I parent ] (Nx.add (Nx.get [ parent ] !moment) child) !moment
  done;
  let tiny = Nx.less_s p.body_subtreemass 1e-15 in
  let denom = Nx.where tiny (Nx.ones_like p.body_subtreemass) p.body_subtreemass in
  let normalized = Nx.div !moment (Nx.unsqueeze ~axes:[ 1 ] denom) in
  let subtree_com = Nx.where (Nx.unsqueeze ~axes:[ 1 ] tiny) d.xipos normalized in
  let com = Nx.get [ 0 ] subtree_com in
  (* cinert: each body's inertia about the centre of mass of its tree. *)
  let root_com = ref (Nx.zeros_like subtree_com) in
  for b = 0 to nbody - 1 do
    root_com := Nx.set [ I b ] (Nx.get [ t.body_rootid.(b) ] subtree_com) !root_com
  done;
  let cinert = ref (Nx.zeros_like d.cinert) in
  for b = 1 to nbody - 1 do
    let dif = Nx.sub (Nx.get [ b ] d.xipos) (Nx.get [ b ] !root_com) in
    let row =
      inert_com
        ~inertia:(Nx.reshape [| 1; 3 |] (Nx.get [ b ] p.body_inertia))
        ~mat:(Nx.reshape [| 1; 9 |] (Nx.get [ b ] d.ximat))
        ~dif:(Nx.reshape [| 1; 3 |] dif)
        ~mass:(Nx.reshape [| 1 |] (Nx.get [ b ] p.body_mass))
    in
    cinert := Nx.set [ I b ] (Nx.reshape [| 10 |] row) !cinert
  done;
  (* cdof: each dof's motion axis in the same frame. *)
  let at = Nx.placement d.qpos in
  let cdof = ref (Nx.zeros_at at f64 [| nv; 6 |]) in
  let unit = zeros3 d.xpos in
  for b = 1 to nbody - 1 do
    for j = t.body_jntadr.(b) to t.body_jntadr.(b) + t.body_jntnum.(b) - 1 do
      let da = t.jnt_dofadr.(j) in
      let offset =
        Nx.sub (Nx.get [ t.body_rootid.(b) ] subtree_com) (Nx.get [ j ] d.xanchor)
      in
      let set_dof k v = cdof := Nx.set [ I (da + k) ] v !cdof in
      let dof_com ~axis ~offset = Nx.concatenate ~axis:0 [ axis; Nx.cross axis offset ] in
      let slide axis = Nx.concatenate ~axis:0 [ zeros3 d.xpos; axis ] in
      match t.jnt_type.(j) with
      | Free ->
        for k = 0 to 2 do
          let e = Nx.zeros_like unit in
          let e = Nx.set [ I k ] (Nx.add_s (Nx.get [ k ] e) 1.0) e in
          set_dof k (Nx.concatenate ~axis:0 [ zeros3 d.xpos; e ])
        done;
        (* The body frame's axes, not the inertial frame's. *)
        let xmat =
          Nx.reshape
            [| 9 |]
            (Rigid.quat_to_mat (Nx.reshape [| 1; 4 |] (Nx.get [ b ] d.xquat)))
        in
        let axis k =
          Nx.stack
            ~axis:0
            [ Nx.get [ k ] xmat; Nx.get [ k + 3 ] xmat; Nx.get [ k + 6 ] xmat ]
        in
        for k = 0 to 2 do
          set_dof (3 + k) (dof_com ~axis:(axis k) ~offset)
        done
      | Ball ->
        let xmat =
          Nx.reshape
            [| 9 |]
            (Rigid.quat_to_mat (Nx.reshape [| 1; 4 |] (Nx.get [ b ] d.xquat)))
        in
        let axis k =
          Nx.stack
            ~axis:0
            [ Nx.get [ k ] xmat; Nx.get [ k + 3 ] xmat; Nx.get [ k + 6 ] xmat ]
        in
        for k = 0 to 2 do
          set_dof k (dof_com ~axis:(axis k) ~offset)
        done
      | Slide -> set_dof 0 (slide (Nx.get [ j ] d.xaxis))
      | Hinge -> set_dof 0 (dof_com ~axis:(Nx.get [ j ] d.xaxis) ~offset)
    done
  done;
  (* crb: composite inertias up the tree, then M from the ancestor pattern. *)
  let crb = ref !cinert in
  for b = nbody - 1 downto 1 do
    let parent = t.body_parentid.(b) in
    if parent > 0
    then
      crb
      := Nx.set [ I parent ] (Nx.add (Nx.get [ parent ] !crb) (Nx.get [ b ] !crb)) !crb
  done;
  let at = Nx.placement d.qpos in
  let mass_matrix = ref (Nx.zeros_at at f64 [| nv; nv |]) in
  for v = 0 to nv - 1 do
    let buf =
      Rigid.mul_inert_vec (Nx.get [ t.dof_bodyid.(v) ] !crb) (Nx.get [ v ] !cdof)
    in
    let j = ref v in
    while !j >= 0 do
      let value =
        Nx.add (Nx.get [ v; !j ] !mass_matrix) (Nx.dot (Nx.get [ !j ] !cdof) buf)
      in
      mass_matrix := Nx.set [ I v; I !j ] value !mass_matrix;
      mass_matrix := Nx.set [ I !j; I v ] value !mass_matrix;
      j := t.dof_parentid.(!j)
    done;
    let diagonal = Nx.add (Nx.get [ v; v ] !mass_matrix) (Nx.get [ v ] p.dof_armature) in
    mass_matrix := Nx.set [ I v; I v ] diagonal !mass_matrix
  done;
  { d with subtree_com; com; cinert = !cinert; cdof = !cdof; mass_matrix = !mass_matrix }

type 'd factor = 'd Nx.float64_t

let factor (a : 'd Nx.float64_t) : 'd factor = Nx.cholesky a

let solve (l : 'a factor) b =
  let y = Nx.solve_triangular ~triangle:Lower l b in
  Nx.solve_triangular ~triangle:Lower ~transpose:Transpose l y
