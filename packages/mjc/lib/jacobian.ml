(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Topo

type 'd f = 'd Nx.float64_t

let dof_width = function
  | Hinge | Slide -> 1
  | Ball -> 3
  | Free -> 6

(* The last dof a body carries, which is the end of the chain a Jacobian walks
   backwards. A body with no joint of its own inherits its nearest ancestor's
   chain: MuJoCo treats such a body as welded to its parent, and so must the
   Jacobian of a point on it. *)
let rec last_dof (t : Topo.t) b =
  if b <= 0
  then (* The world body is its own parent: stop before the recursion does. *)
    -1
  else if t.body_jntnum.(b) > 0
  then (
    let j = t.body_jntadr.(b) + t.body_jntnum.(b) - 1 in
    t.jnt_dofadr.(j) + dof_width t.jnt_type.(j) - 1)
  else last_dof t t.body_parentid.(b)

(* Which dofs move which body.

   It is structure -- a property of the topology, the same every step -- so it is
   one host creation of `[nbody; nv]` rather than a traced `set` per dof of the
   chain. The old form built `nv`-many graph nodes per candidate contact, and a
   model with a thousand candidates spent minutes tracing before it ran; this is
   one node, whatever the chain length. *)
let body_dof_masks ~at (t : Topo.t) =
  Nx.place
    at
    (Nx.init Nx.float64 [| t.nbody; t.nv |] (fun i ->
       let body = i.(0)
       and dof = i.(1) in
       let rec walks j =
         if j < 0 then 0.0 else if Int.equal j dof then 1.0 else walks t.dof_parentid.(j)
       in
       walks (last_dof t body)))

(* A mask over the dofs that move [body], as a row of that table. *)
let chain_mask ~at (t : Topo.t) body = Nx.slice [ I body; A ] (body_dof_masks ~at t)

(* The offset of a point from the reference point of the motion subspace. *)
let offset (t : Topo.t) (d : 'a Data.data) body pos =
  let com = Nx.slice [ I t.body_rootid.(body); A ] d.subtree_com in
  Nx.sub pos com

let ang6 x = Nx.slice [ A; R (0, 3) ] x
let lin6 x = Nx.slice [ A; R (3, 6) ] x

let mask_rows mask x =
  let n = (Nx.shape x).(0) in
  Nx.mul (Nx.reshape [| n; 1 |] mask) x

let as_row v = Nx.reshape [| 1; (Nx.shape v).(0) |] v

let point ?masks (m : 'a Model.model) (d : 'a Data.data) ~body ~pos =
  let t = m.topo in
  let r = offset t d body pos in
  let mask =
    match masks with
    | Some table -> Nx.slice [ I body; A ] table
    | None -> chain_mask ~at:(Nx.placement d.qpos) t body
  in
  let ang = ang6 d.cdof
  and lin = lin6 d.cdof in
  let cross = Nx.cross ang (as_row r) in
  let jacr = mask_rows mask ang in
  let jacp = mask_rows mask (Nx.add lin cross) in
  jacp, jacr

(* [crossMotion] of two 6-vectors, angular part first. *)
let cross6 vel u =
  let g k x = Nx.get [ k ] x in
  let z = g 0 vel
  and o = g 1 vel
  and t = g 2 vel
  and th = g 3 vel
  and fo = g 4 vel
  and fi = g 5 vel in
  let u0 = g 0 u
  and u1 = g 1 u
  and u2 = g 2 u
  and u3 = g 3 u
  and u4 = g 4 u
  and u5 = g 5 u in
  let r0 = Nx.sub (Nx.mul o u2) (Nx.mul t u1) in
  let r1 = Nx.sub (Nx.mul t u0) (Nx.mul z u2) in
  let r2 = Nx.sub (Nx.mul z u1) (Nx.mul o u0) in
  let r3 =
    Nx.add (Nx.sub (Nx.mul o u5) (Nx.mul t u4)) (Nx.sub (Nx.mul fo u2) (Nx.mul fi u1))
  in
  let r4 =
    Nx.add (Nx.sub (Nx.mul t u3) (Nx.mul z u5)) (Nx.sub (Nx.mul fi u0) (Nx.mul th u2))
  in
  let r5 =
    Nx.add (Nx.sub (Nx.mul z u4) (Nx.mul o u3)) (Nx.sub (Nx.mul th u1) (Nx.mul fo u0))
  in
  Nx.stack ~axis:0 [ r0; r1; r2; r3; r4; r5 ]

(* MuJoCo's [mj_jacDot] does not use the stored [cdof_dot] for a quaternion
   joint: it recomputes it from the body's *final* velocity, where
   [mj_comVel] had used the velocity before the joint's own contribution.
   The two differ whenever the joint is not first in its body. *)
let cdof_dot_jac (t : Topo.t) (d : 'a Data.data) =
  let out = ref d.cdof_dot in
  for dof = 0 to t.nv - 1 do
    let j = t.dof_jntid.(dof) in
    let quat =
      match t.jnt_type.(j) with
      | Ball -> true
      | Free -> dof >= t.jnt_dofadr.(j) + 3
      | Hinge | Slide -> false
    in
    if quat
    then (
      let cvel = Nx.slice [ I t.dof_bodyid.(dof); A ] d.cvel in
      let cdof = Nx.slice [ I dof; A ] d.cdof in
      out := Nx.set [ I dof ] (cross6 cvel cdof) !out)
  done;
  !out

let point_dot ?masks (m : 'a Model.model) (d : 'a Data.data) ~body ~pos =
  let t = m.topo in
  let r = offset t d body pos in
  let mask =
    match masks with
    | Some table -> Nx.slice [ I body; A ] table
    | None -> chain_mask ~at:(Nx.placement d.qpos) t body
  in
  let cvel = Nx.slice [ I body; A ] d.cvel in
  let omega = Nx.slice [ R (0, 3) ] cvel
  and vlin = Nx.slice [ R (3, 6) ] cvel in
  let pvel = Nx.add vlin (Nx.cross omega r) in
  let ang = ang6 d.cdof in
  let cdof_dot = cdof_dot_jac t d in
  let d_ang = ang6 cdof_dot
  and d_lin = lin6 cdof_dot in
  let jacr = mask_rows mask d_ang in
  let jacp =
    mask_rows
      mask
      (Nx.add (Nx.add d_lin (Nx.cross d_ang (as_row r))) (Nx.cross ang (as_row pvel)))
  in
  jacp, jacr

let jdotv (m : 'a Model.model) (d : 'a Data.data) ~body ~pos =
  let t = m.topo in
  let jacp, jacr = point_dot m d ~body ~pos in
  let v = Nx.reshape [| t.nv; 1 |] d.qvel in
  let dot x = Nx.reshape [| 3 |] (Nx.matmul (Nx.matrix_transpose x) v) in
  Nx.concatenate ~axis:0 [ dot jacp; dot jacr ]
