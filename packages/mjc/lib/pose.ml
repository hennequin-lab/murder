(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Topo

let qpos_width = function
  | Hinge | Slide -> 1
  | Ball -> 4
  | Free -> 7

let dof_width = function
  | Hinge | Slide -> 1
  | Ball -> 3
  | Free -> 6

let qpos_width (t : Topo.t) j = qpos_width t.jnt_type.(j)
let dof_width (t : Topo.t) j = dof_width t.jnt_type.(j)
let set_scalar x i v = Nx.set [ I i ] v x
let add_scalar x i delta = set_scalar x i (Nx.add (Nx.get [ i ] x) delta)

(* One joint's quaternion advance, written back in place. [adr] is the first
   qpos entry of the quaternion, [vadr] the first of its three velocities. *)
let integrate_quat qpos ~adr ~vadr ~qvel ~dt =
  let quat = Nx.reshape [| 1; 4 |] (Nx.slice [ R (adr, adr + 4) ] qpos) in
  let vel = Nx.reshape [| 1; 3 |] (Nx.slice [ R (vadr, vadr + 3) ] qvel) in
  let advanced = Nx.reshape [| 4 |] (Rigid.quat_integrate quat vel ~scale:dt) in
  Nx.set [ R (adr, adr + 4) ] advanced qpos

let integrate_pos (t : Topo.t) ~qpos ~qvel ~dt =
  let qpos = ref qpos in
  for j = 0 to t.njnt - 1 do
    let adr = t.jnt_qposadr.(j)
    and vadr = t.jnt_dofadr.(j) in
    match t.jnt_type.(j) with
    | Hinge | Slide -> qpos := add_scalar !qpos adr (Nx.mul_s (Nx.get [ vadr ] qvel) dt)
    | Ball -> qpos := integrate_quat !qpos ~adr ~vadr ~qvel ~dt
    | Free ->
      for k = 0 to 2 do
        qpos := add_scalar !qpos (adr + k) (Nx.mul_s (Nx.get [ vadr + k ] qvel) dt)
      done;
      qpos := integrate_quat !qpos ~adr:(adr + 3) ~vadr:(vadr + 3) ~qvel ~dt
  done;
  !qpos

let normalize_at qpos adr =
  let quat = Nx.reshape [| 1; 4 |] (Nx.slice [ R (adr, adr + 4) ] !qpos) in
  let unit = Nx.reshape [| 4 |] (Rigid.quat_normalize quat) in
  qpos := Nx.set [ R (adr, adr + 4) ] unit !qpos

let normalize_quat (t : Topo.t) qpos =
  let qpos = ref qpos in
  for j = 0 to t.njnt - 1 do
    let adr = t.jnt_qposadr.(j) in
    match t.jnt_type.(j) with
    | Hinge | Slide -> ()
    | Ball -> normalize_at qpos adr
    | Free -> normalize_at qpos (adr + 3)
  done;
  !qpos
