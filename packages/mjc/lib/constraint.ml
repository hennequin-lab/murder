(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
module Int32 = Stdlib.Int32
open Topo

let type_equality = 0
let type_friction_dof = 1
let type_friction_tendon = 2
let type_limit_joint = 3
let type_limit_tendon = 4
let type_contact_frictionless = 5
let type_contact_pyramidal = 6

(* One candidate row per (limited hinge or slide joint, side), in MuJoCo's
   order: joints by id, lower side before upper. *)
type cand =
  { joint : int
  ; dof : int
  ; side : float
  ; range_index : int
  }

let limit_candidates (t : Topo.t) : cand array =
  let cands = ref [] in
  for j = 0 to t.njnt - 1 do
    match t.jnt_type.(j) with
    | Hinge | Slide ->
      cands
      := { joint = j; dof = t.jnt_dofadr.(j); side = -1.0; range_index = 0 } :: !cands;
      cands
      := { joint = j; dof = t.jnt_dofadr.(j); side = 1.0; range_index = 1 } :: !cands
    | Ball | Free -> ()
  done;
  Array.of_list (List.rev !cands)

(* A fresh [n]-vector with the dtype and placement of [like]. *)
let head n like = Nx.slice [ R (0, n) ] (Nx.reshape [| -1 |] like)

let limits (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  let nv = t.nv in
  let cands = limit_candidates t in
  let ncand = Array.length cands in
  let violation = ref (Nx.zeros_like (head ncand d.efc_pos)) in
  let margin = ref !violation in
  let active = ref !violation in
  let jac =
    ref (Nx.reshape [| ncand; nv |] (Nx.zeros_like (head (ncand * nv) d.efc_J)))
  in
  for i = 0 to ncand - 1 do
    let c = cands.(i) in
    let value = Nx.get [ t.jnt_qposadr.(c.joint) ] d.qpos in
    let bound = Nx.get [ c.joint; c.range_index ] p.jnt_range in
    let margin_i = Nx.get [ c.joint ] p.jnt_margin in
    let limited = Nx.get [ c.joint ] p.jnt_limited in
    (* [dist = side (bound - value)]: positive inside the range, negative
       beyond it, so a row exists when [dist < margin]. *)
    let dist = Nx.mul_s (Nx.sub bound value) c.side in
    let on = Nx.logical_and (Nx.less dist margin_i) (Nx.greater_s limited 0.5) in
    violation := Nx.set [ I i ] dist !violation;
    margin := Nx.set [ I i ] margin_i !margin;
    active
    := Nx.set
         [ I i ]
         (Nx.where on (Nx.scalar_like dist 1.0) (Nx.scalar_like dist 0.0))
         !active;
    jac := Nx.set [ I i; I c.dof ] (Nx.scalar_like dist (-.c.side)) !jac
  done;
  (* Row index: the number of active rows before this one. An inactive row is
     sent past the buffer, where a scatter writes nothing. *)
  let before = Nx.sub (Nx.cumsum ~axis:0 !active) !active in
  let far = Nx.full_like before (Float.of_int t.nefcmax) in
  let idx = Nx.cast Nx.int64 (Nx.where (Nx.greater_s !active 0.5) before far) in
  let put v x = Nx.scatter ~combine:Set ~axis:0 idx v x in
  (* A scatter is elementwise: the Jacobian's positions are tiled per column. *)
  let idx_rows = Nx.broadcast_to [| ncand; nv |] (Nx.reshape [| ncand; 1 |] idx) in
  let put_rows v x = Nx.scatter ~combine:Set ~axis:0 idx_rows v x in
  let types = Nx.full Nx.int32 [| ncand |] (Int32.of_int type_limit_joint) in
  let ids = Nx.init Nx.int32 [| ncand |] (fun i -> Int32.of_int cands.(i.(0)).joint) in
  let nefc = Nx.cast Nx.int32 (Nx.sum !active) in
  { d with
    nefc
  ; efc_type = put types (head ncand d.efc_type)
  ; efc_id = put ids (head ncand d.efc_id)
  ; efc_pos = put !violation (head ncand d.efc_pos)
  ; efc_margin = put !margin (head ncand d.efc_margin)
  ; efc_J = put_rows !jac (Nx.reshape [| t.nefcmax; nv |] d.efc_J)
  }
