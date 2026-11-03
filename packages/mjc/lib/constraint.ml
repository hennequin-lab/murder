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
  (* Every row past [nefc] is cleared, so a stale row from an earlier step can
     never reach the solver. *)
  { d with
    nefc
  ; efc_type = put types (Nx.zeros_like d.efc_type)
  ; efc_id = put ids (Nx.zeros_like d.efc_id)
  ; efc_pos = put !violation (Nx.zeros_like d.efc_pos)
  ; efc_margin = put !margin (Nx.zeros_like d.efc_margin)
  ; efc_J = put_rows !jac (Nx.zeros_like d.efc_J)
  }

(* {1 Impedance} *)

let min_val = 1e-15
let min_imp = 0.0001
let max_imp = 0.9999
let col k x = Nx.slice [ A; I k ] x
let one_minus x = Nx.sub (Nx.ones_like x) x
let inv x = Nx.div (Nx.ones_like x) x
let pow x p = Nx.pow x p

(* [getimpedance]: the impedance and its derivative at the row's position. *)
let getimpedance ~solimp ~pos ~margin =
  let s0 = col 0 solimp
  and s1 = col 1 solimp
  and s2 = col 2 solimp
  and s3 = col 3 solimp
  and s4 = col 4 solimp in
  let s2p = Nx.maximum_s s2 min_val in
  let x = Nx.div (Nx.sub pos margin) s2p in
  let ax = Nx.abs x in
  let flat = Nx.logical_or (Nx.equal s0 s1) (Nx.less_equal_s s2 min_val) in
  let sat = Nx.logical_or (Nx.greater_equal_s ax 1.0) (Nx.less_equal_s ax 0.0) in
  let imp_flat = Nx.mul_s (Nx.add s0 s1) 0.5 in
  let imp_sat = Nx.where (Nx.greater_equal_s ax 1.0) s1 s0 in
  let p = Nx.maximum_s s4 1.0 in
  let pm1 = Nx.sub p (Nx.ones_like p) in
  let a = inv (pow s3 pm1) in
  let b = inv (pow (one_minus s3) pm1) in
  let low = Nx.less_equal ax s3 in
  let y_lin = ax in
  let y_low = Nx.mul a (pow ax p) in
  let y_high = one_minus (Nx.mul b (pow (one_minus ax) p)) in
  let yp_lin = Nx.ones_like ax in
  let yp_low = Nx.mul (Nx.mul p a) (pow ax pm1) in
  let yp_high = Nx.mul (Nx.mul p b) (pow (one_minus ax) pm1) in
  let is_lin = Nx.equal_s p 1.0 in
  let y = Nx.where is_lin y_lin (Nx.where low y_low y_high) in
  let yp = Nx.where is_lin yp_lin (Nx.where low yp_low yp_high) in
  let imp_raw = Nx.add s0 (Nx.mul y (Nx.sub s1 s0)) in
  let impP_raw = Nx.div (Nx.mul (Nx.mul yp (Nx.sign x)) (Nx.sub s1 s0)) s2p in
  let imp = Nx.where flat imp_flat (Nx.where sat imp_sat imp_raw) in
  let impP =
    Nx.where flat (Nx.zeros_like s0) (Nx.where sat (Nx.zeros_like s0) impP_raw)
  in
  imp, impP

let impedance (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  let nb = t.nefcmax in
  let rows = Nx.arange_f Nx.float64 0.0 (Float.of_int nb) 1.0 in
  let mask = Nx.less rows (Nx.cast Nx.float64 d.nefc) in
  let mask_f = Nx.where mask (Nx.ones_like rows) (Nx.zeros_like rows) in
  ignore mask_f;
  (* Per row, the solver parameters of the object the row names. v1 has limit
     rows only, so the id indexes joints. *)
  let jid = Nx.cast Nx.int64 d.efc_id in
  let solref = Nx.take ~axis:0 jid p.jnt_solref in
  let solimp = Nx.take ~axis:0 jid p.jnt_solimp in
  let solimp =
    let clamp = Nx.clamp ~min:min_imp ~max:max_imp in
    Nx.stack
      ~axis:1
      [ clamp (col 0 solimp)
      ; clamp (col 1 solimp)
      ; Nx.maximum_s (col 2 solimp) 0.0
      ; clamp (col 3 solimp)
      ; Nx.maximum_s (col 4 solimp) 1.0
      ]
  in
  (* solref: one default for the mixed format, and the refsafe clamp. *)
  let dt = Nx.item [] p.timestep in
  let r0 = col 0 solref
  and r1 = col 1 solref in
  let mixed = Nx.logical_xor (Nx.greater_s r0 0.0) (Nx.greater_s r1 0.0) in
  let r0 = Nx.where mixed (Nx.full_like r0 0.02) (Nx.maximum_s r0 (2.0 *. dt)) in
  let r1 = Nx.where mixed (Nx.full_like r1 1.0) r1 in
  let s1 = col 1 solimp in
  let kinv_std =
    inv
      (Nx.maximum_s
         (Nx.mul (Nx.mul (Nx.mul s1 s1) (Nx.mul r0 r0)) (Nx.mul r1 r1))
         min_val)
  in
  let kinv_dir = Nx.div (Nx.neg r0) (Nx.maximum_s (Nx.mul s1 s1) min_val) in
  let k = Nx.where (Nx.greater_s r0 0.0) kinv_std kinv_dir in
  let binv_std = Nx.div (Nx.full_like s1 2.0) (Nx.maximum_s (Nx.mul s1 r0) min_val) in
  let binv_dir = Nx.div (Nx.neg r1) (Nx.maximum_s s1 min_val) in
  let bcoef = Nx.where (Nx.greater_s r1 0.0) binv_std binv_dir in
  let dofadr =
    Nx.cast
      Nx.int64
      (Nx.init Nx.int32 [| t.njnt |] (fun i -> Stdlib.Int32.of_int t.jnt_dofadr.(i.(0))))
  in
  let diagA = Nx.take ~axis:0 (Nx.take ~axis:0 jid dofadr) p.dof_invweight0 in
  let imp, _impP = getimpedance ~solimp ~pos:d.efc_pos ~margin:d.efc_margin in
  let r =
    Nx.maximum_s
      (Nx.div (Nx.mul (one_minus imp) diagA) (Nx.maximum_s imp min_val))
      min_val
  in
  let r = Nx.where mask r (Nx.ones_like r) in
  let dcoef = Nx.where mask (inv r) (Nx.zeros_like r) in
  let vel = Nx.matmul d.efc_J (Nx.reshape [| t.nv; 1 |] d.qvel) in
  let vel = Nx.where mask (Nx.reshape [| nb |] vel) (Nx.zeros_like rows) in
  let aref =
    Nx.sub
      (Nx.neg (Nx.mul bcoef vel))
      (Nx.mul (Nx.mul k imp) (Nx.sub d.efc_pos d.efc_margin))
  in
  let aref = Nx.where mask aref (Nx.zeros_like aref) in
  { d with efc_vel = vel; efc_R = r; efc_D = dcoef; efc_aref = aref }
