(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
module Int32 = Stdlib.Int32
open Topo

type f = Nx.host Nx.float64_t

let type_equality = 0
let type_friction_dof = 1
let type_friction_tendon = 2
let type_limit_joint = 3
let type_limit_tendon = 4
let type_contact_frictionless = 5
let type_contact_pyramidal = 6
let col k x = Nx.slice [ A; I k ] x
let row k x = Nx.slice [ I k; A ] x
let one_minus x = Nx.sub (Nx.ones_like x) x
let inv x = Nx.div (Nx.ones_like x) x
let pow x p = Nx.pow x p
let matvec3 m v = Nx.reshape [| 3 |] (Nx.matmul m (Nx.reshape [| 3; 1 |] v))

(* The quaternion helpers are batched, one rotation per row: a single
   rotation goes in and comes back as a [4]. *)
let as_quat_row q = Nx.reshape [| 1; 4 |] q
let quat_mat q = Nx.reshape [| 3; 3 |] (Rigid.quat_to_mat (as_quat_row q))
let qmul a b = Nx.reshape [| 4 |] (Rigid.quat_mul (as_quat_row a) (as_quat_row b))
let qconj a = Nx.reshape [| 4 |] (Rigid.quat_conj (as_quat_row a))

let qmul_axis a v =
  let zero = Nx.mul_s (Nx.slice [ R (0, 1) ] v) 0.0 in
  qmul a (Nx.concatenate ~axis:0 [ zero; v ])

let qderiv q omega =
  let zero = Nx.mul_s (Nx.slice [ R (0, 1) ] omega) 0.0 in
  qmul (Nx.concatenate ~axis:0 [ zero; omega ]) q |> fun x -> Nx.mul_s x 0.5

let as_row v = Nx.reshape [| 1; (Nx.shape v).(0) |] v
let slice3 at v = Nx.slice [ R (at, at + 3) ] v

(* {1 The row buffer}

   Every kind of constraint writes the same row fields, so each assembler fills
   them in and writes them at its own offset: equality rows first, then limit
   rows. [keep_before] is what makes the buffer reusable: the rows a stage no
   longer produces are cleared, so a stale row can never reach the solver. *)

(** A constraint row: the object it names, what it measures, the inverse
    inertia that sets its regularisation, its solver parameters and its
    Jacobian. *)
type row =
  { kind : int
  ; id : int
  ; pos : f (** 0-d, the violation. *)
  ; margin : f (** 0-d. *)
  ; posnorm : f (** 0-d, what the impedance reads. *)
  ; diagA : f (** 0-d, [1/J M⁻¹ Jᵀ] at the reference configuration. *)
  ; solref : f (** [[2]] *)
  ; solimp : f (** [[5]] *)
  ; jdotv : f (** 0-d, the row's [Jdot qvel]. *)
  ; frictionloss : f (** 0-d, zero but on a friction row. *)
  ; jac : f (** [[nv]] *)
  }

let keep_before (m : Model.host) (d : Data.host) at =
  let rows = Nx.arange_f Nx.float64 0.0 (Float.of_int m.topo.nefcmax) 1.0 in
  let keep = Nx.less rows at in
  let keep_rows = Nx.reshape [| m.topo.nefcmax; 1 |] (Nx.cast Nx.float64 keep) in
  let keep_i x = Nx.where keep x (Nx.zeros_like x) in
  let keep_f x = Nx.where keep x (Nx.zeros_like x) in
  { d with
    efc_type = keep_i d.efc_type
  ; efc_id = keep_i d.efc_id
  ; efc_pos = keep_f d.efc_pos
  ; efc_margin = keep_f d.efc_margin
  ; efc_posnorm = keep_f d.efc_posnorm
  ; efc_frictionloss = keep_f d.efc_frictionloss
  ; efc_diagA = keep_f d.efc_diagA
  ; efc_vel = keep_f d.efc_vel
  ; efc_R = keep_f d.efc_R
  ; efc_D = keep_f d.efc_D
  ; efc_aref = keep_f d.efc_aref
  ; efc_force = keep_f d.efc_force
  ; efc_jdotv = keep_f d.efc_jdotv
  ; efc_solref = Nx.mul d.efc_solref keep_rows
  ; efc_solimp = Nx.mul d.efc_solimp keep_rows
  ; efc_J = Nx.mul d.efc_J keep_rows
  }

let set_row (d : Data.host) ~at (r : row) =
  let i = [ I at ] in
  let i32 = Nx.scalar Nx.int32 in
  { d with
    efc_type = Nx.set i (i32 (Int32.of_int r.kind)) d.efc_type
  ; efc_id = Nx.set i (i32 (Int32.of_int r.id)) d.efc_id
  ; efc_pos = Nx.set i r.pos d.efc_pos
  ; efc_margin = Nx.set i r.margin d.efc_margin
  ; efc_posnorm = Nx.set i r.posnorm d.efc_posnorm
  ; efc_frictionloss = Nx.set i r.frictionloss d.efc_frictionloss
  ; efc_diagA = Nx.set i r.diagA d.efc_diagA
  ; efc_solref = Nx.set i r.solref d.efc_solref
  ; efc_solimp = Nx.set i r.solimp d.efc_solimp
  ; efc_jdotv = Nx.set i r.jdotv d.efc_jdotv
  ; efc_J = Nx.set i r.jac d.efc_J
  }

(* {1 Equality constraints}

   A port of [mj_instantiateEquality] for the three kinds v1 supports. The
   rows of an equality are contiguous and equalities are taken in id order, so
   their row indices are host values; which equalities are on comes from the
   model's [eq_active0], not from the data. *)

let body_xmat (d : Data.host) b = quat_mat (row b d.xquat)
let body_xpos (d : Data.host) b = row b d.xpos

(* The world position and orientation of one side of an equality, and the body
   that carries it. A body anchor is [data]'s slice in the body's frame; a site
   is its own frame, which is the body frame turned by the site's quaternion. *)
let eq_object (m : Model.host) (d : Data.host) i which =
  let t = m.topo
  and p = m.params in
  let data = row i p.eq_data in
  let ids = if which = 0 then p.eq_obj1id else p.eq_obj2id in
  let obj = Int.of_float (Nx.item [ i ] ids) in
  match t.eq_objtype.(i) with
  | Eq_body ->
    let at =
      match t.eq_type.(i), which with
      | Connect, 0 -> 0
      | Connect, _ -> 3
      | Weld, 0 -> 3
      | Weld, _ -> 0
      | Joint, _ -> 0
      (* A tendon equality has no anchor: its objects are tendons, and the row
         reads their lengths directly. *)
      | Tendon, _ -> 0
    in
    let pos = Nx.add (body_xpos d obj) (matvec3 (body_xmat d obj) (slice3 at data)) in
    obj, pos, row obj d.xquat
  | Eq_site ->
    let pos = row obj d.site_xpos in
    let quat = qmul (row t.site_bodyid.(obj) d.xquat) (row obj p.site_quat) in
    t.site_bodyid.(obj), pos, quat

(* [Jdot qvel] of the constraint: the time derivative the reference
   acceleration must cancel, so that a constraint held at zero velocity stays
   at zero. Only connect and weld have one: a joint coupling has none. *)
let equality_jdotv (m : Model.host) (d : Data.host) kind ~body1 ~pos1 ~body2 ~pos2 =
  match kind with
  | Weld | Connect ->
    let j1 = Jacobian.jdotv m d ~body:body1 ~pos:pos1 in
    let j2 = Jacobian.jdotv m d ~body:body2 ~pos:pos2 in
    Some (Nx.sub j1 j2)
  | Joint | Tendon -> None

let equality (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  let nv = t.nv in
  let rows = ref 0 in
  let out = ref d in
  for i = 0 to t.neq - 1 do
    let on = Float.( > ) (Nx.item [ i ] p.eq_active0) 0.5 in
    if on
    then (
      let kind = t.eq_type.(i) in
      let solref = row i p.eq_solref
      and solimp = row i p.eq_solimp in
      let add ~pos ~posnorm ~diagA ~jdotv ~jac =
        out
        := set_row
             !out
             ~at:!rows
             { kind = type_equality
             ; id = i
             ; pos
             ; margin = Nx.scalar Nx.float64 0.0
             ; posnorm
             ; frictionloss = Nx.scalar Nx.float64 0.0
             ; diagA
             ; solref
             ; solimp
             ; jdotv
             ; jac
             };
        Int.incr rows
      in
      match kind with
      | Connect ->
        let b1, pos1, _ = eq_object m d i 0 in
        let b2, pos2, _ = eq_object m d i 1 in
        let j1 = Jacobian.point m d ~body:b1 ~pos:pos1 in
        let j2 = Jacobian.point m d ~body:b2 ~pos:pos2 in
        let jacp1, _ = j1
        and jacp2, _ = j2 in
        let diagA =
          Nx.add
            (Nx.get [ 0 ] (row b1 p.body_invweight0))
            (Nx.get [ 0 ] (row b2 p.body_invweight0))
        in
        let jdotv = equality_jdotv m d kind ~body1:b1 ~pos1 ~body2:b2 ~pos2 in
        let jdotv =
          match jdotv with
          | Some v -> v
          | None -> Nx.zeros Nx.float64 [| 6 |]
        in
        let diff = Nx.sub pos1 pos2 in
        let norm v = Nx.sqrt (Nx.sum (Nx.mul v v)) in
        let posnorm = norm diff in
        for k = 0 to 2 do
          add
            ~posnorm
            ~pos:(Nx.get [ k ] diff)
            ~diagA
            ~jdotv:(Nx.get [ k ] jdotv)
            ~jac:(Nx.sub (col k jacp1) (col k jacp2))
        done
      | Weld ->
        let b1, pos1, quat1 = eq_object m d i 0 in
        let b2, pos2, quat2 = eq_object m d i 1 in
        let jacp1, jacr1 = Jacobian.point m d ~body:b1 ~pos:pos1
        and jacp2, jacr2 = Jacobian.point m d ~body:b2 ~pos:pos2 in
        let data = row i p.eq_data in
        let relpose = Nx.slice [ R (6, 10) ] data in
        let torquescale = Nx.get [ 10 ] data in
        let diff = Nx.sub pos1 pos2 in
        let q1r = qmul quat1 relpose in
        let negq2 = qconj quat2 in
        let qerr = qmul negq2 q1r in
        let rot_pos = Nx.mul (Nx.slice [ R (1, 4) ] qerr) torquescale in
        (* Rotational rows: 0.5 * neg(q2) * (J1 - J2) * q1 * relpose, one row
           per axis of the relative rotation. *)
        let axis = Nx.sub jacr1 jacr2 in
        let q1rv = Nx.slice [ R (1, 4) ] q1r in
        let s0 = Nx.get [ 0 ] q1r in
        let dot = Nx.reshape [| nv; 1 |] (Nx.matmul axis (Nx.reshape [| 3; 1 |] q1rv)) in
        let pure =
          Nx.concatenate
            ~axis:1
            [ Nx.neg dot; Nx.add (Nx.mul axis s0) (Nx.cross axis (as_row q1rv)) ]
        in
        let rot_jac =
          Rigid.quat_mul (Nx.broadcast_to [| nv; 4 |] (as_quat_row negq2)) pure
        in
        let rot_jac =
          Nx.matrix_transpose
            (Nx.mul (Nx.slice [ A; R (1, 4) ] rot_jac) (Nx.mul_s torquescale 0.5))
        in
        let jdotv =
          match equality_jdotv m d kind ~body1:b1 ~pos1 ~body2:b2 ~pos2 with
          | None -> Nx.zeros Nx.float64 [| 6 |]
          | Some jd ->
            let dv = Nx.slice [ R (0, 3) ] jd in
            let djrdv = Nx.slice [ R (3, 6) ] jd in
            let omega1 = Nx.slice [ I b1; R (0, 3) ] d.cvel
            and omega2 = Nx.slice [ I b2; R (0, 3) ] d.cvel in
            let domega = Nx.sub omega1 omega2 in
            let qdot0 = qderiv quat1 omega1 in
            let qdot0r =
              match t.eq_objtype.(i) with
              | Eq_body -> qmul qdot0 relpose
              | Eq_site -> qdot0
            in
            let negqdot1 = qconj (qderiv quat2 omega2) in
            let term1 = qmul (qmul_axis negqdot1 domega) q1r in
            let term2 = qmul (qmul_axis negq2 djrdv) q1r in
            let term3 = qmul (qmul_axis negq2 domega) qdot0r in
            let rot =
              Nx.mul
                (Nx.slice [ R (1, 4) ] (Nx.add term1 (Nx.add term2 term3)))
                (Nx.mul_s torquescale 0.5)
            in
            Nx.concatenate ~axis:0 [ dv; rot ]
        in
        let diagA_t =
          Nx.add
            (Nx.get [ 0 ] (row b1 p.body_invweight0))
            (Nx.get [ 0 ] (row b2 p.body_invweight0))
        in
        let diagA_r =
          Nx.add
            (Nx.get [ 1 ] (row b1 p.body_invweight0))
            (Nx.get [ 1 ] (row b2 p.body_invweight0))
        in
        let norm v = Nx.sqrt (Nx.sum (Nx.mul v v)) in
        let posnorm = norm (Nx.concatenate ~axis:0 [ diff; rot_pos ]) in
        for k = 0 to 2 do
          add
            ~posnorm
            ~pos:(Nx.get [ k ] diff)
            ~diagA:diagA_t
            ~jdotv:(Nx.get [ k ] jdotv)
            ~jac:(Nx.sub (col k jacp1) (col k jacp2))
        done;
        for k = 0 to 2 do
          add
            ~posnorm
            ~pos:(Nx.get [ k ] rot_pos)
            ~diagA:diagA_r
            ~jdotv:(Nx.get [ 3 + k ] jdotv)
            ~jac:(row k rot_jac)
        done
      | Joint ->
        let j1 = Int.of_float (Nx.item [ i ] p.eq_obj1id)
        and j2 = Int.of_float (Nx.item [ i ] p.eq_obj2id) in
        let data = row i p.eq_data in
        let c k = Nx.get [ k ] data in
        let coord j =
          let adr = t.jnt_qposadr.(j) in
          Nx.sub (Nx.get [ adr ] d.qpos) (Nx.get [ adr ] p.qpos0)
        in
        let dof1 = t.jnt_dofadr.(j1) in
        let e dof =
          Nx.set [ I dof ] (Nx.scalar Nx.float64 1.0) (Nx.zeros Nx.float64 [| nv |])
        in
        let dif = if j2 >= 0 then coord j2 else Nx.scalar Nx.float64 0.0 in
        let sq = Nx.mul dif dif in
        let sq_sq = Nx.mul sq sq in
        let cube = Nx.mul sq dif in
        (* The coupling is a quartic in the second joint's displacement. *)
        let poly =
          Nx.add
            (Nx.mul (c 1) dif)
            (Nx.add (Nx.mul (c 2) sq) (Nx.add (Nx.mul (c 3) cube) (Nx.mul (c 4) sq_sq)))
        in
        let deriv =
          Nx.add
            (c 1)
            (Nx.add
               (Nx.mul_s (Nx.mul (c 2) dif) 2.0)
               (Nx.add
                  (Nx.mul_s (Nx.mul (c 3) sq) 3.0)
                  (Nx.mul_s (Nx.mul (c 4) cube) 4.0)))
        in
        let jac =
          if j2 >= 0 then Nx.sub (e dof1) (Nx.mul (e t.jnt_dofadr.(j2)) deriv) else e dof1
        in
        let diagA =
          Nx.add
            (Nx.get [ dof1 ] p.dof_invweight0)
            (if j2 >= 0
             then Nx.get [ t.jnt_dofadr.(j2) ] p.dof_invweight0
             else Nx.scalar Nx.float64 0.0)
        in
        let pos = Nx.sub (Nx.sub (coord j1) (c 0)) poly in
        add ~pos ~posnorm:(Nx.abs pos) ~diagA ~jdotv:(Nx.scalar Nx.float64 0.0) ~jac
      | Tendon ->
        (* The same cubic coupling, with tendon lengths in place of joint
           coordinates. A tendon's length is measured from its value at the
           reference pose, which is a model constant ([ten_length0]) rather than
           something a configuration can supply: [qpos0] does not determine it
           without running the tendon geometry, and MuJoCo stores it for exactly
           that reason. *)
        let obj1 = Int.of_float (Nx.item [ i ] p.eq_obj1id)
        and obj2 = Int.of_float (Nx.item [ i ] p.eq_obj2id) in
        let data = row i p.eq_data in
        let c k = Nx.get [ k ] data in
        let ten = Tendon.forward m d in
        let len t_id =
          Nx.sub (Nx.get [ t_id ] ten.Tendon.length) (Nx.get [ t_id ] p.ten_length0)
        in
        let jac_of t_id = Nx.slice [ I t_id; A ] ten.Tendon.jac in
        let jac0 = jac_of obj1 in
        let pos0 = len obj1 in
        let inv obj = Nx.get [ obj ] p.ten_invweight0 in
        let dif = if obj2 >= 0 then len obj2 else Nx.scalar Nx.float64 0.0 in
        let sq = Nx.mul dif dif in
        let sq_sq = Nx.mul sq sq in
        let cube = Nx.mul sq dif in
        let poly =
          Nx.add
            (Nx.mul (c 1) dif)
            (Nx.add (Nx.mul (c 2) sq) (Nx.add (Nx.mul (c 3) cube) (Nx.mul (c 4) sq_sq)))
        in
        let deriv =
          Nx.add
            (c 1)
            (Nx.add
               (Nx.mul_s (Nx.mul (c 2) dif) 2.0)
               (Nx.add
                  (Nx.mul_s (Nx.mul (c 3) sq) 3.0)
                  (Nx.mul_s (Nx.mul (c 4) cube) 4.0)))
        in
        let jac = if obj2 >= 0 then Nx.sub jac0 (Nx.mul (jac_of obj2) deriv) else jac0 in
        let diagA = if obj2 >= 0 then Nx.add (inv obj1) (inv obj2) else inv obj1 in
        let pos =
          if obj2 >= 0 then Nx.sub (Nx.sub pos0 (c 0)) poly else Nx.sub pos0 (c 0)
        in
        add ~pos ~posnorm:(Nx.abs pos) ~diagA ~jdotv:(Nx.scalar Nx.float64 0.0) ~jac)
  done;
  let d = keep_before m !out (Nx.scalar Nx.float64 (Float.of_int !rows)) in
  { d with nefc = Nx.scalar Nx.int32 (Int32.of_int !rows) }

(* {1 Joint limits}

   One candidate row per (limited hinge or slide joint, side), in MuJoCo's
   order: joints by id, lower side before upper. Which candidates become rows
   is data-dependent, so the assembly is vectorised: activity is a tensor, the
   row index is the number of active rows before it plus the stage's offset,
   and an inactive candidate is sent past the buffer where a scatter writes
   nothing. *)

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
    | (Hinge | Slide) when t.jnt_limited.(j) ->
      cands
      := { joint = j; dof = t.jnt_dofadr.(j); side = -1.0; range_index = 0 } :: !cands;
      cands
      := { joint = j; dof = t.jnt_dofadr.(j); side = 1.0; range_index = 1 } :: !cands
    | Ball | Free | Hinge | Slide -> ()
  done;
  Array.of_list (List.rev !cands)

let limits (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  let nv = t.nv in
  let nb = t.nefcmax in
  let cands = limit_candidates t in
  let ncand = Array.length cands in
  let scalar_like x = Nx.scalar Nx.float64 x in
  let dists = ref (Nx.zeros Nx.float64 [| ncand |])
  and margins = ref (Nx.zeros Nx.float64 [| ncand |])
  and active = ref (Nx.zeros Nx.float64 [| ncand |])
  and jac = ref (Nx.zeros Nx.float64 [| ncand; nv |]) in
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
    dists := Nx.set [ I i ] dist !dists;
    margins := Nx.set [ I i ] margin_i !margins;
    active := Nx.set [ I i ] (Nx.where on (scalar_like 1.0) (scalar_like 0.0)) !active;
    jac := Nx.set [ I i; I c.dof ] (Nx.scalar_like dist (-.c.side)) !jac
  done;
  let start = Nx.cast Nx.float64 d.nefc in
  let before = Nx.sub (Nx.cumsum ~axis:0 !active) !active in
  let far = Nx.full_like before (Float.of_int nb) in
  let idx =
    Nx.cast Nx.int64 (Nx.where (Nx.greater_s !active 0.5) (Nx.add start before) far)
  in
  let put v x = Nx.scatter ~combine:Set ~axis:0 idx v x in
  (* A scatter is elementwise: a multi-column update repeats the row index in
     every column. *)
  let put_cols v x =
    let cols = (Nx.shape v).(1) in
    let idx_rows = Nx.broadcast_to [| ncand; cols |] (Nx.reshape [| ncand; 1 |] idx) in
    Nx.scatter ~combine:Set ~axis:0 idx_rows v x
  in
  let jnt = Nx.init Nx.int32 [| ncand |] (fun i -> Int32.of_int cands.(i.(0)).joint) in
  let dof = Nx.init Nx.int32 [| ncand |] (fun i -> Int32.of_int cands.(i.(0)).dof) in
  let take k x = Nx.take ~axis:0 (Nx.cast Nx.int64 k) x in
  let types = Nx.full Nx.int32 [| ncand |] (Int32.of_int type_limit_joint) in
  let zero = Nx.zeros Nx.float64 [| ncand |] in
  (* Rows from earlier stages survive; anything at or past this stage's offset
     is stale and is cleared before the scatter. *)
  let d = keep_before m d start in
  let d =
    { d with
      efc_type = put types d.efc_type
    ; efc_id = put jnt d.efc_id
    ; efc_pos = put !dists d.efc_pos
    ; efc_margin = put !margins d.efc_margin
    ; efc_posnorm = put !dists d.efc_posnorm
    ; efc_frictionloss = put (Nx.zeros Nx.float64 [| ncand |]) d.efc_frictionloss
    ; efc_diagA = put (take dof p.dof_invweight0) d.efc_diagA
    ; efc_solref = put_cols (take jnt p.jnt_solref) d.efc_solref
    ; efc_solimp = put_cols (take jnt p.jnt_solimp) d.efc_solimp
    ; efc_jdotv = put zero d.efc_jdotv
    ; efc_J = put_cols !jac d.efc_J
    ; nefc = Nx.add d.nefc (Nx.cast Nx.int32 (Nx.sum !active))
    }
  in
  d

(* {1 Tendon limits}

   A limited tendon's range is a row per side, after the joint limits and in
   MuJoCo's order. The row's Jacobian is the tendon's own, which is the whole
   point of having one: [ten_length] is a sum of joint coordinates, so its
   gradient is a vector of gears, and the row pushes on every dof the tendon
   crosses at once. *)

type tcand =
  { tendon : int
  ; side : float
  ; range_index : int
  }

let tendon_candidates (t : Topo.t) : tcand array =
  let cands = ref [] in
  for i = 0 to t.ntendon - 1 do
    if t.ten_limited.(i)
    then (
      cands := { tendon = i; side = -1.0; range_index = 0 } :: !cands;
      cands := { tendon = i; side = 1.0; range_index = 1 } :: !cands)
  done;
  Array.of_list (List.rev !cands)

let tendon_limits (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  let nv = t.nv in
  let nb = t.nefcmax in
  let cands = tendon_candidates t in
  let ncand = Array.length cands in
  if Int.equal ncand 0
  then d
  else (
    let jac_tendon = (Tendon.forward m d).Tendon.jac in
    let scalar_like x = Nx.scalar Nx.float64 x in
    let dists = ref (Nx.zeros Nx.float64 [| ncand |])
    and margins = ref (Nx.zeros Nx.float64 [| ncand |])
    and active = ref (Nx.zeros Nx.float64 [| ncand |])
    and jac = ref (Nx.zeros Nx.float64 [| ncand; nv |]) in
    for i = 0 to ncand - 1 do
      let c = cands.(i) in
      let value = Nx.get [ c.tendon ] d.ten_length in
      let bound = Nx.get [ c.tendon; c.range_index ] p.ten_range in
      let margin_i = Nx.get [ c.tendon ] p.ten_margin in
      (* [dist = side (bound - value)]: positive inside the range, which is the
         same convention the joint limits use, so the impedance below needs to
         know nothing new. *)
      let dist = Nx.mul_s (Nx.sub bound value) c.side in
      let on = Nx.less dist margin_i in
      dists := Nx.set [ I i ] dist !dists;
      margins := Nx.set [ I i ] margin_i !margins;
      active := Nx.set [ I i ] (Nx.where on (scalar_like 1.0) (scalar_like 0.0)) !active;
      jac := Nx.set [ I i ] (Nx.mul_s (Nx.get [ c.tendon ] jac_tendon) (-.c.side)) !jac
    done;
    let start = Nx.cast Nx.float64 d.nefc in
    let before = Nx.sub (Nx.cumsum ~axis:0 !active) !active in
    let far = Nx.full_like before (Float.of_int nb) in
    let idx =
      Nx.cast Nx.int64 (Nx.where (Nx.greater_s !active 0.5) (Nx.add start before) far)
    in
    let put v x = Nx.scatter ~combine:Set ~axis:0 idx v x in
    let put_cols v x =
      let cols = (Nx.shape v).(1) in
      let idx_rows = Nx.broadcast_to [| ncand; cols |] (Nx.reshape [| ncand; 1 |] idx) in
      Nx.scatter ~combine:Set ~axis:0 idx_rows v x
    in
    let ids = Nx.init Nx.int32 [| ncand |] (fun i -> Int32.of_int cands.(i.(0)).tendon) in
    let take k x = Nx.take ~axis:0 (Nx.cast Nx.int64 k) x in
    let types = Nx.full Nx.int32 [| ncand |] (Int32.of_int type_limit_tendon) in
    let zero = Nx.zeros Nx.float64 [| ncand |] in
    { d with
      efc_type = put types d.efc_type
    ; efc_id = put ids d.efc_id
    ; efc_pos = put !dists d.efc_pos
    ; efc_margin = put !margins d.efc_margin
    ; efc_diagA = put (take ids p.ten_invweight0) d.efc_diagA
    ; efc_solref = put_cols (take ids p.ten_solref) d.efc_solref
    ; efc_solimp = put_cols (take ids p.ten_solimp) d.efc_solimp
    ; efc_jdotv = put zero d.efc_jdotv
    ; efc_J = put_cols !jac d.efc_J
    ; nefc = Nx.add d.nefc (Nx.cast Nx.int32 (Nx.sum !active))
    })

(* {1 Impedance}

   [mj_regularizeConstraint] and [mj_referenceConstraint] for the rows already
   in the buffer: the regularisation [R = (1-imp) diagA / imp], its reciprocal
   [D], and the reference acceleration the solver drives the row towards. *)

let min_val = 1e-15
let min_imp = 0.0001
let max_imp = 0.9999

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

(* [dt] is the model's timestep. It is read from the model when the caller does
   not say; the differentiable entry point passes it, for the reason
   {!Solver.forward}'s budget is passed. *)
let impedance ?dt (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  let nb = t.nefcmax in
  let rows = Nx.arange_f Nx.float64 0.0 (Float.of_int nb) 1.0 in
  let mask = Nx.less rows (Nx.cast Nx.float64 d.nefc) in
  let solimp =
    let clamp = Nx.clamp ~min:min_imp ~max:max_imp in
    let s = d.efc_solimp in
    Nx.stack
      ~axis:1
      [ clamp (col 0 s)
      ; clamp (col 1 s)
      ; Nx.maximum_s (col 2 s) 0.0
      ; clamp (col 3 s)
      ; Nx.maximum_s (col 4 s) 1.0
      ]
  in
  let solref = d.efc_solref in
  (* solref: one default for the mixed format, and the refsafe clamp. *)
  let dt =
    match dt with
    | Some x -> x
    | None -> Nx.item [] p.timestep
  in
  let r0 = col 0 solref
  and r1 = col 1 solref in
  let mixed = Nx.logical_xor (Nx.greater_s r0 0.0) (Nx.greater_s r1 0.0) in
  (* The refsafe clamp is for the time-constant form only: a direct-form solref
     is a negative stiffness, and taking its maximum against 2 dt would turn it
     into a four-millisecond time constant — a soft limit on a model that asked
     for a stiff one. *)
  let r0 =
    Nx.where
      mixed
      (Nx.full_like r0 0.02)
      (Nx.where (Nx.greater_s r0 0.0) (Nx.maximum_s r0 (2.0 *. dt)) r0)
  in
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
  let diagA = d.efc_diagA in
  let imp, _impP = getimpedance ~solimp ~pos:d.efc_posnorm ~margin:d.efc_margin in
  let r =
    Nx.maximum_s
      (Nx.div (Nx.mul (one_minus imp) diagA) (Nx.maximum_s imp min_val))
      min_val
  in
  let r = Nx.where mask r (Nx.ones_like r) in
  let dcoef = Nx.where mask (inv r) (Nx.zeros_like r) in
  let vel = Nx.reshape [| nb |] (Nx.matmul d.efc_J (Nx.reshape [| t.nv; 1 |] d.qvel)) in
  let vel = Nx.where mask vel (Nx.zeros_like rows) in
  let aref =
    Nx.sub
      (Nx.sub
         (Nx.neg (Nx.mul bcoef vel))
         (Nx.mul (Nx.mul k imp) (Nx.sub d.efc_pos d.efc_margin)))
      d.efc_jdotv
  in
  let aref = Nx.where mask aref (Nx.zeros_like aref) in
  { d with efc_vel = vel; efc_R = r; efc_D = dcoef; efc_aref = aref }

(* {1 Friction loss}

   One row per dof with a non-zero friction loss, in dof order, appended after
   the equalities. A friction row measures nothing: its position is zero and
   its force is bounded by the loss, which is what turns it from a quadratic
   row into a linear one once the row's velocity error is large enough. *)

(* The rows an equality contributes, which is also where the friction stage
   starts. It is a property of the model, so it is a host count: the equality
   stage writes its rows at host indices. *)
let equality_row_count (m : Model.host) =
  let t = m.topo
  and p = m.params in
  let n = ref 0 in
  for i = 0 to t.neq - 1 do
    if Float.( > ) (Nx.item [ i ] p.eq_active0) 0.5
    then
      n
      := !n
         +
         match t.eq_type.(i) with
         | Connect -> 3
         | Weld -> 6
         | Joint | Tendon -> 1
  done;
  !n

let friction (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  let nv = t.nv in
  let start = equality_row_count m in
  let out = ref d in
  let rows = ref start in
  for dof = 0 to nv - 1 do
    let loss = Nx.get [ dof ] p.dof_frictionloss in
    if Float.( > ) t.dof_frictionloss.(dof) 0.0
    then (
      let solref = row dof p.dof_solref
      and solimp = row dof p.dof_solimp in
      let jac =
        Nx.set [ I dof ] (Nx.scalar Nx.float64 1.0) (Nx.zeros Nx.float64 [| nv |])
      in
      out
      := set_row
           !out
           ~at:!rows
           { kind = type_friction_dof
           ; id = dof
           ; pos = Nx.scalar Nx.float64 0.0
           ; margin = Nx.scalar Nx.float64 0.0
           ; posnorm = Nx.scalar Nx.float64 0.0
           ; frictionloss = loss
           ; diagA = Nx.get [ dof ] p.dof_invweight0
           ; solref
           ; solimp
           ; jdotv = Nx.scalar Nx.float64 0.0
           ; jac
           };
      Int.incr rows)
  done;
  let d = keep_before m !out (Nx.scalar Nx.float64 (Float.of_int !rows)) in
  { d with nefc = Nx.scalar Nx.int32 (Int32.of_int !rows) }

(* {1 Contacts}

   One row per contact while v1 computes frictionless contacts only: the normal
   row, whose position is the penetration depth and whose force may only push.
   A model whose geoms ask for a frictional contact is refused here rather than
   solved with rows that do not match MuJoCo's. *)

(* The uniform path: every geom has the same condim, so every contact owns the
   same number of rows and the row index is a stride. This is the shape the
   compiled step keeps small — a per-contact prefix sum in the graph is correct
   and much slower to compile — and it is what every model but the humanoid
   corpus exercises. *)
let contact_uniform (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  (* A contact contributes one row when it is frictionless and 2*(dim-1) rows
     as a pyramidal cone, one pair of opposing edges per friction axis. Every
     geom of a model must ask for the same dimensionality: mixing them would
     make the row count data-dependent, and a row index has to be a host
     value. Requiring both edges of a pair to carry a non-negative force is
     what produces the cone, so the solver needs nothing new. *)
  let dim = Array.fold t.geom_condim ~init:1 ~f:Int.max in
  (* The cone is a model option, and v1 has one of the two: a pyramidal cone is
     a pair of one-sided rows per friction axis, which is what a box-constrained
     solver can express exactly. An elliptic cone is a quadratic row constraint,
     and running a model that asked for it as if it had the other would be a
     different engine with the same name. *)
  if Float.( > ) (Nx.item [] m.params.cone) 0.5
  then
    raise
      (Invalid_argument
         "contact friction: an elliptic friction cone (cone=\"elliptic\") is not \
          implemented; v1 solves a pyramidal cone");
  let size = if dim <= 1 then 1 else 2 * (dim - 1) in
  if size > 4
  then raise (Invalid_argument "contact friction (condim > 3) is not implemented yet");
  let nb = t.nconmax in
  let start = Nx.cast Nx.float64 d.nefc in
  let slots = Nx.arange_f Nx.float64 0.0 (Float.of_int nb) 1.0 in
  let live = Nx.less slots (Nx.cast Nx.float64 d.ncon) in
  let far = Nx.full_like slots (Float.of_int t.nefcmax) in
  let row_index k =
    let stride = Nx.mul slots (Nx.scalar Nx.float64 (Float.of_int size)) in
    Nx.cast
      Nx.int64
      (Nx.where
         live
         (Nx.add start (Nx.add stride (Nx.scalar Nx.float64 (Float.of_int k))))
         far)
  in
  let put idx v x = Nx.scatter ~combine:Set ~axis:0 idx v x in
  let put_cols idx v x =
    let cols = (Nx.shape v).(1) in
    let idx_rows = Nx.broadcast_to [| nb; cols |] (Nx.reshape [| nb; 1 |] idx) in
    Nx.scatter ~combine:Set ~axis:0 idx_rows v x
  in
  let block k = Nx.slice [ A; R (k * t.nv, (k + 1) * t.nv) ] d.contact_jac in
  let normal_jac = block 0 in
  let mu = Nx.slice [ A; I 0 ] d.contact_friction in
  let weight side column =
    let ids = Nx.cast Nx.int64 (Nx.slice [ A; I side ] d.contact_body) in
    Nx.slice [ A; I column ] (Nx.take ~axis:0 ids p.body_invweight0)
  in
  let tran_weight = Nx.add (weight 0 0) (weight 1 0) in
  let rot_weight = Nx.add (weight 0 1) (weight 1 1) in
  let ids = Nx.init Nx.int32 [| nb |] (fun i -> Int32.of_int i.(0)) in
  let zero = Nx.zeros Nx.float64 [| nb |] in
  let spin = Nx.zeros Nx.float64 [| nb |] in
  ignore spin;
  let base = keep_before m d start in
  let write (out : Data.host ref) idx kind diagA jac =
    { !out with
      efc_type = put idx (Nx.full Nx.int32 [| nb |] (Int32.of_int kind)) !out.efc_type
    ; efc_id = put idx ids !out.efc_id
    ; efc_pos = put idx d.contact_dist !out.efc_pos
    ; efc_margin = put idx d.contact_includemargin !out.efc_margin
    ; efc_posnorm = put idx d.contact_dist !out.efc_posnorm
    ; efc_frictionloss = put idx zero !out.efc_frictionloss
    ; efc_diagA = put idx diagA !out.efc_diagA
    ; efc_solref = put_cols idx d.contact_solref !out.efc_solref
    ; efc_solimp = put_cols idx d.contact_solimp !out.efc_solimp
    ; efc_jdotv = put idx zero !out.efc_jdotv
    ; efc_J = put_cols idx jac !out.efc_J
    }
  in
  let out = ref base in
  if Int.equal size 1
  then out := write out (row_index 0) type_contact_frictionless tran_weight normal_jac
  else
    for k = 1 to dim - 1 do
      let tangent = block k in
      (* MuJoCo indexes the friction axis from zero here: only a condim above
         4 reaches the rotational figure. *)
      ignore (k, rot_weight);
      (* MuJoCo's diagonal for a pyramid edge is not the row's own compliance:
         measured against the C engine over six friction coefficients it is
         [tran * 2 mu^2 (1 + mu^2)], where [tran] is the two bodies' summed
         translational inverse weight and [(1+mu^2)] is the edge row's squared
         norm. The source's per-axis expression could not be reconciled with
         that result, so the port reproduces the result. *)
      let mu2 = Nx.mul mu mu in
      let diagA =
        Nx.mul
          tran_weight
          (Nx.mul (Nx.mul_s mu2 2.0) (Nx.add (Nx.scalar Nx.float64 1.0) mu2))
      in
      for side = 0 to 1 do
        let sign = if Int.equal side 0 then 1.0 else -1.0 in
        let idx = row_index ((2 * (k - 1)) + side) in
        let scale = Nx.reshape [| nb; 1 |] (Nx.mul_s mu sign) in
        let jac = Nx.add normal_jac (Nx.mul tangent scale) in
        out := write out idx type_contact_pyramidal diagA jac
      done
    done;
  { !out with
    nefc = Nx.add d.nefc (Nx.mul d.ncon (Nx.scalar Nx.int32 (Int32.of_int size)))
  }

(* The mixed path: a contact's row count follows its own condim. MuJoCo's rule
   is one row for a frictionless contact and [2*(dim-1)] for a pyramidal one, so
   a model that mixes condim-1 and condim-3 geoms — a humanoid against its
   floor — needs a running sum over the blocks, not a stride. *)
let contact_mixed (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  (* A contact contributes one row when it is frictionless and [2*(dim-1)] rows
     as a pyramidal cone, one pair of opposing edges per friction axis. [dim] is
     the *contact's* own dimensionality — the larger of its two geoms' condims —
     not the model's: a humanoid has condim-1 geoms and condim-3 ones at once,
     and the row count has to follow MuJoCo's. So the block of rows a contact
     owns is [needed c] long, and a row's index is the running sum of the blocks
     before it. Both are tensors, because which contacts exist is data.
     Requiring both edges of a pair to carry a non-negative force is what
     produces the cone, so the solver needs nothing new. *)
  let max_dim = Array.fold t.geom_condim ~init:1 ~f:Int.max in
  (* The cone is a model option, and v1 has one of the two: a pyramidal cone is
     a pair of one-sided rows per friction axis, which is what a box-constrained
     solver can express exactly. An elliptic cone is a quadratic row constraint,
     and running a model that asked for it as if it had the other would be a
     different engine with the same name. *)
  if Float.( > ) (Nx.item [] m.params.cone) 0.5
  then
    raise
      (Invalid_argument
         "contact friction: an elliptic friction cone (cone=\"elliptic\") is not \
          implemented; v1 solves a pyramidal cone");
  if 2 * (max_dim - 1) > 4
  then raise (Invalid_argument "contact friction (condim > 3) is not implemented yet");
  let nb = t.nconmax in
  let start = Nx.cast Nx.float64 d.nefc in
  let slots = Nx.arange_f Nx.float64 0.0 (Float.of_int nb) 1.0 in
  let live = Nx.less slots (Nx.cast Nx.float64 d.ncon) in
  let far = Nx.full_like slots (Float.of_int t.nefcmax) in
  let dims = Nx.cast Nx.float64 d.contact_dim in
  let frictionless = Nx.equal_s dims 1.0 in
  let needed =
    Nx.where
      live
      (Nx.where
         frictionless
         (Nx.ones_like dims)
         (Nx.maximum (Nx.mul_s (Nx.sub_s dims 1.0) 2.0) (Nx.zeros_like dims)))
      (Nx.zeros_like dims)
  in
  (* The running sum, as a matrix product with the strictly lower triangle: the
     per-slot counts times an [nb;nb] mask. *)
  let lower =
    Nx.cast
      Nx.float64
      (Nx.greater (Nx.reshape [| nb; 1 |] slots) (Nx.reshape [| 1; nb |] slots))
  in
  let offset =
    Nx.add start (Nx.reshape [| nb |] (Nx.matmul lower (Nx.reshape [| nb; 1 |] needed)))
  in
  let row_index k exists =
    Nx.cast
      Nx.int64
      (Nx.where exists (Nx.add offset (Nx.scalar Nx.float64 (Float.of_int k))) far)
  in
  let put idx v x = Nx.scatter ~combine:Set ~axis:0 idx v x in
  let put_cols idx v x =
    let cols = (Nx.shape v).(1) in
    let idx_rows = Nx.broadcast_to [| nb; cols |] (Nx.reshape [| nb; 1 |] idx) in
    Nx.scatter ~combine:Set ~axis:0 idx_rows v x
  in
  let block k = Nx.slice [ A; R (k * t.nv, (k + 1) * t.nv) ] d.contact_jac in
  let normal_jac = block 0 in
  let mu = Nx.slice [ A; I 0 ] d.contact_friction in
  let weight side column =
    let ids = Nx.cast Nx.int64 (Nx.slice [ A; I side ] d.contact_body) in
    Nx.slice [ A; I column ] (Nx.take ~axis:0 ids p.body_invweight0)
  in
  let tran_weight = Nx.add (weight 0 0) (weight 1 0) in
  let ids = Nx.init Nx.int32 [| nb |] (fun i -> Int32.of_int i.(0)) in
  let zero = Nx.zeros Nx.float64 [| nb |] in
  let base = keep_before m d start in
  let write (out : Data.host ref) idx kind diagA jac =
    { !out with
      efc_type = put idx kind !out.efc_type
    ; efc_id = put idx ids !out.efc_id
    ; efc_pos = put idx d.contact_dist !out.efc_pos
    ; efc_margin = put idx d.contact_includemargin !out.efc_margin
    ; efc_posnorm = put idx d.contact_dist !out.efc_posnorm
    ; efc_frictionloss = put idx zero !out.efc_frictionloss
    ; efc_diagA = put idx diagA !out.efc_diagA
    ; efc_solref = put_cols idx d.contact_solref !out.efc_solref
    ; efc_solimp = put_cols idx d.contact_solimp !out.efc_solimp
    ; efc_jdotv = put idx zero !out.efc_jdotv
    ; efc_J = put_cols idx jac !out.efc_J
    }
  in
  let mu2 = Nx.mul mu mu in
  (* MuJoCo's diagonal for a pyramid edge is not the row's own compliance:
     measured against the C engine over six friction coefficients it is
     [tran * 2 mu^2 (1 + mu^2)], where [tran] is the two bodies' summed
     translational inverse weight and [(1+mu^2)] is the edge row's squared
     norm. The source's per-axis expression could not be reconciled with that
     result, so the port reproduces the result. *)
  let pyramid_diagA =
    Nx.mul tran_weight (Nx.mul (Nx.mul_s mu2 2.0) (Nx.add (Nx.scalar Nx.float64 1.0) mu2))
  in
  let out = ref base in
  (* One host iteration per possible slot: a tangent block is a slice, so the
     friction axis the slot belongs to has to be known here. Whether the row
     exists does not, and stays a tensor mask. *)
  let max_slots = if max_dim <= 1 then 1 else 2 * (max_dim - 1) in
  for k = 0 to max_slots - 1 do
    let axis = (k / 2) + 1 in
    let sign = if Int.equal (Int.rem k 2) 0 then 1.0 else -1.0 in
    (* Slot [k] exists for a contact that owns more than [k] rows. A single-row
       contact owns one, so slot zero exists and slot one does not. *)
    let exists = Nx.greater needed (Nx.full_like needed (Float.of_int k)) in
    let scale = Nx.reshape [| nb; 1 |] (Nx.mul_s mu sign) in
    let jac_pyramid = Nx.add normal_jac (Nx.mul (block axis) scale) in
    let as_normal = if Int.equal k 0 then frictionless else Nx.zeros_like frictionless in
    let jac = Nx.where (Nx.reshape [| nb; 1 |] as_normal) normal_jac jac_pyramid in
    let diagA = Nx.where as_normal tran_weight pyramid_diagA in
    let kind =
      Nx.where
        as_normal
        (Nx.full Nx.int32 [| nb |] (Int32.of_int type_contact_frictionless))
        (Nx.full Nx.int32 [| nb |] (Int32.of_int type_contact_pyramidal))
    in
    out := write out (row_index k exists) kind diagA jac
  done;
  { !out with nefc = Nx.add d.nefc (Nx.cast Nx.int32 (Nx.sum needed)) }

let all_same_condim t =
  let first = t.geom_condim.(0) in
  Array.for_all t.geom_condim ~f:(Int.equal first)

let contact (m : Model.host) (d : Data.host) : Data.host =
  if all_same_condim m.topo then contact_uniform m d else contact_mixed m d
