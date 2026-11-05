(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base

type t =
  { moment : Nx.host Nx.float64_t
  ; length : Nx.host Nx.float64_t
  ; velocity : Nx.host Nx.float64_t
  }

(* [take x idx] is [x]'s rows [idx], as a tensor, so the values stay in the
   graph. *)
let take (x : Nx.host Nx.float64_t) (idx : int array) =
  let n = Array.length idx in
  Nx.take ~axis:0 (Nx.init Nx.int64 [| n |] (fun i -> Int64.of_int idx.(i.(0)))) x

let zeros n = Nx.zeros Nx.float64 [| n |]
let zeros2 n k = Nx.zeros Nx.float64 [| n; k |]

(* [rows_for] scatters whole rows into a [nu;nv] tensor: MuJoCo's moment matrix
   has one row per actuator, dense here, and a tendon row is the tendon's
   Jacobian scaled by the gear. Flat, because the rows are not contiguous. *)
let rows_for ~nu ~nv (idx : int array) (rows : Nx.host Nx.float64_t) =
  let k = Array.length idx in
  if Int.equal k 0
  then zeros2 nu nv
  else (
    let elem =
      Nx.init
        Nx.int64
        [| k * nv |]
        (fun i ->
           let row = i.(0) / nv in
           Int64.of_int ((idx.(row) * nv) + (i.(0) - (nv * row))))
    in
    Nx.reshape
      [| nu; nv |]
      (Nx.scatter
         ~combine:Add
         ~axis:0
         elem
         (Nx.reshape [| k * nv |] rows)
         (Nx.zeros Nx.float64 [| nu * nv |])))

let forward (m : Model.host) (d : Data.host) =
  let t = m.topo in
  let nu = t.nu
  and nv = t.nv in
  if Int.equal nu 0
  then (
    let z = zeros 0 in
    { moment = zeros2 0 nv; length = z; velocity = z })
  else (
    let tendon = lazy (Tendon.forward m d) in
    let gear = Nx.slice [ A; I 0 ] m.params.actuator_gear in
    let joint_ids = ref []
    and tendon_ids = ref []
    and joint_of = ref []
    and qpos_of = ref [] in
    for i = nu - 1 downto 0 do
      let id = t.actuator_trnid.(2 * i) in
      match t.actuator_trntype.(i) with
      | Topo.Trn_joint ->
        (match t.jnt_type.(id) with
         | Topo.Hinge | Topo.Slide -> ()
         | Topo.Ball | Topo.Free ->
           raise
             (Invalid_argument
                (Printf.sprintf
                   "transmission: actuator %d drives a joint with more than one degree \
                    of freedom, which v1 does not have"
                   i)));
        joint_ids := i :: !joint_ids;
        joint_of := t.jnt_dofadr.(id) :: !joint_of;
        qpos_of := t.jnt_qposadr.(id) :: !qpos_of
      | Topo.Trn_tendon -> tendon_ids := i :: !tendon_ids
    done;
    let joint_ids = Array.of_list !joint_ids
    and tendon_ids = Array.of_list !tendon_ids
    and joint_of = Array.of_list !joint_of
    and qpos_of = Array.of_list !qpos_of in
    let joint_rows =
      if Int.equal (Array.length joint_ids) 0
      then zeros2 nu nv
      else (
        (* one nonzero per row, at the dof the transmission reaches *)
        let lin =
          Nx.init
            Nx.int64
            [| Array.length joint_ids |]
            (fun i -> Int64.of_int ((joint_ids.(i.(0)) * nv) + joint_of.(i.(0))))
        in
        Nx.reshape
          [| nu; nv |]
          (Nx.scatter
             ~combine:Add
             ~axis:0
             lin
             (take gear joint_ids)
             (Nx.zeros Nx.float64 [| nu * nv |])))
    in
    let tendon_rows =
      if Int.equal (Array.length tendon_ids) 0
      then zeros2 nu nv
      else (
        let ten = Lazy.force tendon in
        let ids = Array.map tendon_ids ~f:(fun i -> t.actuator_trnid.(2 * i)) in
        let rows =
          Nx.mul
            (Nx.reshape [| Array.length ids; 1 |] (take gear tendon_ids))
            (take ten.Tendon.jac ids)
        in
        rows_for ~nu ~nv tendon_ids rows)
    in
    let moment = Nx.add joint_rows tendon_rows in
    (* The length is the joint coordinate or the tendon's, scaled by the gear.
       Both are gathered and the type mask picks one, which keeps the two
       paths in one expression rather than two programs. *)
    let by_joint =
      if Int.equal (Array.length joint_ids) 0
      then zeros nu
      else Nx.mul (take gear joint_ids) (take d.qpos qpos_of)
    in
    let by_tendon =
      if Int.equal (Array.length tendon_ids) 0
      then zeros nu
      else (
        let ids = Array.map tendon_ids ~f:(fun i -> t.actuator_trnid.(2 * i)) in
        Nx.mul (take gear tendon_ids) (take (Lazy.force tendon).Tendon.length ids))
    in
    let mask =
      Nx.init Nx.float64 [| nu |] (fun i ->
        match t.actuator_trntype.(i.(0)) with
        | Topo.Trn_joint -> 0.
        | Topo.Trn_tendon -> 1.)
    in
    let length =
      Nx.add (Nx.mul mask by_tendon) (Nx.mul (Nx.sub (Nx.ones_like mask) mask) by_joint)
    in
    let velocity = Nx.sum ~axes:[ 1 ] (Nx.mul moment (Nx.reshape [| 1; nv |] d.qvel)) in
    { moment; length; velocity })

let pp ppf (t : t) =
  Stdlib.Format.fprintf
    ppf
    "@[<v 2>transmission@,moment %a@,length %a@,velocity %a@]"
    Nx.pp
    t.moment
    Nx.pp
    t.length
    Nx.pp
    t.velocity
