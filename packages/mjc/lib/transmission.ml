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

(* [gather_rows x idx] is [x]'s rows [idx.(i)], as a [n] tensor.
   [Nx.take] is a gather along an axis, so the index is a tensor and the values
   stay in the graph. *)
let gather (x : Nx.host Nx.float64_t) (idx : int array) =
  Nx.take
    ~axis:0
    (Nx.init Nx.int64 [| Array.length idx |] (fun i -> Int64.of_int idx.(i.(0))))
    x

let forward (m : Model.host) (d : Data.host) =
  let t = m.topo in
  let nu = t.nu in
  let nv = t.nv in
  if Int.equal nu 0
  then (
    let z = Nx.zeros Nx.float64 [| 0 |] in
    { moment = Nx.zeros Nx.float64 [| 0; nv |]; length = z; velocity = z })
  else (
    let dof = Array.init nu ~f:(fun _ -> 0)
    and qposadr = Array.init nu ~f:(fun _ -> 0) in
    for i = 0 to nu - 1 do
      let id = t.actuator_trnid.(2 * i) in
      match t.actuator_trntype.(i) with
      | Topo.Trn_joint ->
        (match t.jnt_type.(id) with
         | Topo.Hinge | Topo.Slide -> ()
         | Topo.Ball | Topo.Free ->
           raise
             (Invalid_argument
                (Printf.sprintf
                   "transmission: actuator %d drives joint %d, whose type has more than \
                    one degree of freedom"
                   i
                   id)));
        dof.(i) <- t.jnt_dofadr.(id);
        qposadr.(i) <- t.jnt_qposadr.(id)
      | Topo.Trn_tendon ->
        raise
          (Invalid_argument
             (Printf.sprintf
                "transmission: actuator %d drives a tendon, which needs the tendon \
                 Jacobian"
                i))
    done;
    let gear = Nx.slice [ A; I 0 ] m.params.actuator_gear in
    (* One nonzero per row, at the dof the transmission reaches. *)
    let moment =
      let zeros = Nx.zeros Nx.float64 [| nu; nv |] in
      let idx =
        Nx.reshape
          [| nu; 1 |]
          (Nx.init Nx.int64 [| nu |] (fun i -> Int64.of_int dof.(i.(0))))
      in
      let vals = Nx.reshape [| nu; 1 |] gear in
      Nx.scatter ~combine:Set ~axis:1 idx vals zeros
    in
    let scaled = Nx.mul gear (gather d.qpos qposadr) in
    let velocity = Nx.sum ~axes:[ 1 ] (Nx.mul moment (Nx.reshape [| 1; nv |] d.qvel)) in
    { moment; length = scaled; velocity })

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
