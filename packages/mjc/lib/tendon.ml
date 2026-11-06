(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base

type 'a t =
  { jac : 'a Nx.float64_t
  ; length : 'a Nx.float64_t
  ; velocity : 'a Nx.float64_t
  }

let check (t : Topo.t) =
  (* Tendon limits are constraint rows now ({!Constraint.tendon_limits}), so a
     limited tendon is no longer refused here. *)
  Array.iteri t.wrap_type ~f:(fun w kind ->
    match kind with
    | Topo.Wrap_joint -> ()
    | Topo.Wrap_site ->
      raise
        (Invalid_argument
           (Printf.sprintf
              "tendon wrap %d: a site wrap needs the spatial tendon path, which v1 does \
               not have"
              w)))

(* [wraps t] is every path point with the tendon that owns it: a fixed tendon
   is a list of joints, so one pass over the tendons gives the whole scatter. *)
let wraps (t : Topo.t) =
  let rows = ref []
  and dofs = ref []
  and qposadr = ref []
  and ids = ref [] in
  for i = 0 to t.ntendon - 1 do
    for k = 0 to t.tendon_num.(i) - 1 do
      let w = t.tendon_adr.(i) + k in
      let id = t.wrap_objid.(w) in
      rows := i :: !rows;
      dofs := t.jnt_dofadr.(id) :: !dofs;
      qposadr := t.jnt_qposadr.(id) :: !qposadr;
      ids := w :: !ids
    done
  done;
  List.rev !rows, List.rev !dofs, List.rev !qposadr, List.rev !ids

let ints at f xs =
  Nx.place
    at
    (Nx.init
       Nx.int64
       [| List.length xs |]
       (fun i -> Int64.of_int (f (List.nth_exn xs i.(0)))))

let forward (m : 'a Model.model) (d : 'a Data.data) =
  let at = Nx.placement d.qpos in
  let t = m.topo in
  let ntendon = t.ntendon in
  if Int.equal ntendon 0
  then (
    let z = Nx.zeros_at (Nx.placement d.qpos) Nx.float64 [| 0 |] in
    { jac = Nx.zeros_at (Nx.placement d.qpos) Nx.float64 [| 0; t.nv |]
    ; length = z
    ; velocity = z
    })
  else (
    check t;
    let rows, dofs, qposadr, ids = wraps t in
    let n = List.length rows in
    let coef =
      if Int.equal n 0
      then Nx.zeros_at (Nx.placement d.qpos) Nx.float64 [| 0 |]
      else
        Nx.stack ~axis:0 (List.map ids ~f:(fun w -> Nx.slice [ I w ] m.params.wrap_prm))
    in
    let jac =
      if Int.equal n 0
      then Nx.zeros_at (Nx.placement d.qpos) Nx.float64 [| ntendon; t.nv |]
      else (
        let lin =
          ints
            at
            (fun k -> (List.nth_exn rows k * t.nv) + List.nth_exn dofs k)
            (List.range 0 n)
        in
        Nx.reshape
          [| ntendon; t.nv |]
          (Nx.scatter
             ~combine:Add
             ~axis:0
             lin
             coef
             (Nx.zeros_at (Nx.placement d.qpos) Nx.float64 [| ntendon * t.nv |])))
    in
    let length =
      if Int.equal n 0
      then Nx.zeros_at (Nx.placement d.qpos) Nx.float64 [| ntendon |]
      else (
        let qpos_at =
          Nx.take ~axis:0 (ints at (List.nth_exn qposadr) (List.range 0 n)) d.qpos
        in
        Nx.reshape
          [| ntendon |]
          (Nx.scatter
             ~combine:Add
             ~axis:0
             (ints at (List.nth_exn rows) (List.range 0 n))
             (Nx.mul coef qpos_at)
             (Nx.zeros_at (Nx.placement d.qpos) Nx.float64 [| ntendon |])))
    in
    let velocity = Nx.sum ~axes:[ 1 ] (Nx.mul jac (Nx.reshape [| 1; t.nv |] d.qvel)) in
    { jac; length; velocity })
