(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base

type t =
  { jac : Nx.host Nx.float64_t
  ; length : Nx.host Nx.float64_t
  ; velocity : Nx.host Nx.float64_t
  }

let check (t : Topo.t) =
  Array.iteri t.ten_limited ~f:(fun i limited ->
    if limited
    then
      raise
        (Invalid_argument
           (Printf.sprintf
              "tendon %d: a limited tendon is a constraint row, which v1 does not have \
               yet"
              i)));
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

let ints f xs =
  Nx.init
    Nx.int64
    [| List.length xs |]
    (fun i -> Int64.of_int (f (List.nth_exn xs i.(0))))

let forward (m : Model.host) (d : Data.host) =
  let t = m.topo in
  let ntendon = t.ntendon in
  if Int.equal ntendon 0
  then (
    let z = Nx.zeros Nx.float64 [| 0 |] in
    { jac = Nx.zeros Nx.float64 [| 0; t.nv |]; length = z; velocity = z })
  else (
    check t;
    let rows, dofs, qposadr, ids = wraps t in
    let n = List.length rows in
    let coef =
      if Int.equal n 0
      then Nx.zeros Nx.float64 [| 0 |]
      else
        Nx.stack ~axis:0 (List.map ids ~f:(fun w -> Nx.slice [ I w ] m.params.wrap_prm))
    in
    let jac =
      if Int.equal n 0
      then Nx.zeros Nx.float64 [| ntendon; t.nv |]
      else (
        let lin =
          ints
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
             (Nx.zeros Nx.float64 [| ntendon * t.nv |])))
    in
    let length =
      if Int.equal n 0
      then Nx.zeros Nx.float64 [| ntendon |]
      else (
        let qpos_at =
          Nx.take ~axis:0 (ints (List.nth_exn qposadr) (List.range 0 n)) d.qpos
        in
        Nx.reshape
          [| ntendon |]
          (Nx.scatter
             ~combine:Add
             ~axis:0
             (ints (List.nth_exn rows) (List.range 0 n))
             (Nx.mul coef qpos_at)
             (Nx.zeros Nx.float64 [| ntendon |])))
    in
    let velocity = Nx.sum ~axes:[ 1 ] (Nx.mul jac (Nx.reshape [| 1; t.nv |] d.qvel)) in
    { jac; length; velocity })
