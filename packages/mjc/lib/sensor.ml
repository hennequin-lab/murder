(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx

let f64 = Nx.float64

let check (t : Topo.t) =
  Array.iteri t.sensor_refid ~f:(fun i refid ->
    if refid >= 0
    then
      raise
        (Invalid_argument
           (Printf.sprintf
              "sensor %d: a reference frame, which v1 does not have: the measurement \
               would be relative to another object"
              i)))

let three v = Array.init 3 ~f:(fun k -> Nx.get [ k ] v)

let compute (m : Model.host) (d : Data.host) (i : int) =
  let t = m.topo in
  let id = t.sensor_objid.(i) in
  let multi_deg name =
    raise
      (Invalid_argument
         (Printf.sprintf
            "sensor %d: %s of a joint with more than one %s, which v1 does not have"
            i
            name
            (if String.equal name "jointpos" then "coordinate" else "degree of freedom")))
  in
  match t.sensor_type.(i) with
  | Topo.Sens_jointpos ->
    (match t.jnt_type.(id) with
     | Topo.Hinge | Topo.Slide -> [| Nx.get [ t.jnt_qposadr.(id) ] d.qpos |]
     | Topo.Ball | Topo.Free -> multi_deg "jointpos")
  | Topo.Sens_jointvel ->
    (match t.jnt_type.(id) with
     | Topo.Hinge | Topo.Slide -> [| Nx.get [ t.jnt_dofadr.(id) ] d.qvel |]
     | Topo.Ball | Topo.Free -> multi_deg "jointvel")
  | Topo.Sens_actuatorfrc -> [| Nx.get [ id ] d.actuator_force |]
  | Topo.Sens_tendonpos -> [| Nx.get [ id ] d.ten_length |]
  | Topo.Sens_tendonvel -> [| Nx.get [ id ] d.ten_velocity |]
  | Topo.Sens_subtreecom -> three (Nx.get [ id ] d.subtree_com)
  | Topo.Sens_subtreelinvel -> three (Nx.get [ id ] d.subtree_linvel)
  | Topo.Sens_subtreeangmom -> three (Nx.get [ id ] d.subtree_angmom)
  | Topo.Sens_framepos ->
    three
      (match t.sensor_objtype.(i) with
       | Topo.Obj_site -> Nx.get [ id ] d.site_xpos
       | Topo.Obj_body -> Nx.get [ id ] d.xpos
       | Topo.Obj_joint | Topo.Obj_actuator | Topo.Obj_tendon ->
         raise
           (Invalid_argument
              (Printf.sprintf "sensor %d: a frame position of a non-frame object" i)))

let forward (m : Model.host) (d : Data.host) =
  let t = m.topo in
  if Int.equal t.nsensor 0
  then d
  else (
    check t;
    let block = ref (Nx.zeros f64 [| t.nsensordata |]) in
    Array.iteri t.sensor_type ~f:(fun i _ ->
      let parts = compute m d i in
      Array.iteri parts ~f:(fun k v ->
        block := Nx.set [ I (t.sensor_adr.(i) + k) ] v !block));
    { d with Data.sensordata = !block })
