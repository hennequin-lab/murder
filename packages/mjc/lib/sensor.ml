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

(* A body's velocity at a point: MuJoCo's [mj_objectVelocity] for a point
   [origin] on body [b], [v = cvel_tran - (origin - subtree_com(root)) x cvel_rot]. The object type decides which origin MuJoCo uses: a body sensor
   measures at the inertial frame, a body-frame sensor at [xpos]. *)
let point_vel (m : 'a Model.model) (d : 'a Data.data) ~body ~origin =
  let cv = Nx.get [ body ] d.cvel in
  let rot = Nx.reshape [| 3 |] (Nx.slice [ R (0, 3) ] cv)
  and tran = Nx.reshape [| 3 |] (Nx.slice [ R (3, 6) ] cv) in
  let com = Nx.get [ m.topo.body_rootid.(body) ] d.subtree_com in
  Nx.concatenate ~axis:0 [ rot; Nx.sub tran (Nx.cross (Nx.sub origin com) rot) ]

let three v = Array.init 3 ~f:(fun k -> Nx.get [ k ] v)

let compute (m : 'a Model.model) (d : 'a Data.data) (i : int) =
  let t = m.topo in
  let id = t.sensor_objid.(i) in
  let reject what =
    raise
      (Invalid_argument
         (Printf.sprintf
            "sensor %d: %s is not a frame, so it has no pose or velocity"
            i
            what))
  in
  (* The pose of the frame a sensor names: the inertial frame of a body, the
     frame of a body, or a site. *)
  let frame_pos () =
    match t.sensor_objtype.(i) with
    | Topo.Obj_body -> Nx.get [ id ] d.xipos
    | Topo.Obj_xbody -> Nx.get [ id ] d.xpos
    | Topo.Obj_site -> Nx.get [ id ] d.site_xpos
    | Topo.Obj_joint | Topo.Obj_actuator | Topo.Obj_tendon ->
      reject (Printf.sprintf "object type of sensor %d" i)
  in
  let frame_vel () =
    match t.sensor_objtype.(i) with
    | Topo.Obj_body -> point_vel m d ~body:id ~origin:(Nx.get [ id ] d.xipos)
    | Topo.Obj_xbody -> point_vel m d ~body:id ~origin:(Nx.get [ id ] d.xpos)
    | Topo.Obj_site ->
      point_vel m d ~body:t.site_bodyid.(id) ~origin:(Nx.get [ id ] d.site_xpos)
    | Topo.Obj_joint | Topo.Obj_actuator | Topo.Obj_tendon ->
      reject (Printf.sprintf "object type of sensor %d" i)
  in
  match t.sensor_type.(i) with
  | Topo.Sens_jointpos ->
    (match t.jnt_type.(id) with
     | Topo.Hinge | Topo.Slide -> [| Nx.get [ t.jnt_qposadr.(id) ] d.qpos |]
     | Topo.Ball | Topo.Free ->
       raise
         (Invalid_argument
            (Printf.sprintf
               "sensor %d: jointpos of a joint with more than one coordinate, which v1 \
                does not have"
               i)))
  | Topo.Sens_jointvel ->
    (match t.jnt_type.(id) with
     | Topo.Hinge | Topo.Slide -> [| Nx.get [ t.jnt_dofadr.(id) ] d.qvel |]
     | Topo.Ball | Topo.Free ->
       raise
         (Invalid_argument
            (Printf.sprintf
               "sensor %d: jointvel of a joint with more than one degree of freedom, \
                which v1 does not have"
               i)))
  | Topo.Sens_actuatorfrc -> [| Nx.get [ id ] d.actuator_force |]
  | Topo.Sens_tendonpos -> [| Nx.get [ id ] d.ten_length |]
  | Topo.Sens_tendonvel -> [| Nx.get [ id ] d.ten_velocity |]
  | Topo.Sens_subtreecom -> three (Nx.get [ id ] d.subtree_com)
  | Topo.Sens_subtreelinvel -> three (Nx.get [ id ] d.subtree_linvel)
  | Topo.Sens_subtreeangmom -> three (Nx.get [ id ] d.subtree_angmom)
  | Topo.Sens_framepos -> three (frame_pos ())
  | Topo.Sens_framequat ->
    (* The inertial frame's orientation is [ximat]; a body frame's is [xquat]
       already; a site's is its matrix. *)
    let of_mat r9 =
      let q = Rigid.quat_of_mat (Nx.reshape [| 1; 9 |] r9) in
      Array.init 4 ~f:(fun k -> Nx.get [ 0; k ] q)
    in
    (match t.sensor_objtype.(i) with
     | Topo.Obj_body -> of_mat (Nx.get [ id ] d.ximat)
     | Topo.Obj_xbody ->
       let q = Nx.get [ id ] d.xquat in
       Array.init 4 ~f:(fun k -> Nx.get [ k ] q)
     | Topo.Obj_site -> of_mat (Nx.get [ id ] d.site_xmat)
     | Topo.Obj_joint | Topo.Obj_actuator | Topo.Obj_tendon -> reject "a non-frame object")
  | Topo.Sens_framelinvel -> three (Nx.slice [ R (3, 6) ] (frame_vel ()))
  | Topo.Sens_frameangvel -> three (Nx.slice [ R (0, 3) ] (frame_vel ()))

let forward (m : 'a Model.model) (d : 'a Data.data) =
  let t = m.topo in
  if Int.equal t.nsensor 0
  then d
  else (
    check t;
    let block = ref (Nx.zeros_at (Nx.placement d.qpos) f64 [| t.nsensordata |]) in
    Array.iteri t.sensor_type ~f:(fun i _ ->
      let parts = compute m d i in
      Array.iteri parts ~f:(fun k v ->
        block := Nx.set [ I (t.sensor_adr.(i) + k) ] v !block));
    { d with Data.sensordata = !block })
