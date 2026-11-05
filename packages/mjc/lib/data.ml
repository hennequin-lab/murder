(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
module Format = Stdlib.Format

type ('a, 'd) t =
  { time : 'a
  ; qpos : 'a
  ; qvel : 'a
  ; qacc : 'a
  ; act : 'a
  ; act_dot : 'a
  ; ctrl : 'a
  ; qfrc_applied : 'a
  ; xpos : 'a
  ; xquat : 'a
  ; xipos : 'a
  ; ximat : 'a
  ; geom_xpos : 'a
  ; geom_xmat : 'a
  ; site_xpos : 'a
  ; site_xmat : 'a
  ; xanchor : 'a
  ; xaxis : 'a
  ; cvel : 'a
  ; cdof : 'a
  ; cdof_dot : 'a
  ; cinert : 'a
  ; com : 'a
  ; subtree_com : 'a
  ; qfrc_bias : 'a
  ; qfrc_passive : 'a
  ; qfrc_actuator : 'a
  ; actuator_length : 'a
  ; actuator_velocity : 'a
  ; actuator_force : 'a
  ; qfrc_constraint : 'a
  ; qfrc_smooth : 'a
  ; mass_matrix : 'a
  ; ncon : 'd Nx.int32_t
  ; contact_dist : 'a
  ; contact_pos : 'a
  ; contact_frame : 'a
  ; contact_normal : 'a
  ; contact_geom : 'd Nx.int32_t
  ; contact_body : 'd Nx.int32_t
  ; contact_dim : 'd Nx.int32_t
  ; contact_includemargin : 'a
  ; contact_friction : 'a
  ; contact_solref : 'a
  ; contact_jac : 'a
  ; contact_solimp : 'a
  ; nefc : 'd Nx.int32_t
  ; efc_type : 'd Nx.int32_t
  ; efc_id : 'd Nx.int32_t
  ; efc_pos : 'a
  ; efc_margin : 'a
  ; efc_posnorm : 'a
  ; efc_frictionloss : 'a
  ; efc_diagA : 'a
  ; efc_solref : 'a
  ; efc_solimp : 'a
  ; efc_jdotv : 'a
  ; efc_J : 'a
  ; efc_vel : 'a
  ; efc_R : 'a
  ; efc_D : 'a
  ; efc_aref : 'a
  ; efc_force : 'a
  }
[@@deriving ptree]

type host = (Nx.host Nx.float64_t, Nx.host) t
type ('a, 'd) desc = ('a, 'd) t

module Desc = struct
  type ('a, 'd) t = ('a, 'd) desc

  let walk = walk
end

let ptree = Nx.Ptree.instantiate (module Desc)
let nq (m : Model.host) = m.topo.nq
let nv (m : Model.host) = m.topo.nv

let create (m : Model.host) : host =
  let t = m.topo in
  let f64 = Nx.float64 in
  let zeros shape = Nx.zeros f64 shape in
  { time = Nx.scalar f64 0.0
  ; qpos = Nx.copy m.params.qpos0
  ; qvel = zeros [| t.nv |]
  ; qacc = zeros [| t.nv |]
  ; act = zeros [| t.na |]
  ; act_dot = zeros [| t.na |]
  ; ctrl = zeros [| t.nu |]
  ; qfrc_applied = zeros [| t.nv |]
  ; xpos = zeros [| t.nbody; 3 |]
  ; xquat = zeros [| t.nbody; 4 |]
  ; xipos = zeros [| t.nbody; 3 |]
  ; ximat = zeros [| t.nbody; 9 |]
  ; geom_xpos = zeros [| t.ngeom; 3 |]
  ; geom_xmat = zeros [| t.ngeom; 9 |]
  ; site_xpos = zeros [| t.nsite; 3 |]
  ; site_xmat = zeros [| t.nsite; 9 |]
  ; xanchor = zeros [| t.njnt; 3 |]
  ; xaxis = zeros [| t.njnt; 3 |]
  ; cvel = zeros [| t.nbody; 6 |]
  ; cdof = zeros [| t.nv; 6 |]
  ; cdof_dot = zeros [| t.nv; 6 |]
  ; cinert = zeros [| t.nbody; 10 |]
  ; com = zeros [| 3 |]
  ; subtree_com = zeros [| t.nbody; 3 |]
  ; qfrc_bias = zeros [| t.nv |]
  ; qfrc_passive = zeros [| t.nv |]
  ; qfrc_actuator = zeros [| t.nv |]
  ; actuator_length = zeros [| t.nu |]
  ; actuator_velocity = zeros [| t.nu |]
  ; actuator_force = zeros [| t.nu |]
  ; qfrc_constraint = zeros [| t.nv |]
  ; qfrc_smooth = zeros [| t.nv |]
  ; mass_matrix = zeros [| t.nv; t.nv |]
  ; ncon = Nx.scalar Nx.int32 0l
  ; contact_dist = zeros [| t.nconmax |]
  ; contact_pos = zeros [| t.nconmax; 3 |]
  ; contact_frame = zeros [| t.nconmax; 9 |]
  ; contact_normal = zeros [| t.nconmax; 3 |]
  ; contact_geom = Nx.zeros Nx.int32 [| t.nconmax; 2 |]
  ; contact_body = Nx.zeros Nx.int32 [| t.nconmax; 2 |]
  ; contact_dim = Nx.zeros Nx.int32 [| t.nconmax |]
  ; contact_includemargin = zeros [| t.nconmax |]
  ; contact_friction = zeros [| t.nconmax; 5 |]
  ; contact_solref = zeros [| t.nconmax; 2 |]
  ; contact_jac = zeros [| t.nconmax; 3 * t.nv |]
  ; contact_solimp = zeros [| t.nconmax; 5 |]
  ; nefc = Nx.scalar Nx.int32 0l
  ; efc_type = Nx.zeros Nx.int32 [| t.nefcmax |]
  ; efc_id = Nx.zeros Nx.int32 [| t.nefcmax |]
  ; efc_pos = zeros [| t.nefcmax |]
  ; efc_margin = zeros [| t.nefcmax |]
  ; efc_posnorm = zeros [| t.nefcmax |]
  ; efc_frictionloss = zeros [| t.nefcmax |]
  ; efc_diagA = zeros [| t.nefcmax |]
  ; efc_solref = zeros [| t.nefcmax; 2 |]
  ; efc_solimp = zeros [| t.nefcmax; 5 |]
  ; efc_jdotv = zeros [| t.nefcmax |]
  ; efc_J = zeros [| t.nefcmax; t.nv |]
  ; efc_vel = zeros [| t.nefcmax |]
  ; efc_R = zeros [| t.nefcmax |]
  ; efc_D = zeros [| t.nefcmax |]
  ; efc_aref = zeros [| t.nefcmax |]
  ; efc_force = zeros [| t.nefcmax |]
  }

let validate (m : Model.host) (d : host) : (unit, string) Result.t =
  let t = m.topo in
  let problems = ref [] in
  let want name expected x =
    let got = Nx.shape x in
    if not (Poly.equal got expected)
    then
      problems
      := Printf.sprintf
           "%s has shape [%s], expected [%s]"
           name
           (String.concat ~sep:";" (Array.to_list (Array.map got ~f:Int.to_string)))
           (String.concat ~sep:";" (Array.to_list (Array.map expected ~f:Int.to_string)))
         :: !problems
  in
  want "time" [||] d.time;
  want "qpos" [| t.nq |] d.qpos;
  want "qvel" [| t.nv |] d.qvel;
  want "qacc" [| t.nv |] d.qacc;
  want "act" [| t.na |] d.act;
  want "act_dot" [| t.na |] d.act_dot;
  want "ctrl" [| t.nu |] d.ctrl;
  want "qfrc_applied" [| t.nv |] d.qfrc_applied;
  want "xpos" [| t.nbody; 3 |] d.xpos;
  want "xquat" [| t.nbody; 4 |] d.xquat;
  want "xipos" [| t.nbody; 3 |] d.xipos;
  want "ximat" [| t.nbody; 9 |] d.ximat;
  want "geom_xpos" [| t.ngeom; 3 |] d.geom_xpos;
  want "geom_xmat" [| t.ngeom; 9 |] d.geom_xmat;
  want "site_xpos" [| t.nsite; 3 |] d.site_xpos;
  want "site_xmat" [| t.nsite; 9 |] d.site_xmat;
  want "xanchor" [| t.njnt; 3 |] d.xanchor;
  want "xaxis" [| t.njnt; 3 |] d.xaxis;
  want "cvel" [| t.nbody; 6 |] d.cvel;
  want "cdof" [| t.nv; 6 |] d.cdof;
  want "cdof_dot" [| t.nv; 6 |] d.cdof_dot;
  want "cinert" [| t.nbody; 10 |] d.cinert;
  want "com" [| 3 |] d.com;
  want "subtree_com" [| t.nbody; 3 |] d.subtree_com;
  want "qfrc_bias" [| t.nv |] d.qfrc_bias;
  want "qfrc_passive" [| t.nv |] d.qfrc_passive;
  want "qfrc_actuator" [| t.nv |] d.qfrc_actuator;
  want "actuator_length" [| t.nu |] d.actuator_length;
  want "actuator_velocity" [| t.nu |] d.actuator_velocity;
  want "actuator_force" [| t.nu |] d.actuator_force;
  want "qfrc_constraint" [| t.nv |] d.qfrc_constraint;
  want "qfrc_smooth" [| t.nv |] d.qfrc_smooth;
  want "mass_matrix" [| t.nv; t.nv |] d.mass_matrix;
  want "efc_pos" [| t.nefcmax |] d.efc_pos;
  want "efc_margin" [| t.nefcmax |] d.efc_margin;
  want "efc_J" [| t.nefcmax; t.nv |] d.efc_J;
  want "efc_vel" [| t.nefcmax |] d.efc_vel;
  want "efc_R" [| t.nefcmax |] d.efc_R;
  want "efc_D" [| t.nefcmax |] d.efc_D;
  want "efc_aref" [| t.nefcmax |] d.efc_aref;
  want "efc_force" [| t.nefcmax |] d.efc_force;
  match List.rev !problems with
  | [] -> Ok ()
  | problems ->
    Error
      (Printf.sprintf
         "data for model %S is invalid: %s"
         t.name
         (String.concat ~sep:"; " problems))

let pp_vec ppf x =
  Array.iteri (Nx.to_array x) ~f:(fun i v ->
    if i > 0 then Format.pp_print_space ppf ();
    Format.fprintf ppf "%.6g" v)

let pp ppf (d : host) =
  Format.fprintf
    ppf
    "@[<v 2>data t=%.6g@,qpos=%a@,qvel=%a@]"
    (Nx.item [] d.time)
    pp_vec
    d.qpos
    pp_vec
    d.qvel
