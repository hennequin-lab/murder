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
  ; com : 'a
  ; subtree_com : 'a
  ; qfrc_bias : 'a
  ; qfrc_passive : 'a
  ; qfrc_actuator : 'a
  ; qfrc_constraint : 'a
  ; qfrc_smooth : 'a
  ; mass_matrix : 'a
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
  ; com = zeros [| 3 |]
  ; subtree_com = zeros [| t.nbody; 3 |]
  ; qfrc_bias = zeros [| t.nv |]
  ; qfrc_passive = zeros [| t.nv |]
  ; qfrc_actuator = zeros [| t.nv |]
  ; qfrc_constraint = zeros [| t.nv |]
  ; qfrc_smooth = zeros [| t.nv |]
  ; mass_matrix = zeros [| t.nv; t.nv |]
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
  want "com" [| 3 |] d.com;
  want "subtree_com" [| t.nbody; 3 |] d.subtree_com;
  want "qfrc_bias" [| t.nv |] d.qfrc_bias;
  want "qfrc_passive" [| t.nv |] d.qfrc_passive;
  want "qfrc_actuator" [| t.nv |] d.qfrc_actuator;
  want "qfrc_constraint" [| t.nv |] d.qfrc_constraint;
  want "qfrc_smooth" [| t.nv |] d.qfrc_smooth;
  want "mass_matrix" [| t.nv; t.nv |] d.mass_matrix;
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
