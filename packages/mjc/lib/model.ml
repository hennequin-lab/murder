(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
module Format = Stdlib.Format

module Params = struct
  type ('a, 'd) t =
    { body_pos : 'a
    ; body_quat : 'a
    ; body_ipos : 'a
    ; body_iquat : 'a
    ; body_mass : 'a
    ; body_inertia : 'a
    ; body_subtreemass : 'a
    ; jnt_axis : 'a
    ; jnt_pos : 'a
    ; jnt_range : 'a
    ; jnt_limited : 'a
    ; jnt_margin : 'a
    ; jnt_solref : 'a
    ; jnt_solimp : 'a
    ; jnt_stiffness : 'a
    ; dof_armature : 'a
    ; dof_damping : 'a
    ; dof_frictionloss : 'a
    ; qpos0 : 'a
    ; qpos_spring : 'a
    ; geom_pos : 'a
    ; geom_quat : 'a
    ; geom_size : 'a
    ; geom_rgba : 'a
    ; geom_friction : 'a
    ; geom_margin : 'a
    ; geom_gap : 'a
    ; geom_solmix : 'a
    ; geom_solref : 'a
    ; geom_solimp : 'a
    ; site_pos : 'a
    ; site_quat : 'a
    ; site_size : 'a
    ; gravity : 'a
    ; timestep : 'a
    }
  [@@deriving ptree]
end

type ('a, 'd) t =
  { topo : Topo.t [@ptree.skip]
  ; params : ('a, 'd) Params.t
  }
[@@deriving ptree]

type host = (Nx.host Nx.float64_t, Nx.host) t
type ('a, 'd) desc = ('a, 'd) t

module Desc = struct
  type ('a, 'd) t = ('a, 'd) desc

  let walk = walk
end

let ptree = Nx.Ptree.instantiate (module Desc)
let nq (m : _ t) = m.topo.nq
let nv (m : _ t) = m.topo.nv
let nbody (m : _ t) = m.topo.nbody
let ngeom (m : _ t) = m.topo.ngeom

type shape_key =
  { dtype : string
  ; nbody : int
  ; ngeom : int
  ; njnt : int
  ; nq : int
  ; nv : int
  ; nsite : int
  ; ntendon : int
  ; nu : int
  ; na : int
  ; nsensor : int
  ; nconmax : int
  ; nefcmax : int
  ; max_geom_pairs : int
  }

let shape_key (m : _ t) : shape_key =
  let t = m.topo in
  { dtype = "float64"
  ; nbody = t.nbody
  ; ngeom = t.ngeom
  ; njnt = t.njnt
  ; nq = t.nq
  ; nv = t.nv
  ; nsite = t.nsite
  ; ntendon = t.ntendon
  ; nu = t.nu
  ; na = t.na
  ; nsensor = t.nsensor
  ; nconmax = t.nconmax
  ; nefcmax = t.nefcmax
  ; max_geom_pairs = t.max_geom_pairs
  }

let equal_shape_key (a : shape_key) (b : shape_key) = Poly.equal a b

let pp_shape_key ppf k =
  Format.fprintf
    ppf
    "dtype=%s nbody=%d ngeom=%d njnt=%d nq=%d nv=%d nsite=%d ntendon=%d nu=%d na=%d \
     nsensor=%d nconmax=%d nefcmax=%d max_geom_pairs=%d"
    k.dtype
    k.nbody
    k.ngeom
    k.njnt
    k.nq
    k.nv
    k.nsite
    k.ntendon
    k.nu
    k.na
    k.nsensor
    k.nconmax
    k.nefcmax
    k.max_geom_pairs

let validate (m : host) : (unit, string) Result.t =
  let t = m.topo
  and p = m.params in
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
  want "body_pos" [| t.nbody; 3 |] p.body_pos;
  want "body_quat" [| t.nbody; 4 |] p.body_quat;
  want "body_ipos" [| t.nbody; 3 |] p.body_ipos;
  want "body_iquat" [| t.nbody; 4 |] p.body_iquat;
  want "body_mass" [| t.nbody |] p.body_mass;
  want "body_inertia" [| t.nbody; 3 |] p.body_inertia;
  want "body_subtreemass" [| t.nbody |] p.body_subtreemass;
  want "jnt_axis" [| t.njnt; 3 |] p.jnt_axis;
  want "jnt_pos" [| t.njnt; 3 |] p.jnt_pos;
  want "jnt_range" [| t.njnt; 2 |] p.jnt_range;
  want "jnt_limited" [| t.njnt |] p.jnt_limited;
  want "jnt_margin" [| t.njnt |] p.jnt_margin;
  want "jnt_solref" [| t.njnt; 2 |] p.jnt_solref;
  want "jnt_solimp" [| t.njnt; 5 |] p.jnt_solimp;
  want "jnt_stiffness" [| t.njnt |] p.jnt_stiffness;
  want "dof_armature" [| t.nv |] p.dof_armature;
  want "dof_damping" [| t.nv |] p.dof_damping;
  want "dof_frictionloss" [| t.nv |] p.dof_frictionloss;
  want "qpos0" [| t.nq |] p.qpos0;
  want "qpos_spring" [| t.nq |] p.qpos_spring;
  want "geom_pos" [| t.ngeom; 3 |] p.geom_pos;
  want "geom_quat" [| t.ngeom; 4 |] p.geom_quat;
  want "geom_size" [| t.ngeom; 3 |] p.geom_size;
  want "geom_rgba" [| t.ngeom; 4 |] p.geom_rgba;
  want "geom_friction" [| t.ngeom; 3 |] p.geom_friction;
  want "geom_margin" [| t.ngeom |] p.geom_margin;
  want "geom_gap" [| t.ngeom |] p.geom_gap;
  want "geom_solmix" [| t.ngeom |] p.geom_solmix;
  want "geom_solref" [| t.ngeom; 2 |] p.geom_solref;
  want "geom_solimp" [| t.ngeom; 5 |] p.geom_solimp;
  want "site_pos" [| t.nsite; 3 |] p.site_pos;
  want "site_quat" [| t.nsite; 4 |] p.site_quat;
  want "site_size" [| t.nsite; 3 |] p.site_size;
  want "gravity" [| 3 |] p.gravity;
  want "timestep" [||] p.timestep;
  match Topo.validate t, List.rev !problems with
  | Error e, _ -> Error e
  | Ok (), [] -> Ok ()
  | Ok (), problems ->
    Error
      (Printf.sprintf
         "model %S has invalid parameters: %s"
         t.name
         (String.concat ~sep:"; " problems))
