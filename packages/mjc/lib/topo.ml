(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
module Format = Stdlib.Format

type jnt_type =
  | Hinge
  | Slide
  | Ball
  | Free

(** The type for equality-constraint types. MuJoCo's codes are not
    contiguous; the converter maps the ones v1 supports and rejects the rest. *)
type eq_type =
  | Connect (** Ball joint: two anchor points coincide. *)
  | Weld (** Rigid: relative position and orientation. *)
  | Joint (** Scalar coupling between two joints, through a cubic. *)

(** The type for the objects an equality constrains. *)
type eq_objtype =
  | Eq_body
  | Eq_site

type geom_type =
  | Plane
  | Hfield
  | Sphere
  | Capsule
  | Ellipsoid
  | Cylinder
  | Box
  | Mesh

type t =
  { name : string
  ; mujoco_version : string
  ; nbody : int
  ; ngeom : int
  ; njnt : int
  ; neq : int (** Equality constraints. *)
  ; nsite : int
  ; ntendon : int
  ; nu : int
  ; na : int
  ; nsensor : int
  ; nmesh : int
  ; nq : int
  ; nv : int
  ; nconmax : int
  ; nefcmax : int
  ; max_geom_pairs : int
  ; body_parentid : int array
  ; body_jntadr : int array
  ; body_jntnum : int array
  ; body_geomadr : int array
  ; body_geomnum : int array
  ; body_rootid : int array
  ; jnt_type : jnt_type array
  ; jnt_bodyid : int array
  ; jnt_qposadr : int array
  ; jnt_dofadr : int array
  ; dof_bodyid : int array
  ; dof_jntid : int array
  ; dof_parentid : int array
  ; geom_bodyid : int array
  ; geom_type : geom_type array
  ; geom_contype : int array
  ; geom_conaffinity : int array
  ; geom_priority : int array
  ; site_bodyid : int array
  ; jnt_limited : bool array (** [[njnt]] *)
  ; dof_frictionloss : float array (** [[nv]] *)
  ; eq_type : eq_type array (** [[neq]] *)
  ; eq_objtype : eq_objtype array (** [[neq]] *)
  ; body_names : string array
  ; jnt_names : string array
  ; geom_names : string array
  ; site_names : string array
  }

let body_name (t : t) i =
  if i >= 0 && i < Array.length t.body_names then t.body_names.(i) else ""

let geom_name (t : t) i =
  if i >= 0 && i < Array.length t.geom_names then t.geom_names.(i) else ""

let jnt_name (t : t) i =
  if i >= 0 && i < Array.length t.jnt_names then t.jnt_names.(i) else ""

let equal (a : t) (b : t) = Poly.equal a b

(* Validation. Collect every problem rather than reporting the first: a
   converter that dropped a field usually dropped more than one. *)

let length_ok problems name expected xs =
  let got = Array.length xs in
  if got <> expected
  then
    problems
    := Printf.sprintf "%s has length %d, expected %d" name got expected :: !problems

let budget_ok problems name x =
  if x <= 0
  then problems := Printf.sprintf "%s is %d, expected positive" name x :: !problems

let check_blocks problems ~what ~adr ~num ~count =
  let seen = Array.create ~len:count false in
  for i = 0 to Array.length adr - 1 do
    let start = adr.(i)
    and n = num.(i) in
    if n = 0
    then (
      if start <> -1
      then
        problems
        := Printf.sprintf "%s %d has no %s but its address is %d" what i what start
           :: !problems)
    else if start < 0 || start + n > count
    then
      problems
      := Printf.sprintf
           "%s %d's %ss [%d,%d) leave [0,%d)"
           what
           i
           what
           start
           (start + n)
           count
         :: !problems
    else
      for k = start to start + n - 1 do
        if seen.(k)
        then
          problems := Printf.sprintf "%s %d claims %s %d twice" what i what k :: !problems
        else seen.(k) <- true
      done
  done;
  Array.iteri seen ~f:(fun k claimed ->
    if not claimed
    then problems := Printf.sprintf "%s %d belongs to no body" what k :: !problems)

let qpos_width_of = function
  | Hinge | Slide -> 1
  | Ball -> 4
  | Free -> 7

let dof_width_of = function
  | Hinge | Slide -> 1
  | Ball -> 3
  | Free -> 6

let validate (t : t) : (unit, string) Result.t =
  let problems = ref [] in
  let add fmt = Printf.ksprintf (fun s -> problems := s :: !problems) fmt in
  let len : string -> int -> int array -> unit = length_ok problems in
  len "body_parentid" t.nbody t.body_parentid;
  len "body_jntadr" t.nbody t.body_jntadr;
  len "body_jntnum" t.nbody t.body_jntnum;
  len "body_geomadr" t.nbody t.body_geomadr;
  len "body_geomnum" t.nbody t.body_geomnum;
  len "body_rootid" t.nbody t.body_rootid;
  if Array.length t.jnt_type <> t.njnt
  then add "jnt_type has length %d, expected %d" (Array.length t.jnt_type) t.njnt;
  len "jnt_bodyid" t.njnt t.jnt_bodyid;
  len "jnt_qposadr" t.njnt t.jnt_qposadr;
  len "jnt_dofadr" t.njnt t.jnt_dofadr;
  len "dof_bodyid" t.nv t.dof_bodyid;
  len "dof_jntid" t.nv t.dof_jntid;
  len "dof_parentid" t.nv t.dof_parentid;
  len "geom_bodyid" t.ngeom t.geom_bodyid;
  if Array.length t.geom_type <> t.ngeom
  then add "geom_type has length %d, expected %d" (Array.length t.geom_type) t.ngeom;
  length_ok problems "jnt_limited" t.njnt (Array.map t.jnt_limited ~f:Bool.to_int);
  length_ok problems "dof_frictionloss" t.nv t.dof_frictionloss;
  if Array.length t.eq_type <> t.neq
  then add "eq_type has length %d, expected %d" (Array.length t.eq_type) t.neq;
  if Array.length t.eq_objtype <> t.neq
  then add "eq_objtype has length %d, expected %d" (Array.length t.eq_objtype) t.neq;
  len "geom_contype" t.ngeom t.geom_contype;
  len "geom_conaffinity" t.ngeom t.geom_conaffinity;
  len "geom_priority" t.ngeom t.geom_priority;
  len "site_bodyid" t.nsite t.site_bodyid;
  let len_names : string -> int -> string array -> unit = length_ok problems in
  len_names "body_names" t.nbody t.body_names;
  len_names "jnt_names" t.njnt t.jnt_names;
  len_names "geom_names" t.ngeom t.geom_names;
  len_names "site_names" t.nsite t.site_names;
  (* Counts are non-negative, budgets positive. *)
  List.iter
    [ "nbody", t.nbody
    ; "ngeom", t.ngeom
    ; "njnt", t.njnt
    ; "neq", t.neq
    ; "nsite", t.nsite
    ; "ntendon", t.ntendon
    ; "nu", t.nu
    ; "na", t.na
    ; "nsensor", t.nsensor
    ; "nmesh", t.nmesh
    ; "nq", t.nq
    ; "nv", t.nv
    ]
    ~f:(fun (name, value) ->
      if value < 0 then add "%s is %d, expected non-negative" name value);
  budget_ok problems "nconmax" t.nconmax;
  budget_ok problems "nefcmax" t.nefcmax;
  budget_ok problems "max_geom_pairs" t.max_geom_pairs;
  (* The world body is body 0 and has no parent. *)
  if t.nbody > 0
  then (
    (* MuJoCo gives the world body itself as its parent. *)
    if t.body_parentid.(0) <> 0
    then add "body_parentid.(0) is %d, expected 0" t.body_parentid.(0);
    for i = 1 to t.nbody - 1 do
      let p = t.body_parentid.(i) in
      if p < 0 || p >= i then add "body %d's parent %d is not before it" i p
    done);
  (* Joints and dofs: bodies exist, addresses tile qpos and qvel, widths match. *)
  let nq = ref 0
  and nv = ref 0 in
  for j = 0 to t.njnt - 1 do
    let body = t.jnt_bodyid.(j) in
    if body < 0 || body >= t.nbody then add "joint %d's body %d is out of range" j body;
    let wq = qpos_width_of t.jnt_type.(j)
    and wv = dof_width_of t.jnt_type.(j) in
    let aq = t.jnt_qposadr.(j)
    and av = t.jnt_dofadr.(j) in
    if aq <> !nq then add "joint %d starts at qpos %d, expected %d" j aq !nq;
    if av <> !nv then add "joint %d starts at qvel %d, expected %d" j av !nv;
    nq := !nq + wq;
    nv := !nv + wv
  done;
  if !nq <> t.nq then add "joints own %d qpos entries, nq is %d" !nq t.nq;
  if !nv <> t.nv then add "joints own %d qvel entries, nv is %d" !nv t.nv;
  for v = 0 to t.nv - 1 do
    let j = t.dof_jntid.(v) in
    if j < 0 || j >= t.njnt then add "dof %d's joint %d is out of range" v j;
    let b = t.dof_bodyid.(v) in
    if b < 0 || b >= t.nbody then add "dof %d's body %d is out of range" v b;
    let p = t.dof_parentid.(v) in
    if p < -1 || p >= v then add "dof %d's parent %d is not before it" v p
  done;
  check_blocks problems ~what:"joint" ~adr:t.body_jntadr ~num:t.body_jntnum ~count:t.njnt;
  check_blocks
    problems
    ~what:"geom"
    ~adr:t.body_geomadr
    ~num:t.body_geomnum
    ~count:t.ngeom;
  (* A free joint is its body's only joint and sits on the world. *)
  for j = 0 to t.njnt - 1 do
    match t.jnt_type.(j) with
    | Free ->
      let body = t.jnt_bodyid.(j) in
      if body >= 0 && body < t.nbody
      then (
        if t.body_jntnum.(body) <> 1
        then
          add
            "body %d has a free joint and %d other joints"
            body
            (t.body_jntnum.(body) - 1);
        if t.body_parentid.(body) <> 0
        then add "free joint %d's body %d is not a child of the world" j body)
    | Hinge | Slide | Ball -> ()
  done;
  (* Geoms and sites point at bodies. *)
  for g = 0 to t.ngeom - 1 do
    let b = t.geom_bodyid.(g) in
    if b < 0 || b >= t.nbody then add "geom %d's body %d is out of range" g b
  done;
  for s = 0 to t.nsite - 1 do
    let b = t.site_bodyid.(s) in
    if b < 0 || b >= t.nbody then add "site %d's body %d is out of range" s b
  done;
  match List.rev !problems with
  | [] -> Ok ()
  | problems ->
    Error
      (Printf.sprintf "model %S is invalid: %s" t.name (String.concat ~sep:"; " problems))

let pp_jnt_type ppf = function
  | Hinge -> Format.pp_print_string ppf "hinge"
  | Slide -> Format.pp_print_string ppf "slide"
  | Ball -> Format.pp_print_string ppf "ball"
  | Free -> Format.pp_print_string ppf "free"

let pp_geom_type ppf = function
  | Plane -> Format.pp_print_string ppf "plane"
  | Hfield -> Format.pp_print_string ppf "hfield"
  | Sphere -> Format.pp_print_string ppf "sphere"
  | Capsule -> Format.pp_print_string ppf "capsule"
  | Ellipsoid -> Format.pp_print_string ppf "ellipsoid"
  | Cylinder -> Format.pp_print_string ppf "cylinder"
  | Box -> Format.pp_print_string ppf "box"
  | Mesh -> Format.pp_print_string ppf "mesh"

let pp ppf (t : t) =
  Format.fprintf
    ppf
    "@[<v 2>%s (MuJoCo %s)@,\
     nbody=%d ngeom=%d njnt=%d nsite=%d ntendon=%d nu=%d na=%d nsensor=%d nmesh=%d@,\
     nq=%d nv=%d nconmax=%d nefcmax=%d max_geom_pairs=%d@,"
    t.name
    t.mujoco_version
    t.nbody
    t.ngeom
    t.njnt
    t.nsite
    t.ntendon
    t.nu
    t.na
    t.nsensor
    t.nmesh
    t.nq
    t.nv
    t.nconmax
    t.nefcmax
    t.max_geom_pairs;
  for b = 0 to t.nbody - 1 do
    Format.fprintf
      ppf
      "@,body %d %S parent=%d joints=%d geoms=%d"
      b
      (body_name t b)
      t.body_parentid.(b)
      t.body_jntnum.(b)
      t.body_geomnum.(b)
  done;
  for j = 0 to t.njnt - 1 do
    Format.fprintf
      ppf
      "@,  joint %d %S %a body=%d qpos=%d qvel=%d"
      j
      (jnt_name t j)
      pp_jnt_type
      t.jnt_type.(j)
      t.jnt_bodyid.(j)
      t.jnt_qposadr.(j)
      t.jnt_dofadr.(j)
  done;
  for g = 0 to t.ngeom - 1 do
    Format.fprintf
      ppf
      "@,  geom %d %S %a body=%d"
      g
      (geom_name t g)
      pp_geom_type
      t.geom_type.(g)
      t.geom_bodyid.(g)
  done;
  Format.fprintf ppf "@]"
