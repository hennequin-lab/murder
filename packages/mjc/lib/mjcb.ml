(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
module Buffer = Stdlib.Buffer
module Bytes = Stdlib.Bytes
module Int32 = Stdlib.Int32
module Int64 = Stdlib.Int64

let magic = "MJCB"
let magic_end = "MJCE"
let version = 9

(* Layout, in writing order:

   "MJCB" u16 version u16 flags
   string name
   string mujoco_version
   u32 count, then the counts
   u32 count, then that many int arrays, each u32 length and i32 elements
   four name tables, each u32 count, then u32 length and bytes per name
   u32 count, then that many tensors: string name, u32 rank, u32 dims,
   f64 payload, row major
   "MJCE"

   Every array and tensor is named, so a reader checks presence rather than
   position, and a field added by a later milestone is a missing-field error
   in an older one. *)

let counts =
  [ "nbody"
  ; "ngeom"
  ; "njnt"
  ; "neq"
  ; "nsite"
  ; "ntendon"
  ; "nu"
  ; "na"
  ; "nsensor"
  ; "nmesh"
  ; "nq"
  ; "nv"
  ; "nconmax"
  ; "nefcmax"
  ; "max_geom_pairs"
  ]

let int_arrays =
  [ "body_parentid"
  ; "body_jntadr"
  ; "body_jntnum"
  ; "body_geomadr"
  ; "body_geomnum"
  ; "body_rootid"
  ; "jnt_type"
  ; "jnt_bodyid"
  ; "jnt_qposadr"
  ; "jnt_dofadr"
  ; "dof_bodyid"
  ; "dof_jntid"
  ; "dof_parentid"
  ; "geom_bodyid"
  ; "geom_type"
  ; "geom_contype"
  ; "geom_conaffinity"
  ; "geom_priority"
  ; "site_bodyid"
  ; "eq_type"
  ; "eq_objtype"
  ]

let name_tables = [ "body_names"; "jnt_names"; "geom_names"; "site_names" ]

(* Parameter tensors, in [Model.Params] order. *)
let fields =
  [ "body_pos"
  ; "body_quat"
  ; "body_ipos"
  ; "body_iquat"
  ; "body_mass"
  ; "body_inertia"
  ; "body_subtreemass"
  ; "body_invweight0"
  ; "jnt_axis"
  ; "jnt_pos"
  ; "jnt_range"
  ; "jnt_limited"
  ; "jnt_stiffness"
  ; "jnt_margin"
  ; "jnt_solref"
  ; "jnt_solimp"
  ; "dof_armature"
  ; "dof_damping"
  ; "dof_frictionloss"
  ; "dof_solref"
  ; "dof_solimp"
  ; "dof_invweight0"
  ; "qpos0"
  ; "qpos_spring"
  ; "geom_pos"
  ; "geom_quat"
  ; "geom_size"
  ; "geom_condim"
  ; "geom_rbound"
  ; "geom_rgba"
  ; "geom_friction"
  ; "geom_margin"
  ; "geom_gap"
  ; "geom_solmix"
  ; "geom_solref"
  ; "geom_solimp"
  ; "site_pos"
  ; "site_quat"
  ; "site_size"
  ; "eq_obj1id"
  ; "eq_obj2id"
  ; "eq_data"
  ; "eq_solref"
  ; "eq_solimp"
  ; "eq_active0"
  ; "gravity"
  ; "timestep"
  ; "integrator"
  ; "disableflags"
  ; "iterations"
  ; "tolerance"
  ]

let jnt_type_code = function
  | Topo.Hinge -> 0
  | Topo.Slide -> 1
  | Topo.Ball -> 2
  | Topo.Free -> 3

let geom_type_code = function
  | Topo.Plane -> 0
  | Topo.Hfield -> 1
  | Topo.Sphere -> 2
  | Topo.Capsule -> 3
  | Topo.Ellipsoid -> 4
  | Topo.Cylinder -> 5
  | Topo.Box -> 6
  | Topo.Mesh -> 7

let jnt_type_of_code c =
  match c with
  | 0 -> Topo.Hinge
  | 1 -> Topo.Slide
  | 2 -> Topo.Ball
  | 3 -> Topo.Free
  | c -> raise (Invalid_argument (Printf.sprintf "unknown joint type %d" c))

let geom_type_of_code c =
  match c with
  | 0 -> Topo.Plane
  | 1 -> Topo.Hfield
  | 2 -> Topo.Sphere
  | 3 -> Topo.Capsule
  | 4 -> Topo.Ellipsoid
  | 5 -> Topo.Cylinder
  | 6 -> Topo.Box
  | 7 -> Topo.Mesh
  | c -> raise (Invalid_argument (Printf.sprintf "unknown geom type %d" c))

let integrator_of_code = function
  | 0 -> Topo.Euler
  | 3 -> Topo.Implicit_fast
  | c -> raise (Invalid_argument (Printf.sprintf "unsupported integrator %d" c))

let eq_type_code = function
  | Topo.Connect -> 0
  | Topo.Weld -> 1
  | Topo.Joint -> 2

let eq_objtype_code = function
  | Topo.Eq_body -> 1
  | Topo.Eq_site -> 6

let eq_type_of_code = function
  | 0 -> Topo.Connect
  | 1 -> Topo.Weld
  | 2 -> Topo.Joint
  | c -> raise (Invalid_argument (Printf.sprintf "unsupported equality type %d" c))

(* A joint equality names joints, not bodies or sites, and MuJoCo leaves its
   object type at zero: both codes mean "a body" here. *)
let eq_objtype_of_code = function
  | 0 | 1 -> Topo.Eq_body
  | 6 -> Topo.Eq_site
  | c -> raise (Invalid_argument (Printf.sprintf "unsupported equality object %d" c))

(* {1 Writing} *)

let put_i32 buf v =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.of_int v);
  Buffer.add_bytes buf b

let put_u16 buf v =
  let b = Bytes.create 2 in
  Bytes.set_uint16_le b 0 v;
  Buffer.add_bytes buf b

let put_f64 buf v =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.bits_of_float v);
  Buffer.add_bytes buf b

let put_string buf s =
  put_i32 buf (String.length s);
  Buffer.add_string buf s

let put_ints buf xs =
  put_i32 buf (Array.length xs);
  Array.iter xs ~f:(put_i32 buf)

let put_names buf xs =
  put_i32 buf (Array.length xs);
  Array.iter xs ~f:(put_string buf)

let put_tensor buf (x : Nx.host Nx.float64_t) =
  let shape = Nx.shape x in
  put_i32 buf (Array.length shape);
  Array.iter shape ~f:(put_i32 buf);
  Array.iter (Nx.to_array x) ~f:(put_f64 buf)

let to_string (m : Model.host) : (string, string) Result.t =
  match Model.validate m with
  | Error e -> Error e
  | Ok () ->
    let t = m.topo
    and p = m.params in
    let buf = Buffer.create 4096 in
    Buffer.add_string buf magic;
    put_u16 buf version;
    put_u16 buf 0;
    put_string buf t.name;
    put_string buf t.mujoco_version;
    put_i32 buf (List.length counts);
    List.iter
      [ t.nbody
      ; t.ngeom
      ; t.njnt
      ; t.neq
      ; t.nsite
      ; t.ntendon
      ; t.nu
      ; t.na
      ; t.nsensor
      ; t.nmesh
      ; t.nq
      ; t.nv
      ; t.nconmax
      ; t.nefcmax
      ; t.max_geom_pairs
      ]
      ~f:(put_i32 buf);
    put_i32 buf (List.length int_arrays);
    List.iter
      [ t.body_parentid
      ; t.body_jntadr
      ; t.body_jntnum
      ; t.body_geomadr
      ; t.body_geomnum
      ; t.body_rootid
      ; Array.map t.jnt_type ~f:jnt_type_code
      ; t.jnt_bodyid
      ; t.jnt_qposadr
      ; t.jnt_dofadr
      ; t.dof_bodyid
      ; t.dof_jntid
      ; t.dof_parentid
      ; t.geom_bodyid
      ; Array.map t.geom_type ~f:geom_type_code
      ; t.geom_contype
      ; t.geom_conaffinity
      ; t.geom_priority
      ; t.site_bodyid
      ; Array.map t.eq_type ~f:eq_type_code
      ; Array.map t.eq_objtype ~f:eq_objtype_code
      ]
      ~f:(put_ints buf);
    put_i32 buf (List.length name_tables);
    List.iter [ t.body_names; t.jnt_names; t.geom_names; t.site_names ] ~f:(put_names buf);
    put_i32 buf (List.length fields);
    List.iter2_exn
      fields
      [ p.body_pos
      ; p.body_quat
      ; p.body_ipos
      ; p.body_iquat
      ; p.body_mass
      ; p.body_inertia
      ; p.body_subtreemass
      ; p.body_invweight0
      ; p.jnt_axis
      ; p.jnt_pos
      ; p.jnt_range
      ; p.jnt_limited
      ; p.jnt_stiffness
      ; p.jnt_margin
      ; p.jnt_solref
      ; p.jnt_solimp
      ; p.dof_armature
      ; p.dof_damping
      ; p.dof_frictionloss
      ; p.dof_solref
      ; p.dof_solimp
      ; p.dof_invweight0
      ; p.qpos0
      ; p.qpos_spring
      ; p.geom_pos
      ; p.geom_quat
      ; p.geom_size
      ; p.geom_condim
      ; p.geom_rbound
      ; p.geom_rgba
      ; p.geom_friction
      ; p.geom_margin
      ; p.geom_gap
      ; p.geom_solmix
      ; p.geom_solref
      ; p.geom_solimp
      ; p.site_pos
      ; p.site_quat
      ; p.site_size
      ; p.eq_obj1id
      ; p.eq_obj2id
      ; p.eq_data
      ; p.eq_solref
      ; p.eq_solimp
      ; p.eq_active0
      ; p.gravity
      ; p.timestep
      ; p.integrator
      ; p.disableflags
      ; p.iterations
      ; p.tolerance
      ]
      ~f:(fun name x ->
        put_string buf name;
        put_tensor buf x);
    Buffer.add_string buf magic_end;
    Ok (Buffer.contents buf)

(* {1 Reading} *)

exception Bad of string

let bad fmt = Printf.ksprintf (fun msg -> raise (Bad msg)) fmt

type cursor =
  { src : string
  ; mutable pos : int
  }

let need c n = if c.pos + n > String.length c.src then bad "truncated at byte %d" c.pos

let get_i32 c =
  need c 4;
  let v = Bytes.get_int32_le (Bytes.unsafe_of_string c.src) c.pos in
  c.pos <- c.pos + 4;
  Int32.to_int v

let get_u16 c =
  need c 2;
  let v = Bytes.get_uint16_le (Bytes.unsafe_of_string c.src) c.pos in
  c.pos <- c.pos + 2;
  v

let get_f64 c =
  need c 8;
  let v = Bytes.get_int64_le (Bytes.unsafe_of_string c.src) c.pos in
  c.pos <- c.pos + 8;
  Int64.float_of_bits v

let get_raw c n =
  need c n;
  let s = String.sub c.src ~pos:c.pos ~len:n in
  c.pos <- c.pos + n;
  s

let get_string c =
  let n = get_i32 c in
  if n < 0 then bad "negative length %d at byte %d" n c.pos;
  need c n;
  let s = String.sub c.src ~pos:c.pos ~len:n in
  c.pos <- c.pos + n;
  s

let get_ints c =
  let n = get_i32 c in
  if n < 0 then bad "negative array length %d" n;
  Array.init n ~f:(fun _ -> get_i32 c)

let get_names c =
  let n = get_i32 c in
  if n < 0 then bad "negative name count %d" n;
  Array.init n ~f:(fun _ -> get_string c)

let get_tensor c =
  let rank = get_i32 c in
  if rank < 0 || rank > 4 then bad "rank %d out of range" rank;
  let shape = Array.init rank ~f:(fun _ -> get_i32 c) in
  let numel = Array.fold shape ~init:1 ~f:( * ) in
  Array.init numel ~f:(fun _ -> get_f64 c), shape

let expect name expected got =
  if expected <> got then bad "%s has %d entries, expected %d" name got expected

let field name xs =
  match List.Assoc.find xs ~equal:String.equal name with
  | Some v -> v
  | None -> bad "missing field %S" name

let parse (c : cursor) : Model.host =
  if not (Poly.equal (get_raw c 4) magic) then bad "not a .mjcb file (bad magic)";
  let v = get_u16 c in
  if not (Int.equal v version) then bad "format version %d, this build reads %d" v version;
  ignore (get_u16 c : int);
  let name = get_string c in
  let mujoco_version = get_string c in
  let ncounts = get_i32 c in
  expect "counts" (List.length counts) ncounts;
  let counts_map = List.map counts ~f:(fun k -> k, get_i32 c) in
  let n = get_i32 c in
  expect "int arrays" (List.length int_arrays) n;
  let ints = List.map int_arrays ~f:(fun k -> k, get_ints c) in
  let n = get_i32 c in
  expect "name tables" (List.length name_tables) n;
  let names = List.map name_tables ~f:(fun k -> k, get_names c) in
  let n = get_i32 c in
  expect "fields" (List.length fields) n;
  let tensors =
    List.map fields ~f:(fun field ->
      let name = get_string c in
      if not (String.equal name field) then bad "expected field %S, found %S" field name;
      name, get_tensor c)
  in
  (* The trailer is the last four bytes. *)
  if not (Poly.equal (get_raw c 4) magic_end) then bad "missing trailer at byte %d" c.pos;
  let f64 = Nx.float64 in
  let tensor name =
    let data, shape = field name tensors in
    Nx.create f64 shape data
  in
  let i name = field name ints in
  let frictionloss = Nx.to_array (tensor "dof_frictionloss") in
  let integrator =
    integrator_of_code (Int.of_float (Nx.to_array (tensor "integrator")).(0))
  in
  let limited = Nx.to_array (tensor "jnt_limited") in
  let disable = Int.of_float (Nx.to_array (tensor "disableflags")).(0) in
  let condim = Nx.to_array (tensor "geom_condim") in
  let topo =
    { Topo.name
    ; mujoco_version
    ; nbody = field "nbody" counts_map
    ; ngeom = field "ngeom" counts_map
    ; njnt = field "njnt" counts_map
    ; neq = field "neq" counts_map
    ; nsite = field "nsite" counts_map
    ; ntendon = field "ntendon" counts_map
    ; nu = field "nu" counts_map
    ; na = field "na" counts_map
    ; nsensor = field "nsensor" counts_map
    ; nmesh = field "nmesh" counts_map
    ; nq = field "nq" counts_map
    ; nv = field "nv" counts_map
    ; nconmax = field "nconmax" counts_map
    ; nefcmax = field "nefcmax" counts_map
    ; max_geom_pairs = field "max_geom_pairs" counts_map
    ; body_parentid = i "body_parentid"
    ; body_jntadr = i "body_jntadr"
    ; body_jntnum = i "body_jntnum"
    ; body_geomadr = i "body_geomadr"
    ; body_geomnum = i "body_geomnum"
    ; body_rootid = i "body_rootid"
    ; jnt_type = Array.map (i "jnt_type") ~f:jnt_type_of_code
    ; jnt_bodyid = i "jnt_bodyid"
    ; jnt_qposadr = i "jnt_qposadr"
    ; jnt_dofadr = i "jnt_dofadr"
    ; dof_bodyid = i "dof_bodyid"
    ; dof_jntid = i "dof_jntid"
    ; dof_parentid = i "dof_parentid"
    ; geom_bodyid = i "geom_bodyid"
    ; geom_type = Array.map (i "geom_type") ~f:geom_type_of_code
    ; geom_contype = i "geom_contype"
    ; geom_conaffinity = i "geom_conaffinity"
    ; geom_priority = i "geom_priority"
    ; site_bodyid = i "site_bodyid"
    ; integrator
    ; geom_condim = Array.map condim ~f:Int.of_float
    ; contacts_enabled = Int.( land ) disable (1 lsl 4) = 0
    ; jnt_limited = Array.map limited ~f:(fun x -> Stdlib.( > ) x 0.5)
    ; dof_frictionloss = frictionloss
    ; eq_type = Array.map (i "eq_type") ~f:eq_type_of_code
    ; eq_objtype = Array.map (i "eq_objtype") ~f:eq_objtype_of_code
    ; body_names = field "body_names" names
    ; jnt_names = field "jnt_names" names
    ; geom_names = field "geom_names" names
    ; site_names = field "site_names" names
    }
  in
  let params =
    { Model.Params.body_pos = tensor "body_pos"
    ; body_quat = tensor "body_quat"
    ; body_ipos = tensor "body_ipos"
    ; body_iquat = tensor "body_iquat"
    ; body_mass = tensor "body_mass"
    ; body_inertia = tensor "body_inertia"
    ; body_subtreemass = tensor "body_subtreemass"
    ; body_invweight0 = tensor "body_invweight0"
    ; jnt_axis = tensor "jnt_axis"
    ; jnt_pos = tensor "jnt_pos"
    ; jnt_range = tensor "jnt_range"
    ; jnt_limited = tensor "jnt_limited"
    ; jnt_stiffness = tensor "jnt_stiffness"
    ; jnt_margin = tensor "jnt_margin"
    ; jnt_solref = tensor "jnt_solref"
    ; jnt_solimp = tensor "jnt_solimp"
    ; dof_armature = tensor "dof_armature"
    ; dof_damping = tensor "dof_damping"
    ; dof_frictionloss = tensor "dof_frictionloss"
    ; dof_solref = tensor "dof_solref"
    ; dof_solimp = tensor "dof_solimp"
    ; dof_invweight0 = tensor "dof_invweight0"
    ; qpos0 = tensor "qpos0"
    ; qpos_spring = tensor "qpos_spring"
    ; geom_pos = tensor "geom_pos"
    ; geom_quat = tensor "geom_quat"
    ; geom_size = tensor "geom_size"
    ; geom_condim = tensor "geom_condim"
    ; geom_rbound = tensor "geom_rbound"
    ; geom_rgba = tensor "geom_rgba"
    ; geom_friction = tensor "geom_friction"
    ; geom_margin = tensor "geom_margin"
    ; geom_gap = tensor "geom_gap"
    ; geom_solmix = tensor "geom_solmix"
    ; geom_solref = tensor "geom_solref"
    ; geom_solimp = tensor "geom_solimp"
    ; site_pos = tensor "site_pos"
    ; site_quat = tensor "site_quat"
    ; site_size = tensor "site_size"
    ; eq_obj1id = tensor "eq_obj1id"
    ; eq_obj2id = tensor "eq_obj2id"
    ; eq_data = tensor "eq_data"
    ; eq_solref = tensor "eq_solref"
    ; eq_solimp = tensor "eq_solimp"
    ; eq_active0 = tensor "eq_active0"
    ; gravity = tensor "gravity"
    ; timestep = tensor "timestep"
    ; integrator = tensor "integrator"
    ; disableflags = tensor "disableflags"
    ; iterations = tensor "iterations"
    ; tolerance = tensor "tolerance"
    }
  in
  { Model.topo; params }

let of_string ?(source = "<string>") (s : string) : (Model.host, string) Result.t =
  let c = { src = s; pos = 0 } in
  match parse c with
  | m -> Ok m
  | exception Bad msg -> Error (Printf.sprintf "mjcb: %s: %s" source msg)
  | exception Invalid_argument msg -> Error (Printf.sprintf "mjcb: %s: %s" source msg)

let read path : (Model.host, string) Result.t =
  match Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all with
  | s -> of_string ~source:path s
  | exception Sys_error msg -> Error (Printf.sprintf "mjcb: %s: %s" path msg)

let write path (m : Model.host) : (unit, string) Result.t =
  match to_string m with
  | Error e -> Error e
  | Ok s ->
    (match
       Stdlib.Out_channel.with_open_bin path (fun oc ->
         Stdlib.Out_channel.output_string oc s)
     with
     | () -> Ok ()
     | exception Sys_error msg -> Error (Printf.sprintf "mjcb: %s: %s" path msg))
