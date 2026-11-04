(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Model topology.

    A model has two halves. The {e topology} is host data: counts, parent
    indices, object types and names. It is the same for every dtype, never
    differentiated, and read at trace time to build index tensors and to check
    shapes. The {e parameters} are tensors and live in {!Model}.

    The field names and array layouts follow MuJoCo's [mjModel], so a reader
    of its header can map one to the other without a table. Arrays indexed by
    object id have that object's count as their length, and [-1] means none.
    A variant type stands for an integer code that a validated model can
    carry. *)

(** {1:types Object types} *)

(** The type for joint types. MuJoCo's codes are [0] to [3] in this order. *)
type jnt_type =
  | Hinge (** One rotational degree of freedom about an axis. *)
  | Slide (** One translational degree of freedom along an axis. *)
  | Ball (** Three rotational degrees of freedom, stored as a quaternion. *)
  | Free (** Three translational and three rotational degrees of freedom. *)

(** The type for geom types. MuJoCo's codes are [0] to [7] in this order. Only
    the first six are supported by the v1 kernel; a model may still carry a
    mesh or height field so that it loads and is inspected. *)
type geom_type =
  | Plane
  | Hfield
  | Sphere
  | Capsule
  | Ellipsoid
  | Cylinder
  | Box
  | Mesh

(** The type for equality-constraint types. Modules of v1 support connect,
    weld and joint; the other MuJoCo kinds are rejected when a model loads. *)
type eq_type =
  | Connect (** Two anchor points coincide: a ball joint. *)
  | Weld (** Rigid attachment: relative position and orientation. *)
  | Joint (** A scalar coupling between two joints, through a cubic. *)

(** The type for the integration scheme. v1 integrates with [Euler] and
    [Implicit_fast]; the other MuJoCo schemes are rejected when a model
    loads. *)
type integrator =
  | Euler (** Semi-implicit, damping integrated implicitly. *)
  | Implicit_fast (** Semi-implicit, velocity solved in [M + h diag(B)]. *)

(** The type for what an equality constrains. *)
type eq_objtype =
  | Eq_body
  | Eq_site

(** {1:model The structure} *)

type t =
  { name : string (** The model's name, as the converter recorded it. *)
  ; mujoco_version : string (** The MuJoCo release the converter compiled with. *)
  ; nbody : int (** Bodies, including the world body [0]. *)
  ; ngeom : int
  ; njnt : int
  ; neq : int (** Equality constraints. *)
  ; nsite : int
  ; ntendon : int
  ; nu : int (** Actuators. *)
  ; na : int (** Activation states, summed over actuators. *)
  ; nsensor : int
  ; nmesh : int
  ; nq : int (** Position coordinates. *)
  ; nv : int (** Velocity coordinates, also the number of degrees of freedom. *)
  ; nconmax : int (** Compile-time contact budget, per world. *)
  ; nefcmax : int (** Compile-time constraint-row budget, per world. *)
  ; max_geom_pairs : int (** Compile-time candidate-pair budget. *)
  ; body_parentid : int array (** [[nbody]], [0] for the world body. *)
  ; body_jntadr : int array (** [[nbody]], first joint, [-1] if none. *)
  ; body_jntnum : int array (** [[nbody]], number of joints. *)
  ; body_geomadr : int array (** [[nbody]], first geom, [-1] if none. *)
  ; body_geomnum : int array (** [[nbody]], number of geoms. *)
  ; body_rootid : int array (** [[nbody]], the root of the kinematic tree. *)
  ; jnt_type : jnt_type array (** [[njnt]] *)
  ; jnt_bodyid : int array (** [[njnt]] *)
  ; jnt_qposadr : int array (** [[njnt]], first address in [qpos]. *)
  ; jnt_dofadr : int array (** [[njnt]], first address in [qvel]. *)
  ; dof_bodyid : int array (** [[nv]] *)
  ; dof_jntid : int array (** [[nv]] *)
  ; dof_parentid : int array (** [[nv]], previous dof on the path, [-1]. *)
  ; geom_bodyid : int array (** [[ngeom]] *)
  ; geom_type : geom_type array (** [[ngeom]] *)
  ; geom_contype : int array (** [[ngeom]], collision mask. *)
  ; geom_conaffinity : int array (** [[ngeom]], collision mask. *)
  ; geom_priority : int array (** [[ngeom]], contact override. *)
  ; site_bodyid : int array (** [[nsite]] *)
  ; jnt_limited : bool array
    (** [[njnt]], whether the joint's range is active. Host data, for the
        same reason as the friction losses below. *)
  ; dof_frictionloss : float array
    (** [[nv]], the friction loss of each dof. Host data, because which
        dofs get a friction row is decided when the program is built, not
        when it runs. *)
  ; integrator : integrator
  ; contacts_enabled : bool
    (** Whether MuJoCo's contact-disable flag is set. A model that disables
        contacts produces no rows and no contact array. *)
  ; eq_type : eq_type array (** [[neq]] *)
  ; eq_objtype : eq_objtype array
    (** [[neq]], what [eq_obj1id] and [eq_obj2id] index in the model. *)
  ; body_names : string array (** [[nbody]] *)
  ; jnt_names : string array (** [[njnt]] *)
  ; geom_names : string array (** [[ngeom]] *)
  ; site_names : string array (** [[nsite]] *)
  }

(** {1:accessors Accessors} *)

(** [body_name t i] is body [i]'s name, or [""] when the model has none. *)
val body_name : t -> int -> string

(** [geom_name t i] is geom [i]'s name, or [""] when the model has none. *)
val geom_name : t -> int -> string

(** [jnt_name t i] is joint [i]'s name, or [""] when the model has none. *)
val jnt_name : t -> int -> string

(** {1:checks Checks} *)

(** [validate t] is [Ok ()] when [t] satisfies the invariants every kernel
    assumes, and [Error message] otherwise. The message names the array, the
    index and the offending value.

    The invariants are: every array has the length its count implies; the
    world body is its own parent and every other body's parent precedes it;
    joints, dofs, geoms and sites point at bodies and at addresses inside
    [qpos] and [qvel]; each body's joint and geom blocks lie inside their
    arrays and do not overlap their neighbours; a free joint is a body's only
    joint; and every budget is positive. *)
val validate : t -> (unit, string) result

(** [equal a b] is [true] iff [a] and [b] have the same topology, names
    included. *)
val equal : t -> t -> bool

(** [pp] formats the counts and the object names. *)
val pp : Format.formatter -> t -> unit
