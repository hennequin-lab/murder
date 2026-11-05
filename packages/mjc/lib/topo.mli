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

(** The type for a sensor. v1 has the state-projection and subtree families;
    the ones that need the acceleration stage (gyro, accelerometer) or a
    ray-geometry query (touch) are refused when a model loads. *)
type sensor_type =
  | Sens_jointpos
  | Sens_jointvel
  | Sens_framepos
  | Sens_framequat
  | Sens_framelinvel
  | Sens_frameangvel
  | Sens_actuatorfrc
  | Sens_tendonpos
  | Sens_tendonvel
  | Sens_subtreecom
  | Sens_subtreelinvel
  | Sens_subtreeangmom

(** The type for what a sensor measures. *)
type obj_type =
  | Obj_body (** The inertial frame: [xipos] and [ximat]. *)
  | Obj_xbody (** The body frame: [xpos] and [xquat]. *)
  | Obj_site
  | Obj_joint
  | Obj_actuator
  | Obj_tendon

(** The type for what a tendon wraps. v1 has fixed tendons over joints; a site
    or a geometric wrap needs the spatial-tendon path, which is not v1. *)
type wrap_type =
  | Wrap_joint (** The joint's coordinate, scaled by [wrap_prm.[0]]. *)
  | Wrap_site (** A site, which v1 does not have. *)

(** The type for what an actuator drives. v1 supports a joint's single
    degree of freedom and a fixed tendon; a site and a ball joint's three
    degrees of freedom are rejected when a model loads. *)
type trn_type =
  | Trn_joint (** A hinge or slide joint, through [gear.[0]]. *)
  | Trn_tendon (** A fixed tendon, through [gear.[0]]. *)

(** The type for how an actuator's gain depends on its own length and
    velocity. MuJoCo's muscle and DC-motor gains are not v1. *)
type gain_type =
  | Gain_fixed (** [gain = gainprm.[0]] *)
  | Gain_affine (** [gain = gainprm . (1, length, velocity)] *)

(** The type for the same, for the bias. *)
type bias_type =
  | Bias_none (** [bias = 0] *)
  | Bias_affine (** [bias = biasprm . (1, length, velocity)] *)

(** The type for an actuator's internal dynamics. A muscle or a PID is not
    v1. *)
type dyn_type =
  | Dyn_none (** The input is the control. *)
  | Dyn_integrator (** [act' = ctrl], so the input is the integrated setpoint. *)
  | Dyn_filter (** [act' = (ctrl - act) / max(dynprm.[0], minval)] *)

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
  ; nwrap : int (** Tendon path points, summed over the tendons. *)
  ; nu : int (** Actuators. *)
  ; na : int (** Activation states, summed over actuators. *)
  ; nsensor : int
  ; nsensordata : int (** The width of the sensor block. *)
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
  ; geom_condim : int array
    (** [[ngeom]], the contact dimensionality each geom asks for. Host
        data: which contact rows a model can produce is decided when the
        program is built. *)
  ; contacts_enabled : bool
    (** Whether MuJoCo's contact-disable flag is set. A model that disables
        contacts produces no rows and no contact array. *)
  ; sensor_type : sensor_type array (** [[nsensor]] *)
  ; sensor_objtype : obj_type array (** [[nsensor]] *)
  ; sensor_objid : int array
    (** [[nsensor]], the joint, site, body, actuator or tendon it measures,
        depending on its kind. *)
  ; sensor_adr : int array (** [[nsensor]], its first slot in the block. *)
  ; sensor_dim : int array (** [[nsensor]] *)
  ; sensor_refid : int array
    (** [[nsensor]], the reference object, [-1] for none. v1 has no reference
        frames and refuses one. *)
  ; tendon_adr : int array (** [[ntendon]], first wrap of each tendon. *)
  ; tendon_num : int array (** [[ntendon]], how many wraps it has. *)
  ; wrap_type : wrap_type array (** [[nwrap]] *)
  ; wrap_objid : int array (** [[nwrap]], the joint or site wrapped. *)
  ; exclude_pairs : (int * int) array
    (** [[ [nexclude] ]] body pairs whose geoms never touch, sorted with the
        smaller id first. *)
  ; ten_limited : bool array (** [[ntendon]], whether its range is a row. *)
  ; actuator_trntype : trn_type array (** [[nu]], what each actuator drives. *)
  ; actuator_trnid : int array
    (** [[nu;2]], flattened: the joint or tendon index, and MuJoCo's second
        id, which v1 requires to be zero. Host data, because which moment
        arm to build is decided when the program is built. *)
  ; actuator_gaintype : gain_type array (** [[nu]] *)
  ; actuator_biastype : bias_type array (** [[nu]] *)
  ; actuator_dyntype : dyn_type array (** [[nu]] *)
  ; actuator_actadr : int array
    (** [[nu]], the first activation state of each actuator, [-1] when it has
        none. *)
  ; actuator_actearly : bool array
    (** [[nu]], whether the actuator reads the next activation instead of the
        current one. v1 does not have it; the loader carries it so the kernel
        can refuse a model that asks, rather than quietly using the wrong
        state. *)
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
