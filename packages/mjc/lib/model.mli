(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Model parameters.

    A {!t} pairs a {!Topo.t} with the model's numbers. The numbers are tensors
    because they are what a differentiated step and a compiled step read: a
    mass, an inertia, a geom size or the timestep is a leaf of the parameter
    structure, so {!Nx.Ptree}-driven transformations see it.

    The field names and shapes follow MuJoCo's [mjModel], one tensor per
    [mjModel] array. Shapes are written after the field, with [nbody], [njnt],
    [nv] and so on from the {!Topo.t}. *)

(** {1:params Parameters} *)

(** The parameter structure. Its leaves are tensors on any dtype and device
    set; [mjc] instantiates it at [float64] on the host. *)
module Params : sig
  type ('a, 'd) t =
    { body_pos : 'a (** [[nbody;3]], body frame in the parent frame. *)
    ; body_quat : 'a (** [[nbody;4]], body frame in the parent frame. *)
    ; body_ipos : 'a (** [[nbody;3]], inertial frame in the body frame. *)
    ; body_iquat : 'a (** [[nbody;4]], inertial frame in the body frame. *)
    ; body_mass : 'a (** [[nbody]] *)
    ; body_inertia : 'a (** [[nbody;3]], diagonal inertia, principal axes. *)
    ; body_subtreemass : 'a (** [[nbody]], the mass of the subtree at a body. *)
    ; body_invweight0 : 'a
      (** [[nbody;2]], approximate [1/J M⁻¹ Jᵀ] for translation and
          rotation at the reference configuration. *)
    ; jnt_axis : 'a (** [[njnt;3]], joint axis in the body frame. *)
    ; jnt_pos : 'a (** [[njnt;3]], joint anchor in the body frame. *)
    ; jnt_range : 'a (** [[njnt;2]], limits. *)
    ; jnt_limited : 'a (** [[njnt]], [1] where [jnt_range] is active. *)
    ; jnt_margin : 'a (** [[njnt]], the distance at which a limit is a row. *)
    ; jnt_solref : 'a (** [[njnt;2]], the limit's stiffness and damping. *)
    ; jnt_solimp : 'a (** [[njnt;5]], the limit's impedance. *)
    ; jnt_stiffness : 'a (** [[njnt]], spring stiffness. *)
    ; dof_armature : 'a (** [[nv]] *)
    ; dof_damping : 'a (** [[nv]] *)
    ; dof_frictionloss : 'a (** [[nv]] *)
    ; dof_invweight0 : 'a (** [[nv]], [1 / M_ii] at the reference configuration. *)
    ; qpos0 : 'a (** [[nq]], the reference configuration. *)
    ; qpos_spring : 'a (** [[nq]], the spring reference configuration. *)
    ; geom_pos : 'a (** [[ngeom;3]], geom frame in the body frame. *)
    ; geom_quat : 'a (** [[ngeom;4]], geom frame in the body frame. *)
    ; geom_size : 'a (** [[ngeom;3]], size parameters, by geom type. *)
    ; geom_rgba : 'a (** [[ngeom;4]], colour, for the viewer. *)
    ; geom_friction : 'a (** [[ngeom;3]], sliding, torsional, rolling. *)
    ; geom_margin : 'a (** [[ngeom]], contact detection distance. *)
    ; geom_gap : 'a (** [[ngeom]], contact inclusion distance. *)
    ; geom_solmix : 'a (** [[ngeom]], mixing weight for solver parameters. *)
    ; geom_solref : 'a (** [[ngeom;2]], constraint stiffness and damping. *)
    ; geom_solimp : 'a (** [[ngeom;5]], constraint impedance. *)
    ; site_pos : 'a (** [[nsite;3]], site frame in the body frame. *)
    ; site_quat : 'a (** [[nsite;4]], site frame in the body frame. *)
    ; site_size : 'a (** [[nsite;3]] *)
    ; eq_obj1id : 'a (** [[neq]], the body or site of the first object. *)
    ; eq_obj2id : 'a (** [[neq]], the second, [-1] when there is none. *)
    ; eq_data : 'a (** [[neq;12]], anchors, reference pose and coefficients. *)
    ; eq_solref : 'a (** [[neq;2]] *)
    ; eq_solimp : 'a (** [[neq;5]] *)
    ; eq_active0 : 'a (** [[neq]], [1] where the constraint is on. *)
    ; gravity : 'a (** [[3]] *)
    ; timestep : 'a (** [[]], a 0-d tensor. *)
    ; iterations : 'a (** [[]], the solver's iteration budget. *)
    ; tolerance : 'a (** [[]], and its tolerance. *)
    }
  [@@deriving ptree]
end

(** {1:model The model} *)

(** The type for models. [topo] is skipped by the derived walk: it is host
    data, the same for every dtype and every placement. *)
type ('a, 'd) t =
  { topo : Topo.t [@ptree.skip]
  ; params : ('a, 'd) Params.t
  }
[@@deriving ptree]

(** The type for a [float64] model on the host, which is what every v1 kernel
    takes. *)
type host = (Nx.host Nx.float64_t, Nx.host) t

(** {1:structure Structure} *)

(** [ptree] describes a {!t} to the transformations: its parameters are
    leaves, its topology is a constant. *)
val ptree : (host, Nx.host) Nx.Ptree.t

(** {1:accessors Accessors} *)

(** [nq m] is the number of position coordinates. *)
val nq : _ t -> int

(** [nv m] is the number of velocity coordinates. *)
val nv : _ t -> int

(** [nbody m] is the number of bodies, including the world body. *)
val nbody : _ t -> int

(** [ngeom m] is the number of geoms. *)
val ngeom : _ t -> int

(** {1:checks Checks} *)

(** [validate m] is [Ok ()] when the topology is valid and every parameter
    tensor has the shape its name and the topology imply, and
    [Error message] otherwise. The message names the field, the expected
    shape and the shape found. *)
val validate : host -> (unit, string) result

(** {1:shape_keys Shape keys} *)

(** The compile-time shape of a model: what a compiled step is specialised on.
    Two models with equal keys run the same compiled program, so every value
    here must change the traced program and nothing else may. *)
type shape_key =
  { dtype : string (** ["float64"] or ["float32"]. *)
  ; nbody : int
  ; ngeom : int
  ; njnt : int
  ; neq : int
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

(** [shape_key m] is the key a compiled step is specialised on. *)
val shape_key : _ t -> shape_key

(** [equal_shape_key a b] is [true] iff [a] and [b] are the same key. *)
val equal_shape_key : shape_key -> shape_key -> bool

(** [pp_shape_key] formats a key on one line, as [nv=27 ngeom=20 ...]. *)
val pp_shape_key : Format.formatter -> shape_key -> unit
