(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Data: the state a step advances, and the results it leaves behind.

    A {!t} is one structure, part state and part output. Everything that
    {!Nx.Ptree} walks is a tensor of the model's dtype: the state at the top,
    then the derived quantities a test, a viewer or a later stage reads.

    All shapes are unbatched. [qpos] is [[nq]], [xpos] is [[nbody;3]], and so
    on; batching is a leading axis that {!Rune.vmap} adds to the leaves and
    removes inside the mapped function. *)

(** {1:data The structure} *)

type ('a, 'd) t =
  { time : 'a (** [[]], seconds. *)
  ; qpos : 'a (** [[nq]], joint coordinates. *)
  ; qvel : 'a (** [[nv]], joint velocities. *)
  ; qacc : 'a (** [[nv]], joint accelerations. *)
  ; act : 'a (** [[na]], actuator activation states. *)
  ; act_dot : 'a (** [[na]], their time derivative, from the actuator dynamics. *)
  ; ctrl : 'a (** [[nu]], actuator controls. *)
  ; qfrc_applied : 'a (** [[nv]], external forces, set by the caller. *)
  ; xpos : 'a (** [[nbody;3]], body frames, world. *)
  ; xquat : 'a (** [[nbody;4]], body frames, world. *)
  ; xipos : 'a (** [[nbody;3]], inertial frames, world. *)
  ; ximat : 'a (** [[nbody;9]], inertial frames, world. *)
  ; geom_xpos : 'a (** [[ngeom;3]], geom frames, world. *)
  ; geom_xmat : 'a (** [[ngeom;9]], geom frames, world. *)
  ; site_xpos : 'a (** [[nsite;3]], site frames, world. *)
  ; site_xmat : 'a (** [[nsite;9]], site frames, world. *)
  ; xanchor : 'a (** [[njnt;3]], joint anchors, world. *)
  ; xaxis : 'a (** [[njnt;3]], joint axes, world. *)
  ; cvel : 'a (** [[nbody;6]], spatial velocities in the body frame. *)
  ; cdof : 'a (** [[nv;6]], motion subspace of each dof. *)
  ; cdof_dot : 'a (** [[nv;6]], its time derivative. *)
  ; cinert : 'a (** [[nbody;10]], inertia about the subtree centre of mass. *)
  ; com : 'a (** [[3]], centre of mass. *)
  ; subtree_com : 'a (** [[nbody;3]], centre of mass of each subtree. *)
  ; subtree_linvel : 'a (** [[nbody;3]], its velocity, world. *)
  ; subtree_angmom : 'a (** [[nbody;3]], angular momentum about its centre. *)
  ; qfrc_bias : 'a (** [[nv]], Coriolis, centrifugal and gravity forces. *)
  ; qfrc_passive : 'a (** [[nv]], springs, dampers and fluid forces. *)
  ; qfrc_actuator : 'a (** [[nv]], actuator forces. *)
  ; actuator_length : 'a (** [[nu]], each transmission's length. *)
  ; actuator_velocity : 'a (** [[nu]], its velocity. *)
  ; actuator_force : 'a (** [[nu]], the scalar force each actuator makes. *)
  ; qfrc_constraint : 'a (** [[nv]], constraint forces. *)
  ; qfrc_smooth : 'a (** [[nv]], the total force before constraints. *)
  ; mass_matrix : 'a (** [[nv;nv]], the mass matrix. *)
  ; ncon : 'd Nx.int32_t (** [[]], the number of contacts. *)
  ; contact_dist : 'a (** [[nconmax]], signed distance, negative when touching. *)
  ; contact_pos : 'a (** [[nconmax;3]], the contact point, world. *)
  ; contact_frame : 'a
    (** [[nconmax;9]], the contact frame's three rows: the normal, then two
        orthonormal tangents. The pyramid's friction rows are built from it, so
        a viewer that wants to decompose the contact force needs it too. *)
  ; contact_normal : 'a
    (** [[nconmax;3]], the normal, from geom 1's surface towards geom 2. *)
  ; contact_geom : 'd Nx.int32_t (** [[nconmax;2]], the geoms that made it. *)
  ; contact_body : 'd Nx.int32_t (** [[nconmax;2]], their bodies. *)
  ; contact_dim : 'd Nx.int32_t (** [[nconmax]], the contact dimensionality. *)
  ; contact_includemargin : 'a
    (** [[nconmax]], the distance inside which the contact is real. *)
  ; contact_friction : 'a (** [[nconmax;5]], in MuJoCo's slot layout. *)
  ; contact_solref : 'a (** [[nconmax;2]], mixed from the two geoms. *)
  ; contact_jac : 'a
    (** [[nconmax;nv]], the normal row's Jacobian. It is computed where the
        contact geometry is, with the bodies as host values, because a
        Jacobian is a walk up a body's dof chain. *)
  ; contact_solimp : 'a (** [[nconmax;5]] *)
  ; nefc : 'd Nx.int32_t (** [[]], the number of constraint rows. *)
  ; efc_type : 'd Nx.int32_t (** [[nefcmax]], row type. *)
  ; efc_id : 'd Nx.int32_t (** [[nefcmax]], the joint or other object. *)
  ; efc_pos : 'a (** [[nefcmax]], constraint position (violation). *)
  ; efc_margin : 'a (** [[nefcmax]], the position at which it activates. *)
  ; efc_posnorm : 'a
    (** [[nefcmax]], the position the impedance is read at: the row's own
        violation, except for an equality, which reads the norm over its
        block of rows. *)
  ; efc_frictionloss : 'a
    (** [[nefcmax]], the friction loss of the row: zero for every kind but
        a friction row, where it bounds the force. *)
  ; efc_diagA : 'a
    (** [[nefcmax]], the row's approximate inverse inertia [1/J M⁻¹ Jᵀ],
        which sets its regularisation. *)
  ; efc_solref : 'a (** [[nefcmax;2]], the row's stiffness and damping. *)
  ; efc_solimp : 'a (** [[nefcmax;5]], the row's impedance. *)
  ; efc_jdotv : 'a
    (** [[nefcmax]], the row's [Jdot qvel]: the reference acceleration
        must cancel it or the constraint would drift. *)
  ; efc_J : 'a (** [[nefcmax;nv]], the Jacobian rows. *)
  ; efc_vel : 'a (** [[nefcmax]], the row velocity [J qvel]. *)
  ; efc_R : 'a (** [[nefcmax]], the impedance's regularisation. *)
  ; efc_D : 'a (** [[nefcmax]], the impedance's diagonal weight. *)
  ; efc_aref : 'a (** [[nefcmax]], the reference acceleration. *)
  ; efc_force : 'a (** [[nefcmax]], the force, also the warmstart. *)
  }
[@@deriving ptree]

(** The type for [float64] data on the host, which is what every v1 kernel
    takes. *)
type host = (Nx.host Nx.float64_t, Nx.host) t

(** {1:structure Structure} *)

(** [ptree] describes a {!t} to the transformations. *)
val ptree : (host, Nx.host) Nx.Ptree.t

(** {1:construction Construction and accessors} *)

(** [create m] is the state of model [m] at rest: [qpos] is [m.qpos0], every
    other leaf is zero, and [time] is zero with a zero time step. *)
val create : Model.host -> host

(** [nq m] and [nv m] are the model's counts, repeated for convenience. *)
val nq : Model.host -> int

(** [nv m] is the number of velocity coordinates. *)
val nv : Model.host -> int

(** {1:checks Checks} *)

(** [validate m d] is [Ok ()] when every leaf of [d] has the shape [m]'s
    topology implies, and [Error message] otherwise, naming the field, the
    expected shape and the shape found. *)
val validate : Model.host -> host -> (unit, string) result

(** [pp] formats the state's counts and the time. *)
val pp : Format.formatter -> host -> unit
