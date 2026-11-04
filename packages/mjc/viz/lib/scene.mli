(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The scene: what a model and a state look like, with no renderer in sight.

    This is the intermediate representation the viewer draws and the tests
    read. It holds host floats in world coordinates, so a test can assert on
    it without a display and a backend can be swapped without touching the
    translation from {!Mjc.Data}.

    The shapes are MuJoCo's v1 geom set, resolved from the geom's world frame
    and size: a capsule knows its two endpoints, a box its centre, axes and
    half-extents, so no backend has to know the convention that a capsule lies
    along its local z. *)

(** {1:types Types} *)

type p3 =
  { x : float
  ; y : float
  ; z : float
  }

type colour =
  { r : float (** red, in [0, 1]. *)
  ; g : float
  ; b : float
  ; a : float (** opacity, in [0, 1]. *)
  }

(** The type for shapes, in world coordinates. *)
type shape =
  | Sphere of
      { centre : p3
      ; radius : float
      }
  | Capsule of
      { a : p3 (** one end of the axis. *)
      ; b : p3 (** the other end. *)
      ; radius : float
      }
  | Box of
      { centre : p3
      ; axes : p3 * p3 * p3 (** the three unit axes, world. *)
      ; half : float * float * float (** half-extents along them. *)
      }
  | Cylinder of
      { centre : p3
      ; axis : p3
      ; radius : float
      ; half_height : float
      }
  | Ellipsoid of
      { centre : p3
      ; axes : p3 * p3 * p3
      ; radii : float * float * float
      }
  | Plane of
      { normal : p3
      ; offset : float (** the signed distance from the origin. *)
      ; half : float * float (** half-sizes of the drawn patch. *)
      }
  | Unsupported of string (** a geom type v1 does not draw, by name. *)

type geom =
  { name : string
  ; shape : shape
  ; colour : colour
  }

(** The type for body frames: the origin and the three axis tips, world. *)
type frame =
  { origin : p3
  ; x : p3
  ; y : p3
  ; z : p3
  }

type body =
  { name : string
  ; frame : frame
  }

(** The type for a contact, as the solver sees it. *)
type contact =
  { point : p3
  ; normal : p3 (** unit, from geom 1 towards geom 2. *)
  ; depth : float (** signed distance; negative is penetration. *)
  ; dim : int
  }

(** The type for a force arrow: a point and a vector. *)
type arrow =
  { from : p3
  ; vector : p3
  }

(** The type for the kinds of joint, which is the model's [jnt_type] under a
    name that does not make the viewer read {!Mjc.Topo}. *)
type jnt =
  | Hinge
  | Slide
  | Ball
  | Free

(** The type for an inertia box: the box about a body's principal axes whose
    diagonal inertia is the body's. It is drawn, not simulated. *)
type inertia =
  { centre : p3
  ; axes : p3 * p3 * p3 (** the principal axes, world. *)
  ; half : float * float * float (** half-extents along them. *)
  }

(** The type for a joint's world placement: where it is and which way it acts. *)
type joint =
  { name : string
  ; kind : jnt
  ; anchor : p3 (** the joint anchor, world. *)
  ; axis : p3 (** the axis, world; unit for a hinge or a slide. *)
  }

(** The type for a state's mechanical energy, in the model's units. *)
type energy =
  { kinetic : float (** [1/2 qvel' M qvel]. *)
  ; potential : float (** [-sum m g . xipos], the world's gravity. *)
  }

(** The type for scenes. *)
type t =
  { time : float
  ; bodies : body list
  ; geoms : geom list
  ; contacts : contact list
  ; forces : arrow list
  ; inertia : inertia list
  ; com : p3 list (** the subtree centre of mass of every body with mass. *)
  ; joints : joint list
  ; velocity : arrow list (** per-body velocity at the centre of mass. *)
  ; energy : energy option
  }

(** {1:build Building} *)

(** [of_state ?frames ?contacts ?forces ?inertia ?com ?joints ?velocity ?energy m d] is the scene of [d]: every geom of [m] at its world pose in
    [d], and, when asked for, the overlays. A layer that is not asked for is
    the empty list, so a backend can draw the record without a case for
    absent.

    [energy] is the only option that costs a pass of its own — it builds the
    mass matrix — and it is the only one that reads [qvel], so the others are
    correct on a state whose kinematics were re-run without a step. *)
val of_state
  :  ?frames:bool
  -> ?contacts:bool
  -> ?forces:bool
  -> ?inertia:bool
  -> ?com:bool
  -> ?joints:bool
  -> ?velocity:bool
  -> ?energy:bool
  -> Mjc.Model.host
  -> Mjc.Data.host
  -> t

(** {1:overlays Overlays}

    The overlays are pure functions of a model and a state, which is what
    makes them testable without a display. {!of_state} calls them; they are
    exposed so a test can compare one number at a time. *)

(** [inertia_boxes m d] is one box per body with mass, about the body's principal
    axes at its centre of mass. The half-extents are those of the box with the
    same diagonal inertia: [h_x^2 = 3/(2m) (I_y + I_z - I_x)] and its cyclic
    permutations, which is what MuJoCo's own viewer draws. A non-physical
    inertia — one where that expression is negative — is clamped at a hair
    above zero rather than drawn inside out. *)
val inertia_boxes : Mjc.Model.host -> Mjc.Data.host -> inertia list

(** [subtree_com m d] is [subtree_com] of every body with mass: the centre of mass of
    the bodies a body carries, itself included. *)
val subtree_com : Mjc.Model.host -> Mjc.Data.host -> p3 list

(** [joint_axes m d] is every joint's anchor and axis in world coordinates. *)
val joint_axes : Mjc.Model.host -> Mjc.Data.host -> joint list

(** [velocities ?scale m d] is one arrow per body with mass, at its own centre
    of mass, along that point's velocity, which is the Jacobian of the point
    times [qvel]. (The engine's [cvel] is com-based over a body's *subtree*, so
    composing it down a chain needs a shift; the Jacobian has no such
    subtlety.) [scale] multiplies the velocity and defaults to [1.]; a model in
    metres per second needs a larger one to show at all. *)
val velocities : ?scale:float -> Mjc.Model.host -> Mjc.Data.host -> arrow list

(** [mechanical_energy m d] is the kinetic energy — summed over bodies as
    [1/2 m |v|^2 + 1/2 w' I w], which is [1/2 qvel' M qvel] without building
    the matrix — and the potential energy [-sum m g . xipos], both in the
    model's units. It needs a state whose kinematics and velocities are
    current, which is what a step or {!Mjc.Forward.forward} leaves; a bare
    [Kinematics.forward] after a joint nudge has stale velocities. *)
val mechanical_energy : Mjc.Model.host -> Mjc.Data.host -> energy

(** {1:bounds Bounds} *)

(** [extent s] is the box [(lo, hi)] of what a backend would draw, leaving out
    planes — the ground, not the subject — and anything v1 cannot draw. A scene
    with nothing to frame gives a unit box. *)
val extent : t -> p3 * p3

(** [framing s] is [(eye, target)], the view both backends open on: the centre
    of {!extent}, with the eye offset by [(2, -2.8, 1.6)] times the box's
    largest side, which is far enough out that a model of any size fits. *)
val framing : t -> p3 * p3

(** [pp] formats a summary: the time, the geom and contact counts. *)
val pp : Format.formatter -> t -> unit
