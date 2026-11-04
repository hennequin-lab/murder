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

(** The type for scenes. *)
type t =
  { time : float
  ; bodies : body list
  ; geoms : geom list
  ; contacts : contact list
  ; forces : arrow list
  }

(** {1:build Building} *)

(** [of_state ?bodies ?contacts ?forces m d] is the scene of [d]: every geom
    of [m] at its world pose in [d], and, when asked for, the body frames, the
    contact points and the constraint forces. *)
val of_state
  :  ?frames:bool
  -> ?contacts:bool
  -> ?forces:bool
  -> Mjc.Model.host
  -> Mjc.Data.host
  -> t

(** [pp] formats a summary: the time, the geom and contact counts. *)
val pp : Format.formatter -> t -> unit
