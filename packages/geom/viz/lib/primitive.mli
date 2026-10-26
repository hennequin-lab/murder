(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Neutral three-dimensional primitives for the renderer.

    This module is the boundary between the Clifford algebras of [geom] and
    the drawing layer. The renderer consumes only {!t} values; the adapters at
    the bottom of this interface turn [Cga.t] and [Pga.t] objects into
    primitives. Hand-built primitives are equally valid, which keeps the
    renderer testable without the algebra.

    Coordinates are plain host floats in world units chosen by the caller.
    No camera, page or clipping information appears here. *)

(** {1:vectors Points and vectors} *)

(** The type for a point or vector in three dimensions. The fields are
    public so that callers can pattern-match and update them. *)
type p3 =
  { x : float
  ; y : float
  ; z : float
  }

(** [p3 x y z] is the point [(x, y, z)]. *)
val p3 : float -> float -> float -> p3

(** [add a b] is the vector sum. *)
val add : p3 -> p3 -> p3

(** [sub a b] is the vector difference [a - b]. *)
val sub : p3 -> p3 -> p3

(** [scale a s] is [a] multiplied by the scalar [s]. *)
val scale : p3 -> float -> p3

(** [dot a b] is the scalar product [a · b]. *)
val dot : p3 -> p3 -> float

(** [cross a b] is the cross product [a × b]. *)
val cross : p3 -> p3 -> p3

(** [norm a] is the Euclidean length of [a]. *)
val norm : p3 -> float

(** [normalize a] is [a] divided by its length. A zero vector is returned
    unchanged. *)
val normalize : p3 -> p3

(** {1:objects Primitives} *)

(** The type for circles in three dimensions. *)
type circle =
  { center : p3
  ; normal : p3 (** the plane normal; the circle lies in a plane with it *)
  ; radius : float
  }

(** The type for spheres. *)
type sphere =
  { center : p3
  ; radius : float
  }

(** The type for planes. The normal is expected to be a unit vector and
    [offset] is the signed distance from the origin:
    [{x : normal · x = offset}]. *)
type plane =
  { normal : p3
  ; offset : float
  ; patch : p3 list
    (** a finite polygon lying in the plane, drawn instead of the
        infinite surface *)
  }

(** The type for coordinate triads. The three endpoints are the tips of the
    [x], [y] and [z] axes. *)
type frame =
  { origin : p3
  ; x : p3
  ; y : p3
  ; z : p3
  }

(** The type for primitives. *)
type t =
  | Point of p3 (** a dot, drawn with the style's fill and stroke *)
  | Segment of p3 * p3 (** a straight line between two points *)
  | Polyline of p3 list (** an open sequence of connected segments *)
  | Polygon of p3 list (** a closed, filled polygon *)
  | Circle of circle (** a circle, sampled and drawn as a closed curve *)
  | Sphere of sphere (** a sphere, drawn as the silhouette seen from the camera *)
  | Plane of plane (** a finite patch of a plane *)
  | Frame of frame (** a coordinate triad, with red, green and blue axes *)

(** [transform m p] applies the rigid motion [m] to the primitive [p].

    Points are transformed with {!Geom.Pga.apply_motor}. A sphere keeps its
    radius; a circle keeps its radius and has its normal carried by the
    rotation; a plane has its patch transformed and its normal and offset
    rebuilt from a transformed point of the plane. This is a rigid action:
    dilations and transversions have no [Motor.t], so conformal
    transformations must be applied to the algebra object and re-extracted
    with an adapter. *)
val transform : Geom.Motor.t -> t -> t

(** [plane_patch ~normal ~offset] is a square of side four in the plane
    [{x : normal · x = offset}], centred on the point of the plane closest to
    the origin. It is the default patch used by the PGA plane adapter. The
    orientation of the square is arbitrary but deterministic. *)
val plane_patch : normal:p3 -> offset:float -> p3 list

(** {1:adapters Adapters from the algebras} *)

(** [p3_of_tensor v] reads an unbatched tensor of shape [[3]] as a point. It
    is the low-level helper used by the adapters; the adapters assume
    unbatched algebra values. *)
val p3_of_tensor : Nx.float32_t -> p3

(** [scalar_of_tensor v] reads an unbatched scalar tensor as a float. *)
val scalar_of_tensor : Nx.float32_t -> float

(** [of_cga_point p] is the point at [Cga.coords p]. An ideal point has no
    finite coordinates and produces an unspecified position. *)
val of_cga_point : Geom.Cga.t -> t

(** [of_cga_sphere s] is the sphere of the CGA sphere [s]. *)
val of_cga_sphere : Geom.Cga.t -> t

(** [of_cga_plane pi] is the plane of the CGA plane [pi], normalized with
    [Cga.normalize_plane] and given a default {!plane_patch}. *)
val of_cga_plane : Geom.Cga.t -> t

(** [of_cga_line ?extent l] is a segment of half-length [extent] (default
    [2.5]) around the point of the finite CGA line [l] closest to the origin.
    A line is infinite; a picture is not. *)
val of_cga_line : ?extent:float -> Geom.Cga.t -> t

(** [of_cga_circle c] is the circle of the CGA circle [c]. *)
val of_cga_circle : Geom.Cga.t -> t

(** [of_pga_point p] is the projective point [p] at its Euclidean
    coordinates. The point is divided by its weight, so a rescaled
    representative gives the same primitive; an ideal point has zero weight
    and produces an unspecified position. *)
val of_pga_point : Geom.Pga.t -> t

(** [of_pga_plane pi] is the plane of the PGA plane [pi]. The plane is
    normalized projectively, so a rescaled representative gives the same
    primitive, and the finite patch is built by {!plane_patch}. *)
val of_pga_plane : Geom.Pga.t -> t

(** [of_pga_line ?extent l] is a segment of half-length [extent] (default
    [2.5]) around the point of the finite PGA line [l] closest to the origin,
    [direction × moment]. The line must not be ideal. *)
val of_pga_line : ?extent:float -> Geom.Pga.t -> t
