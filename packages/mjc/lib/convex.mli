(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Convex support functions.

    MuJoCo's convex collision path -- the one that handles every geom pair
    {!Collision} does not have a primitive formula for, including all four
    heightfield pairs and the three ellipsoid pairs -- is a support-mapped
    algorithm: it needs, from a geom, only the point of its convex hull that is
    farthest along a direction. That point is analytic for every primitive, which
    is what this module is, and it is the part of the path that can be written
    and checked without the iteration on top of it.

    The functions follow MuJoCo's own ({!mjc_sphereSupport} and its neighbours in
    [engine_collision_convex.c]) term for term, including the local-frame
    rotation through [matᵀ] and the tie-breaking choices at a zero direction,
    because the iteration downstream is only reproducible if its inputs are.

    Every function takes the direction as either one point ([3]) or a batch
    ([k; 3]) and returns the same shape: the contact stage computes a whole
    geom-type pair's candidates at once, and the iteration that follows will have
    to be a fixed-length masked loop for the same reason the solver's rounds are. *)

(** The shapes a support point is defined for. [Point] is what a pair with no
    extent becomes -- the centre of a sphere used as a probe. *)
type shape =
  | Point
  | Sphere
  | Line (** a capsule's axial segment, without its radius *)
  | Capsule
  | Ellipsoid
  | Cylinder
  | Box

(** [support shape ~pos ~mat ~size ~dir] is the point of [shape]'s convex hull
    farthest along [dir], in the world frame.

    [pos] is the geom's centre, [mat] its [3x3] orientation (columns are the geom
    axes), [size] its MuJoCo size triple, and [dir] the direction: a point, or a
    batch of them. A direction of zero length is the degenerate case each shape
    resolves the way MuJoCo does -- the sphere and capsule answer [pos], the box
    its positive corner, the cylinder its top rim, the ellipsoid its [x] extent. *)
val support
  :  shape
  -> pos:('s, 'a) Rigid.f
  -> mat:('s, 'a) Rigid.f
  -> size:('s, 'a) Rigid.f
  -> dir:('s, 'a) Rigid.f
  -> ('s, 'a) Rigid.f
