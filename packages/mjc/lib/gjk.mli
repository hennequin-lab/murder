(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The convex contact test: GJK over {!Convex.support}.

    This is the first half of the convex path, the one that answers "do these
    two touch". MuJoCo 3.3.7 does not use libccd for this any more -- it has a
    native implementation in [engine_collision_gjk.c] (2365 lines: GJK for the
    contact test, EPA for the depth and the witness points, and a polytope with
    face adjacency for the rest) -- and the two conventions that matter here are
    its: the support point of each shape is pushed out by **half its margin**
    along the direction, and with a zero distance cutoff GJK never reports a
    distance, only whether the origin is inside the inflated Minkowski
    difference. Separated pairs are the caller's business, exactly as the
    primitive pairs already treat a distance beyond the margin as "no contact".

    Written on float arrays rather than tensors, and that is the honest state of
    it: the iteration is data-dependent in a way a compiled step cannot express,
    so the batched form is a fixed-length masked loop and a separate piece of
    work. What this buys first is an oracle: every convex pair can now be
    decided, and the answer can be checked against the analytic pairs the engine
    already has before any of it is batched or ported.

    See [doc/adr/0013-convex-path.md] for the order of the rest. *)

(** A convex shape as the iteration needs it. *)
type obj =
  { support : float array -> float array
    (** The farthest point along a **unit** direction, world frame. *)
  ; margin : float
    (** Half of it inflates a support point, which is MuJoCo's rule: a caller
        passes the *pair* margin to both objects, so the pair's threshold is
        that margin. *)
  }

(** [of_geom shape ~pos ~mat ~size ~margin] is the {!Convex.support} function of
    one geom, with [mat] a row-major [3x3]. *)
val of_geom
  :  Convex.shape
  -> pos:float array
  -> mat:float array
  -> size:float array
  -> margin:float
  -> obj

(** [closest_point ps] is the point of the convex hull of [ps] nearest the
    origin, for one to four points. It is the distance subalgorithm on its own,
    which is the part of GJK that is pure geometry and can be checked against a
    brute force. *)
val closest_point : float array list -> float array

(** [intersect ~iterations ~tolerance a b] is whether [a] and [b] touch, the
    origin being inside the inflated Minkowski difference. [iterations] bounds
    the loop and [tolerance] the duality gap; MuJoCo's defaults are 100 and
    1e-6. A pair that cannot be decided inside the budget is reported as
    touching, which is the conservative answer for a physics engine and the one
    the primitive paths take at a degenerate distance.

    [trace] prints the iteration to stdout, which is what a disagreement between
    this and a closed form is diagnosed with. *)
val intersect : ?iterations:int -> ?tolerance:float -> ?trace:bool -> obj -> obj -> bool
