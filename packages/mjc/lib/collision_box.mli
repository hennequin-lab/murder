(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Box-box contact manifolds, ported from MuJoCo's [mjc_BoxBox].

    This is the geometry kernel only: given two boxes' world poses and
    half-extents it returns the contacts between them, as MuJoCo 3.17 computes
    them. It is host code, not tensor code, and it is deliberately separate
    from {!Mjc.Collision} until it is tested: the manifold's control flow is
    data-dependent (a separating axis, then either one edge contact or a
    clipped polygon of up to twelve), and a traced version of that is a
    different, harder problem than a faithful port.

    The algorithm, in the C engine's two stages:

    - {b Separating-axis test.} Fifteen candidate axes — three face normals per
      box and the nine cross products of edge directions. If the boxes are
      further apart than [margin] along any of them there is no contact. Face
      axes win near-ties, because a face gives a multi-point manifold, and an
      edge axis must beat the best face by {!edge_bias}, because resting stacks
      otherwise flip between the two codes on rounding noise.
    - {b Manifold.} A face axis makes the face's owner the reference box: the
      other box's most opposed face is clipped against the reference face's
      four side planes (Sutherland-Hodgman) and every surviving vertex within
      the margin becomes a contact, at the midpoint between the two surfaces.
      An edge axis gives one contact at the midpoint of the closest points
      between the two supporting edge segments.

    Every constant below is the C engine's, because each one is a tie-break
    that only shows up in whether a stack sits still. *)

(** {1:tuning Rounding scales}

    These are MuJoCo's, in [mjtNum] epsilon units, and they are not to be
    "tidied": they decide which of two nearly-equal axes wins, and the stacks
    that made them necessary are exactly the ones this kernel exists for. *)

(** Slack on the separation tests, times the sum of the half-extents. *)
val sep_eps : float

(** [sin^2] below which an edge-cross axis is rounding noise. *)
val par_eps : float

(** An axis component below which a supporting corner is ambiguous. *)
val sgn_eps : float

(** Squared relative radius for dropping near-duplicate clipped vertices. *)
val dupe_eps : float

(** Relative penalty an edge axis pays on a near-tie with a face axis. *)
val edge_bias : float

(** Vertex capacity of the clip polygon, and the most contacts one pair can
    make. *)
val max_verts : int

(** {1:manifold Manifold} *)

(** The type for one contact: the signed distance along the normal (negative
    is penetration), the world point, and the world unit normal, from the first
    box towards the second. *)
type contact =
  { dist : float
  ; pos : float array (** [[3]] *)
  ; normal : float array (** [[3]], unit. *)
  }

(** [manifold ~pos1 ~mat1 ~size1 ~pos2 ~mat2 ~size2 ~margin] is the contacts
    between two boxes, in the C engine's order, or an empty array when they are
    separated by more than [margin] along every candidate axis.

    [pos] is a world position, [mat] a row-major rotation matrix and [size] the
    half-extents, which is the layout of [Data.geom_xpos], [Data.geom_xmat] and
    [Params.geom_size] — a caller can hand it three slices and no copying. *)
val manifold
  :  pos1:float array
  -> mat1:float array
  -> size1:float array
  -> pos2:float array
  -> mat2:float array
  -> size2:float array
  -> margin:float
  -> contact array
