(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Plane-based projective geometric algebra, [Cl(3, 0, 1)].

    The algebra has three Euclidean directions [e_1, e_2, e_3] with square
    [+1] and one degenerate direction [e_0] with square [0]. Its objects are
    planes (grade 1), lines (grade 2) and points (grade 3): a point is the
    intersection of three planes, so it is a trivector. Ideal points are the
    points at infinity; a point has a weight and is ideal when that weight
    vanishes.

    The pseudoscalar is not invertible ([I^2 = 0]), so the Hodge {!dual}
    inherited from {!Algebra} raises for this signature. Use {!complement} for
    the Poincaré dual, and {!meet} and {!join} for the incidence calculus.
    Incidence is the wedge, for example [p and pi = 0] for a point [p] on a
    plane [pi]; unlike the conformal model, the degenerate metric does not
    reduce it to a scalar contraction of the dual point with the plane.

    The packed {!Motor} type is the even subalgebra of this algebra and is
    shared with {!Cga}; {!of_motor} and {!apply_motor} bridge the two
    representations.

    {b Normalisation.} The inherited {!Algebra.normalize} is the metric
    normalisation [a / norm a]. The projective normalisations are
    {!normalize_point} and {!normalize_plane}. *)

include Algebra.ALGEBRA with type kind = Nx.float32_elt

(** {1:basis Basis} *)

(** [e1] is the first Euclidean basis vector. *)
val e1 : t

(** [e2] is the second Euclidean basis vector. *)
val e2 : t

(** [e3] is the third Euclidean basis vector. *)
val e3 : t

(** [e0] is the degenerate basis vector, with [e0^2 = 0]. It is orthogonal to
    the Euclidean directions. *)
val e0 : t

(** {1:points Points} *)

(** [point x] is the projective point of the Euclidean coordinates [x], of
    shape [batch_shape @ [|3|]], represented by a trivector. *)
val point : Nx.float32_t -> t

(** [point3 x y z] is [point] of the single coordinate vector [(x, y, z)]. *)
val point3 : float -> float -> float -> t

(** [point_coords p] extracts the Euclidean coordinates of the point [p] as a
    tensor of shape [batch_shape @ [|3|]]. The point is divided by its weight;
    an ideal point has no finite coordinates and the result is unspecified. *)
val point_coords : t -> Nx.float32_t

(** [weight p] is the homogeneous weight of the point [p], the coefficient of
    the Euclidean pseudoscalar [e_123], of shape [batch_shape]. *)
val weight : t -> Nx.float32_t

(** [is_ideal p] is [true] where the weight of [p] vanishes, that is, where
    [p] is a point at infinity (a direction). *)
val is_ideal : t -> Nx.bool_t

(** [normalize_point p] divides the point [p] by its weight, so that the
    coefficient of the Euclidean pseudoscalar is [1]. An ideal point has zero
    weight and cannot be normalised. *)
val normalize_point : t -> t

(** [ideal_point d] is the ideal point (direction) of the Euclidean vector [d],
    of shape [batch_shape @ [|3|]], represented by a trivector of zero weight.
    It is the point at infinity in the direction [d]. *)
val ideal_point : Nx.float32_t -> t

(** {1:planes Planes} *)

(** [plane ~normal ~offset] is the plane [{x : n dot x = offset}], with [n] the
    given normal (shape [batch_shape @ [|3|]]) and the given scalar offset
    (shape [batch_shape]). The normal is not normalised; use
    {!normalize_plane} before comparing offsets. *)
val plane : normal:Nx.float32_t -> offset:Nx.float32_t -> t

(** [plane3 nx ny nz d] is [plane] of the single normal [(nx, ny, nz)] and
    offset [d]. *)
val plane3 : float -> float -> float -> float -> t

(** [plane_normal pi] extracts the normal of the plane [pi] as a tensor of shape
    [batch_shape @ [|3|]]. *)
val plane_normal : t -> Nx.float32_t

(** [plane_offset pi] extracts the offset of the plane [pi], of shape
    [batch_shape]. *)
val plane_offset : t -> Nx.float32_t

(** [normalize_plane pi] divides [pi] by the norm of its normal, so that its
    Euclidean part is a unit vector and its offset is a signed distance. *)
val normalize_plane : t -> t

(** {1:lines Lines} *)

(** [line_through p q] is the projective line through the two points [p] and
    [q], represented by a bivector. This is the join of the two points. *)
val line_through : t -> t -> t

(** [line_of_planes pi1 pi2] is the line where the two planes [pi1] and [pi2]
    meet, represented by their bivector. This is the meet of the two planes. *)
val line_of_planes : t -> t -> t

(** [line_direction l] is the Euclidean direction of the line [l] as a tensor
    of shape [batch_shape @ [|3|]]. *)
val line_direction : t -> Nx.float32_t

(** [line_moment l] is the moment of the line [l] with respect to the origin,
    as a tensor of shape [batch_shape @ [|3|]]. The Plücker coordinates of the
    line are [(direction, moment)]. *)
val line_moment : t -> Nx.float32_t

(** {1:incidence Incidence and composition} *)

(** [point_on_plane p pi] is [true] where the point [p] lies on the plane [pi],
    that is, [p and pi = 0]. *)
val point_on_plane : t -> t -> Nx.bool_t

(** [point_on_line p l] is [true] where the point [p] lies on the line [l].
    The test uses the complement dual, since the plain wedge of a point
    (grade 3) and a line (grade 2) is always zero in four dimensions. *)
val point_on_line : t -> t -> Nx.bool_t

(** [line_in_plane l pi] is [true] where the line [l] lies in the plane [pi]. *)
val line_in_plane : t -> t -> Nx.bool_t

(** [meet a b] is the outer product [a and b], the projective intersection
    when the operands are constraint objects of the right grades: two planes
    meet in a line, and a plane and a line meet in a point. The result has the
    sum of the grades; for other combinations it may be zero or the
    pseudoscalar. Use the named incidence predicates for cases where the meet
    is not a blade of the expected type. *)
val meet : t -> t -> t

(** [join a b] is the regressive product [a or b], the projective span when
    the operands are subspaces of the right grades: two points span a line, and
    a point and a line span a plane. *)
val join : t -> t -> t

(** {1:motors Motors} *)

(** [of_motor m] embeds the packed motor [m] into the dense even subalgebra of
    this algebra. The dual part of the packed basis ([e_i e_0] and
    [e_123 e_0]) is embedded with the opposite sign, the [epsilon -> -epsilon]
    automorphism of the dual quaternions; this aligns the packed translator
    with the PGA point convention, so that [apply_motor (translate t)] moves a
    point by [+t]. *)
val of_motor : Motor.t -> t

(** [apply_motor m x] is the action of the motor [m] on [x],
    [m x tilde(m)], without materialising the dense motor. Object grades and
    incidence relations are preserved. *)
val apply_motor : Motor.t -> t -> t
