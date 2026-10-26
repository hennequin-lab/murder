(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The Euclidean algebra [Cl(3, 0, 0)].

    This is the rotation algebra of three-dimensional Euclidean space: vectors
    are grade-1 elements, bivectors are rotation planes, and the even
    subalgebra is the quaternions. It is the smallest interesting model for the
    generic {!Algebra} and a useful orientation-only layer on its own.

    The coefficients are those of the orthonormal basis [e_1, e_2, e_3]. *)

include Algebra.ALGEBRA with type kind = Nx.float32_elt

(** {1:basis Basis} *)

(** [e1] is the first basis vector. *)
val e1 : t

(** [e2] is the second basis vector. *)
val e2 : t

(** [e3] is the third basis vector. *)
val e3 : t

(** {1:vectors Vectors} *)

(** [vector v] embeds a Euclidean vector of shape [batch_shape @ [|3|]] as a
    grade-1 multivector. *)
val vector : Nx.float32_t -> t

(** [vector3 x y z] is [vector] of the single vector [(x, y, z)]. *)
val vector3 : float -> float -> float -> t

(** [to_vector a] extracts the grade-1 part of [a] as a tensor of shape
    [batch_shape @ [|3|]]. Higher-grade parts are ignored. *)
val to_vector : t -> Nx.float32_t

(** {1:rotors Rotors} *)

(** [rotor ~axis ~angle] is the unit even versor

    {[
     R = exp (-(angle / 2) B),
    ]}

    where [B] is the unit bivector of the rotation plane normal to [axis]. The
    axis has shape [batch_shape @ [|3|]] and is normalised internally; the
    angle has shape [batch_shape]. The sandwich [x -> R x R~] rotates a vector
    by [angle] about the axis, with the right-hand rule.

    Raises [Invalid_argument] if an axis is zero. *)
val rotor : axis:Nx.float32_t -> angle:Nx.float32_t -> t

(** [apply_rotor r x] is [r x tilde(r)], the action of the rotor [r] on the
    multivector [x]. It is {!Algebra.sandwich} with the inverse replaced by the
    reverse, since [r] is unit. *)
val apply_rotor : t -> t -> t
