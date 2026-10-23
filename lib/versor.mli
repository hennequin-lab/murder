(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Exponential and logarithm of versors.

    A versor is an invertible element of the algebra, usually a product of
    vectors, that acts by the sandwich [x -> v x v^(-1)]. The exponential of a
    bivector is a versor; the logarithm of a versor is a bivector (or, for a
    general element, a multivector in the Lie algebra).

    This module is a functor over an algebra so that the same exponential code
    serves {!E3}, {!Pga} and {!Cga}. The primary robotics path is the packed
    {!Motor} type, which implements the [SE(3)] exponential directly; this
    module is the dense-algebra counterpart.

    {b Status.} The first implementation provides the closed form for a simple
    bivector ({!exp_simple}) and the [SE(3)] exponential through {!Motor};
    {!exp} and {!log} for a general bivector are staged behind this interface.
    Until the general routine lands they raise [Invalid_argument] for inputs
    outside the supported subset. The signatures will not change. *)

(** {1:versors Versors} *)

module type VERSOR = sig
  (** The scalar kind of the underlying algebra. *)
  type kind

  (** The type for multivectors of the underlying algebra. *)
  type t

  (** [exp b] is the exponential of the bivector [b], the versor
      [sum_(k >= 0) b^k / k!]. For a simple bivector this is the closed form of
      the notes, §1.6; for the [SE(3)] subalgebra it is the motor exponential.

      Raises [Invalid_argument] if [b] is outside the subset supported by the
      current implementation. *)
  val exp : t -> t

  (** [log v] is a bivector whose exponential is the versor [v], with the
      branch chosen near the identity. The current implementation supports a
      versor whose non-scalar part is a single simple bivector (a rotation,
      boost or null rotation); for a general even versor with several
      independent bivector components the result is the simple approximation
      and need not invert {!exp}. General even-versor logarithms are a later
      refinement.

      Raises [Invalid_argument] if [v] is not a versor or is outside the
      supported subset. *)
  val log : t -> t

  (** [exp_simple b] is the exponential of a simple bivector, using the closed
      form

      {[
        exp b =
          cos theta + (sin theta / theta) b,   if b^2 = -theta^2 < 0,
          cosh theta + (sinh theta / theta) b, if b^2 = +theta^2 > 0,
          1 + b,                               if b^2 = 0,
      ]}

      with the limiting value at [theta = 0]. It is always defined for a
      simple bivector, including the null generators of translations,
      transversions and shears. *)
  val exp_simple : t -> t

  (** [slerp v0 v1 alpha] interpolates between the versors [v0] and [v1] along
      the group exponential:

      {[
       slerp v0 v1 alpha = exp (alpha * log (v0^(-1) v1)) v0.
      ]}

      [alpha] is a tensor of shape [batch_shape] with values in [[0];[1]].
      [alpha = 0] gives [v0] and [alpha = 1] gives [v1]. The interpolation is
      the screw linear interpolation (ScLERP) of rigid motions when the versors
      are motors. *)
  val slerp : t -> t -> (float, kind) Nx.t -> t
end

(** [Make A] is the exponential and logarithm layer of the algebra [A]. *)
module Make (A : Algebra.ALGEBRA) : VERSOR with type t = A.t and type kind = A.kind
