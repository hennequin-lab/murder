(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dense Clifford algebras.

    An algebra is a signature together with a real scalar kind. A multivector is
    an {!Nx.t} whose {e last} axis is the blade axis, of length [2^n], and whose
    leading axes are ordinary batch axes. The coefficient at blade index [k] is
    the coefficient of {!Blade.to_int}[ k] in the basis blade of that index. A
    single multivector therefore has shape [[|2^n|]], a batch of [b]
    multivectors has shape [[|b; 2^n|]], and so on.

    The coefficient basis is the orthonormal basis of the signature. In the
    conformal model this is [e_1, ..., e_5], not the null basis: the null
    vectors [e_0] and [e_infinity] are derived multivectors. This keeps the
    bitmask product rule valid and sparse (see {!val-mul_sign}); the model
    module {!Cga} exposes the null vectors by name.

    The module is a functor over the signature and the scalar kind, so the
    structure constants are computed once per instantiation, at module
    initialisation. The functions are pure {!Nx} operations: no host read of a
    tensor value and no data-dependent OCaml branch, so they are compatible with
    [Rune.jit] and [Rune.grad]. *)

(** {1:parameters Parameters} *)

module type SCALAR = sig
  (** The buffer element kind of {!Nx}, such as {!Nx.float32_elt}. *)
  type kind

  (** The data type descriptor used to build constants. *)
  val dtype : (float, kind) Nx.dtype
end

module type SIGNATURE = sig
  (** The metric signature of the algebra. *)
  val signature : Signature.t
end

(** {1:algebras Algebras} *)

module type ALGEBRA = sig
  (** The scalar kind of the algebra. *)
  type kind

  (** The type for multivectors. The last axis is the blade axis. *)
  type t = (float, kind) Nx.t

  (** The data type descriptor of the scalar kind. *)
  val dtype : (float, kind) Nx.dtype

  (** The metric signature. *)
  val signature : Signature.t

  (** [n] is the ambient dimension, {!Signature.dim} of the signature. *)
  val n : int

  (** [size] is the number of blades, [2^n]. *)
  val size : int

  (** {1:construction Construction} *)

  (** [scalar x] is the unbatched scalar multivector with coefficient [x] on
      the identity blade. *)
  val scalar : float -> t

  (** [one] is [scalar 1.]. *)
  val one : t

  (** [zero] is [scalar 0.]. *)
  val zero : t

  (** [basis b] is the unbatched multivector with coefficient [1.] on blade
      [b] and zero elsewhere.

      Raises [Invalid_argument] if [Blade.to_int b >= size]. *)
  val basis : Blade.t -> t

  (** [e i] is the basis vector [e_i], that is, [basis (Blade.of_basis i)].

      Raises [Invalid_argument] if [i] is outside [[1];[n]]. *)
  val e : int -> t

  (** [pseudoscalar] is [e_1 ... e_n]. *)
  val pseudoscalar : t

  (** [of_array a] is the unbatched multivector with coefficients [a], in
      increasing blade-index order.

      Raises [Invalid_argument] if [Array.length a <> size]. *)
  val of_array : float array -> t

  (** [to_array a] is the coefficient array of an unbatched multivector, in
      increasing blade-index order.

      Raises [Invalid_argument] if [a] has batch axes. *)
  val to_array : t -> float array

  (** [of_nx x] views [x] as a multivector, checking that its last axis has
      length [size].

      Raises [Invalid_argument] if the last axis is not [size]. *)
  val of_nx : (float, kind) Nx.t -> t

  (** [to_nx a] is [a] as an ordinary tensor. It is the identity; the function
      exists to make the representation explicit at API boundaries. *)
  val to_nx : t -> (float, kind) Nx.t

  (** {1:shape Shape and coefficients} *)

  (** [shape a] is the shape of [a], including the trailing blade axis. *)
  val shape : t -> int array

  (** [batch_shape a] is the shape of [a] without the trailing blade axis. *)
  val batch_shape : t -> int array

  (** [coeff a b] is the coefficient of blade [b] in [a]: a tensor of shape
      [batch_shape a], without the blade axis. *)
  val coeff : t -> Blade.t -> (float, kind) Nx.t

  (** [with_coeff a b c] is [a] with the coefficient of [b] replaced by [c].
      [c] must have shape [batch_shape a], or broadcast to it.

      Raises [Invalid_argument] if the shapes are incompatible. *)
  val with_coeff : t -> Blade.t -> (float, kind) Nx.t -> t

  (** [grade k a] keeps the grade-[k] part of [a] and zeros the rest. For [k]
      outside [[0];[n]] the result is zero. *)
  val grade : int -> t -> t

  (** [even a] keeps the even-grade part of [a]. *)
  val even : t -> t

  (** [odd a] keeps the odd-grade part of [a]. *)
  val odd : t -> t

  (** {1:products Products} *)

  (** [gp a b] is the geometric product [a b]. Batch axes broadcast according
      to {!Nx}. *)
  val gp : t -> t -> t

  (** [op a b] is the outer (exterior) product [a and b]. *)
  val op : t -> t -> t

  (** [inner a b] is the left contraction [a dot b]. For vectors it is the
      ordinary dot product. *)
  val inner : t -> t -> t

  (** [scalar_prod a b] is the scalar product [<tilde(a) b>_0], a tensor of
      shape [batch_shape] after broadcasting. *)
  val scalar_prod : t -> t -> (float, kind) Nx.t

  (** [commutator a b] is [(a b - b a) / 2], the Lie bracket of bivectors. *)
  val commutator : t -> t -> t

  (** {1:involutions Involutions} *)

  (** [reverse a] reverses the order of every product in [a]. On a grade-[k]
      blade it acts as [(-1)^(k (k - 1) / 2)]. *)
  val reverse : t -> t

  (** [grade_involution a] multiplies the grade-[k] part of [a] by [(-1)^k]. *)
  val grade_involution : t -> t

  (** [conjugate a] is the Clifford conjugate of [a], the composition of
      {!reverse} and {!grade_involution}. *)
  val conjugate : t -> t

  (** {1:duals Duals} *)

  (** [dual a] is the right Hodge dual [a I^(-1)], of grade [n - k] when [a]
      has grade [k].

      Raises [Invalid_argument] if the signature is degenerate, in which case
      [I] is not invertible. For a degenerate algebra use {!complement}. *)
  val dual : t -> t

  (** [undual a] is [a I], the inverse of {!dual}:
      [undual (dual a) = a].

      Raises [Invalid_argument] if the signature is degenerate. *)
  val undual : t -> t

  (** [complement a] is the signed Poincaré complement: on a basis blade
      [e_b], it is [complement_sign ~n b * e_(complement ~n b)]. It is defined
      for every signature and is the dual used in projective geometric
      algebra. *)
  val complement : t -> t

  (** [regressive a b] is the regressive (meet) product of [a] and [b],
      [a or b], defined through {!complement}:

      {[
       a or b = complement (complement a and complement b)
      ]} *)
  val regressive : t -> t -> t

  (** {1:norms Norms and inverses} *)

  (** [norm2 a] is the scalar [<tilde(a) a>_0], of shape [batch_shape a]. For a
      blade it is the squared magnitude up to sign. *)
  val norm2 : t -> (float, kind) Nx.t

  (** [norm a] is [sqrt (abs (norm2 a))], the magnitude of a blade. *)
  val norm : t -> (float, kind) Nx.t

  (** [normalize a] is [a] divided by {!norm}[ a]. For a null blade the result
      is unspecified (division by zero); check {!norm2} first. *)
  val normalize : t -> t

  (** [inverse a] is the inverse of a blade or versor,
      [tilde(a) / <tilde(a) a>_0]. For a general multivector the inverse may
      not exist; the function does not detect this. *)
  val inverse : t -> t

  (** {1:versors Versors and actions} *)

  (** [sandwich v x] is [v x v^(-1)], the action of the versor [v] on [x]. The
      inverse is computed from [tilde(v) v]; for a unit versor it is
      [tilde(v)] up to sign. *)
  val sandwich : t -> t -> t

  (** [adjoint v] is the matrix of the linear map [x -> v x v^(-1)] in the blade
      basis. Its shape is [batch_shape v @ [|size; size|]]; for a single versor
      it is [size] by [size]. *)
  val adjoint : t -> (float, kind) Nx.t

  (** {1:printing Printing} *)

  (** [pp ppf a] formats [a] for debugging. For a batched value it prints the
      shape and the coefficients of each batch element. *)
  val pp : Format.formatter -> t -> unit
end

(** [Make (S) (K)] is the algebra of signature [S.signature] and scalar kind
    [K.kind]. The structure constants are computed at module initialisation. *)
module Make (S : SIGNATURE) (K : SCALAR) : ALGEBRA with type kind = K.kind
