(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Basis blades.

    A blade is an ordered product [e_i e_j ...] of distinct basis vectors. The
    basis vectors are numbered from [1] to [n]; the blade is encoded by the
    integer whose bit [i - 1] is set exactly when [e_i] occurs in it. The empty
    product is the scalar [1], encoded by [0].

    This module is independent of the metric: it knows the index sets, the
    grade, the order in which a product of two blades must be rearranged, and
    the complement of an index set. The metric factor of a product is handled by
    {!Algebra}; see {!val-mul_sign} for the sign that remains. *)

(** {1:types Types} *)

(** The type for blades. The integer is the bitmask of the basis vectors. *)
type t = private int

(** {1:constructors Constructors and accessors} *)

(** [of_int i] is the blade with bitmask [i].

    Raises [Invalid_argument] if [i] is negative. Bits above the ambient
    dimension are not rejected here; the {!Algebra} that owns the dimension
    checks the shape when it uses the blade. *)
val of_int : int -> t

(** [to_int b] is the bitmask of [b]. *)
val to_int : t -> int

(** [of_basis i] is the basis vector [e_i], that is, the blade with only bit
    [i - 1] set.

    Raises [Invalid_argument] if [i < 1]. *)
val of_basis : int -> t

(** [contains b i] is [true] iff [e_i] occurs in [b].

    Raises [Invalid_argument] if [i < 1]. *)
val contains : t -> int -> bool

(** [is_scalar b] is [true] iff [b] is the empty product, [1]. *)
val is_scalar : t -> bool

(** {1:properties Properties} *)

(** [grade b] is the number of basis vectors in [b], that is, the population
    count of its bitmask. *)
val grade : t -> int

(** [equal b b'] is [true] iff [b] and [b'] have the same bitmask. *)
val equal : t -> t -> bool

(** [compare b b'] is a total order on blades, compatible with {!equal}. *)
val compare : t -> t -> int

(** [hash b] is a hash of [b], compatible with {!equal}. *)
val hash : t -> int

(** [all n] is the array of the [2^n] blades of an [n]-dimensional algebra, in
    increasing bitmask order. *)
val all : int -> t array

(** {1:signs Signs}

    The sign functions below return [1] or [-1]. They are the signs of the
    involutions and of the reordering of a product in an orthonormal basis. *)

(** [reverse_sign b] is [(-1)^(k (k - 1) / 2)] for a blade of grade [k], the
    sign of the reverse of [b]. *)
val reverse_sign : t -> int

(** [grade_involution_sign b] is [(-1)^k] for a blade of grade [k]. *)
val grade_involution_sign : t -> int

(** [clifford_conjugation_sign b] is [(-1)^(k (k + 1) / 2)] for a blade of
    grade [k], the sign of the Clifford conjugate of [b]. *)
val clifford_conjugation_sign : t -> int

(** [mul_sign b c] is the sign of the permutation that sorts the concatenation
    of [b] and [c] into the blade [b xor c]. It is the sign contributed by the
    reordering alone; the metric factor of the common basis vectors is applied
    by the algebra. In particular,

    {[
    e_b e_c = mul_sign b c * metric factor * e_ (b xor c)
    ]}

    for an orthonormal basis. *)
val mul_sign : t -> t -> int

(** {1:complements Complements}

    The complement is the index-set complement used by the Poincaré dual of a
    degenerate algebra. It is a signed permutation of the coefficients; the
    signs are a convention fixed by {!complement_sign}. *)

(** [complement ~n b] is the blade whose index set is the complement of [b] in
    [{1, ..., n}]. The result has grade [n - grade b]. *)
val complement : n:int -> t -> t

(** [complement_sign ~n b] is the sign [s] such that

    {[
     e_b and e_(complement ~n b) = s * I
    ]}

    where [I = e_1 ... e_n] is the pseudoscalar. Equivalently,
    [s * e_(complement ~n b)] is the signed complement [b^c] for which
    [e_b and b^c = I]. The sign is [+1] for the scalar and the pseudoscalar. *)
val complement_sign : n:int -> t -> int

(** {1:printing Printing} *)

(** [pp ppf b] formats [b] for debugging. *)
val pp : Format.formatter -> t -> unit

(** [to_string b] is a human-readable name of [b], such as ["1"], ["e1"] or
    ["e12"], with the basis vectors in increasing order. *)
val to_string : t -> string
