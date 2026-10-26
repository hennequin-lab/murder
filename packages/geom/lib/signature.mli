(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Metric signatures.

    A signature is the diagonal of the metric in an orthonormal basis: an array
    of length [n] whose entries are [+1], [-1] or [0]. The entry [i] is the
    square of the basis vector [e_(i+1)]:

    {[
     e_i e_j + e_j e_i = 2 g_(i j), quad g = "diag"(sigma_1, ..., sigma_n).
    ]}

    A code [sigma = +1] is a Euclidean direction, [-1] a negative (timelike)
    direction, and [0] a null (degenerate) direction. The number of each is the
    signature [(p, q, r)]. *)

(** {1:types Types} *)

(** The type for signatures. Invariant: every entry is [+1], [-1] or [0]. *)
type t

(** {1:constructors Constructors} *)

(** [make sigma] is the signature with diagonal [sigma].

    Raises [Invalid_argument] if an entry is not [-1], [0] or [1]. *)
val make : int array -> t

(** [euclidean n] is the signature [(n, 0, 0)]: [n] positive directions. *)
val euclidean : int -> t

(** [projective n] is the signature of [n]-dimensional projective geometric
    algebra, [n] positive directions followed by one null direction:
    [sigma = [|1; ...; 1; 0|]] with [n + 1] entries. *)
val projective : int -> t

(** [conformal n] is the signature of the conformal model of [RR^n]:
    [n + 1] positive directions followed by one negative direction,

    {[
     sigma = [|1; ...; 1; -1|]
    ]}

    with [n + 2] entries. For [n = 3] this is the [Cl(4, 1)] model of the
    notes. *)
val conformal : int -> t

(** [conformal3] is [conformal 3], the [Cl(4, 1)] signature
    [|[|1; 1; 1; 1; -1|]|]. *)
val conformal3 : t

(** {1:properties Properties} *)

(** [to_array sigma] is a copy of the diagonal of [sigma]. *)
val to_array : t -> int array

(** [dim sigma] is the ambient dimension [n], the length of the diagonal. *)
val dim : t -> int

(** [size sigma] is the number of basis blades, [2^n]. *)
val size : t -> int

(** [positives sigma] is the number of [+1] entries. *)
val positives : t -> int

(** [negatives sigma] is the number of [-1] entries. *)
val negatives : t -> int

(** [nulls sigma] is the number of [0] entries. *)
val nulls : t -> int

(** [is_degenerate sigma] is [true] iff at least one entry is [0]. *)
val is_degenerate : t -> bool

(** [pseudoscalar_square sigma] is the square of the pseudoscalar
    [I = e_1 ... e_n] in the algebra of [sigma]: [1] or [-1] for a
    non-degenerate signature, and [0] when the signature is degenerate.

    For a non-degenerate signature [(p, q, 0)],

    {[
     I^2 = (-1)^(n (n - 1) / 2) * (-1)^q.
    ]}

    In particular [I^2 = -1] for [Cl(4, 1)] and [I^2 = 0] for [Cl(3, 0, 1)]. *)
val pseudoscalar_square : t -> int

(** {1:comparing Comparing and printing} *)

(** [equal sigma sigma'] is [true] iff the diagonals are equal. *)
val equal : t -> t -> bool

(** [pp ppf sigma] formats [sigma] for debugging. *)
val pp : Format.formatter -> t -> unit

(** [to_string sigma] formats [sigma] as a signature, such as ["(3,0,0)"] or
    ["(4,1,0)"]. *)
val to_string : t -> string
