(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Isotypic kinds of one permutation group, and their canonical test vectors.

   A single group acting on [k] occurrences of a tensor axis of dimension [n]
   decomposes [V^{⊗k}] ([V = ℂ^n] the permutation representation) into
   isotypic components

   [V^{⊗k} ≅ ⊕_λ S^{[n-|λ|, λ]} ⊗ M_λ]

   indexed by partitions of at most [k] (the kinds); the multiplicity
   [m_lambda] is the dimension of the multiplicity space, and the commutant
   algebra is the direct sum of the matrix algebras [M_m] over the kinds.

   The [k ≤ 2] cases used by the compiler:

   - k = 0: kinds [[]], multiplicities [1];
   - k = 1: kinds [[]] (trivial) and [[1]] (standard [W]), multiplicities
     [1] each;
   - k = 2: kinds [[]] (2), [[1]] (3), [[2]] (1) and [[1; 1]] (1), for [n]
     in the stable range (n >= 4); below it kinds drop out, which is exactly
     the non-semisimplicity of the partition algebra.

   For each kind and copy we give one canonical *test vector*: a vector of
   [V^{⊗k}] that is the image of a fixed internal vector under the copy's
   canonical embedding. Copies are the multiplicity-space basis vectors; the
   internal vector is the same abstract element on both sides of a compilation,
   so the block entries (inner products of a left test vector with the image
   of a right one) recover the operator's matrix on the multiplicity space, up
   to the left copies' Gram matrix. *)

open Base
open Nx

(* The partition tail of a kind: [] is the trivial [S^(n)], [1] the standard
   [S^(n-1,1)], [2] the symmetric [S^(n-2,2)] and [1;1] the antisymmetric
   [S^(n-2,1,1)]. *)
type t = int list

let compare = List.compare Int.compare
let equal = List.equal Int.equal

let to_string t =
  match t with
  | [] -> "t"
  | [ 1 ] -> "w"
  | [ 2 ] -> "s"
  | [ 1; 1 ] -> "a"
  | _ -> "?" ^ String.concat ~sep:"," (List.map t ~f:Int.to_string)

(* Every kind of [V^{⊗k}] for [k <= 2], in a canonical order: the kinds that
   survive at any dimension are listed, and the ones whose irrep vanishes at a
   given [n] are removed by {!kinds}. *)
let all ~k =
  match k with
  | 0 -> [ [] ]
  | 1 -> [ []; [ 1 ] ]
  | 2 -> [ []; [ 1 ]; [ 2 ]; [ 1; 1 ] ]
  | k ->
    invalid_arg
      (Printf.sprintf "Symo.Kind: %d permuted axes per group (at most 2 supported)" k)

(* The dimension of the irrep [S^(n-|λ|, λ)]. It is zero — the kind is then
   absent from [V^{⊗k}] — when the partition does not fit in [n], which for
   [k = 2] means [n < 4] for [λ = [2]] and [λ = [1; 1]], and for [k = 1]
   means [n = 1] for [λ = [1]]. The two-row values are the hook-length
   formula's [n(n-3)/2] and [(n-1)(n-2)/2]. *)
let dim ~n : t -> int = function
  | [] -> 1
  | [ 1 ] -> n - 1
  | [ 2 ] -> n * (n - 3) / 2
  | [ 1; 1 ] -> (n - 1) * (n - 2) / 2
  | _ -> 0

(* The kinds of [V^{⊗k}] at dimension [n]: {!all}, less the ones whose irrep
   vanishes there. Below the stable range a kind disappears rather than
   changing; the multiplicities of the survivors are {!multiplicity}'s. *)
let kinds ~k ~n = List.filter (all ~k) ~f:(fun kind -> dim ~n kind > 0)

(* The multiplicity of a kind in [V^{⊗k}], the dimension of its multiplicity
   space: the number of copies the block decomposition carries. These are the
   stable-range values, and they stay right at every dimension where {!check}
   holds ([1] for the trivial kind at [k = 1]; [2], [3], [1] and [1] at
   [k = 2]); [n = 2] with two permuted axes is where they do not, and it is
   refused. *)
let multiplicity ~k : t -> int = function
  | [] ->
    (match k with
     | 0 -> 1
     | 1 -> 1
     | 2 -> 2
     | _ -> assert false)
  | [ 1 ] ->
    (match k with
     | 1 -> 1
     | 2 -> 3
     | _ -> 0)
  | [ 2 ] ->
    (match k with
     | 2 -> 1
     | _ -> 0)
  | [ 1; 1 ] ->
    (match k with
     | 2 -> 1
     | _ -> 0)
  | _ -> 0

(* [check ~k ~n] accepts the dimensions at which the surviving kinds account
   for [V^{⊗k}], [Σ_λ dim S_λ · m_λ = n^k], and refuses the others: at
   [n = 2] with two permuted axes the survivors span 5 of the 4 dimensions of
   [V ⊗ V], because [V] is the sum of the trivial and the sign representation
   and extra relations appear. Every [n >= 3] passes for [k = 2], and every
   [n >= 1] for [k <= 1]. *)
let check ~k ~n =
  let total =
    List.fold (kinds ~k ~n) ~init:0 ~f:(fun acc kind ->
      acc + (dim ~n kind * multiplicity ~k kind))
  in
  let expected = Int.(n ** k) in
  if not (Int.equal total expected)
  then
    invalid_arg
      (Printf.sprintf
         "Symo.Kind: %d permuted axes of dimension %d: the isotypic kinds account for %d \
          of the %d dimensions of the tensor power, and the extra relations there have \
          no block decomposition"
         k
         n
         total
         expected)

(* ------------------------------------------------------------------------
   Canonical test vectors
   ------------------------------------------------------------------------ *)

let helmert n =
  (* [h_1 = (1, -1, 0, …, 0) / √2], a canonical unit vector of [1^⊥]. *)
  let v = Array.create ~len:n 0.0 in
  v.(0) <- 1.0 /. Float.sqrt 2.0;
  v.(1) <- -1.0 /. Float.sqrt 2.0;
  create float32 [| n |] v

let ones n = Nx.ones float32 [| n |]
let eye n = Nx.eye float32 n
let outer a b = Infix.(reshape [| -1; 1 |] a * reshape [| 1; -1 |] b)

(* A canonical vector of [S(n-2,2)]: a symmetric, zero-diagonal matrix whose
   row sums vanish (a 4-cycle), hence traceless and orthogonal to the diagonal
   [W] copy. Requires [n >= 4]. *)
let s_vector n =
  let a = Array.create ~len:(n * n) 0.0 in
  let set i j v = a.((i * n) + j) <- v in
  set 0 1 1.0;
  set 1 0 1.0;
  set 2 3 1.0;
  set 3 2 1.0;
  set 1 2 (-1.0);
  set 2 1 (-1.0);
  set 3 0 (-1.0);
  set 0 3 (-1.0);
  create float32 [| n; n |] a

let elementary n i j =
  let a = Array.create ~len:(n * n) 0.0 in
  a.((i * n) + j) <- 1.0;
  create float32 [| n; n |] a

(* Project an antisymmetric matrix onto [Λ²W]: remove the [1 ∧ x] component,
   which is [-(1/n)(1 yᵀ - y 1ᵀ)] with [y] the zero-sum part of [M 1]. *)
let project_antisymmetric m =
  let n = (shape m).(0) in
  let nn = Float.of_int n in
  let y = sum ~axes:[ 1 ] m in
  let y = Infix.(y - (ones n *$ (item [] (sum y) /. nn))) in
  let y = reshape [| n; 1 |] y in
  let one_row = reshape [| 1; n |] (ones n) in
  let w = Infix.(((one_row * transpose y) - (y * one_row)) *$ (-1. /. nn)) in
  Infix.(m - w)

let diagonal_part n x =
  (* the [W]-component of [Diag(x)]: [Diag(x) - (x 1ᵀ + 1 xᵀ)/n]. *)
  let nn = Float.of_int n in
  let diag = Nx.diag x in
  let x = reshape [| n; 1 |] x in
  let spread =
    Infix.(
      ((x * reshape [| 1; n |] (ones n)) + ((ones n |> reshape [| n; 1 |]) * transpose x))
      *$ (1. /. nn))
  in
  Infix.(diag - spread)

(* [test_vectors ~k ~n] is one canonical test vector per (kind, copy), in a
   canonical order, as a tensor of shape [[|n; …; n|]] ([k] axes). The copy
   index is the copy's position in the canonical order of its kind.

   Every vector is normalized to unit norm: the copy basis is then the same
   abstract basis at every size, so the block coordinates are comparable across
   sizes (the blocks are the operator's matrix in this basis). *)
let normalize v =
  let norm = Float.sqrt (item [] (sum (mul v v))) in
  Infix.(v *$ (1. /. norm))

let test_vectors ~k ~n =
  check ~k ~n;
  let alive kind = dim ~n kind > 0 in
  let vectors =
    match k with
    | 0 -> [ [], 0, scalar float32 1.0 ]
    | 1 ->
      let standard = if alive [ 1 ] then [ [ 1 ], 0, helmert n ] else [] in
      [ [], 0, ones n ] @ standard
    | 2 ->
      let x = helmert n in
      let spread = Infix.(Nx.ones float32 [| n; n |] *$ (1. /. Float.of_int n)) in
      let diagonal = Infix.(eye n - spread) in
      let row = outer x (ones n) in
      let column = outer (ones n) x in
      let diag_w = diagonal_part n x in
      let symmetric_traceless =
        (* needs a four-cycle: [n >= 4], which is exactly when the kind is
           alive *)
        if alive [ 2 ] then [ [ 2 ], 0, s_vector n ] else []
      in
      let antisymmetric =
        if alive [ 1; 1 ]
        then (
          let e = elementary n 0 1 in
          [ [ 1; 1 ], 0, project_antisymmetric Infix.(e - transpose e) ])
        else []
      in
      [ [], 0, spread; [], 1, diagonal ]
      @ [ [ 1 ], 0, row; [ 1 ], 1, column; [ 1 ], 2, diag_w ]
      @ symmetric_traceless
      @ antisymmetric
    | _ -> assert false
  in
  List.map vectors ~f:(fun (kind, copy, v) -> kind, copy, normalize v)
