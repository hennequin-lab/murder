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

(* The kinds of [V^{⊗k}] with the multiplicities of the stable range. *)
let stable_kinds ~k =
  match k with
  | 0 -> [ [] ]
  | 1 -> [ []; [ 1 ] ]
  | 2 -> [ []; [ 1 ]; [ 2 ]; [ 1; 1 ] ]
  | k ->
    invalid_arg
      (Printf.sprintf "Symo.Kind: %d permuted axes per group (at most 2 supported)" k)

let stable_multiplicity ~k : t -> int = function
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

(* The smallest dimension for which the stable multiplicities hold: the
   partition algebra is semisimple for [n ≥ 2k]. *)
let stable_dim ~k = 2 * k

let check_stable ~k ~n =
  if n < stable_dim ~k
  then
    invalid_arg
      (Printf.sprintf
         "Symo.Kind: %d permuted axes need dimension >= %d for a faithful block \
          decomposition (got %d)"
         k
         (stable_dim ~k)
         n)

(* ------------------------------------------------------------------------
   Canonical test vectors
   ------------------------------------------------------------------------ *)

let helmert n =
  (* [h_1 = (1, -1, 0, …, 0) / √2], a canonical unit vector of [1^⊥]. *)
  let v = Array.create ~len:n 0.0 in
  v.(0) <- 1.0 /. Float.sqrt 2.0;
  v.(1) <- -1.0 /. Float.sqrt 2.0;
  Nx.create Nx.float32 [| n |] v

let ones n = Nx.ones Nx.float32 [| n |]
let eye n = Nx.eye Nx.float32 n
let outer a b = Nx.mul (Nx.reshape [| -1; 1 |] a) (Nx.reshape [| 1; -1 |] b)

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
  Nx.create Nx.float32 [| n; n |] a

let elementary n i j =
  let a = Array.create ~len:(n * n) 0.0 in
  a.((i * n) + j) <- 1.0;
  Nx.create Nx.float32 [| n; n |] a

(* Project an antisymmetric matrix onto [Λ²W]: remove the [1 ∧ x] component,
   which is [-(1/n)(1 yᵀ - y 1ᵀ)] with [y] the zero-sum part of [M 1]. *)
let project_antisymmetric m =
  let n = (Nx.shape m).(0) in
  let nn = Float.of_int n in
  let y = Nx.sum ~axes:[ 1 ] m in
  let y = Nx.sub y (Nx.mul_s (ones n) (Nx.item [] (Nx.sum y) /. nn)) in
  let y = Nx.reshape [| n; 1 |] y in
  let one_row = Nx.reshape [| 1; n |] (ones n) in
  let w =
    Nx.mul_s (Nx.sub (Nx.mul one_row (Nx.transpose y)) (Nx.mul y one_row)) (-1. /. nn)
  in
  Nx.sub m w

let diagonal_part n x =
  (* the [W]-component of [Diag(x)]: [Diag(x) - (x 1ᵀ + 1 xᵀ)/n]. *)
  let nn = Float.of_int n in
  let diag = Nx.diag x in
  let x = Nx.reshape [| n; 1 |] x in
  let spread =
    Nx.mul_s
      (Nx.add
         (Nx.mul x (Nx.reshape [| 1; n |] (ones n)))
         (Nx.mul (ones n |> Nx.reshape [| n; 1 |]) (Nx.transpose x)))
      (1. /. nn)
  in
  Nx.sub diag spread

(* [test_vectors ~k ~n] is one canonical test vector per (kind, copy), in a
   canonical order, as a tensor of shape [[|n; …; n|]] ([k] axes). The copy
   index is the copy's position in the canonical order of its kind.

   Every vector is normalized to unit norm: the copy basis is then the same
   abstract basis at every size, so the block coordinates are comparable across
   sizes (the blocks are the operator's matrix in this basis). *)
let normalize v =
  let norm = Float.sqrt (Nx.item [] (Nx.sum (Nx.mul v v))) in
  Nx.mul_s v (1. /. norm)

let test_vectors ~k ~n =
  check_stable ~k ~n;
  let vectors =
    match k with
    | 0 -> [ [], 0, Nx.scalar Nx.float32 1.0 ]
    | 1 ->
      let x = helmert n in
      [ [], 0, ones n; [ 1 ], 0, x ]
    | 2 ->
      let x = helmert n in
      let spread = Nx.mul_s (Nx.ones Nx.float32 [| n; n |]) (1. /. Float.of_int n) in
      let diagonal = Nx.sub (eye n) spread in
      let row = outer x (ones n) in
      let column = outer (ones n) x in
      let diag_w = diagonal_part n x in
      let e = elementary n 0 1 in
      let s = s_vector n in
      let a = project_antisymmetric (Nx.sub e (Nx.transpose e)) in
      [ [], 0, spread
      ; [], 1, diagonal
      ; [ 1 ], 0, row
      ; [ 1 ], 1, column
      ; [ 1 ], 2, diag_w
      ; [ 2 ], 0, s
      ; [ 1; 1 ], 0, a
      ]
    | _ -> assert false
  in
  List.map vectors ~f:(fun (kind, copy, v) -> kind, copy, normalize v)
