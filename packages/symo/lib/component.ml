(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Basis components of the invariant subspace (a single term, or a sum).

   A component is either a single term, or a sum of a term and its transpose —
   the symmetric case, where the estimator is required to be symmetric. The
   compiler works with lists of components; each carries one factor. *)

open Base

type t =
  | Single of Term.t
  | Sum of Term.t list
[@@deriving compare, equal, sexp]

let transpose = function
  | Single term -> Single (Term.transpose term)
  | Sum terms -> Sum (List.map terms ~f:Term.transpose)

let inner_product ~dims a b =
  let open Float in
  let dp = Term.inner_product ~dims in
  match a, b with
  | Single c1, Single c2 -> dp c1 c2
  | Single c1, Sum c2s -> List.fold c2s ~init:0. ~f:(fun accu c2 -> accu + dp c1 c2)
  | Sum c1s, Single c2 -> List.fold c1s ~init:0. ~f:(fun accu c1 -> accu + dp c1 c2)
  | Sum c1s, Sum c2s ->
    List.fold c1s ~init:0. ~f:(fun accu c1 ->
      List.fold c2s ~init:accu ~f:(fun accu' c2 -> accu' + dp c1 c2))

(* [coefficient ~dims comp data] is the component's coefficient tensor: the
   term's coefficient, or the sum of the terms' for a [Sum]. *)
let coefficient ~dims comp data =
  let coef = Term.coefficient ~dims in
  match comp with
  | Single c -> coef c data
  | Sum cs ->
    List.fold cs ~init:(Nx.scalar Nx.float32 0.0) ~f:(fun accu c ->
      Nx.add accu (coef c data))

(* The Gram matrix of the components, [B_ij = <c_i, c_j>]. Symmetric and
   positive semidefinite by construction, positive definite when the
   components are linearly independent. Computed once, at compile time. *)
let design_matrix ~dims components =
  let n = List.length components in
  let entries = Array.create ~len:(n * n) 0.0 in
  List.iteri components ~f:(fun i c1 ->
    List.iteri components ~f:(fun j c2 ->
      entries.((i * n) + j) <- inner_product ~dims c1 c2));
  Nx.create Nx.float32 [| n; n |] entries
