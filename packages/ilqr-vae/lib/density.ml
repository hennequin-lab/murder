(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* One Gaussian log density, written once for every form the covariance is
   held in: a scalar variance, a diagonal, the Cholesky factor of the
   covariance, or the Cholesky factor of the precision. The last two are the
   two factors a caller can have — Σ = L Lᵀ as Covariance.factor builds, and
   P = UᵀU as Nx.cholesky builds — and each case takes the route its factor
   allows:

   Σ = L Lᵀ        quadratic = ‖Δ L⁻ᵀ‖²      one triangular solve, L lower
   Σ = (UᵀU)⁻¹     quadratic = ‖Δ Uᵀ‖²       one product, U upper

   A diagonal is the elementwise case of the first, with the factor already
   known, and a scalar is the same again with one scale; both skip the
   factor's O (d²) work entirely. Nothing is densified. *)

open Base
open Nx
open Types

type ('w, 's) covariance =
  | Isotropic of ('w, 's) value
  | Diagonal of { log_stds : ('w, 's) value }
  | Cholesky of { factor : ('w, 's) value }
  | Precision_cholesky of { factor : ('w, 's) value }

let log_2pi = Float.log (2.0 *. Float.pi)

(* A scalar is a one-dimensional distribution: the last axis of a rank-0
   tensor is the dimension it does not spell out. *)
let dim t =
  let shape = Nx.shape t in
  if Array.is_empty shape then 1 else shape.(Array.length shape - 1)

let sum_rows t = sum ~axes:[ -1 ] t
let numel t = Array.fold (Nx.shape t) ~init:1 ~f:( * )

(* [rows t] is the number of rows of [t]: every axis but the last, flattened. *)
let rows t = numel t / dim t

(* [as_rows diff ~f] applies [f] to [diff] read as a [rows; d] matrix and puts
   the result back in [diff]'s shape. A factor is [d; d] while [diff] may have
   any number of leading axes, and Nx's triangular solve wants the factor's and
   the right-hand side's ranks to match, so the rows are flattened for the
   solve. *)
let as_rows diff ~f =
  let shape = Nx.shape diff in
  reshape shape (f (reshape [| rows diff; dim diff |] diff))

let logp ?mean ~covariance x =
  (* The deviation is [mean − x]: its sign cannot reach any term below (each is
     a norm or a determinant), and taking the mean first keeps the subtraction
     in the order tolk's rewrite pass handles best on the graphs the encoder's
     scans build. *)
  let diff =
    match mean with
    | None -> x
    | Some mean -> Infix.(mean - x)
  in
  let open Infix in
  let normalizer = scalar_like x (Float.of_int (dim x) *. log_2pi) in
  let finish ~quadratic ~log_det = (quadratic + (log_det + normalizer)) *$ -0.5 in
  match covariance with
  | Isotropic variance ->
    finish
      ~quadratic:(sum_rows (diff * diff / variance))
      ~log_det:(log variance *$ Float.of_int (dim x))
  | Diagonal { log_stds } ->
    let scaled = diff / exp log_stds in
    finish ~quadratic:(sum_rows (scaled * scaled)) ~log_det:(sum_rows (log_stds *$ 2.0))
  | Cholesky { factor } ->
    let whitened =
      as_rows diff ~f:(fun flat ->
        matrix_transpose (solve_triangular ~triangle:Lower factor (matrix_transpose flat)))
    in
    finish
      ~quadratic:(sum_rows (whitened * whitened))
      ~log_det:(sum_rows (log (diag factor) *$ 2.0))
  | Precision_cholesky { factor } ->
    let whitened = diff *@ matrix_transpose factor in
    finish
      ~quadratic:(sum_rows (whitened * whitened))
      ~log_det:(sum_rows (log (diag factor) *$ -2.0))
