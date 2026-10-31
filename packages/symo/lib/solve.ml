(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The host-side estimator solve (Eq. 10 of the paper).

   The optimizer uses the blockwise entry point, {!hessian_inverse_blocks}: it
   applies the damped symmetric powers to each isotypic block of [S_w] and
   [S_g] (see {!Blocks}), which is exact at any size and small. The dense
   entry points below are the same formulas on a dense operator, kept for
   tests and diagnostics.

   [H] and [H_inv] are built from the two dense matrices of the Taylor
   estimator with damped symmetric powers,

     sw12   = S_w^{1/2~},      swm12 = S_w^{-1/2~}
     inner  = sw12 S_g sw12
     H      = swm12 inner^{1/2~} swm12
     H_inv  = sw12  inner^{-1/2~} sw12

   with [X^{p~} = U diag((damping * s_max + s)^p) Uᵀ] for [X = U diag(s) Uᵀ].

   The damping reads the spectrum's maximum with [Nx.item], which a traced
   value refuses, so this module is the host boundary of the step: it runs on
   the small block matrices, never inside the traced [prepare]/[finish]
   halves. *)

open Base
open Nx

(* [svd64 x] is the economy SVD of [x]. The factorization runs in float64 (as
   the old Owl implementation did) and its factors come back in float32, the
   working dtype of the rest of the step. The singular values are float64, as
   the call asks, and are cast along with the rest. *)
let svd64 x =
  let u, s, _ = svd float64 (cast float64 x) in
  cast float32 u, cast float32 s

(* [spectrum64 x] is [svd64 x] for a symmetric [x]. A symmetric matrix's
   singular values are its eigenvalues' magnitudes and its singular vectors are
   its eigenvectors, so [Nx.eigh] gives the factorization [Nx.svd] would,
   [x = u diag(±s) uᵀ], at about half the cost. *)
let spectrum64 x =
  let w, u = eigh float64 (cast float64 x) in
  cast float32 u, cast float32 (abs w)

(* [damp_spectrum ~damping s] is [damping * s_max + s]. The relative damping
   keeps the smallest directions of a singular operator finite. *)
let damp_spectrum ~damping s =
  let smax = item [] (max s) in
  Infix.(s +$ (damping *. smax))

(* [symmetric_power ~damping ~power x] is the damped symmetric power of a
   symmetric matrix [x = U diag(s) Uᵀ]. A negative power of a zero singular
   value is taken to be zero (pseudo-inverse semantics): a zero operator
   carries no information, and its inverse contributes nothing rather than
   [inf]-times-zero. *)
let symmetric_power ~damping ~power x =
  let u, s = spectrum64 x in
  let stilde = damp_spectrum ~damping s in
  let z =
    if Float.(power >= 0.)
    then pow_s stilde power
    else
      where (greater stilde (scalar float32 0.)) (pow_s stilde power) (zeros_like stilde)
  in
  Infix.(u * reshape [| 1; -1 |] z *@ transpose u)

(* The estimator [H] itself. The optimizer only needs its inverse; [H] is kept
   for the tests and for debug dumps. *)
let hessian ~damping sigma_w sigma_g =
  let sw12 = symmetric_power ~damping ~power:0.5 sigma_w in
  let swm12 = symmetric_power ~damping ~power:(-0.5) sigma_w in
  let inner = Infix.(sw12 *@ sigma_g *@ transpose sw12) in
  let inner12 = symmetric_power ~damping ~power:0.5 inner in
  Infix.(swm12 *@ inner12 *@ transpose swm12)

(* [hessian_inverse ~damping sigma_w sigma_g] is Eq. 10's preconditioner: the
   step is [theta - eta * H_inv g]. *)
let hessian_inverse ~damping sigma_w sigma_g =
  let sw12 = symmetric_power ~damping ~power:0.5 sigma_w in
  let inner = Infix.(sw12 *@ sigma_g *@ transpose sw12) in
  let inner_inv12 = symmetric_power ~damping ~power:(-0.5) inner in
  Infix.(sw12 *@ inner_inv12 *@ transpose sw12)

(* ------------------------------------------------------------------------
   Blockwise
   ------------------------------------------------------------------------ *)

(* The largest singular value over a list of blocks. *)
let global_smax blocks =
  List.fold blocks ~init:0.0 ~f:(fun acc x ->
    let w, _ = eigh float64 (cast float64 x) in
    Float.max acc (item [] (max (abs w))))

(* The damped symmetric power of every block, with one global [s_max]. *)
let symmetric_power_blocks ~damping ~smax ~power blocks =
  List.map blocks ~f:(fun x ->
    let w, u = eigh float64 (cast float64 x) in
    let u = cast float32 u in
    let w = cast float32 (abs w) in
    let stilde = Infix.(w +$ (damping *. smax)) in
    let z =
      if Float.(power >= 0.)
      then pow_s stilde power
      else
        where
          (greater stilde (scalar float32 0.))
          (pow_s stilde power)
          (zeros_like stilde)
    in
    Infix.(u * reshape [| 1; -1 |] z *@ transpose u))

(* [hessian_inverse_blocks ~damping sigma_w sigma_g] is Eq. 10's
   preconditioner, blockwise: the blocks are damped and inverted kind by kind.
   Each damped power uses its own matrix's largest singular value (the maximum
   over the blocks), as the dense formula does. *)
let hessian_inverse_blocks ~damping sigma_w sigma_g =
  let sw12 =
    symmetric_power_blocks ~damping ~smax:(global_smax sigma_w) ~power:0.5 sigma_w
  in
  let inner = List.map2_exn sw12 sigma_g ~f:(fun a b -> Infix.(a *@ b *@ transpose a)) in
  let inner_inv12 =
    symmetric_power_blocks ~damping ~smax:(global_smax inner) ~power:(-0.5) inner
  in
  List.map2_exn sw12 inner_inv12 ~f:(fun a b -> Infix.(a *@ b *@ transpose a))
