(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The host-side estimator solve (Eq. 10 of the paper).

   [H] and [H_inv] are built from the two dense surrogates of the Taylor
   estimator with damped symmetric powers,

     sw12   = S_w^{1/2~},      swm12 = S_w^{-1/2~}
     inner  = sw12 S_g sw12
     H      = swm12 inner^{1/2~} swm12
     H_inv  = sw12  inner^{-1/2~} sw12

   with [X^{p~} = U diag((damping * s_max + s)^p) Uᵀ] for [X = U diag(s) Uᵀ].

   [Nx.svd] is the one operation [Rune.jit] refuses, so this module is the host
   boundary of the step: it runs on the small dense surrogate matrices, never
   inside the traced [prepare]/[finish] halves. *)

open Base

(* [svd64 x] is the economy SVD of [x]. The factorization runs in float64 (as
   the old Owl implementation did) and its factors come back in float32, the
   working dtype of the rest of the step. Singular values are float64 in [Nx]
   and are cast along with the rest. *)
let svd64 x =
  let u, s, _ = Nx.svd (Nx.cast Nx.float64 x) in
  Nx.cast Nx.float32 u, Nx.cast Nx.float32 s


(* [damp_spectrum ~damping s] is [damping * s_max + s]. The relative damping
   keeps the smallest directions of a singular surrogate finite. *)
let damp_spectrum ~damping s =
  let smax = Nx.item [] (Nx.max s) in
  Nx.add_s s (damping *. smax)

(* [symmetric_power ~damping ~power x] is the damped symmetric power of a
   symmetric matrix [x = U diag(s) Uᵀ]. A negative power of a zero singular
   value is taken to be zero (pseudo-inverse semantics): a zero surrogate
   carries no information, and its inverse contributes nothing rather than
   [inf]-times-zero. *)
let symmetric_power ~damping ~power x =
  let u, s = svd64 x in
  let stilde = damp_spectrum ~damping s in
  let z =
    if Float.(power >= 0.)
    then Nx.pow_s stilde power
    else
      Nx.where
        (Nx.greater stilde (Nx.scalar Nx.float32 0.))
        (Nx.pow_s stilde power)
        (Nx.zeros_like stilde)
  in
  Nx.matmul (Nx.mul u (Nx.reshape [| 1; -1 |] z)) (Nx.transpose u)

(* The estimator [H] itself. The optimizer only needs its inverse; [H] is kept
   for the tests and for debug dumps. *)
let hessian ~damping sigma_w sigma_g =
  let sw12 = symmetric_power ~damping ~power:0.5 sigma_w in
  let swm12 = symmetric_power ~damping ~power:(-0.5) sigma_w in
  let inner = Nx.matmul (Nx.matmul sw12 sigma_g) (Nx.transpose sw12) in
  let inner12 = symmetric_power ~damping ~power:0.5 inner in
  Nx.matmul (Nx.matmul swm12 inner12) (Nx.transpose swm12)

(* [hessian_inverse ~damping sigma_w sigma_g] is Eq. 10's preconditioner: the
   step is [theta - eta * H_inv g]. *)
let hessian_inverse ~damping sigma_w sigma_g =
  let sw12 = symmetric_power ~damping ~power:0.5 sigma_w in
  let inner = Nx.matmul (Nx.matmul sw12 sigma_g) (Nx.transpose sw12) in
  let inner_inv12 = symmetric_power ~damping ~power:(-0.5) inner in
  Nx.matmul (Nx.matmul sw12 inner_inv12) (Nx.transpose sw12)
