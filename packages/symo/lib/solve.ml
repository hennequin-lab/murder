(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The host-side estimator solves.

   The optimizer's preconditioner is the root-damped inverse of the gradient's
   second moment,

     P = (S_g^{1/2} + damping · I)^{-1}

   applied blockwise: one small eigendecomposition per isotypic kind, with the
   same absolute [damping] in every block. The blocks are the operator's
   matrices in orthonormal bases of the isotypic components (see {!Blocks}), so
   the shift is [damping · I] on the whole parameter space — one global,
   absolute damping, not a scale taken from each block's spectrum. The root and
   the shift put [damping] where Adam's [epsilon] sits, so it is in gradient
   units, and they cap the largest eigenvalue of [P] at [1/damping] at every
   size: the step is bounded and increasing [damping] shrinks it monotonically.
   The blockwise form reads no value with [Nx.item] and would trace; it is kept
   host-side because the eigendecompositions run in float64.

   The Taylor estimator of Eq. 10 is kept below with *relative* damping,
   [damping · s_max + s]: it is the paper's estimator and the dense reference
   of the tests, but the optimizer no longer uses it. Its damping reads the
   spectrum's maximum with [Nx.item], which a traced value refuses, so it is
   host-only.

   [H] and [H_inv] are built from the two dense matrices of the Taylor
   estimator,

     sw12   = S_w^{1/2~},      swm12 = S_w^{-1/2~}
     inner  = sw12 S_g sw12
     H      = swm12 inner^{1/2~} swm12
     H_inv  = sw12  inner^{-1/2~} sw12

   with [X^{p~} = U diag((damping * s_max + s)^p) Uᵀ] for [X = U diag(s) Uᵀ]. *)

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

(* ------------------------------------------------------------------------
   Absolute damping: the optimizer's preconditioner
   ------------------------------------------------------------------------ *)

(* [damped_inverse_root ~damping x] is
   [((proj x)^{1/2} + damping · I)^{-1}], where [proj x] is [x] with its
   negative eigenvalues set to zero. [x] is a second moment, positive
   semidefinite by construction, so the projection only removes roundoff; a
   zero eigenvalue becomes [damping], and the inverse has no pseudo-inverse
   branch. [damping] is absolute and sits where Adam's [epsilon] sits, at the
   root: it is not scaled by the spectrum, it caps the result's largest
   eigenvalue at [1/damping] whatever the operator's scale, and raising
   [damping] shrinks every direction. The root and the shift are taken in
   float64 ({!eigh} runs there); only the shifted, strictly positive
   denominators are cast. *)
let damped_inverse_root ~damping x =
  if Float.(damping <= 0.)
  then
    invalid_arg
      "Symo.Solve.damped_inverse_root: the damping must be positive (it shifts the root \
       spectrum, and is also the gain cap)";
  let w, u = eigh float64 (cast float64 x) in
  let d = Infix.(sqrt (maximum w (zeros_like w)) +$ damping) in
  let z = cast float32 Infix.(ones_like d / d) in
  let u = cast float32 u in
  Infix.(u * reshape [| 1; -1 |] z *@ transpose u)

(* The same per isotypic block: the blocks are an operator's matrices in
   orthonormal bases of the isotypic components, so adding the same [damping]
   to each root spectrum is adding [damping · I] on the whole parameter space. *)
let damped_inverse_root_blocks ~damping blocks =
  List.map blocks ~f:(damped_inverse_root ~damping)

(* ------------------------------------------------------------------------
   Relative damping: the Taylor estimator of Eq. 10
   ------------------------------------------------------------------------ *)

(* [damp_spectrum ~relative_damping s] is [relative_damping * s_max + s]. The
   relative damping keeps the smallest directions of a singular operator
   finite, at a scale read from the spectrum's own maximum. *)
let damp_spectrum ~relative_damping s =
  let smax = item [] (max s) in
  Infix.(s +$ (relative_damping *. smax))

(* [symmetric_power ~relative_damping ~power x] is the damped symmetric power
   of a symmetric matrix [x = U diag(s) Uᵀ]. A negative power of a zero
   singular value is taken to be zero (pseudo-inverse semantics): a zero
   operator carries no information, and its inverse contributes nothing
   rather than [inf]-times-zero. *)
let symmetric_power ~relative_damping ~power x =
  let u, s = spectrum64 x in
  let stilde = damp_spectrum ~relative_damping s in
  let z =
    if Float.(power >= 0.)
    then pow_s stilde power
    else
      where (greater stilde (scalar float32 0.)) (pow_s stilde power) (zeros_like stilde)
  in
  Infix.(u * reshape [| 1; -1 |] z *@ transpose u)

(* The estimator [H] itself. The optimizer only needs its inverse; [H] is kept
   for the tests and for debug dumps. *)
let hessian ~relative_damping sigma_w sigma_g =
  let sw12 = symmetric_power ~relative_damping ~power:0.5 sigma_w in
  let swm12 = symmetric_power ~relative_damping ~power:(-0.5) sigma_w in
  let inner = Infix.(sw12 *@ sigma_g *@ transpose sw12) in
  let inner12 = symmetric_power ~relative_damping ~power:0.5 inner in
  Infix.(swm12 *@ inner12 *@ transpose swm12)

(* [hessian_inverse ~relative_damping sigma_w sigma_g] is Eq. 10's
   preconditioner: the step is [theta - eta * H_inv g]. *)
let hessian_inverse ~relative_damping sigma_w sigma_g =
  let sw12 = symmetric_power ~relative_damping ~power:0.5 sigma_w in
  let inner = Infix.(sw12 *@ sigma_g *@ transpose sw12) in
  let inner_inv12 = symmetric_power ~relative_damping ~power:(-0.5) inner in
  Infix.(sw12 *@ inner_inv12 *@ transpose sw12)

(* ------------------------------------------------------------------------
   Blockwise
   ------------------------------------------------------------------------ *)

(* The largest singular value over a list of blocks. *)
let global_smax blocks =
  List.fold blocks ~init:0.0 ~f:(fun acc x ->
    let w, _ = eigh float64 (cast float64 x) in
    Float.max acc (item [] (max (abs w))))

(* The damped symmetric power of every block, with one [s_max] over the blocks
   and the same relative damping in each. *)
let symmetric_power_blocks ~relative_damping ~smax ~power blocks =
  List.map blocks ~f:(fun x ->
    let w, u = eigh float64 (cast float64 x) in
    let u = cast float32 u in
    let w = cast float32 (abs w) in
    let stilde = Infix.(w +$ (relative_damping *. smax)) in
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

(* [hessian_inverse_blocks ~relative_damping sigma_w sigma_g] is Eq. 10's
   preconditioner, blockwise: the blocks are damped and inverted kind by kind.
   Each damped power uses its own matrix's largest singular value (the maximum
   over the blocks), as the dense formula does. *)
let hessian_inverse_blocks ~relative_damping sigma_w sigma_g =
  let sw12 =
    symmetric_power_blocks
      ~relative_damping
      ~smax:(global_smax sigma_w)
      ~power:0.5
      sigma_w
  in
  let inner = List.map2_exn sw12 sigma_g ~f:(fun a b -> Infix.(a *@ b *@ transpose a)) in
  let inner_inv12 =
    symmetric_power_blocks ~relative_damping ~smax:(global_smax inner) ~power:(-0.5) inner
  in
  List.map2_exn sw12 inner_inv12 ~f:(fun a b -> Infix.(a *@ b *@ transpose a))
