(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Curvature descriptions: the Hessian of a mini loss as a linear map on its
   prediction.

   A little loss l = ℓ(y) contributes two things to a sketch: its value and
   tangent, and the block YᵀHY of the generalized Gauss-Newton matrix, where
   H = ∂²ℓ/∂y² and Y is the tangent batch of y. H is only ever used through its
   action on tangent directions, so a description is exactly that action: a
   linear map on tensors of the prediction's shape, applied to the whole
   [k]-lane batch at once — the lane axis is one more leading axis for every
   structured case, so the application needs no per-lane loop.

   A description is kept as a small variant rather than an opaque closure, so
   that the one dtype adaptation every case needs — a tangent batch whose
   element type may differ from the prediction's, since a loss can be evaluated
   in a wider type than the model — happens in exactly one place, [apply],
   instead of at every construction site. Nothing here derives a Hessian by
   hand: the suite checks every constructor against [Rune.hessian'] of the
   corresponding mini loss as a function of a free [y]. *)

(* A tensor whose dtype is hidden; the describing constructors accept tensors
   at any element type and [apply] casts them to the tangent's. *)
type packed = Packed : ('a, 'b) Nx.t -> packed

type t =
  | Scale of float
  | Diag : ('a, 'b) Nx.t -> t
  | Softmax_ce : float * packed -> t
  | Hvp : (('a, 'b) Nx.t -> ('a, 'b) Nx.t) * ('a, 'b) Nx_core.Dtype.t -> t

(* [apply t v] is H v for a tangent batch [v] of the prediction's shape: one
   leading lane axis, any number of trailing dimensions. *)
let apply : type c d. t -> (c, d) Nx.t -> (c, d) Nx.t =
  fun t v ->
  match t with
  | Scale s -> Nx.mul_s v (Nx_core.Dtype.of_float (Nx.dtype v) s)
  | Diag w -> Nx.mul v (Nx.cast (Nx.dtype v) w)
  | Softmax_ce (scale, Packed p) ->
    (* p⊙v − p (p·v), contracted per row over the class axis. *)
    let p = Nx.cast (Nx.dtype v) p in
    let pv = Nx.sum (Nx.mul p v) ~axes:[ -1 ] ~keepdims:true in
    let hv = Nx.sub (Nx.mul p v) (Nx.mul p pv) in
    if scale = 1.0 then hv else Nx.mul_s hv (Nx_core.Dtype.of_float (Nx.dtype hv) scale)
  | Hvp (f, dt) ->
    (match Nx_core.Dtype.equal_witness dt (Nx.dtype v) with
     | Some Type.Equal ->
       (* The usual case: the closure is written at this very dtype. *)
       Rune.vmap' f v
     | None -> Nx.cast (Nx.dtype v) (Rune.vmap' f (Nx.cast dt v)))

(* H = s·I: the curvature of a mean-squared loss, and of any loss whose second
   derivative in [y] is constant. The scalar is produced in the argument's own
   element type, so nothing is promoted or converted. *)
let scale s = Scale s

(* H = diag w: the curvature of a weighted mean-squared loss, or of a negative
   log-likelihood in a natural parameterisation. [w] is cast to the prediction's
   dtype when applied, so a float64 weight can describe a float32 prediction. *)
let diag w = Diag w

(* The softmax cross-entropy curvature, for [p = softmax y] over the last axis:
   H v = p⊙v − p (p·v), contracted per row, times [scale] (a loss averaged over
   rows contributes 1/rows). [p] is the probability tensor of the little loss,
   so the value and the curvature describing it cannot disagree. *)
let softmax_ce ?(scale = 1.0) p = Softmax_ce (scale, Packed p)

(* An escape hatch for curvatures with no structured form: [f] is the Hessian
   action on a single direction of the prediction's shape, and the library lifts
   it over the lane axis — one vectorized translation of the closure rather than
   a loop over lanes. The closure must be linear in its argument and
   side-effect free. [y] is the prediction whose loss this describes; it fixes
   the element type [f] is written at, so captured tensors need no casting. *)
let hvp ~(y : ('a, 'b) Nx.t) (f : ('a, 'b) Nx.t -> ('a, 'b) Nx.t) = Hvp (f, Nx.dtype y)
