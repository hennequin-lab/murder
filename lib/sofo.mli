(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sketched second-order training.

    A {e sketch} measures a model's loss and curvature along a random
    [k]-dimensional subspace of parameter space instead of in all of it. The
    subspace is the span of [k] sampled directions Θ, held as a
    parameter-shaped structure whose every leaf stacks its [k] directions on a
    leading axis; a sketch reports the loss, the gradient contracted with Θ
    (C = Θᵀ∇c), the generalized Gauss-Newton matrix contracted on both sides
    (ΘᵀJᵀHJΘ, a [k×k] matrix), and [apply], which turns a [k]-vector back into
    a parameter-space direction. An update rule then solves a [k]-dimensional
    problem and steps along its result — the cheap part, and not this library's
    business.

    The gradient and the curvature are not obtained by differentiating a
    sketch: the loss is differentiated once, in batched forward mode along all
    [k] directions at once, and the curvature is accumulated from little losses
    the user marks with {!observe} or produces with the packaged losses
    ({!mse}, {!softmax_ce}). So a sketch costs one forward pass plus one small
    [k×k] accumulation per marked little loss.

    Three modes, one loss function — swap the enclosing driver, not the model:

    - {!sketch}: loss, C and the sketched GGN (curvature observations are
      read);
    - [Rune.jvp_k] alone: loss and C (observations are inert);
    - [Rune.value_and_grad]: loss and the exact gradient (observations are
      inert).

    Observations are inert by construction: {!observe} performs an effect that
    only a sketch collector intercepts, and any other handler lets it pass.
    Marking a little loss therefore costs nothing outside a sketch.

    {[
    let loss params =
      let y = predict params x in
      Nx.add (Sofo.mse y target) (Nx.sum (Nx.mul w w))
    in
    let sk = Sofo.sketch (module Params) ~k:32 loss params in
    (* sk.c, sk.ggn : the sketched gradient and GGN; sk.apply : sketch → params *)
    ]} *)

module Curv : sig
  (** Curvature descriptions: the Hessian of a little loss as a linear map on
      its prediction.

      A little loss l = ℓ(y) contributes the block YᵀHY to the sketched GGN,
      with H = ∂²ℓ/∂y² and Y the tangent batch of y. Descriptions therefore
      only ever need to {e apply} H to a batch of directions, which every
      constructor does in one batched pass over the lanes.

      Every constructor's linearity is what makes the batched application
      sound, and each is checked in the test suite against [Rune.hessian'] of
      the corresponding loss rather than against a hand-derived formula. *)

  type t

  (** [apply c v] is H v: the curvature applied to a tangent batch [v] of the
      prediction's shape, whose leading axis holds the sketch's lanes. The
      application is linear in [v] and keeps the lane axis, so it serves a
      whole batch in one pass. *)
  val apply : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t

  (** [scale s] is H = s·I: the curvature of a mean-squared loss, and of any
      loss whose second derivative in [y] is constant. *)
  val scale : float -> t

  (** [diag w] is H = diag w: the curvature of a weighted mean-squared loss, or
      of any negative log-likelihood in a natural parameterisation whose
      Hessian is diagonal. [w] is cast to the prediction's dtype. *)
  val diag : ('a, 'b) Nx.t -> t

  (** [softmax_ce ~scale p] is the cross-entropy curvature, for [p = softmax y]
      over the last axis: H v = p⊙v − p (p·v), contracted per row, times
      [scale] (a loss averaged over rows contributes [1/rows]). [p] is the
      probability tensor of the little loss, so the value and the curvature it
      describes cannot disagree. *)
  val softmax_ce : ?scale:float -> ('a, 'b) Nx.t -> t

  (** [hvp ~y f] describes an arbitrary curvature through its action [f] on a
      single direction of the prediction's shape; the library lifts it over the
      directions' lane axis. [f] must be linear in its argument and
      side-effect free, and [y] fixes the element type [f] is written at, so
      captured tensors need no casting. This is the escape hatch for losses
      with no structured form, and it costs one vectorized translation of [f]
      per observation. *)
  val hvp : y:('a, 'b) Nx.t -> (('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> t
end

(** [observe ~y ~curv l] marks the scalar [l] as a little loss in the
    prediction [y], whose Hessian with respect to [y] is [curv]. A sketch
    accumulating curvature reads [l]'s tangent and [y]'s tangent batch and adds
    the block YᵀHY; any other handler lets the mark pass, so calling this is a
    no-op outside a sketch.

    [l] must be a scalar computed from [y] by differentiable operations — the
    user's own arithmetic defines both the value and its tangent. A little loss
    that depends on several predictions may be observed once per prediction
    (the blocks add) or jointly, by observing a concatenation of them with a
    matching curvature. *)
val observe : y:('a, 'b) Nx.t -> curv:Curv.t -> ('a, 'b) Nx.t -> unit

(** [mse ?w y target] is the mean weighted squared error
    [mean (w * (y - target)²)], observed with its exact curvature
    (H = diag (2 w / numel y), or [2 / numel y] times the identity without
    weights), and returned as a scalar to accumulate into the loss. *)
val mse : ?w:('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t

(** [softmax_ce y labels] is the cross entropy of the last axis of [y] against
    class indices [labels] — which have [y]'s shape without its last axis —
    averaged over the leading axes, observed with its exact curvature
    ([diag p − p pᵀ] per row, divided by the number of rows) and returned as a
    scalar. Labels must lie in [\[0, num_classes)]; the checked cases are the
    user's to validate up front. *)
val softmax_ce : ('a, 'b) Nx.t -> Nx.int32_t -> ('a, 'b) Nx.t

(** What the collector observed, for diagnosis: {!check} compares the first two
    with the sketch's own loss and tangent. *)
type diagnostics =
  { observed_loss : Nx.float64_t
    (** Σ l over the observed little losses, as a float64 scalar. *)
  ; observed_c : Nx.float64_t
    (** Σ ċ over the observed little losses, shape [k]: the tangent the
        collector saw where it counts. *)
  ; blocks : int (** Observations that contributed a curvature block. *)
  ; skipped : int
    (** Observations whose prediction was a constant of the differentiation.
        Y = 0 makes their block zero, so they are skipped rather than
        accumulated as zeros. *)
  }

(** A sketch's results. *)
type 'p sketch =
  { k : int (** The number of directions the sketch spans. *)
  ; loss : Nx.float64_t (** The primal total loss, as a float64 scalar. *)
  ; c : Nx.float64_t
    (** C = Θᵀ∇c, shape [k]: the gradient sketched onto the sampled
        directions. *)
  ; ggn : Nx.float64_t
    (** The sketched generalized Gauss-Newton matrix ΘᵀJᵀHJΘ, shape [k;k].
        Blocks are accumulated in float64 and the result symmetrized. *)
  ; apply : Nx.float64_t -> 'p
    (** [apply z] is the parameter-space direction Θz for [z : [k]], each
        leaf in its parameter's dtype. *)
  ; diagnostics : diagnostics
  }

(** [sketch (module P) ~k loss params] sketches [loss] at [params] along [k]
    sampled directions.

    [loss] must return a scalar — it is the whole objective, including any
    terms the user accumulates but does not observe. Little losses are marked
    with {!observe} or produced by {!mse}/{!softmax_ce}, and every term that
    contributes to the loss should be marked, or the sketch's curvature will
    only cover the marked ones ({!check} reports the discrepancy).

    Directions are sampled per leaf as standard normal [k]-lane batches from
    the ambient [Rng] scope, so wrapping the call in [Rune.Rng.with_key] makes
    them reproducible; non-float leaves get zero directions and do not
    participate. [sketch_sampler] replaces the default sampler — for
    Rademacher directions, per-leaf scaling, or a caller-owned RNG — and is
    given [k] and [params] and must return a parameter structure whose leaves
    stack [k] directions on a leading axis.

    [strict] (default [false]) runs {!check} and raises [Invalid_argument] on a
    discrepancy. It reads tensor values, so it concretizes the sketch's
    numbers: inside a [Rune.jit]ed function it raises [Rune.Jit_error] instead.
    Check the returned sketch when compiling.

    Raises [Invalid_argument] if [k < 1], if [loss] does not return a scalar, or
    if a marked prediction's tangent is not a [k]-lane batch (marking a loss
    differentiated with [Rune.jvp] rather than [Rune.jvp_k]). *)
val sketch
  :  (module Nx.Ptree.S with type t = 'p)
  -> k:int
  -> ?sketch_sampler:(int -> 'p -> 'p)
  -> ?strict:bool
  -> ('p -> ('c, 'd) Nx.t)
  -> 'p
  -> 'p sketch

(** [check sk] compares the little losses the collector observed with the loss
    the driver returned: [Ok ()] when they agree within a small relative
    tolerance, [Error msg] describing the disagreement otherwise. A loss term
    accumulated into the loss but never marked is absent from the sketch; a term
    marked but never accumulated is in the sketch but not in the loss.

    Reading the values concretizes them, so call this outside a [Rune.jit]ed
    function. Skipped observations are not an error: a prediction that is a
    constant of the differentiation has a zero block by construction. *)
val check : 'p sketch -> (unit, string) result
