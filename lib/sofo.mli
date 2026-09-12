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
      per observation. Under an enclosing [Rune.vmap] the direction [f]
      receives carries the map's batch axes, like every other curvature here:
      a shape-dependent [f] must accept them, as [Scale], [Diag] and
      [Softmax_ce] do by construction. *)
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
    matching curvature.

    {b Inside [Rune.vmap].} Everything the mapped function performs happens
    once, on physically batched tensors, so an observation inside a map carries
    one little loss per mapped element, packed along the batch axis. The
    collector sums their values and tangents — each packed little loss
    contributes to the total, and the single k×k block contracts over the whole
    batch — so the observation accounts for the batch's contribution to the
    loss, whatever reduction the user applies outside. A mean over the map's
    axis is therefore a factor of [M] away from the observations, which
    {!check} reports (and [strict] refuses): fold the factor into the little
    loss (scale the value and its curvature by [1/M], and reduce the map's
    outputs with a sum), or observe the batched prediction after the map. *)
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

    The same loss function runs under the other drivers without changing: the
    observations are inert to {!Rune.value_and_grad} and to {!Rune.jvp_k} alone.
    A [Rune.scan] inside the loss folds inside the collector, so a per-step
    observation fires for every step; a [Rune.vmap] inside the loss batches the
    trials inside the tangent axis (put batch dimensions there, never outside
    the sketch — {!Rune.vmap} around [sketch] is a lane error, not a batch of
    sketches). Inside a [Rune.jit]ed function a sketch is correct but unrolled:
    the collector claims each scan and its accumulations are traced beside the
    primal operations, so compile time grows with the horizon where a plain
    compiled rollout stages the same scan as a loop.

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

(** {1 The optimizer: Algorithm 1's update}

    {!sketch} measures; this turns a measurement into a step. It is the whole
    of SOFO beyond the sketching — the damped subspace solve and the parameter
    update — and it is deliberately small: everything expensive or
    differentiable belongs to the sketch.

    Two halves, split by what compiles rather than by taste. The sketching half
    is differentiable and jits ({!Compiled}), while the update needs the SVD of
    the sketched GGN, which does not compile: it runs eagerly on the host at
    O(k³), negligible beside a model with P ≫ k parameters — the premise of
    the algorithm. A deployment therefore compiles one program and keeps the
    update outside it:

    {[
    module O = Sofo.Optim.Compiled (Params)

    (* traced once, replayed for every step: the compiled half *)
    let sketch_step = Rune.jit2 (module O.In) (module O.Out) (O.sketch ~k loss)

    (* the loop: one replay, one eager solve, one eager step *)
    let params, state =
      let out = sketch_step { O.params; key = state.Sofo.Optim.key } in
      (match O.check out with
       | Error m -> failwith m
       | Ok () -> ());
      O.update ~lr ~damping params out, Sofo.Optim.next state
    ]}

    The state is only the direction stream — a key carried as a tensor, so it
    rides the compiled step as an ordinary input leaf and one compilation
    serves the whole run — and an iteration counter. Directions are drawn
    inside the compiled step, from that key, so every replay starts from a fresh
    random subspace without a retrace. *)

module Optim : sig
  (** {2 State} *)

  type state =
    { key : Nx.int32_t (** The stream the directions Θ are drawn from. *)
    ; step : int (** Iterations completed. *)
    }

  (** [init ?key ()] is a fresh state: step 0, and [key] as the stream's root,
      or a subkey of the ambient {!Nx.Rng} scope when [key] is omitted. The
      whole run follows from this key. *)
  val init : ?key:Nx.Rng.key -> unit -> state

  (** [next st] is the state of the iteration after [st]: the counter advances
      and the key becomes the subkey the counter indexes, so no two steps share
      a direction draw. *)
  val next : state -> state

  (** {2 Directions} *)

  (** [directions (module P) ?key ~k params] is Θ: one standard-normal [k]-lane
      batch per float leaf, and zero lanes for leaves that cannot carry a
      direction (an RNG key, an index, a counter — they do not participate).
      A pure function of [key] (or of the ambient scope when [key] is
      omitted), so the compiled step can draw it from a key that is an input
      leaf, and a caller can reproduce a step's subspace by name. This is
      {!sketch}'s own sampler, keyed. *)
  val directions
    :  (module Nx.Ptree.S with type t = 'p)
    -> ?key:Nx.Rng.key
    -> k:int
    -> 'p
    -> 'p

  (** [apply (module P) ~k thetas z] is Θz: the parameter-space direction the
      sketch coordinates [z] denote, each leaf in its parameter's dtype.
      {!sketch} returns the same function as the [apply] field of its record;
      this one takes the directions as a value, because a compiled step cannot
      return a closure. *)
  val apply : (module Nx.Ptree.S with type t = 'p) -> k:int -> 'p -> Nx.float64_t -> 'p

  (** {2 The update} *)

  (** [coordinates ?damping ggn c] is the damped sketched solve
      [U (S + λ·s_max I)⁻¹ Vᵀ C] of the sketched normal equations [ggn·z = c],
      from the SVD of [ggn] — Algorithm 1, lines 10–12. [damping] (λ, default
      [1e-6]) is relative to the largest singular value, as the paper's is; the
      solve is eager and O(k³), and it is the one place the library needs a
      factorization.

      {b Note.} The SVD is not differentiable and does not compile, so this is
      for the host side of a step, not inside a [Rune.jit]ed program. *)
  val coordinates : ?damping:float -> Nx.float64_t -> Nx.float64_t -> Nx.float64_t

  (** [update (module P) ~lr ?damping sk params] is one SOFO step:
      [θ ← θ − η·Θ U (S + λ·s_max I)⁻¹ Vᵀ C], from the sketch [sk] measured at
      [params]. [lr] defaults to [1.0], which together with [damping = 0.] is
      the exact Newton step inside the sketched subspace. *)
  val update
    :  (module Nx.Ptree.S with type t = 'p)
    -> ?lr:float
    -> ?damping:float
    -> 'p sketch
    -> 'p
    -> 'p

  (** {2 One eager step} *)

  (** [step (module P) ~k ~lr ?damping ?strict st ~loss ~params] sketches
      [loss] at [params] along directions drawn from [st]'s key, applies the
      update, and returns the new parameters, the next state and the sketch —
      the loss, C, G̃ and the diagnostics, for logging or {!check}. One call is
      one training iteration, eager end to end; use {!Compiled} when the
      sketching half should be compiled. *)
  val step
    :  (module Nx.Ptree.S with type t = 'p)
    -> k:int
    -> lr:float
    -> ?damping:float
    -> ?strict:bool
    -> state
    -> loss:('p -> ('c, 'd) Nx.t)
    -> params:'p
    -> 'p * state * 'p sketch

  (** {2 The compiled half}

      [Compiled (P)] is the sketching computation as a pure function of
      [(params, key)], with the structure types a compiled step needs. Wrap it
      once —

      {[
      let step = Rune.jit2 (module O.In) (module O.Out) (O.sketch ~k loss)
      ]}

      — and every later call replays: the shapes of [in_] do not change across
      a training run, and the directions come from the key the caller threads,
      so a fresh subspace costs nothing. What comes back is everything the
      update needs and nothing that needs a factorization. *)
  module Compiled (P : Nx.Ptree.S) : sig
    type in_ =
      { params : P.t
      ; key : Nx.int32_t
      }

    (** [In] is [in_] as a parameter tree: one input structure for the whole
        compiled step, parameters and key together. *)
    module In : Nx.Ptree.S with type t = in_

    type out =
      { loss : Nx.float64_t (** The primal total, as a scalar. *)
      ; c : Nx.float64_t (** C = Θᵀ∇c, shape [k]. *)
      ; ggn : Nx.float64_t (** ΘᵀJᵀHJΘ, shape [k;k]. *)
      ; dirs : P.t (** Θ, the directions the sketch was measured along. *)
      ; observed_loss : Nx.float64_t
      ; observed_c : Nx.float64_t
      }

    module Out : Nx.Ptree.S with type t = out

    (** [sketch ~k loss] draws Θ from [in_.key], sketches [loss] at
        [in_.params], and returns the numbers together with the directions.
        Pure, differentiable in nothing, and safe to compile. *)
    val sketch : k:int -> (P.t -> ('c, 'd) Nx.t) -> in_ -> out

    (** [update ?lr ?damping params out] is {!Optim.update} on a compiled
        step's output: the eager half. *)
    val update : ?lr:float -> ?damping:float -> P.t -> out -> P.t

    (** [check out] is {!check} on a compiled step's output — the same
        statement about the observed little losses, made where the values are
        readable. A compiled trace cannot read them, so this is how a
        compiled deployment keeps the cross-check. *)
    val check : out -> (unit, string) result
  end
end
