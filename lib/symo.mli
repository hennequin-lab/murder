(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Weight-space symmetry for curvature estimation.

    A network is [G]-invariant when [L(A w) = L(w)] for every group element
    [A], with [G] acting orthogonally on the vectorised parameters [w]. The
    gradient is then equivariant, [∇L(A w) = A ∇L(w)], so one gradient knows
    the gradient everywhere on the orbit and group averages are themselves
    invariant objects. SYMO builds the two orbit averages the optimizer needs
    — the first-order [R1(v) = E_G[(⊗ A) v]] and the second-order
    [R2(v, v') = E_G[(⊗ A) v v'ᵀ (⊗ A)ᵀ]] — in a compact basis of the
    commutant algebra, estimates the coefficients of that basis (the
    {e factors}) from data, and uses them to precondition the gradient with
    Eq. 10 of

    > Artemev, Xia, Boyd, Yu, Dangel, Hennequin, Bernacchia,
    > {e Exploiting weight-space symmetries for approximating curvature},
    > ICML 2026.

    {2 Symmetry specifications}

    A parameter tensor is described by one {!Symmetry.spec} per axis:
    [Symmetry.Id] for an axis the group leaves alone (a "free" axis),
    [Symmetry.Perm i] for an axis transformed by group [i] (axes sharing [i]
    are transformed by the same group element), and [Symmetry.Absent] for a
    side of the commutation equation that has no axis. A model supplies one
    specification per leaf, alongside the leaf dimensions:

    {[
    module Model = struct
      type 'a t = { w : 'a } [@@deriving ptree]

      let dims = { w = [ 100; 78 ] }
      let symmetries = { w = [ Symmetry.Id; Symmetry.Perm 0 ] }
      let surrogate_dim = 2
    end
    ]}

    {!Make} turns that declaration into the orbit machinery and the Taylor
    optimizer for the whole parameter tree. The estimator works on a {e surrogate} network: [Id] axes keep their dimension, permuted axes shrink
    to {!Orbit.Model.surrogate_dim} (2 in the paper). The surrogate group must
    stay non-trivial — collapsing it to a single element makes the estimated
    curvature vanish identically — which is why the dimension is explicit
    rather than derived from [dims].

    {2 The Taylor step}

    With [S_w = R2(δ_w, δ_w)] and [S_g = R2(δ_g, δ_g)] built from the
    non-invariant parts of the parameters and the averaged gradient, the
    estimator is

    {v
      H     = S_w^{-1/2̃} (S_w^{1/2̃} S_g S_w^{1/2̃})^{1/2̃} S_w^{-1/2̃}
      H_inv = S_w^{ 1/2̃} (S_w^{1/2̃} S_g S_w^{1/2̃})^{-1/2̃} S_w^{ 1/2̃}
    v}

    with damped symmetric powers [X^{p̃} = U diag((damping·s_max + s)^p) Uᵀ],
    and the step is [θ ← θ − η · H_inv · ḡ]. {!Solve} performs the
    factorization, the only step outside [Rune.jit]'s reach; {!Optim}
    composes momentum, orbit averages, the dense surrogates, the estimator
    solve and the parameter shift.

    {2 Compiled step}

    The step is split so that everything but the estimator solve is
    traceable: [prepare] (momentum, orbit averages, dense [S_w]/[S_g], EMA
    and bias counters) and [finish] (factors of [H_inv], apply to the
    momentum, shift the parameters). {!Optim.Make.Compiled} wraps the two
    halves in [Rune.jit2] with the whole state threaded through input and
    output leaves, so a training loop traces once and then replays:

    {[
    let config =
      { Optim.learning_rate = Some 0.1; beta_1 = 0.9; beta_2 = 0.99; damping = 1e-4 }
    in
    let compiled = S.Compiled.create ~config in
    let rec train state n =
      if n = 0
      then state
      else (
        let _, grads = Rune.value_and_grad S.ptree loss state.S.State.theta in
        train (S.Compiled.step compiled ~config ~state ~grads) (n - 1))
    in
    train (S.init ~config params) 1000
    ]}

    Compiled functions are not thread-safe and cache on the leaf signature of
    the state; changing the tree's shape or dtype retraces. *)

(** The two sides of a commutation equation: the left and right copies an
    orbit average pairs. *)
module Sides : module type of Sides

(** Axis indices of a commutation equation: [Left i] names the [i]-th axis of
    the left side, [Right i] the [i]-th axis of the right side. Indices
    compare structurally, collect in sets, and compile to einsum characters. *)
module Index : module type of Index

(** Symmetry specifications: one {!Symmetry.spec} per axis, and the surrogate
    dimensions a specification induces. *)
module Symmetry : module type of Symmetry

(** Basis terms: index ties (Kronecker deltas) plus the free axes that carry a
    factor. The pure part of the compiler also lives here — term ordering,
    transposition, normalization, and the symbolic and dense projections. *)
module Term : module type of Term

(** Basis components: a single term, or a term summed with its transpose in
    the symmetric case. *)
module Component : module type of Component

(** The symbolic basis of the invariant subspace: its components, the axes
    tied by each group, and the group ids that order them. *)
module Basis : module type of Basis

(** The per-tensor compiler: a symmetry specification at fixed dimensions
    becomes closures for factor estimation, dense blocks and batched
    block-vector products. The design matrix is factorized once, at compile
    time, by Cholesky (with a jittered retry and an SVD pseudo-inverse
    fallback for degenerate bases). *)
module Compiler : module type of Compiler

(** Orbit machinery: first- and second-order averages over a parameter tree.

    Everything here is a composition of tree traversals and [Nx] operations —
    no host read of a tensor value, no dynamic shapes, no RNG — so the eager
    functions are traceable as they stand. *)
module Orbit : sig
  module type Model = Orbit.Model
  module type S = Orbit.S

  (** [Make (M)] lifts the per-tensor compiler to the whole parameter tree:
      one compiler per leaf at full size and at surrogate size, factor
      estimation from parameters and from dense surrogates, and the orbit
      averages [R1] and [R2]. *)
  module Make (M : Model) : S with type 'a t = 'a M.t
end

(** The host-side estimator solve. *)
module Solve : sig
  (** [svd64 x] is the economy SVD of [x], computed in float64 and returned in
      float32. *)
  val svd64 : Nx.float32_t -> Nx.float32_t * Nx.float32_t

  (** [damp_spectrum ~damping s] is [damping * s_max + s]. *)
  val damp_spectrum : damping:float -> Nx.float32_t -> Nx.float32_t

  (** [symmetric_power ~damping ~power x] is the damped symmetric power of a
      symmetric matrix, [U diag((damping·s_max + s)^power) Uᵀ] for
      [x = U diag(s) Uᵀ]. A negative power of a zero singular value is zero:
      a zero surrogate yields a zero direction rather than NaN. *)
  val symmetric_power : damping:float -> power:float -> Nx.float32_t -> Nx.float32_t

  (** [hessian ~damping sigma_w sigma_g] is the estimator [H] itself. *)
  val hessian : damping:float -> Nx.float32_t -> Nx.float32_t -> Nx.float32_t

  (** [hessian_inverse ~damping sigma_w sigma_g] is Eq. 10's preconditioner
      [H_inv]. This is the one routine [Rune.jit] refuses ([Nx.svd]), kept
      host-side on the small dense surrogate matrices. *)
  val hessian_inverse : damping:float -> Nx.float32_t -> Nx.float32_t -> Nx.float32_t
end

(** The Taylor optimizer on the orbit machinery: bias-corrected momentum, an
    EMA of the gradient surrogate with debiasing, the estimator solve, the
    parameter shift, and the compiled [prepare]/[finish] pair. *)
module Optim : module type of Optim

(** [Make (M)] is the library entry point: the orbit machinery ({!Orbit.Make})
    and the Taylor optimizer ({!Optim.Make}) for a model [M], plus [ptree],
    the packed traversal {!Rune.grad} and {!Rune.jit} take.

    The parameter tree is [Nx.float32_t M.t] throughout; [dims] and
    [symmetries] describe each leaf, and [surrogate_dim] the mock dimension
    permuted axes take at surrogate size. *)
module Make (M : Orbit.Model) : sig
  (** The packed traversal of the parameter tree. *)
  val ptree : (module Nx.Ptree.S with type t = Nx.float32_t M.t)

  include Orbit.S with type 'a t := 'a M.t

  (** The orbit machinery as a named submodule, alongside its [include]d
      contents. *)
  module Orbit : Orbit.S with type 'a t = 'a M.t

  include Optim.S with type 'a tree := 'a M.t
end
