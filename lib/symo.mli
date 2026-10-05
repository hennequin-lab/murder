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
    end
    ]}

    {!Make} turns that declaration into the orbit machinery and the Taylor
    optimizer for the whole parameter tree.

    {2 Blocks and sizes}

    The second-order operator lives in the commutant algebra of the group
    action, which is a direct sum of small matrix algebras — one {e block} per
    isotypic kind ([Blocks]). The factors the compiler estimates are a
    size-dependent {e encoding} of that algebra; the blocks are the
    size-independent description. {!Orbit.S.Second_order.blocks_of_factors}
    decodes them and {!Orbit.S.Second_order.transport} re-encodes an operator
    at another size, preserving its blocks, its distinct spectrum and any
    function of it (only multiplicities, hence trace and determinant, change).
    The optimizer works on the blocks directly, so no surrogate is involved.

    A permuted axis must be in the stable range ([n >= 2k] for [k] permuted
    axes per group, i.e. at least 2, 4 and 6 for one, two and three permuted
    axes); below it the algebra is not semisimple and the decomposition is not
    faithful. The compiler raises [Invalid_argument] when a spec violates it.

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
    factorization on the assembled blocks, the only step outside [Rune.jit]'s
    reach; {!Optim} composes momentum, orbit averages, the factors of
    [S_w]/[S_g], the estimator solve and the parameter shift.

    {2 Compiled step}

    The step is split so that everything but the estimator solve is
    traceable: [prepare] (momentum, orbit averages, the packed factors of
    [S_w]/[S_g], EMA and bias counters) and [finish] (the factors of [H_inv],
    applied to the momentum, shift the parameters). {!Optim.Make.Compiled} wraps the two
    halves in [Rune.jit] with the whole state threaded through input and
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

(** Isotypic kinds of one permutation group, with their canonical test
    vectors: the [k ≤ 2] decomposition of [V^{⊗k}] into [⊕_λ S^{[n-|λ|,λ]} ⊗ M_λ] that the block coordinates are built on.
*)
module Kind : module type of Kind

(** Block (Wedderburn) coordinates of the commutant algebra of a compiled
    pair: [to_blocks]/[of_blocks] convert factor lists to blocks and back,
    and [transport] re-encodes an operator at another size, preserving its
    blocks, its distinct spectrum and any function of it. *)
module Blocks : module type of Blocks

(** Orbit machinery: first- and second-order averages over a parameter tree.

    Everything here is a composition of tree traversals and [Nx] operations —
    no host read of a tensor value, no dynamic shapes, no RNG — so the eager
    functions are traceable as they stand. *)
module Orbit : sig
  module type Model = Orbit.Model
  module type S = Orbit.S

  (** [Make (M)] lifts the per-tensor compiler to the whole parameter tree:
      one compiler per leaf at full size and at surrogate size, factor
      estimation from parameters and from dense surrogates, the orbit
      averages [R1] and [R2], and a random group element applied to the tree
      ([random_transform]). *)
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
      [H_inv] on two dense matrices. It needs [Nx.eigh], which [Rune.jit] does
      not compile, so it is kept host-side. Its arguments must be on the
      host. *)
  val hessian_inverse : damping:float -> Nx.float32_t -> Nx.float32_t -> Nx.float32_t

  (** [hessian_inverse_blocks ~damping sigma_w sigma_g] is the same
      preconditioner applied blockwise: one square matrix per isotypic kind,
      with the damping scale shared across blocks. This is what the optimizer
      uses; it is exact at the model's size and small. *)
  val hessian_inverse_blocks
    :  damping:float
    -> Nx.float32_t list
    -> Nx.float32_t list
    -> Nx.float32_t list
end

(** The Taylor optimizer on the orbit machinery: bias-corrected momentum, an
    EMA of the gradient's factors with debiasing, the blockwise estimator
    solve, the parameter shift, and the compiled [prepare]/[finish] pair. *)
module Optim : module type of Optim

(** [Make (M)] is the library entry point: the orbit machinery ({!Orbit.Make})
    and the Taylor optimizer ({!Optim.Make}) for a model [M], plus [ptree],
    the packed traversal {!Rune.grad} and {!Rune.jit} take.

    The parameter tree is [Nx.float32_t M.t] throughout; [dims] and
    [symmetries] describe each leaf. *)
module Make (M : Orbit.Model) : sig
  (** The packed traversal of the parameter tree. *)
  val ptree : Nx.float32_t M.t Nx.Ptree.t

  include Orbit.S with type 'a t := 'a M.t

  (** The orbit machinery as a named submodule, alongside its [include]d
      contents. *)
  module Orbit : Orbit.S with type 'a t = 'a M.t

  include Optim.S with type 'a tree := 'a M.t
end
