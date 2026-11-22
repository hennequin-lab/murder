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
      type ('a, _) t = { w : 'a } [@@deriving ptree]

      let dims = { w = [ 100; 78 ] }
      let symmetries = { w = [ Symmetry.Id; Symmetry.Perm 0 ] }
    end
    ]}

    {!Make} turns that declaration into the orbit machinery and the
    damped-whitening optimizer for the whole parameter tree.

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

    {2 The damped GGN step}

    [S_g] is the second-order orbit average of the Gauss–Newton matrix,
    EMA'd with [beta_2] and debiased. {!Optim.S.ggn} builds its samples from
    the model's outputs: one per-output gradient per element of the output,
    whose mean orbit-averaged outer product is the GGN. The preconditioner is

    {[
      P = (S_g + λ I)^{-1}            (Inverse, the default: λ in gradient² units)
      P = (S_g^{1/2} + λ I)^{-1}      (Root, optional: λ in gradient units)
      θ ← θ − η · P · ḡ
    ]}

    [λ] is one absolute [damping]: it is added to every isotypic block's
    spectrum, or to its square root. The blocks are orthonormal bases of the
    isotypic components, so that is a multiple of the identity on the whole
    parameter space — across kinds and leaves — and the gain is capped at
    [1/λ] in either form. {!Solve} runs the eigendecompositions on the
    assembled blocks; it reads no host value, so the whole step is one
    [Rune.jit] program ({!Optim.S.Compiled}).

    The paper's Taylor estimator [H]/[H_inv] (Eq. 10) is still available from
    {!Solve}. It uses a relative damping,
    [X^{p̃} = U diag((damping·s_max + s)^p) Uᵀ], and is not the
    optimizer's step.

    {2 The jitted step}

    Nothing in the step reads a host value, so {!Optim.Make.Compiled} wraps the
    whole of it in [Rune.jit]: the state, the gradients and the GGN samples
    cross the boundary as input leaves and the next state comes back as the
    output leaf. [ggn] traces too, so the forward pass, the loss gradient, the
    metric and the update can share one program:

    {[
    let config = S.default ~learning_rate:(Some 0.1) ~damping:1e-2
    let compiled = S.Compiled.create ~config
    let rec train state n =
      if n = 0
      then state
      else (
        let _, grads = Rune.value_and_grad S.ptree loss state.S.State.theta in
        let samples = S.ggn ~outputs state.S.State.theta in
        train (S.Compiled.step compiled ~state ~grads ~samples) (n - 1))
    in
    train (S.init ~config params) 1000
    ]}

    [S.default] is the damped GGN step: [beta_1 = 0.9], [beta_2 = 0.99], and
    [`Inverse], so [damping] is in gradient-squared units. The config is a
    constant of the compiled step, so changing it means a new
    {!Optim.S.Compiled.create}. Compiled functions are not thread-safe and
    cache on the leaf signature of the state; changing the tree's shape or
    dtype retraces. *)

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
  module Make (M : Model) : S with type ('a, 'd) t = ('a, 'd) M.t
end

(** The estimator solves. *)
module Solve : sig
  (** [svd64 x] is the economy SVD of [x], computed in float64 and returned in
      float32. *)
  val svd64 : Nx.host Nx.float32_t -> Nx.host Nx.float32_t * Nx.host Nx.float32_t

  (** [damped_inverse_root ~damping x] is
      [((proj x)^{1/2} + damping·I)^{-1}], where [proj x] zeroes [x]'s
      negative eigenvalues (roundoff on a second moment). [damping] is
      absolute and sits at the root, in gradient units: it is not scaled by
      the spectrum, so the result's largest eigenvalue is capped at
      [1/damping]. It must be positive. It reads no host value, so it traces. *)
  val damped_inverse_root : damping:float -> 'd Nx.float32_t -> 'd Nx.float32_t

  (** [damped_inverse_root_blocks ~damping blocks] is the same on each block.
      Because the blocks are an operator's matrices in orthonormal bases of the
      isotypic components, the shared [damping] is [damping·I] on the whole
      parameter space. *)
  val damped_inverse_root_blocks
    :  damping:float
    -> 'd Nx.float32_t list
    -> 'd Nx.float32_t list

  (** [damped_inverse ~damping x] is [(proj x + damping·I)^{-1}], the linear
      cousin of {!damped_inverse_root}, with [damping] in the units of [x]'s
      spectrum. The result's largest eigenvalue is capped at [1/damping]. It
      must be positive. *)
  val damped_inverse : damping:float -> 'd Nx.float32_t -> 'd Nx.float32_t

  (** [damped_inverse_blocks ~damping blocks] is the same on each block. *)
  val damped_inverse_blocks
    :  damping:float
    -> 'd Nx.float32_t list
    -> 'd Nx.float32_t list

  (** [damp_spectrum ~relative_damping s] is [relative_damping * s_max + s]. *)
  val damp_spectrum
    :  relative_damping:float
    -> Nx.host Nx.float32_t
    -> Nx.host Nx.float32_t

  (** [symmetric_power ~relative_damping ~power x] is the damped symmetric
      power of a symmetric matrix [x = U diag(s) Uᵀ], namely
      [U diag((relative_damping·s_max + s)^power) Uᵀ]. A negative power of a
      zero singular value is zero: a zero surrogate yields a zero direction
      rather than NaN. *)
  val symmetric_power
    :  relative_damping:float
    -> power:float
    -> Nx.host Nx.float32_t
    -> Nx.host Nx.float32_t

  (** [hessian ~relative_damping sigma_w sigma_g] is the Taylor estimator [H]
      itself. *)
  val hessian
    :  relative_damping:float
    -> Nx.host Nx.float32_t
    -> Nx.host Nx.float32_t
    -> Nx.host Nx.float32_t

  (** [hessian_inverse ~relative_damping sigma_w sigma_g] is Eq. 10's
      preconditioner [H_inv] on two dense matrices. It reads the spectrum's
      maximum with [Nx.item], which a traced value refuses, so it is kept
      host-side. Its arguments must be on the host. *)
  val hessian_inverse
    :  relative_damping:float
    -> Nx.host Nx.float32_t
    -> Nx.host Nx.float32_t
    -> Nx.host Nx.float32_t

  (** [hessian_inverse_blocks ~relative_damping sigma_w sigma_g] is the same
      preconditioner applied blockwise: one square matrix per isotypic kind,
      with the damping scale shared across blocks. The optimizer no longer
      uses it; its root-damped counterpart is
      {!damped_inverse_root_blocks}. *)
  val hessian_inverse_blocks
    :  relative_damping:float
    -> Nx.host Nx.float32_t list
    -> Nx.host Nx.float32_t list
    -> Nx.host Nx.float32_t list
end

(** The damped-GGN optimizer on the orbit machinery: bias-corrected momentum,
    the second-order orbit average of the GGN with debiasing, the blockwise
    [(S_g + λI)^{-1}] (default) or [(S_g^{1/2} + λI)^{-1}] solve, the parameter
    shift, and the one [Rune.jit] step. *)
module Optim : module type of Optim

(** [Make (M)] is the library entry point: the orbit machinery ({!Orbit.Make})
    and the damped-GGN optimizer ({!Optim.Make}) for a model [M], plus [ptree],
    the packed traversal `Rune.grad` and `Rune.jit` take.

    The parameter tree is [('d Nx.float32_t, 'd) M.t] throughout, on any
    device set ['d]: the compiled tables are host values, placed beside the
    tree's arrays where they meet them, and the step runs where the arrays
    lie (it needs a set whose kernels compute [Nx.eigh]). [dims] and
    [symmetries] describe each leaf. *)
module Make (M : Orbit.Model) : sig
  (** The packed traversal of the parameter tree on the host. A description is
      at one set: [Nx.Ptree.instantiate (module M)] describes the tree on
      another. *)
  val ptree : ((Nx.host Nx.float32_t, Nx.host) M.t, Nx.host) Nx.Ptree.t

  include Orbit.S with type ('a, 'd) t := ('a, 'd) M.t

  (** The orbit machinery as a named submodule, alongside its [include]d
      contents. *)
  module Orbit : Orbit.S with type ('a, 'd) t = ('a, 'd) M.t

  include Optim.S with type ('a, 'd) tree := ('a, 'd) M.t
end
