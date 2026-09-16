# SYMO on the modern Raven stack — port plan

Status: M0 (pure front end) and M1 (compiler on Nx) are implemented in
`symo/lib/` with 18 passing tests (`dune runtest`); M2-M4 are pending. Two Nx
`einsum` defects shaped the tensor-side design (§10). `old_code/` is
reference-only (excluded from the build via `(data_only_dirs old_code)` and
gitignored); we will **not** try to compile it.

Reference for the algorithm:

> Artemev, Xia, Boyd, Yu, Dangel, Hennequin, Bernacchia,
> *Exploiting weight-space symmetries for approximating curvature*, ICML 2026,
> [arXiv:2606.00442](https://arxiv.org/abs/2606.00442).

`old_code/` is an earlier, float32 OCaml implementation of the same idea on
`owl` + torch bindings + a `prms` typed-parameter-tree library. The goal is to
re-express it on `nx`/`rune`, keeping the compiler's logic essentially as is
and changing only the tensor layer, the parameter-tree layer and the AD layer.

Scope decisions since the first draft:

- of the three estimators in `old_code` (`Original`, `Taylor`, `Global`), only
  **`Taylor`** is ported — it is Eq. 10 of the paper and what was published.
  `Original` and `Global` (and with them the Lyapunov solve) are dropped.
- no model definitions in the library. `models.ml` is not ported here; tests
  use small local fixtures, and examples come later, outside the library.
- the eventual **JIT compilation of the step is a design constraint from the
  start**: everything that can be traced is kept traceable, and the one
  non-traceable routine (the SVD of the estimator) is isolated behind a host
  boundary.
- the design-matrix pseudo-inverse is replaced by a **Cholesky factorization
  plus triangular solves** (see §2.3 and §3.5).
- some compiler internals are renamed for clarity; the proposals are listed in
  §4.1 and used throughout this document.

---

## 0. Scope

In scope, in order:

1. the **compiler**: from a symbolic symmetry specification to runnable
   closures computing orbit averages, factor estimation and curvature blocks;
2. the **first- and second-order orbit machinery** built on those closures;
3. the **`Taylor` update** (Eq. 10), as an eager step and as a jitted step;
4. tests (compiler vs. brute force, orbit averages, optimizer, jit parity).

Out of scope:

- `Original` and `Global` estimators, and the continuous Lyapunov solve;
- model definitions (`MLP`, `RNN`, …); they belong in examples or downstream
  code, not in `symo`;
- `Pinned` / `Free` / `Bounded` parameter wrappers (`prms`-specific; see §6.6);
- devices, `pmap`, multi-GPU (eager `nx` is CPU; `rune` device work can come later);
- a fully jitted estimator (SVD replaced by matrix iterations); see §3.5.

---

## 1. The algorithm in one page

### 1.1 Symmetry, orbit averages

A network is invariant under a group `G` acting on the vectorised parameters
`w` by orthogonal matrices `A`: `L(A w) = L(w)`. Then gradients are equivariant,
`∇L(A w) = A ∇L(w)`, so one gradient gives the gradient anywhere on the orbit.
Averaging over the orbit gives group-invariant objects. The old code (and
`Term`/`Component`/`Compiler`) computes, for tensors of any order,

- first order: `R1(v) = E_G[(⊗_k A_i(k)) v]`  (Eq. 1 of the paper);
- second order: `R2(v, v') = E_G[(⊗A) v v'ᵀ (⊗A)ᵀ]` (Eq. 2).

Each parameter tensor carries a **symmetry spec**: a list, per axis, of
`Absent` (no axis; pads a side in a compilation), `Id` (the axis exists and is
not transformed; a "free" axis), or `Perm i` (the axis is transformed by group
`i`). Axes tied to the same `Perm i` are transformed by the *same* group
element. `Sides.t = { left; right }` holds the two sides of the commutation
equation.

### 1.2 Structure of the orbit average: terms and factors

Invariance (`A S Aᵀ = S`, Eq. 6) forces `S` to live in the commutant algebra.
Its basis elements are sparse binary tensors built from Kronecker products of
identities and all-ones tensors, i.e. from **tied indices** (groups of indices
forced equal by Kronecker deltas) plus **free axes** (axes of `Id` type). A
single basis element — a `Term.t` — is

```ocaml
type Term.t =
  { ties : Index.t list list (* groups of tied indices *)
  ; free : Index.t list      (* untied axes carrying the factors *)
  }
```

and the orbit average is a linear superposition of such terms weighted by
small **factors** (scalars or matrices over the free axes). This is the compact
representation: we never materialise the full `S`; we estimate the factors.

### 1.3 The estimator: `Taylor` (Eq. 10)

With `S_w = R2(δ_w, δ_w)` and `S_g = R2(δ_g, δ_g)`, where `δ_w = w - R1(w)` and
`δ_g = ḡ - R1(ḡ)` for the momentum-averaged gradient `ḡ`:

- `S_w` is used as is; `S_g` is EMA'd with `β₂` and debiased;
- for a symmetric PSD `X` with singular values `s`, the damped symmetric
  powers are `X^{±1/2̃} = U diag(s̃^{±1/2}) Uᵀ`, `s̃ = damping·s_max + s`;
- the estimator and its inverse are

```
H     = S_w^{-1/2̃} (S_w^{1/2̃} S_g S_w^{1/2̃})^{1/2̃} S_w^{-1/2̃}
H_inv = S_w^{ 1/2̃} (S_w^{1/2̃} S_g S_w^{1/2̃})^{-1/2̃} S_w^{ 1/2̃}
```

- the step is `θ ← θ - η · H_inv · ḡ`.

The old code also forms `H` itself but only saves it for debugging; the port
computes `H_inv` only (keep `H` behind a debug flag if we want the old
numerics-checkable artifacts). This estimator is `Taylor` in `old_code` and
Eq. 10 of the paper; it is the only one ported.

### 1.4 What the compiler must produce

For a `(spec, dims)` pair, `Compiler.compile` returns a record of closures:

```ocaml
type compiled =
  { basis : Basis.t
  ; dims : int list Sides.t
  ; apply_block : factors:Nx.float32_t list -> Nx.float32_t -> Nx.float32_t
  ; dense_block : factors:Nx.float32_t list -> Nx.float32_t
  ; estimate_factors :
      [ `full of Nx.float32_t | `outer_product of Nx.float32_t * Nx.float32_t ]
      -> Nx.float32_t list
  ; transform : perms:Nx.int32_t list -> Nx.float32_t -> Nx.float32_t
  }
```

- `apply_block ~factors v`: batched matrix-vector product by one curvature
  block (the block acts on the left axes; `v` is batch × right axes);
- `dense_block ~factors`: the dense block, obtained by applying `apply_block`
  to an identity;
- `estimate_factors data`: least-squares estimation of the factor list from a
  data tensor, through the constant design matrix of the basis (`B`, or its
  Cholesky factor) — see §2.3;
- `transform ~perms`: test/debug helper, applies group elements to tied axes.

`First_order` compiles one `compiled` per leaf; `Second_order` compiles one per
pair of leaves. `dims` and `surrogate_dims` give the same specifications at
full size and at factor size.

---

## 2. The compiler pipeline (old code, modern reading)

This is the heart of the port, and it is almost entirely pure OCaml; only the
last three stages touch tensors.

### 2.1 From symmetry spec to basis

`Compiler.basis_of_spec ~symmetric spec` (old `invariance_from_spec`):

1. **free axes** — the `Id` axes of both sides;
2. **unique perms** — for each `Perm i`, the list of `Index.t` occurrences
   (axis positions on left/right, sorted);
3. **partitions** — for each group, all set partitions of its occurrences
   into blocks of size ≥ 2 (singletons dropped). A block becomes one tie; the
   rest of the term is the free axes;
4. **cartesian product** across groups, then `Term.sort`;
5. for second-order terms only, drop terms that tie ≥ 2 occurrences of the
   same side in one block (`ties_one_side`, old `binds_one_side`). The old
   comment marks this as empirical ("to be confirmed theoretically") — keep
   it, test it, revisit;
6. if `~symmetric`, merge a term with its transpose into a `Sum`
   (`bundle_transposes`), else wrap each as a `Single`.

The result is a `Basis.t = { label; symmetric; components; group_axes }` (old
`Invariance.t`). The old code `print`s the component count here — port without
the print (or log behind a flag); the count is already recoverable from
`components`.

### 2.2 From terms to contractions

The old IR (repeated einsum labels plus identity tensors) is deliberately not
ported: ties are compiled into explicit **delta operands** and every
contraction is pairwise. See §10 for the two Nx `einsum` defects that forced
this. Concretely, `compile_apply_term` (old `compile_bmv_prod_single_term`):

- gives every axis of the vector operand its own character (`Index.to_char`,
  unique per axis, `'z'` reserved for the batch axis);
- compiles each tie into a `Delta.tensor ~order dim` operand carrying the tied
  axes' characters;
- contracts the deltas one at a time into the running tensor
  (`Contract.binary`, which materializes each intermediate), updating the
  label list to the symmetric difference of the two operands' labels;
- contracts the factor last, against the free-axis labels, keeping the
  involved left axes as the output;
- views the result to `batch × dims.left`, scales by `Term.normalization` and
  broadcasts the axes the term does not involve.

`Term.normalization` (old `compute_normalization`) is
`1 / sqrt(prod of dims of indices not involved in the term)`; it is part of the
convention and ported exactly.

`Component.t` is either a `Single` term or a `Sum` (used for the symmetric
case). `compile_apply` sums the per-component contributions weighted by their
factors.

### 2.3 Factor estimation without SVD

`Component.design_matrix ~dims components` is the symmetric Gram matrix
`B_ij = <c_i, c_j>` using the *symbolic* inner product `Term.inner_product`
(which merges ties and multiplies axis dimensions). It is computed **once per
`(spec, dims)`**, at compile time, and is a constant of the compiled closure.

The old runtime path is

```ocaml
let b = List.map components ~f:(fun c -> Component.coefficient ~dims c data) in
let v = B_pinv * b in
List.map components ~f:(fun _ -> reshape (row v i) factor_shape)
```

where `B_pinv` was computed at compile time by an SVD of `B` in double
precision (`u / s *@ uᵀ`, valid because `B` is symmetric).

**Port: Cholesky instead of the SVD pinv.**

- `B` is a Gram matrix, hence symmetric PSD; it is positive-definite exactly
  when the basis components are linearly independent, which is the generic
  case. For PD `B`, `B⁺ = B⁻¹`, so the Cholesky solve is the same linear
  problem the old code solved, with a better algorithm.
- At compile time, factor `B` in float64: `L = Nx.cholesky B64` (with the
  fallback below).
- At run time, either
  (a) *solve*: `y = solve_triangular L b`, `v = solve_triangular ~transpose:true L y`
      — two small triangular solves per call, fully traceable, or
  (b) *inverse*: precompute `B_inv` at compile time (by two triangular solves
      against the identity) and keep the old `B_inv *@ b` matmul.
  Plan: **(a) as the default** (it avoids materialising an inverse and is what
  we want numerically), (b) as a fallback if the unrolled solves bloat a
  jitted trace — `Second_order` calls factor estimation once per pair, so the
  trace contains O(n²) of them.
- Both `cholesky` and `solve_triangular` are jittable in `rune` for float
  dtypes (see §3.5); the old `svd`/`pinv` is not.
- **Rank deficiency.** Degenerate dimensions (e.g. an axis of size 1) can make
  distinct basis components coincide, so `B` is PSD but not PD and Cholesky
  raises `Not_positive_definite`. Policy, at compile time only: retry with a
  relative jitter `B + ε·(tr(B)/n)·I`; if that still fails, fall back to the
  old SVD pseudo-inverse and warn. The fallback stays compile-time, so it does
  not threaten jit-ability; `Nx.pinv`/`Nx.svd` never enter the traced program.
- **Precision.** Keep the factorization in float64 (the old code used double
  for the pinv) and cast `L` or `B_inv` to float32 once at compile time, so
  the runtime solve happens at the parameter dtype. A `~precision:` knob can
  promote the solve to float64 later if needed.

### 2.4 Surrogates: parameter space vs. dense surrogate

The vectorised parameter tree is concatenated per leaf into one dense
"surrogate" vector of size `Σ_i prod(surrogate_dims_i)`, split back apart by
fixed-size splits. The same construction gives a dense surrogate matrix for the
second order. Working on the surrogate is what makes the small-matrix solve
cheap: its size is the number of factor coefficients, not the number of
parameters.

The old code asks the user for `surrogate_dims`. From reading the compiler,
they are forced to be `dims` with every `Perm` axis set to `1` (the `Id` axes
survive; `Absent` axes do not exist). Reason: `apply_block` broadcasts the
uninvolved (`Perm`) left axes to their dimension, so a `Perm` axis left at `n`
would make a block `n×` too large; the factor space itself only spans the free
axes. **Plan: derive them (`Symmetry.collapse_dims`) and check the equality
numerically.** Keep an explicit override for safety.

---

## 3. Target architecture

### 3.1 Uniform ptrees everywhere metadata lives

Raven's answer to `prms` is `Nx.Ptree`: a payload-generic record/list derives
`map`, `map2`, `iter`, `fold`, `fold2`, `names` with `[@@deriving ptree]`
(`ppx_ptree`, local package in `raven/`), and `Nx.Ptree.instantiate` turns it
into the packed `Nx.Ptree.S` walker that `Rune.grad`/`jit` take. So, exactly as
in the old `Prms.T` design:

```ocaml
module Layer = struct type 'a t = { w : 'a; b : 'a } [@@deriving ptree] end

module Params = struct
  type 'a t = { layers : 'a Layer.t list }
  [@@deriving ptree]
end
```

and the symmetry metadata are values of the same structure:

```ocaml
val dims           : int list Params.t
val symmetries     : Symmetry.spec list Params.t
val surrogate_dims : int list Params.t   (* derived, see §2.4 *)
```

The central design idea for the port: **keep metadata and compiled closures in
the same uniform-tree shape as the parameters**, so every traversal aligns
structurally and no leaf ordering is hand-maintained:

- `dims`, `symmetries`, `surrogate_dims`: `'a Params.t` values;
- per-leaf compiled entities: `Compiler.t Params.t` (first order) and, for
  second order, `Compiler.t array Params.t` — row `i` of the pair matrix
  indexed by leaf `i`;
- packed tensor data: `Nx.float32_t Params.t`;
- zip data with metadata via `Params.map2`, flatten in traversal order with
  `Params.fold` where a dense concatenation is genuinely needed.

Only genuinely dense things are flat: the surrogate vector/matrix and the
per-leaf factor lists that index the *other* leaves.

### 3.2 Dtype policy

`old_code/` is float32 throughout (`to_bigarray ~kind:Bigarray.float32`; Owl
single-precision AD), with double precision used only for the factorization
inside the solver. Keep that:

- parameters, factors, blocks, einsums, the runtime solves: `Nx.float32_t`;
- the design-matrix Cholesky (and the SVDs of the estimator): `Nx.cast
  Nx.float64` in, cast back to float32 after.

`Nx.svd` returns singular values as `float64_t` already, which matches that
split. The optimizer step is host-side and not differentiated, so the casts
never meet `Rune`'s tape.

### 3.3 AD: Rune replaces `Owl.Algodiff`

The old `P.AD` module (forward/reverse tags, `value_and_grad`, `jvp`,
`hessian_v`, `vectorize`) does not belong inside the compiler; it is exactly
what `Rune` provides:

| old (`Owl.Algodiff`)         | new (`Rune`)                                           |
|------------------------------|--------------------------------------------------------|
| `P.value_and_grad x ~f`      | `Rune.value_and_grad (module P) f params`              |
| `jvp x ~f ~v`                | `Rune.jvp (module P) f params v`                       |
| `hessian_v x ~f ~v`          | `Rune.hvp (module P) f params v` (and `Rune.hessian'`) |
| `vectorize theta` (AD-aware) | tree traversal + `Nx.reshape`/`Nx.concatenate`         |

The old `Make` functor required `module P : Prms.T` *and* exposed an AD module;
the new `Make` only needs the traversal + metadata, and callers differentiate
the loss with `Rune` themselves. This keeps the optimizer step pure
(`grads -> state -> state`) and testable without AD.

### 3.4 RNG

`old_code` draws random permutations with torch's ambient RNG (`transform`,
`random_transform`) inside the compiler. Modern `nx` wants explicit keys
(`Nx.Rng.key`, `split`, `fold_in`, scope). Port rule: any function that used
to draw takes an explicit key (or pre-computed permutations), never reaches
for a global generator. Tests then replay exactly. The optimizer itself is
deterministic — no RNG in the step.

### 3.5 JIT plan

The step is designed so its heavy, parameter-space part can be traced. The
facts, checked against raven's source:

- `Nx.einsum` is a frontend decomposition into reshape/permute/matmul/sum/
  broadcast primitives (`nx/lib/core/frontend.ml`, `Einsum.calculate`), so a
  jitted program never sees an einsum op — it traces the decomposition. All
  compiler einsums are jittable.
- `Rune.jit` lowers `cholesky` and `solve_triangular` at trace time into
  fixed-step Tolk compositions for float dtypes (`rune/lib/jit.ml:1189` and
  `:1207`). A non-PD input to a *jitted* Cholesky yields NaNs instead of an
  exception — a reason to keep the design-matrix factorization at compile
  time, where mistakes are catchable.
- `Rune.jit` refuses `svd`, `eig`, `eigvals`, `eigh`, `eigvalsh` and `qr`
  (`rune/lib/jit.ml:1202-1206`). The estimator's `svd64` is therefore the one
  routine that cannot be traced.
- Tensors closed over by the compiled function are compile-time constants
  bound at first compile. The design-matrix Cholesky factor is exactly that.

So: **factor estimation becomes traceable (Cholesky solves), and the estimator
solve stays host-side on the small dense surrogate matrices.** Everything else
in the step — momentum, orbit averages, the dense surrogate assembly, the
factor application and the parameter shift — is pure `Nx` and traceable,
provided:

- no `Nx.item` or other host read of a traced value;
- no OCaml control flow that depends on traced values (the structure — number
  of leaves, components, equations — is static OCaml data, so `M.fold`/`map2`
  unroll fine at trace time);
- shapes are static (they are: `dims` and the tree structure are static);
- no RNG draws in the step.

Step decomposition:

```
                   jitted prepare                        host solve                  jitted finish
state, grads ──────────────────────────►  mid  ─────────────────────────►  h_inv  ─────────────────►  state'
        momentum, orbit averages,        S_w, S_g, δ's, g_avg,           svd64 + damping,          factors of h_inv,
        dense S_w/S_g, EMA, betas        debiased ḡ, sigma_g_avg,        symmetric powers          apply to ḡ, shift θ
                                         beta counters
```

Suggested types (all traversable with `Nx.Ptree.S`; dense matrices and scalars
are tensor leaves):

```ocaml
module State : sig
  type t =
    { theta : P.t                (* parameters *)
    ; g_avg : P.t                (* momentum buffer *)
    ; sigma_g_avg : Nx.float32_t (* dense surrogate, n_factors × n_factors *)
    ; beta_1_t : Nx.float32_t    (* scalars carried as 0-d tensors *)
    ; beta_2_t : Nx.float32_t
    }
  include Nx.Ptree.S with type t := t
end

type mid =
  { theta : P.t
  ; g_avg : P.t
  ; g_avg_debias : P.t
  ; sigma_w : Nx.float32_t       (* S_w *)
  ; sigma_g : Nx.float32_t       (* debiased S_g *)
  ; sigma_g_avg : Nx.float32_t   (* next state's EMA *)
  ; beta_1_t : Nx.float32_t
  ; beta_2_t : Nx.float32_t
  }

val prepare : State.t * P.t -> mid               (* traceable *)
val solve : damping:float -> mid -> Nx.float32_t (* host: Nx.svd *)
val finish : mid * Nx.float32_t -> State.t       (* traceable *)
```

and the compiled front-end:

```ocaml
module Compiled : sig
  val prepare : (State.t * P.t) -> mid    (* Rune.jit2 (module In) (module Mid) *)
  val finish : (mid * Nx.float32_t) -> State.t
  val step : config -> State.t -> P.t -> State.t (* prepare; solve; finish *)
end
```

Notes:

- `solve` is eager and small (the surrogate is the number of factor
  coefficients, e.g. a few dozen at most); it is the only place `Nx.svd`
  appears in the step.
- The compiled functions are built once (`Rune.jit` caches on the partial
  application and traces on first call); because the whole state is threaded
  through input/output leaves, a training loop never re-traces.
- `~donate:true` is available for a state-to-state loop that owns its buffers.
- A **fully jitted step** is possible in principle: replace the estimator's
  SVDs with fixed-iteration Newton–Schulz symmetric square roots and a
  power-iteration damping scale. That changes numerics and buys a solver that
  is already tiny; it is explicitly deferred. The host boundary is the design
  unless profiling says otherwise.

### 3.6 Proposed library surface

```ocaml
(* symo.mli (sketch) *)
module Sides    : sig type 'a t = { left : 'a; right : 'a } [@@deriving sexp] end
module Index    : sig type t = Left of int | Right of int ...
                        val to_char : ?shifted:bool -> t -> char ... end
module Symmetry : sig type spec = Absent | Id | Perm of int ...
                        val label_of : spec list Sides.t -> string
                        val collapse_dims : spec list -> int list -> int list end
module Term     : sig type t = { ties : Index.t list list; free : Index.t list } ...
                        val normalization : dims:int list Sides.t -> t -> float
                        val coefficient : dims:int list Sides.t -> t
                          -> [ `full of Nx.float32_t
                             | `outer_product of Nx.float32_t * Nx.float32_t ]
                          -> Nx.float32_t
                        val inner_product : dims:int list Sides.t -> t -> t -> float end
module Component : sig ... end
module Basis     : sig ... end
module Compiler  : sig ... (* type t, compile, compile_manual *) end

module Make (M : sig
    include Nx.Ptree.Uniform
    val dims : int list t
    val symmetries : Symmetry.spec list t
  end) : sig
  (* typed view of the tree at the working dtype *)
  module P : Nx.Ptree.S with type t = Nx.float32_t M.t

  val surrogate_dims : int list M.t (* derived once *)

  (** nearest invariant point: the first-order orbit average *)
  val orbit_average : Nx.float32_t M.t -> Nx.float32_t M.t

  module First_order : sig ... end
  module Second_order : sig ... end

  module Optim : sig
    type config =
      { learning_rate : float option
      ; beta_1 : float
      ; beta_2 : float
      ; damping : float
      }
    type state
    val init : config:config -> P.t -> state
    val step : config:config -> state -> grads:P.t -> state
  end

  module Compiled : sig ... end (* jitted prepare/finish/step, §3.5 *)
end
```

The functor is the closest analogue of the old `Symo.Make` and makes the
(expensive, compile-time) `basis_of_spec` enumeration happen once per model.
If an explicit value-based API is preferred later, `compile` can return a
record of functions; the internals are the same.

### 3.7 `Compiled.t` is a value, not a module

`Compiler.compile` returns the record of closures described in §1.4. Tests
reach into `basis`/`dims`; the optimizer uses `apply_block`,
`estimate_factors` and `dense_block`; `transform` is for tests and debugging.
Test-only helpers (random factors, brute-force group enumeration) live in
`test/support/`, not in the library.

---

## 4. Module-by-module port plan

| old file | LOC | port plan |
| --- | --- | --- |
| `sides.ml` | 7 | verbatim |
| `index.ml` | 87 | verbatim; generalise `to_char`, keep `Comparator` |
| `symmetry.ml` | 23 | renames + `collapse_dims` (`surrogate_dims` derivation) |
| `invariance.ml` | 20 | renames (`Basis.t`) |
| `term.ml` | 174 | pure parts verbatim modulo renames; `coefficient` → `Nx.einsum` |
| `component.ml` | 35 | pure; `design_matrix` → `Nx.create`/`Nx.float32` |
| `compiler.ml` | 440 | tensor port: einsum IR, identities, Cholesky estimation, blocks, transform |
| `models.ml` | 263 | **not ported**: model definitions belong in examples/downstream |
| `symo.ml` | 623 | split into `First_order`, `Second_order` (new `orbit.ml`), `Optim` (Taylor only), `Solve` |

### 4.1 Proposed renames

The old names are terse and in places misleading (`jit` is not a jit,
`deltas` hides that they are index ties, primes proliferate). Proposals (used
throughout the plan; confirmed at implementation start):

| old | new | note |
| --- | --- | --- |
| `Symmetry.spec`: `Nil \| I \| P i` | `Absent \| Id \| Perm i` | no axis / trivial action / group `i` |
| `Term.deltas` | `Term.ties` | blocks of tied indices |
| `Term.all_indices_involved` | `Term.indices_involved` | returns an `Index.Set` |
| `Term.dot_product` | `Term.inner_product` | term ⊗ term |
| `Term.dense_dot_product` | `Term.coefficient` | dense data projected onto a term |
| `Component.Normal \| Bundle` | `Component.Single \| Sum` | symmetric case sums with the transpose |
| `Component.dense_dot_product` | `Component.coefficient` | |
| `Component.dot_product` | `Component.inner_product` | |
| `Invariance.t` | `Basis.t` | symbolic basis of the invariant subspace |
| `Invariance.permuted_axes` | `Basis.group_axes` | axes tied per group |
| `Compiler.jit` / `jit_manual` | `Compiler.compile` / `compile_manual` | reserve "jit" for `Rune.jit` |
| `Compiler.compiled` | `Compiler.t` | |
| `compiled.bmv_prod` | `compiled.apply_block` | applies one block to a batch |
| `compiled.block` | `compiled.dense_block` | materialises the block |
| `compiled.factor_estimation` | `compiled.estimate_factors` | verb |
| `compiled.random_transform` | `compiled.transform` | applies group elements; takes perms |
| `compiled.dummy_factors` | (moved to `test/support`) | test-only |
| `ir.v_eq` / `ir.out_eq` | `ir.input_eq` / `ir.output_eq` | |
| `ir.output_identities` | `ir.identity_tensors` | |
| `process_delta_group` | `process_tie` | |
| `compile_bmv_prod*` | `compile_apply*` | |
| `compile_dummy_factors` | (test support) | |
| `compile_factor_estimation` | `compile_estimate_factors` | |
| `merge_primal_transpose` | `bundle_transposes` | |
| `binds_one_side` | `ties_one_side` | |
| `invariance_from_spec` | `basis_of_spec` | |
| `block_dims` | `side_counts` | returns `(n_left, n_right)`, not dims |
| `First_order.estimate_factors` | `First_order.factors_of_params` | from a parameter tree |
| `First_order.estimate_factors'` | `First_order.factors_of_dense` | from the dense surrogate |
| `First_order.build_surrogate` | `First_order.dense_of_factors` | |
| `First_order.expand_surrogate` | `First_order.params_of_dense` | |
| `First_order.jit_with` | `First_order.compile_with` | |
| `Second_order.estimate_factors` | `Second_order.factors_of_pair` | cross-covariance |
| `Second_order.estimate_factors'` | `Second_order.factors_of_dense` | |
| `Second_order.build_surrogate` | `Second_order.dense_of_factors` | |
| `Second_order.mvp` | `Second_order.apply` | |
| `type 'a sized = Small \| Large` | dropped | the function names carry the space |
| `split` / `split_surrogate` | `split_params` / `split_dense` | |
| `Optim.Taylor` | `Optim` | only estimator left |
| `dsvd` | `Solve.svd64` | |
| `relative_damp` | `Solve.damp_spectrum` | `η·s_max + s` |
| `apply_momentum` | `ema` | |
| `update` | `shift` | parameter update |
| `Taylor.step ~info:g` | `step ~grads:g` | |
| `save_mat` | `debug_save` (off by default) | uses `Nx_io.save_npy` |

Concrete notes follow.

**`Index`** — `to_char` currently covers 4 axes per side (`a..d` left, `e..h`
right, plus a shifted alphabet). The port generates chars from a pool
(excluding the batch char `'z'`) and raises a clear error past the limit,
rather than inheriting a silent 4-axis ceiling.

**`Term`** — `inner_product` merges tie groups with a fixed-point loop; port
as is. `coefficient` builds `input_eq` with `Bytes.set` and then `Nx.einsum`:

- `` `full x ``: `view x ~size:(dims.left @ dims.right)`, equation
  `left_chars ^ right_chars ^ "->" ^ free_chars`;
- `` `outer_product (xl, xr) ``: view `xl` by `dims.left`, `xr` by
  `dims.right`, equation `left_chars ^ "," ^ right_chars ^ "->" ^ free_chars`.

Scale by the normalization and reshape to the free dims. `Nx.slice [Nx.I i]`
or `Nx.take` replaces torch's row slicing.

**`Component.design_matrix`** — build a float array as before and
`Nx.create Nx.float32 [| n; n |]`.

**`Compiler`**

- `initial_ir` / `ir_of` / `process_tie`: pure, port verbatim modulo renames.
  Keep the long explanatory comment about the semantics — it is the spec.
- identity tensors: replace `init_nd` with `delta_tensor : int list ->
  Nx.float32_t` built by chaining `Nx.eye` (`δ_ijk… = 1` iff all indices
  equal), reshaped to the left dims, batch axis added, multiplied in.
- `compile_apply_term`: same equations with `Nx.einsum`; `Nx.reshape`,
  `Nx.broadcast_to`, `Nx.mul_s`; `Nx.contiguous` before a reshape if the
  preceding op may return a view.
- `compile_estimate_factors`: `B` in float64, Cholesky (with the jitter/SVD
  fallback of §2.3), solves (or precomputed inverse) at float32; `b` by
  `Component.coefficient`; rows via `Nx.slice`. Every tensor op here is
  traceable.
- `transform`: take per-group `int32_t` permutations (or an `Nx.Rng.key`);
  use `Nx.take ~axis`; apply the same permutation to every axis tied to the
  same `Perm i`.

**`First_order`** (new `orbit.ml`)

- `compile_with dims` and `compile_with surrogate_dims` → `Compiler.t M.t`
  trees;
- `factors_of_params : Nx.float32_t M.t -> Nx.float32_t list M.t` via
  `M.map2 (fun x c -> c.estimate_factors (`outer_product (x, ones))) g large`;
- `dense_of_factors : Nx.float32_t list M.t -> Nx.float32_t` by folding
  `small.dense_block` over the leaves in traversal order and
  `Nx.concatenate ~axis:0`;
- `factors_of_dense : Nx.float32_t -> Nx.float32_t list M.t` by splitting the
  dense vector at precomputed offsets and mapping `small.estimate_factors`;
- `params_of_dense` and `orbit_average` accordingly;
- precompute leaf offsets from `prod surrogate_dims` with one `M.fold`.

**`Second_order`**

- `compile_with ~symmetric dims`: pair matrix `Compiler.t array M.t` (row per
  leaf, array over the flat index of the second leaf);
- `factors_of_pair`/`factors_of_dense`, `dense_of_factors` (concatenate rows
  into a block matrix, optional `0.5 · (X + Xᵀ)` when symmetric), `apply`:
  for each leaf `i`, sum over `j` of `c_ij.apply_block ~factors:f_ij v_j`,
  where `v_j` comes from `M.fold` in traversal order and the batch axis is
  added as in the old code;
- drop `symmetrize_factors` (dead) and `damped_inverse` (commented out; it
  belonged to `Global`).

**`Optim` and `Solve`** (new `optim.ml`, `solve.ml`)

- `Solve.svd64`, `Solve.symmetric_power` (damped `X^{±1/2̃}`), and
  `Solve.hessian_inverse ~damping sigma_w sigma_g` implementing §1.3; the only
  non-traceable module.
- `Optim` = the `Taylor` update, split for jit exactly as §3.5:
  `prepare`, `solve`, `finish`, plus the eager `step` composing them;
- state holds the parameter tree, the momentum tree, the dense
  `sigma_g_avg` and the two bias counters (tensors, so the state threads
  through compiled code);
- `shift` applies `θ ← θ - η·δ` leafwise (bounds to be added later);
- `ema`, `debias`, `Solve.damp_spectrum` as in the old code;
- keep `debug_save` optional and off by default.

---

## 5. Tensor-op mapping

| old | new |
| --- | --- |
| `Tensor.shape` | `Nx.shape` |
| `Tensor.reshape` / `Tensor.view` | `Nx.reshape` (add `Nx.contiguous` where needed) |
| `Tensor.transpose ~dim0 ~dim1` | `Nx.swapaxes` / `Nx.transpose ~axes` |
| `Tensor.concat ~dim` | `Nx.concatenate ~axis` |
| `Tensor.split_sizes ~dim ~split_size` | `Nx.array_split ~axis:(`Indices …)` |
| `Tensor.slice ~dim ~start ~end_ ~step` | `Nx.slice [Nx.Rs (start, stop, step)]` / `Nx.I i` |
| `Tensor.index_select ~dim ~index` | `Nx.take ~axis ~indices` |
| `Tensor.einsum ~path:None ~equation xs` | `Nx.einsum equation xs` (frontend decomposition; jittable) |
| `Tensor.matmul`, `C.( *@ )` | `Nx.matmul` |
| `Tensor.broadcast_to` | `Nx.broadcast_to` |
| `Tensor.rand/randn`, `Owl` rand | `Nx.randn`/`Nx.rand` with an explicit `Nx.Rng.key` |
| `Mat.eye`, `Owl.Dense.Ndarray.S.init_nd` | `Nx.eye` + a `delta_tensor` helper |
| `to_bigarray ~kind:float32` | `Nx.cast Nx.float64` (and back) |
| `Owl.Linalg.D.svd` — design-matrix pinv (compile time) | `Nx.cholesky` + `Nx.solve_triangular` (fallback: `Nx.svd`) |
| `Owl.Linalg.D.svd` — estimator (per step, host) | `Nx.svd` on float64; host-side, not jitted |
| `Owl.Dense.Matrix.S.save_txt` | `Nx_io.save_npy` (debug, off by default) |
| `Owl.Algodiff.S` (`value_and_grad`, `jvp`, `hessian_v`) | `Rune.value_and_grad`, `Rune.jvp`, `Rune.hvp`/`hessian'` |
| `Prms.T` map/fold/iter/iter2 | `Nx.Ptree.Uniform` map/map2/iter/fold/fold2 |
| `Prms.Pinned/Free/Bounded`, `P.value` | dropped in phase 1 (see §6.6) |
| `Torch.Device.t`, `to_device`, `Tensor.grad` | dropped; Rune handles devices later, AD replaces `.grad` |
| `C.( $* )`, `AD.Maths.*` | direct `Nx` arithmetic |

Removed with the dropped optimizers: `Owl.Linalg.D.lyapunov` (continuous
Sylvester), `Owl.Linalg.D.eig`/`damped_inverse`, `ssvd`.

---

## 6. Design decisions and deviations

1. **Keep the compiler's math untouched** except for the factorization swap in
   factor estimation. Char counting, normalization, partition enumeration,
   `ties_one_side`, term sorting and `bundle_transposes` are ported as-is;
   only tensor calls and names change. Any further "improvement" gets its own
   explicitly documented commit and a test.
2. **Metadata live in uniform trees.** Per-leaf compiled closures and dims
   specs travel with the parameter tree, so `map2`/`fold2` replace the old
   index bookkeeping (`Int.incr i; P.iter …`). Flat arrays survive only for
   the pair matrix of `Second_order` and the dense surrogate.
3. **Derive `surrogate_dims`.** `Symmetry.collapse_dims` sets every `Perm`
   axis to `1`. This removes a user-supplied parameter that must otherwise be
   kept in sync with `dims` and `symmetries`. Verify against exact orbit
   averages before deleting the explicit form.
4. **Cholesky instead of SVD pinv for the design matrix**, with a compile-time
   jitter/SVD fallback for rank-deficient bases (§2.3). Rationale: `B` is a
   Gram matrix, so Cholesky is the natural factorization; the solves are
   traceable; the old double-precision SVD dependency disappears from the
   construction path.
5. **One host boundary: the estimator's `svd64`.** The step is written as
   `prepare` (traced) → `solve` (host) → `finish` (traced), so the only
   non-traceable op sits on the small dense surrogate matrices. No `Nx.item`,
   value-dependent branching, dynamic shapes or RNG inside the traced halves.
6. **No debug prints.** `basis_of_spec` returns components; logging is the
   caller's business. `debug_save` is opt-in.
7. **Parameter constraints are a separate, later layer.** `prms`'s
   `Pinned/Free/Bounded` is orthogonal to symmetry. Modern plan: a uniform
   tree of constraints (`[`Free | `Bounds of float * float ]`) applied
   leafwise in `shift`, or vega-style gradient transforms. Until then, every
   float leaf moves.
8. **Implement eager first, then jit.** Milestones M3 (eager) and M4 (jitted)
   share the same `prepare`/`solve`/`finish` decomposition, so M4 adds only
   `Rune.jit2` wrappers and parity tests; nothing is rewritten to become
   traceable.
9. **Reuse `nx.io`** for debug dumps, off by default.
10. **Keep the established module names** (`Sides`, `Index`, `Term`,
    `Component`, `Basis`, `Compiler`, `First_order`, `Second_order`) so the
    diff against `old_code` stays reviewable; §4.1 lists the proposed
    function/field renames.

---

## 7. Validation plan (no old-code compilation)

The old implementation cannot be run here (missing deps), so correctness rests
on references computable with raven alone:

**Pure front-end (no tensors)**

- term sorting/equality is order-invariant; `transpose ∘ transpose = id`;
- `partitions` produces exactly the set partitions of size ≥ 2 (check counts
  against Bell numbers minus singleton cases for small `n`);
- normalization weights are symmetric and positive; `inner_product` is
  commutative.

**Compiler against brute force (the critical tests)**

- For small dims (2–4) and one tensor, enumerate the **whole group** (all
  permutations of `S_n` for `n ≤ 4`) and compute exact `R1`/`R2`; compare with
  `orbit_average` and with `dense_of_factors (factors_of_dense X)`. This
  validates partitions, the IR, normalization and factor estimation together;
- larger groups: Monte-Carlo the orbit average with a fixed RNG key and
  compare within tolerance;
- `dense_block ~factors` and `apply_block ~factors v` against the same exact
  averages;
- invariance: `transform`ed blocks satisfy `A S Aᵀ = S`;
- round-trips: `estimate_factors (outer_product …)` of a transformed parameter
  equals the transformed factors;
- `Second_order.apply ~factors v` against the dense surrogate matrix times the
  vectorised `v` for small trees.

**Cholesky estimation**

- for random PD Gram matrices, `solve L (solve Lᵀ b) ≈ B_inv b ≈` the old SVD
  pinv formula, within tolerance;
- singular `B` (degenerate dims) takes the jitter/fallback path and still
  reproduces the old pinv result;
- the factorization is compile-time: assert no `Nx.svd` runs during a step
  (covered by the jit parity test below).

**Optimizer (`Taylor` only)**

- check `H_inv` against a dense reference on an invariant quadratic
  `L(w) = ½ (w - w*)ᵀ H (w - w*)` with `H` in the commutant, and check the
  exchange identity `S_g ≈ H S_w H` numerically;
- check the step against a direct Newton step in the invariant subspace;
- check `ema`/`debias` schedules against hand-computed values (including the
  old `max 1e-4` floor on the bias counters);
- end-to-end: train a small local fixture (a 2-layer record defined in the
  test, no library `Models`) on a regression with `Rune.value_and_grad`; the
  loss decreases and gradients come back in the parameter tree's structure.

**JIT**

- parity: eager `Optim.step` and `Compiled.step` agree over several
  iterations (state in, state out) within float32 tolerance;
- traceability: the jitted halves raise no `Jit_error`, trace once (repeated
  calls with varying inputs reuse the compilation — observable through
  timing or `Rune` debug env vars), and accept the whole state as input
  leaves;
- captured constants: the Cholesky factors are captured, not retraced, and
  do not mutate between calls.

**AD integration**

- `Rune.value_and_grad` gradients checked against finite differences on the
  fixture model;
- gradient equivariance (`∇L(Aw) = A∇L(w)`) checked with `transform`.

Where a PyTorch implementation of the paper is available, cross-check factor
values on shared inputs; otherwise the brute-force group tests above are the
primary oracle.

---

## 8. Milestones

**M0 — pure front-end (done).** `Sides`, `Index`, `Symmetry`, `Basis`, `Term` (pure
parts), `Component` (pure parts). Tests as in §7 (first bullet). Acceptance:
`dune runtest` green; the diff against old files is mechanical modulo renames.

**M1 — compiler on Nx (done).** `Term.coefficient`, `Compiler`
Cholesky estimation/`dense_block`/`apply_block`/`transform`. As built, the IR
and identity tensors were replaced by delta operands and pairwise contraction
(§10).
Tests: dense brute force on one tensor for specs `[Id;Perm 0]`,
`[Perm 0;Perm 0]`, `[Perm 0;Perm 1]`, `[Id;Perm 0;Id;Perm 0]`, plus the
Cholesky-vs-SVD equivalence tests. Acceptance: every compiled closure matches
the exact enumeration.

**M2 — orbit machinery.** `First_order`, `Second_order`, offset/surrogate
layout, `surrogate_dims` derivation. Tests: single- and multi-leaf trees,
hidden-permutation and autoencoder specs, `S_n` enumeration for `n ≤ 4` plus
MC for larger `n`. Acceptance: `orbit_average` and the surrogate round-trips
within float32 tolerance; `A S Aᵀ = S`.

**M3 — eager Taylor step.** `Solve` (host `svd64`, damped symmetric powers,
`hessian_inverse`), `Optim.prepare`/`solve`/`finish`/`step`, pure and
structured for jit. Tests: invariant quadratics, exchange identity, EMA/
debias schedules, end-to-end fixture. Acceptance: Newton agreement on
quadratics; stable training on the fixture.

**M4 — jitted step.** `Compiled.prepare`/`finish`/`step` via `Rune.jit2`,
state threaded as input/output leaves. Tests: eager/jit parity, no
`Jit_error`, no retracing across iterations. Acceptance: bitwise-ish parity
over 100 steps and a measurable speedup on a larger fixture.

**M5 — polish and performance.** Write `symo.mli` with the doc style of
`sofo.mli`; memoise `basis_of_spec` per `(spec, dims)`; profile the step and
decide between the solve and precomputed-inverse variants of §2.3; optional
`donate`/beam tuning. Examples live outside the library, later.

---

## 9. Open questions / risks

1. **`Nx.einsum` fidelity.** The generated equations include repeated inputs,
   scalar operands, a synthetic batch char and broadcasting via views. M1
   starts by pinning down exactly which forms `Nx.einsum` accepts and how it
   handles rank-0/1 operands; fall back to `Nx.matmul`+`reshape` rewrites where
   the equation is unsupported.
2. **Cholesky failure policy.** Jitter size, whether to warn or raise when the
   fallback triggers, and whether the fallback result should be exposed for
   inspection. Degenerate dims are the concrete case to test.
3. **`ties_one_side`.** Ported as an empirical filter; keep the old comment and
   add a test showing what breaks if it is removed.
4. **`surrogate_dims`** derivation needs numerical confirmation before the
   explicit user argument is removed.
5. **Combinatorial compile cost.** Partition enumeration is factorial in the
   number of tied axes; second order compiles `n²` entities. Memoisation per
   `(spec, dims)` and a component-count warning are probably needed for
   realistic models.
6. **Leaf order stability.** The folded surrogate layout assumes `fold`
   visits leaves in the same order every time (`ppx_ptree` guarantees this for
   records/lists, but lists must not change shape between compile and run).
7. **`Nx.svd` float32/double split** may differ from Owl's LAPACK path in the
   last digits; tests use tolerances, not bit equality.
8. **`learning_rate : float option`.** The old "None = measure only" is a
   convenient switch; decide later whether to keep it or split
   `direction`/`shift` so a measurement never has to fabricate a state step.
9. **`Pinned/Bounded`** semantics (e.g. re-parameterised parameters) must be
   recovered eventually; confirm they are not needed for the first port.
10. **JIT coverage.** Confirm that every op in the traced halves lowers
    (`einsum` decomposition, `slice`, `take`, `concatenate`, `cholesky`,
    `solve_triangular`); if a lowering is missing, the offending step moves to
    the host half or is rewritten.
11. **Full-step jit** (Newton–Schulz square roots + power-iteration damping)
    is an explicit non-goal now; revisit only if the host solver shows up in
    profiles.

---

## 10. Implementation notes: Nx `einsum` defects (M0–M1)

Two defects in `Nx.einsum` surfaced while porting the compiler; both are
worked around in `Delta`/`Contract`, and both look worth fixing upstream.

1. **Repeated labels within one operand contract the wrong axes.** Nx extracts
   a repeated label's diagonal by calling `diagonal`, which moves the diagonal
   axis to the *end* of the shape, while the label list drops the duplicate in
   place (`nx/lib/core/frontend.ml`, the `process` loop of
   `handle_repeated_indices`). Minimal reproductions:

   ```ocaml
   (* [1..8] reshaped [2;2;2]: Σ_i T[i,i,j] is [8;10] *)
   Nx.einsum "iij->j" [| t |]      (* Nx gives [3;15] *)
   (* a [2;2;2;2] arange: Σ_b A[a,b,a,b] is [5;25] *)
   Nx.einsum "abab->a" [| a |]     (* Nx gives [11;19] *)
   ```

2. **Multi-operand contractions can reshape a non-contiguous intermediate.**
   `contract_pair` transposes an intermediate to the output order, and a later
   contraction reshapes it without materializing it, raising
   `Invalid_argument("reshape: cannot reshape …, incompatible strides …")`
   (`nx/lib/core/frontend.ml:3397`).

Workaround (implemented, `lib/delta.ml`, `lib/contract.ml`):

- `Delta.tensor ~order dim` builds the tie tensor;
- `Contract.binary` contracts exactly two operands and materializes the result;
- `Contract.permute_sum` does the final permutation and summation manually;
- the compiler contracts deltas one at a time, tracking labels as characters.

Every contraction is binary and every operand is contiguous, so the jit story
is unchanged: `einsum` still lowers to primitives, and `contiguous` is a
no-op where Nx would otherwise fail.

Status: M0 and M1 are done. `symo/lib/` now has `Sides`, `Index`, `Symmetry`,
`Term`, `Component`, `Basis`, `Delta`, `Contract` and `Compiler`, with 18 tests
in `symo/test/`, all green. Next: M2 (`First_order`/`Second_order` orbit
machinery), then the `Taylor` step and its `Rune.jit` halves.

---

## Appendix A — `old_code/` inventory

| file | lines | notes |
| --- | ---: | --- |
| `sides.ml` | 7 | `{ left; right }` |
| `index.ml` | 87 | `Left/Right`, chars, dims/axes lookup, filters |
| `symmetry.ml` | 23 | `Nil \| I \| P of int`, labels |
| `invariance.ml` | 20 | label, components, group axes |
| `term.ml` | 174 | ties + free axes; symbolic and dense dot products |
| `component.ml` | 35 | `Normal`/`Bundle`; design matrix |
| `compiler.ml` | 440 | spec → components → einsum closures |
| `models.ml` | 263 | MLP/RNN definitions — **out of scope for the library** |
| `symo.ml` | 623 | `Make`: AD plumbing, `First_order`, `Second_order`, `Taylor`, `Global`, `Original` |
| `symo.mli.unused` | 79 | earlier public API (`beta_21`/`beta_22`, `info` triple) |
| dead code | — | empty `Adam` functor, unused `symmetrize_factors`, commented `damped_inverse`, `ssvd` |

`Global` (continuous Lyapunov), `Original` (Eq. 11) and the model definitions
are deliberately not ported.

## Appendix B — the pipeline, compressed

```
Symmetry.spec list Sides.t
  │  basis_of_spec
  ▼
Basis.t { components = Single/Sum Term.t list; group_axes }
  │  compile_apply / compile_estimate_factors / dense_block
  ▼
Compiler.t (einsum closures; Cholesky solves for estimation)
  │  First_order: factors_of_params → dense_of_factors → params_of_dense
  │  Second_order: factors_of_pair → dense_of_factors → apply
  ▼
orbit averages R1(w), R2(δw, δw), R2(δg, δg)
  │  Taylor/Eq. 10: damped symmetric powers and H_inv  [host: svd64]
  ▼
θ ← θ − η · H_inv · ḡ , with EMA + debiasing + relative damping

jit boundary: everything above except the bracketed svd64 is traceable;
the step is prepare (jitted) → solve (host) → finish (jitted).
```
