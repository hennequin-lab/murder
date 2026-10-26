# SYMO — size-independent curvature: blocks, transport, and the end of `ensure_size_invariance`

Status: **implemented** (M1–M2 and the `k ≤ 2` part of M3; see the "As built"
note below). This document is the response to the
[blog post](https://claude.ai/artifact/PtY9gVYZkTDnQx7o9iq2mA) *Inverting an
n² × n² equivariant matrix with a 2×2 and a 3×3 inverse* (a PDF copy is at
`~/symo-brainstorm.pdf`), which gives the principled solution to the
surrogate-spectrum problem that `ensure_size_invariance` papers over.

## As built

- `lib/kind.ml`: the isotypic kinds of one permutation group for `k ≤ 2` and
  their canonical (orthonormal) test vectors, built from the `1`/`W` split,
  the `W`-copies (row, column, diagonal), the `4`-cycle for `S(n-2,2)` and the
  projected antisymmetric vector for `Λ²W`.
- `lib/blocks.ml`: the per-group tables, the full factor↔block map and its
  inverse, the Gram matrices (identity, since the copies are orthonormal),
  `block_matrices`/`blocks_of_matrices`, and `transport`.
- `lib/compiler.ml`: the `ties_one_side` filter and the
  `ensure_size_invariance` flag are gone; the basis is the full commutant
  basis.
- `lib/orbit.ml`: the surrogate (`surrogate_dims`, the `small` compilers) is
  gone. `Second_order` builds the per-pair `Blocks.t`, exposes
  `kinds`, `blocks_of_factors`/`factors_of_blocks`, `transport`, and packs
  the factor lists into one flat tensor (`pack`/`unpack`) so the state and
  the jit boundary stay tensor-valued.
- `lib/solve.ml`: `hessian_inverse_blocks` applies Eq. 10 blockwise, each
  damped power with its own matrix's largest singular value, as the dense
  formula does.
- `lib/optim.ml`: `prepare` carries the packed factors of `S_w`/`S_g` (and the
  EMA of `S_g` in factor space), `solve` runs the blockwise estimator on the
  host, `finish` applies the factors at full size.
- Tests: `test/test_blocks.ml` (spectrum, transport, round trips, free axes,
  several groups, blockwise vs dense estimator) plus an Orbit-level transport
  test in `test_optim.ml`; the old suites pass unchanged.
- Remaining: the general `k ≥ 3` kinds and the numerical-Wedderburn route
  (M3), and a fully jitted solve (M4). `Blocks.build` raises a clear
  `Invalid_argument` for `k ≥ 3` and for dims below the stable range.

Reference for the algorithm (unchanged):

> Artemev, Xia, Boyd, Yu, Dangel, Hennequin, Bernacchia,
> *Exploiting weight-space symmetries for approximating curvature*, ICML 2026,
> [arXiv:2606.00442](https://arxiv.org/abs/2606.00442).

---

## 1. Summary

The estimator currently inverts curvature in a **small surrogate** built with
the *same diagram-basis factors* as the full-size operator. That is wrong:
the factors are a size-dependent encoding of the commutant algebra, so
`Φ_s(w) ≠ Φ_n(w)` and the surrogate's spectrum is not the full-size spectrum.
`ensure_size_invariance` deletes the terms (ties within one side) for which
the mismatch shows, at the cost of discarding genuine curvature information.

The fix is to represent curvature operators by their **blocks** — the
Wedderburn coordinates of the commutant algebra, whose sizes are the
multiplicities of the isotypic kinds — and to **transport** between sizes by
re-encoding blocks:

```
blocks_n = Φ_n(w_n)            (decode)
w_m      = Φ_m^{-1}(blocks_n)  (re-encode at size m)
```

`Φ_r` is an algebra isomorphism for every `r` in the stable range
(`r ≥ k + ℓ` per group, i.e. `r ≥ 2k` for square specs). Transport therefore
preserves products, inverses, distinct eigenvalues, invertibility and any
function of the operator; only multiplicities (hence trace, determinant,
norms) change with size.

This plan removes the surrogate from the optimizer step entirely and computes
the estimator's matrix functions **blockwise**, on matrices of size equal to
the multiplicities (2, 3, … — for the paper's models). It deletes
`ensure_size_invariance` and the `ties_one_side` filter, and adds a public
transport capability for size generalization.

---

## 2. Diagnosis: where the size dependence lives

### 2.1 The current pipeline

`Optim.prepare → Solve.hessian_inverse → Optim.finish`:

1. **Estimate factors at full size.** `Second_order.factors_of_pair` runs
   `Compiler.estimate_factors` on each leaf pair at the leaf's real `dims`:
   a least-squares projection of `δ_w`/`δ_g` onto the diagram basis (design
   matrix `B_ij = <c_i, c_j>` from `Component.design_matrix`).
2. **Assemble a dense surrogate at size s.** `Second_order.dense_of_factors`
   uses the `small` compilers (`surrogate_dims`: every `Perm` axis collapses
   to `Model.surrogate_dim`, 2 or 4) and the *same factors*, giving
   `S_r = Σ_i w_i F_i(r)` at `r = s`.
3. **Host-side functional calculus on the surrogate.** `Solve` runs
   `eigh`/symmetric powers on the dense `s² × s²` (or leaf-concatenated)
   matrix, after `Nx.place Nx.Placement.host`.
4. **Re-estimate factors at size s and apply at full size.**
   `factors_of_dense` solves for the factors of `H_inv` at the surrogate
   dims, and `apply` (with the `large` compilers) uses them at full size.

Steps 2 and 4 assume the diagram-basis factors mean the same operator at
sizes `n` and `s`. They do not.

### 2.2 Why: the diagram basis is a size-dependent encoding

For a fixed spec, the diagram basis spans the commutant algebra `A_r` of the
group action at size `r`. By Schur's lemma (and, for permutation groups, the
partition algebra),

```
A_r ≅ ⊕_λ M_{m_λ(r)}        (kinds λ = isotypic components, m_λ = multiplicity)
```

and the map

```
Φ_r : (diagram-basis weights w) ↦ (blocks B_λ ∈ M_{m_λ})_λ
```

is an algebra isomorphism **whose entries depend on `r`** (through `√(r−1)`,
`√(r−2)`, `1/r`, …, from the overlap between the diagonal and the spread-out
directions). The blocks `B_λ` are the size-independent description of the
operator; the weights are its size-dependent coordinates.

So `Φ_s(w) ≠ Φ_n(w)` in general. The surrogate is a different operator, with
a different distinct spectrum, and inverting it does not give the inverse of
the full-size operator. The `ties_one_side` filter keeps only the terms whose
blocks happen to agree across sizes; the old code marked this as empirical
("to be confirmed theoretically") and it is over-aggressive.

### 2.3 Numerical evidence

A scratch program (`[P;P]` spec on both sides, one group, `symmetric:true`,
random factors) compared the dense operators at `n = 8` and at the surrogate
`s = 4` built with the **same factors**:

```
full  n=8: 0.1921(x1) 0.4224(x7) 0.6216(x7) 0.9939(x21) 1.2779(x20) 3.6183(x7) 4.5973(x1)
surro s=4: 0.3750(x3) 0.4201(x1) 0.8784(x3) 0.9939(x3) 1.2779(x2) 3.7066(x3) 4.1343(x1)
```

The multiplicities match the theory exactly at `n = 8`
(`1, 1` for the 2×2 block; `7, 7, 7 = n−1` for the 3×3 block;
`20 = n(n−3)/2` for `S`; `21 = (n−1)(n−2)/2` for `A`), and the *distinct*
eigenvalues differ at `s = 4`. This is the bug.

### 2.4 A second, independent problem: the surrogate size is below the stable range

The partition algebra `P_k(r)` is semisimple only for `r ≥ 2k` (for the
two-sided stable range `r ≥ k + ℓ`). Below it the diagram elements are not
linearly independent — for `k = 2`, 8 of the 15 survive at `r = 2` and 14 at
`r = 3` — so `Φ_r` is not an isomorphism and no transport exists. The
default `surrogate_dim = 2` is therefore below the stable range for every
second-order spec with two permuted axes (the RNN's `w` leaf), and even
`surrogate_dim = 4` is only borderline for `k = 2` and too small for `k = 3`.

The block architecture avoids this entirely by working at the model's real
size (where `n ≥ 8` in practice); transport, where needed, is between two
*stable* sizes.

---

## 3. The principle: blocks and transport

For a spec with `k` permuted occurrences on one side (per group), the space
decomposes into isotypic components indexed by partitions `λ`:

```
V^{⊗k} ≅ ⊕_λ S^{[n−|λ|, λ]} ⊗ M_{λ}        (|λ| ≤ k, M_λ the multiplicity space)
```

- the **kind** is `λ`; its dimension is `d_λ(n) = dim S^{[n−|λ|,λ]}` (a
  polynomial in `n`; the multiplicities `m_λ` are independent of `n` in the
  stable range);
- an equivariant operator acts as `I_{d_λ} ⊗ B_λ` on kind `λ`; the **block**
  `B_λ` is a `m_λ × m_λ` matrix (rectangular `m_{L,λ} × m_{R,λ}` for mixed
  specs);
- the diagram-basis weights number `Σ_λ m_λ² = B_{k+ℓ}` (Bell number), the
  blocks' entries number the same;
- **transport** `Ψ_{n→m} = Φ_m^{-1} ∘ Φ_n` is an algebra isomorphism for
  stable `n, m`, so
  `f(X_m) = Ψ_{n→m}(f(X_n))` for the inverse, powers, square roots, resolvents
  and every other function of the operator. Multiplicities `d_λ(n)` change,
  so `tr`, `det` and norms do not transport.

### 3.1 The worked case (`k = ℓ = 2`, one group) — the blog's formulas

Kinds and multiplicities in `V^{⊗2}`: trivial ×2, standard `W` ×3,
`S(n−2,2)` ×1, `Λ²W = S(n−2,1,1)` ×1; blocks `R` (2×2), `T` (3×3),
`σ_S`, `σ_A`; `15 = 2² + 3² + 1 + 1 = B_4`. With `a = √(n−1)`, `b = √(n−2)`
and the blog's weight vector `w_0 … w_14` (its diagram basis):

```
R = [ w0+…+w9+w11+w12+w13 + (w10+w14)/n      a(w11+w12+w13 + (w10+w14)/n) ]
    [ a(w7+w8+w9 + (w10+w14)/n)              w0+w2 + (n−1)/n (w10+w14)    ]

T = [ w0+w3+w8+w12+w14/n    w2+w5+w9+w12+w14/n   b(w12+w14/n) ]
    [ w2+w6+w8+w13+w14/n    w0+w4+w9+w13+w14/n   b(w13+w14/n) ]
    [ b(w8+w14/n)           b(w9+w14/n)          w0+w2+(n−2)/n w14 ]

σ_S = w0 + w2        σ_A = w0 − w2
```

`Φ_n^{-1}` is triangular in the same ordering (blog §6): each line reads one
block entry backwards, so no linear system has to be solved.

These formulas are for the blog's *orthonormal* basis; SYMO's diagram basis
has its own normalization (`Term.normalization = 1/√D`), so the M0 spike must
establish the basis correspondence before using them as a test oracle.

### 3.2 General specs

- **Several groups.** Kinds are tuples `(λ^{(g)})_g`, one partition per group;
  multiplicities multiply. This is the tensor product of the per-group
  partition algebras.
- **Free (`Id`) axes.** A free axis of dimension `f` is a trivial
  representation: it multiplies the multiplicity of every kind by `f` (and
  the block entries carry the free-axis indices, as the factors do today).
- **`Absent` padding.** A side with no axes has `k = 0`: only the empty kind,
  multiplicity 1.
- **Several leaves.** A second-order operator is a block matrix over leaf
  pairs `(i, j)`; for kind `λ`, assemble the pair blocks into the
  `(Σ_i m_{i,λ}) × (Σ_j m_{j,λ})` matrix indexed by `(leaf, copy)`. Functions
  act on these assembled blocks; splitting back gives the pair factors.
  This is exactly the current `Second_order.factors = float32_t list array t`
  layout, with the extra kind index.

---

## 4. Target architecture

```
Compiler          Orbit.Second_order            Solve                Optim
---------         -------------------            -----                -----
kinds + Φ_r       blocks_of_factors              damped powers        prepare: factors -> blocks
factors <->       factors_of_blocks              on each assembled    solve:   blockwise H_inv
blocks            transport                     block                finish:  blocks -> factors
transport         (assembled over leaves)                             apply at full size
```

- **`Compiler`** gains the kind/block metadata for a `(spec, dims)` pair and
  the linear maps `factors_to_blocks` / `blocks_to_factors`; both are
  `D × D` linear maps (matmuls on the factor coordinates), hence traceable.
  `dims` may be any *stable* size; `transport ~from ~to` composes
  `Φ_to^{-1} ∘ Φ_from`.
- **`Orbit.Second_order`** gains `blocks_of_factors` / `factors_of_blocks`
  (assembling per-kind matrices across leaves) and `transport_factors`.
  The `small`/`large` split survives only where it is genuinely useful
  (tests, debug), not in the step.
- **`Solve`** runs `eigh` and the damped symmetric powers **per block**:
  sizes 2, 3, …, far below the current dense surrogate. The host boundary
  stays (it is small), and a later milestone may replace it with closed
  forms or Newton–Schulz iterations for a fully jitted step.
- **`Optim`** carries blocks in `Mid` instead of dense `sigma_w`/`sigma_g`;
  `prepare` converts factors to blocks, `finish` converts `H_inv` blocks back
  to factors and applies them at full size.
- **`Model`** loses `surrogate_dim` and `ensure_size_invariance`.
  `Symmetry.surrogate_dims` and `Orbit.surrogate_dims` leave the step
  (kept as test utilities if useful).

### 4.1 Why this is also the minimal correct fix

Even if one keeps the surrogate (smallest possible change), the surrogate
must be built with **transported** factors `w_s = Ψ_{n→s}(w_n)`, and its
inverse transported back, and `s` must be in the stable range. That is
Milestone M1 below. M2 goes further and removes the surrogate, which is
simpler, cheaper and makes the size independence structural rather than
compensated.

---

## 5. Computing the kinds and Φ

Three routes. They are complementary; M0–M2 use A, M3 adds B, and C is an
independent oracle.

### Route A — canonical test vectors (recommended for M0–M2)

For a spec pair at stable size `r`, construct a small set of canonical test
tensors spanning the isotypic components with **consistent copy labels**:

- per group, split each occurrence into `1` vs `W` via `P_1 = J/r` and
  `P_W = I − J/r`;
- inside `W^{⊗j}`, use Young symmetrizers / the seminormal basis to label
  copies by `λ` (the `W^{⊗j}` decomposition is the Brauer-algebra structure,
  not just Young projectors, because of the contractions);
- tensor in the other groups, the free axes and the other side.

For each basis component `F_i` the block coordinates are the inner products
`⟨v_a, F_i v_b⟩`; stacking them gives `Φ_r` as a `D × D` float64 matrix at
compile time, inverted by a small solve. The same construction at any stable
`r` gives `Φ_r`; copy labels are canonical across sizes and leaves.

*Pros*: fits the existing compiler (which already applies basis elements to
vectors), no algebra decomposition, gives the copy labels needed for
multi-leaf assembly. *Cons*: the test-vector construction needs care for
general `k`.

### Route B — structure constants + numerical Wedderburn (general endpoint)

- Compute the algebra's multiplication table in the diagram basis
  **combinatorially**: composing two diagrams gives a diagram times
  `r^{#loops}`; for several groups/leaves the composition sums over the
  middle leaf and the middle indices, giving coefficients that are products
  of dimensions.
- From the structure constants, compute the center `Z(A)` (solve
  `[z, F_i] = 0` for all `i`), its primitive idempotents `z_λ`, and a basis of
  each simple component `z_λ A`; this is `Φ_r`. Since the structure constants
  are functions of `r`, the same computation gives `Φ_m`.

*Pros*: fully general (all specs, groups, leaves, `Absent`, free axes), no
representation-theoretic case analysis, compile-time only, `D` is small
(15 for `k = 2`, 203 for `k = 3`). *Cons*: implementing a robust numerical
Wedderburn decomposition and the diagram composition; needs the stable range.

### Route C — matrix-free functional calculus (oracle, not transport)

Compute `f(S)` directly at full size from the minimal polynomial: build the
Krylov sequence `v, Sv, S²v, …` (each application is the compiled
`apply_block`, `O(N)`), find the first linear dependence (degree ≤ `D`),
interpolate `f` at the distinct eigenvalues (roots of the small companion
matrix), and apply the resulting polynomial to `g_avg`. No kinds, no `Φ`,
works at any size including the non-semisimple range.

*Pros*: simplest to implement; an independent oracle for Routes A/B; a
fallback for specs below the stable range. *Cons*: no blocks/transport; a
numerical rank decision; not the target architecture.

### Recommendation

| milestone | route |
|---|---|
| M0 | A, for `k ≤ 2`, single group, as a spike |
| M1–M2 | A, extended to the specs used by the repo's models |
| M3 | B, for the general case |
| all | C as a test oracle (blockwise `f(S)` == matrix-free `f(S)`) |

---

## 6. Milestones

### M0 — validation spike (no library changes)

- Keep the diagnostic (same factors at `n` and `s` give different distinct
  spectra) as a test; it documents the bug.
- Implement Route A for the `k = 2`, single-group square spec in a scratch
  module: build the blog's isotypic basis numerically at any stable `r`,
  compute `Φ_r`, and check `Φ_r` against the blog's `R`, `T`, `σ_S`, `σ_A`
  formulas (after fixing the basis correspondence and normalization).
- Check transport: `Φ_m(Ψ_{n→m} w) = Φ_n(w)` and equal distinct spectra.
- Go/no-go for M1.

### M1 — transport in the existing surrogate pipeline (minimal correct fix)

- `Basis`/`Compiler`: kinds, `Φ_r`, `Φ_r^{-1}`, `transport` for the specs
  used by the models (`[I]`, `[P]`, `[P;P]`, `[P;I]`, `[I;P]`, …).
- `Orbit.Second_order`: `transport_factors`; `dense_of_factors` transports
  `w_n → w_s` before assembling the surrogate; `factors_of_dense` transports
  the inverse's factors `s → n` before returning.
- Require a **stable** surrogate size (`s ≥ max over groups of k_g + ℓ_g`),
  replacing the fixed `surrogate_dim = 2`; keep the old path behind a flag
  for A/B tests.
- Delete the `ties_one_side` filter and `ensure_size_invariance` from the
  call sites.
- Tests: blog formulas, transport identities, dense cross-checks, existing
  suites green; the diagnostic test flips from failing to passing.

### M2 — estimator on blocks (target architecture)

- `Solve`: blockwise damped symmetric powers on each assembled block.
- `Optim`: `Mid` carries blocks; `prepare`/`finish` use
  `blocks_of_factors`/`factors_of_blocks`; no dense surrogate in the step.
- Delete `Model.surrogate_dim`, `Symmetry.surrogate_dims`,
  `Orbit.surrogate_dims` (or keep the last as a test utility) and the
  `small` compilers from the step.
- Tests: blockwise `H_inv` == dense `H_inv` at small `n`; eager/jit parity;
  `student_teacher` and the RNN tutorial train; size-generalization test.
- Docs: `symo.mli`, README, this plan's status line.

### M3 — general specs (Route B)

- Structure constants + numerical Wedderburn; kinds for several groups, free
  axes, mixed `k_L`/`k_R`, several leaves; assembled blocks across leaves.
- Cross-check Route B against Route A on all specs where A exists; dense
  cross-checks; property tests for every spec in the repo.

### M4 — polish

- Closed-form or blockwise-jittable solve (2×2/3×3), possibly a fully jitted
  step.
- Benchmarks: block solve vs the surrogate `eigh`; step time; memory.
- A short `doc/` note on the block algebra and transport, derived from the
  blog.

---

## 7. API sketch

```ocaml
(* Compiler *)
module Kind : sig
  type t (* per group: a partition λ; plus the free axes *)
end

type block =
  { kind : Kind.t
  ; matrix : Nx.float32_t (* m_L × m_R in an orthonormal copy basis *)
  }

type t =
  { basis : Basis.t
  ; dims : int list Sides.t
  ; kinds : Kind.info Sides.t      (* kinds and multiplicities per side *)
  ; apply_block : ...
  ; dense_block : ...
  ; estimate_factors : ...
  ; transform : ...
  ; factors_to_blocks : Nx.float32_t list -> block list
  ; blocks_to_factors : block list -> Nx.float32_t list
  }

val transport :
  from_dims:int list Sides.t ->
  to_dims:int list Sides.t ->
  Nx.float32_t list ->
  Nx.float32_t list
```

```ocaml
(* Orbit.Second_order *)
type blocks (* per kind λ, assembled over (leaf, copy) indices *)

val blocks_of_factors : ?symmetric:bool -> factors -> blocks
val factors_of_blocks : ?symmetric:bool -> blocks -> factors
val transport : from_dims:int list t -> to_dims:int list t -> factors -> factors
```

```ocaml
(* Solve *)
val hessian_inverse_blocks : damping:float -> blocks -> blocks
```

```ocaml
(* Model: removed *)
(* val surrogate_dim : int *)
(* val ensure_size_invariance : bool *)
```

---

## 8. Tests and acceptance criteria

**M0/M1**
- For `[P;P]` square specs at `n = 4 … 64`, `Φ_n` equals `Qᵀ F_i Q` with `Q`
  the blog's orthonormal basis; `Φ_n^{-1}` matches the explicit triangular
  formulas; the block entries match `R`, `T`, `σ_S`, `σ_A`.
- Transport: for random `w` and stable `n, m`,
  `Φ_m(Ψ_{n→m} w) = Φ_n(w)` to `1e-6`; `Ψ` is an algebra homomorphism on
  random products; distinct eigenvalues of `X_n(w)` and `X_m(Ψ w)` agree
  (multiplicities may differ).
- Regression: the same-factors diagnostic only passes with transport.
- `dune runtest` green; no `ensure_size_invariance` anywhere.

**M2**
- For small `n`, blockwise `H_inv` equals the dense `H_inv` from `Nx.eigh` on
  the full operator (1e-5).
- Eager/jit parity unchanged; `student_teacher` and `mlp_landscape` smoke
  tests; RNN example trains.
- Size generalization: a curvature operator estimated at `n = 8` and
  transported to `m = 16` has the same blocks; the step via transport matches
  a direct estimate at `m` for the same data (up to the data's own size
  dependence).

**M3**
- Property tests for every spec appearing in the repo's models: `Φ` from
  Route A == Route B; blockwise `f(S)` == Route C; transport identities.

---

## 9. Risks and open questions

1. **Stable range.** The block algebra is semisimple only for `r ≥ k + ℓ`
   per group (`k = 2 → r ≥ 4`, `k = 3 → r ≥ 6`). Real models satisfy this;
   small test dims (2, 3) do not. Policy: block path for stable specs;
   Route C (matrix-free) as the fallback below the threshold; do not
   transport to a non-semisimple size.
2. **Copy matching across leaves and groups.** Assembled blocks need
   canonical copy labels. Route A gives them by construction; Route B must
   identify the same kind across different specs (central character,
   idempotent trace, dimension). Verify on the multi-leaf test models.
3. **`symmetric:true` components.** The symmetric compilation bundles
   `term + transpose` (11 components instead of 15 for `[P;P]`). The
   symmetric elements form a Jordan, not associative, algebra; `Φ` restricted
   to them must have full rank and land in the symmetric block coordinates
   (`R = Rᵀ`, `T = Tᵀ`). Check that functions of symmetric elements stay in
   the span (they do: Cayley–Hamilton in the ambient algebra).
4. **Normalization conventions.** SYMO's diagram basis (`1/√D`) differs from
   the blog's orthonormal basis. Transport is convention-independent as long
   as `Φ` is computed consistently at both sizes, but the M0 cross-check must
   translate the basis explicitly.
5. **Numerical robustness.** `Φ` inverses and the Wedderburn decomposition
   run in float64 at compile time; nearly degenerate specs may be
   ill-conditioned. Fall back to Route C (or the old filtered path) with a
   warning, as the design-matrix fallback does today.
6. **JIT.** `factors_to_blocks`/`blocks_to_factors` are matmuls (traceable);
   the functional calculus stays host-side on small matrices. A fully jitted
   step needs closed forms or Newton–Schulz per block.
7. **Performance.** Assembled block sizes `M_λ = Σ_i m_{i,λ}` grow with the
   number of leaves and the number of permuted axes per group; still small
   for the paper's models, but check the language-model-scale cases.
8. **Multiplicity semantics.** The estimator's `damping` uses `s_max` and the
   pseudo-inverse treats zero singular values as zero; both must be defined
   blockwise (max over blocks; zero blocks contribute zero). The optimizer's
   update is unaffected by multiplicities, which is the point — but any
   diagnostic that reads trace/determinant must account for them.

---

## 10. What is removed, kept, added

**Removed** (eventually): `Model.ensure_size_invariance`,
`Model.surrogate_dim`, `Compiler.basis_of_spec`'s `ties_one_side` filter,
the surrogate from the optimizer step (`small` compilers,
`dense_of_factors`/`factors_of_dense` as used by `Optim`),
`Symmetry.surrogate_dims` from the step, `Solve`'s dense factorization path
(`eigh` on the surrogate, kept as a fallback until M2 is validated).

**Kept**: the diagram basis, `estimate_factors` (design matrix + Cholesky /
pseudo-inverse fallback), `apply_block`/`dense_block`, `random_transform`,
the `Make` entry point, the eager/jitted split.

**Added**: kinds and `Φ`/`Φ^{-1}` in `Compiler`; `blocks_of_factors` /
`factors_of_blocks` / `transport` in `Orbit.Second_order`; blockwise `Solve`;
the public `transport` API; tests and the M0 spike.

---

## 11. References

- Blog: *Inverting an n² × n² equivariant matrix with a 2×2 and a 3×3
  inverse* — the transport formulas (§5–§7), the seven kinds and their copies
  (§4), the worked `Φ_n`/`Φ_n^{-1}` (§5–§6).
- Artemev et al., *Exploiting weight-space symmetries for approximating
  curvature*, ICML 2026, [arXiv:2606.00442](https://arxiv.org/abs/2606.00442)
  — the estimator (Eq. 10), the commutant algebra, the factors.
- T. Halverson, A. Ram, *Partition algebras*, European J. Combin. 26 (2005)
  — the algebra and its diagram basis.
- *A seminormal form for partition algebras*,
  [arXiv:1102.2047](https://arxiv.org/abs/1102.2047) — explicit irreducible
  representations.
- *Fast computation of permutation equivariant layers with the partition
  algebra*, [arXiv:2303.06208](https://arxiv.org/abs/2303.06208) — orbit vs
  diagram basis and fast algebra operations.
- *Dimensions of irreducible modules for partition algebras and tensor power
  multiplicities*, [arXiv:1605.06543](https://arxiv.org/abs/1605.06543) —
  multiplicity formulas.
- *Partition algebras and the invariant theory of the symmetric group*,
  [arXiv:1709.07751](https://arxiv.org/abs/1709.07751) — stable range and
  transport of diagram coordinates.
- H. Maron et al., *Invariant and equivariant graph networks*, ICLR 2019 —
  the 15-element basis for `n × n` arrays.
