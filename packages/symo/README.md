# SYMO — weight-space symmetry for curvature estimation

SYMO exploits the symmetries of a neural network to estimate its curvature from
orbit averages of gradients, and uses the estimate to precondition gradient
descent. This repo is the OCaml implementation of

> Artemev, Xia, Boyd, Yu, Dangel, Hennequin, Bernacchia,
> _Exploiting weight-space symmetries for approximating curvature_, ICML 2026,
> [arXiv:2606.00442](https://arxiv.org/abs/2606.00442).

built on the [Raven](https://github.com/raven-ml/raven) stack (`nx` tensors, `rune` autodiff/JIT, `ppx_ptree` parameter trees).

## The idea

If a network is invariant under a group `G` acting on the vectorised parameters
`w` by orthogonal matrices `A` — `L(A w) = L(w)` — then its gradient is
equivariant, `∇L(A w) = A ∇L(w)`, and group averages are invariant objects.
SYMO builds the two orbit averages the optimizer needs,

```text
R1(v)     = E_G[(⊗ A) v]                         (first order)
R2(v, v') = E_G[(⊗ A) v v'ᵀ (⊗ A)ᵀ]              (second order)
```

in a compact basis of the commutant algebra: sparse binary tensors (Kronecker
deltas plus free axes) whose coefficients, the **factors**, are estimated by
least squares from data. For a weight-space symmetry the group elements are
permutations of hidden units, so the factors are small matrices over the axes
the group leaves alone.

The optimizer forms the second-order orbit average of the non-invariant part of
the averaged gradient and preconditions the step with its root-damped inverse:

```text
S_g = R2(δ_g, δ_g),   δ_g = ḡ - R1(ḡ)      (EMA'd with β₂, debiased)
P   = (S_g^{1/2} + λI)^{-1}

θ ← θ - η · P · ḡ
```

`λ` is one **absolute** damping term, added to the square root of every isotypic
block's spectrum: one scalar for all kinds and leaves, never a fraction of a
spectrum. It sits where Adam's `ε` sits, so it is in gradient (RMS) units. `P`
is positive definite and its gain is capped at `1/λ`, so `‖Δθ‖ ≤ η‖ḡ‖/λ` and the
step is always a descent direction; a large `λ` tends to plain gradient descent.
The paper's Taylor estimator (Eq. 10, with relative damping `damping·s_max + s`)
is kept in `Solve` for comparison and tests, but the optimizer no longer uses it.

## A model in three declarations

A model declares its parameter tree with `[@@deriving ptree]` (one `walk`
over the tree's parts, from which `Nx.Ptree` derives maps, folds and compiled
keys) and one symmetry specification per leaf:

```ocaml
open Symo

module Mlp = struct
  type ('a, _) t =
    { w1 : 'a   (* hidden x input *)
    ; w2 : 'a   (* output x hidden *)
    }
  [@@deriving ptree]

  let dims : (int list, _) t = { w1 = [ 32; 8 ]; w2 = [ 1; 32 ] }

  (* One permutation group ties the hidden axis of [w1] to the hidden axis of
     [w2]; the other axes are [Id] (free). *)
  let symmetries : (Symmetry.spec list, _) t =
    { w1 = [ Symmetry.Perm 0; Symmetry.Id ]
    ; w2 = [ Symmetry.Id; Symmetry.Perm 0 ]
    }
end

module S = Symo.Make (Mlp)
```

`Symo.Make (M)` returns the orbit machinery (`S.First_order`,
`S.Second_order`, `S.orbit_average`, and the block coordinates
`S.Second_order.blocks_of_factors` / `transport`), the optimizer (`S.init`,
`S.step`, `S.prepare`/`S.solve`/`S.finish`), the compiled step
(`S.Compiled`), and
`S.ptree`, the packed traversal `Rune.grad` and `Rune.jit` take.

## Training

Differentiate the loss with `Rune.value_and_grad` over `S.ptree` and step the
optimizer state; the compiled step is the one to use in a training loop. With
`params` an initial `(Nx.host Nx.float32_t, Nx.host) Mlp.t` and `loss` any
scalar function of it (the example below regresses a student against a
teacher):

```ocaml
let config : S.config =
  { learning_rate = Some 0.1; beta_1 = 0.9; beta_2 = 0.99; damping = 1e-1 }
  (* damping is λ, absolute and in gradient units: the gain of
     (S_g^{1/2} + λI)^{-1} is capped at 1/λ. *)

let compiled = S.Compiled.create ~config
let state = ref (S.init ~config params)

let train_step () =
  let _, grads = Rune.value_and_grad S.ptree loss !state.S.State.theta in
  state := S.Compiled.step compiled ~config ~state:!state ~grads
```

`S.step ~config ~state ~grads` is the eager equivalent. The step is split so
only the preconditioner solve is host-side:

```text
prepare (jitted)                 solve (host)                  finish (jitted)
momentum, orbit average of g,    blocks of S_g, eigh of        apply P to ḡ,
factors of S_g, EMA, debias      (S_g^{1/2} + λI)^{-1}         shift θ
```

`S.Compiled.create` is cheap; the first call traces and compiles, later calls
replay. The whole state, the gradients and the preconditioner cross the boundary as
input/output leaves, so a training loop never retraces.

## Example

A student-teacher MLP with a permutable hidden layer (the same model as
above, runnable):

```sh
dune exec examples/student_teacher.exe
```

```text
student-teacher MLP: 32 hidden units, 8 inputs, 128 examples
    step           mse
       0      2.840415
      50      0.013507
     100      0.005589
     200      0.002241
     300      0.001100
     400      0.000678
     500      0.000471
```

`examples/rnn.ml` trains an RNN with `S.Compiled` and writes its diagnostics
and plots to the directory named by `-d`:

```sh
dune exec examples/rnn.exe -- -d /tmp
```

## Building and testing

This package is part of the `murder` monorepo. From the repository root:

```sh
dune build @packages/symo/runtest
dune exec packages/symo/examples/student_teacher.exe
dune exec packages/symo/examples/rnn.exe -- -d /tmp
```

The suites are:

| suite           | what it checks                                                                                                        |
| --------------- | --------------------------------------------------------------------------------------------------------------------- |
| `test_term`     | term ordering, normalization, symbolic inner products                                                                 |
| `test_compiler` | compiled closures against brute-force sums over the index space, the Cholesky design solve, invariance (`A S Aᵀ = S`) |
| `test_orbit`    | `R1`/`R2` and dense round trips against exact enumeration of the symmetry group                                       |
| `test_blocks`   | blocks against the dense spectrum, transport across sizes, the blockwise preconditioner against the dense one         |
| `test_optim`    | preconditioner identities, the damped step on an invariant quadratic, block transport, end-to-end training            |
| `test_jit`      | 50-step eager/compiled parity, state threading, replay-not-retrace                                                    |

## Library layout

| module                       | role                                                                                                                   |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `Sides`, `Index`             | the two sides of a commutation equation; axis indices and einsum characters                                            |
| `Symmetry`                   | `Absent`/`Id`/`Perm i` specs                                                                                           |
| `Kind`, `Blocks`             | isotypic kinds and canonical test vectors; block coordinates and transport between sizes                               |
| `Term`, `Component`, `Basis` | basis terms (ties + free axes), components, and the symbolic basis of the invariant subspace                           |
| `Compiler`                   | one specification at fixed dimensions → closures for factor estimation, dense blocks and batched block-vector products |
| `Orbit`                      | lifts the compiler to the whole parameter tree (`First_order`, `Second_order`)                                         |
| `Solve`                      | the host-side solves (root-damped inverse, and the Taylor estimator's damped powers)                                   |
| `Optim`                      | the damped-whitening step, eager and jitted (`State`, `Mid`, `Compiled`)                                               |
| `Make`                       | the entry point: orbit machinery + optimizer + packed traversal                                                        |

## Design notes

- **Blocks, not surrogates.** The second-order operator lives in the
  commutant algebra, which is a direct sum of small matrix algebras — one block
  per isotypic kind (`Kind`, `Blocks`). The factors are a size-dependent
  *encoding* of that algebra; the blocks are the size-independent description.
  The optimizer works on the blocks directly, and `transport` re-encodes an
  operator at another size, preserving its blocks, its distinct spectrum and
  any function of it. Only multiplicities (trace, determinant) change with the
  size.
- **The symmetric basis falls back where bundling loses.** Bundling a term
  with its transpose symmetrizes the permuted axes but leaves the free axes'
  factor arbitrary, so it spans `Sym(W) ⊗ End(F)` and misses the
  `Λ²(W) ⊗ Λ²(F)` part of the symmetric operators. That part is empty when the
  free factor holds one entry or no group acts on two axes of one side;
  elsewhere `Compiler.compile ~symmetric:true` uses the full basis, so the
  estimator stays the projection onto the invariant subspace (and a second
  moment stays positive semidefinite).
- **The damping is absolute and sits on the root.** `damping` is `λ` in
  `(S_g^{1/2} + λI)^{-1}`: one scalar added to the *square root* of every
  isotypic block's spectrum, in gradient units (where Adam's `ε` sits), so the
  gain cap `1/λ` and the step bound `η‖ḡ‖/λ` are global. The Taylor estimator's
  relative damping (`damping·s_max`) remains in `Solve`.
- **Permuted axes must be in the stable range.** A group with `k` permuted
  axes needs dimension `n >= 2k` (2, 4 and 6 for one, two and three axes) for
  the commutant to be semisimple and the block decomposition faithful; the
  compiler raises `Invalid_argument` below it. Free (`Id`) axes have no such
  constraint.
- **The design matrix is factorized once, at compile time**, by Cholesky in
  float64 (with a jittered retry and an eigendecomposition pseudo-inverse
  fallback for degenerate bases). The per-call factor estimation is two small
  triangular solves, which stay traceable under `Rune.jit`.
- **float32 tensors, float64 factorizations.** Parameters, factors, blocks and
  the runtime solves are float32; only the compile-time Cholesky and the
  host-side estimator eigendecompositions run in float64.
- **No global RNG.** Everything that draws takes an explicit `Nx.Rng.t`, so
  tests and runs replay exactly; the optimizer step itself is deterministic.
- **The library contains no model definitions.** Models live in downstream
  code or examples (`examples/student_teacher.ml`); only generic parameter-tree
  machinery is here.
- The `Nx.einsum` repeated-label defect that shaped the tensor layer — and
  its upstream fix, which retired the `Delta`/`Contract` workaround — is
  documented in `PLAN.md`, §10.

`PLAN.md` is the port's design document and keeps the full milestone history;
`PLAN-transport.md` records the block/transport design that replaced the
surrogate estimator and the `ensure_size_invariance` filter.

## License

ISC. See the SPDX headers in the sources.
