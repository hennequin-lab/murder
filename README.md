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

The estimator (`Taylor`, Eq. 10 of the paper) forms

```text
S_w   = R2(δ_w, δ_w),   δ_w = w - R1(w)
S_g   = R2(δ_g, δ_g),   δ_g = ḡ - R1(ḡ)     (EMA'd with β₂, debiased)

H     = S_w^{-1/2̃} (S_w^{1/2̃} S_g S_w^{1/2̃})^{1/2̃} S_w^{-1/2̃}
H_inv = S_w^{ 1/2̃} (S_w^{1/2̃} S_g S_w^{1/2̃})^{-1/2̃} S_w^{ 1/2̃}

θ ← θ - η · H_inv · ḡ
```

with damped symmetric powers `X^{p̃} = U diag((damping·s_max + s)^p) Uᵀ`.

## A model in three declarations

A model declares its parameter tree with `[@@deriving ptree]` (which provides
the `map`/`map2`/`iter`/`fold`/`names` traversals), one symmetry specification
per leaf, and the surrogate dimension:

```ocaml
open Symo

module Mlp = struct
  type 'a t =
    { w1 : 'a   (* hidden x input *)
    ; w2 : 'a   (* output x hidden *)
    }
  [@@deriving ptree]

  let dims : int list t = { w1 = [ 32; 8 ]; w2 = [ 1; 32 ] }

  (* One permutation group ties the hidden axis of [w1] to the hidden axis of
     [w2]; the other axes are [Id] (free). *)
  let symmetries : Symmetry.spec list t =
    { w1 = [ Symmetry.Perm 0; Symmetry.Id ]
    ; w2 = [ Symmetry.Id; Symmetry.Perm 0 ]
    }

  (* The estimator works on a surrogate network: free axes keep their size,
     permuted axes shrink to 2. Must stay >= 2 (see Design notes). *)
  let surrogate_dim = 2
end

module S = Symo.Make (Mlp)
```

`Symo.Make (M)` returns the orbit machinery (`S.First_order`,
`S.Second_order`, `S.orbit_average`), the optimizer (`S.init`, `S.step`,
`S.prepare`/`S.solve`/`S.finish`), the compiled step (`S.Compiled`), and
`S.ptree`, the packed traversal `Rune.grad` and `Rune.jit` take.

## Training

Differentiate the loss with `Rune.value_and_grad` over `S.ptree` and step the
optimizer state; the compiled step is the one to use in a training loop. With
`params` an initial `Nx.float32_t Mlp.t` and `loss` any scalar function of it
(the example below regresses a student against a teacher):

```ocaml
let config : S.config =
  { learning_rate = Some 0.1; beta_1 = 0.9; beta_2 = 0.99; damping = 1e-4 }

let compiled = S.Compiled.create ~config
let state = ref (S.init ~config params)

let train_step () =
  let _, grads = Rune.value_and_grad S.ptree loss !state.S.State.theta in
  state := S.Compiled.step compiled ~config ~state:!state ~grads
```

`S.step ~config ~state ~grads` is the eager equivalent. The step is split so
only the estimator solve is host-side:

```text
prepare (jitted)                solve (host)              finish (jitted)
momentum, orbit averages,  →    svd64 + damped       →    factors of H_inv,
dense S_w/S_g, EMA, betas       symmetric powers          apply to ḡ, shift θ
```

`S.Compiled.create` is cheap; the first call traces and compiles, later calls
replay. The whole state, the gradients and the estimator cross the boundary as
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
       0      4.189058
      50      0.035483
     100      0.007039
      ...
     500      0.000165
```

## Building and testing

This repository is a standalone project inside the `raven-and-friends` dune
workspace; `nx`, `rune` and `ppx_ptree` come from the local `raven/packages`.
From this directory:

```sh
dune build @check     # type-check everything
dune runtest          # 34 tests
```

The suites are:

| suite           | what it checks                                                                                                        |
| --------------- | --------------------------------------------------------------------------------------------------------------------- |
| `test_term`     | term ordering, normalization, symbolic inner products                                                                 |
| `test_compiler` | compiled closures against brute-force sums over the index space, the Cholesky design solve, invariance (`A S Aᵀ = S`) |
| `test_orbit`    | `R1`/`R2` and surrogate round trips against exact enumeration of the symmetry group                                   |
| `test_optim`    | estimator identities, Newton step on an invariant quadratic, end-to-end training                                      |
| `test_jit`      | 50-step eager/compiled parity, state threading, replay-not-retrace                                                    |

## Library layout

| module                       | role                                                                                                                   |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `Sides`, `Index`             | the two sides of a commutation equation; axis indices and einsum characters                                            |
| `Symmetry`                   | `Absent`/`Id`/`Perm i` specs and the surrogate dimension they induce                                                   |
| `Term`, `Component`, `Basis` | basis terms (ties + free axes), components, and the symbolic basis of the invariant subspace                           |
| `Compiler`                   | one specification at fixed dimensions → closures for factor estimation, dense blocks and batched block-vector products |
| `Orbit`                      | lifts the compiler to the whole parameter tree (`First_order`, `Second_order`)                                         |
| `Solve`                      | the host-side estimator (`svd64`, damped symmetric powers, `H`/`H_inv`)                                                |
| `Optim`                      | the Taylor step, eager and jitted (`State`, `Mid`, `Compiled`)                                                         |
| `Make`                       | the entry point: orbit machinery + optimizer + packed traversal                                                        |

## Design notes

- **The surrogate dimension must be at least 2.** The estimator runs on a
  surrogate network where permuted axes take `surrogate_dim` values. With a
  one-element group the distinct basis components coincide, the surrogate
  degenerates and the estimated curvature vanishes (the step is zero/NaN). The
  paper uses 2.
- **The design matrix is factorized once, at compile time**, by Cholesky in
  float64 (with a jittered retry and an SVD pseudo-inverse fallback for
  degenerate bases). The per-call factor estimation is two small triangular
  solves, which stay traceable under `Rune.jit`.
- **float32 tensors, float64 factorizations.** Parameters, factors, blocks and
  the runtime solves are float32; only the compile-time Cholesky and the
  host-side estimator SVDs run in float64.
- **No global RNG.** Everything that draws takes an explicit `Nx.Rng.key`, so
  tests and runs replay exactly; the optimizer step itself is deterministic.
- **The library contains no model definitions.** Models live in downstream
  code or examples (`examples/student_teacher.ml`); only generic parameter-tree
  machinery is here.
- The `Nx.einsum` repeated-label defect that shaped the tensor layer — and
  its upstream fix, which retired the `Delta`/`Contract` workaround — is
  documented in `PLAN.md`, §10.

`PLAN.md` is the port's design document and keeps the full milestone history,
including two corrections found while porting (the surrogate convention above,
and a dropped scalar factor in the compiler).

## License

ISC. See the SPDX headers in the sources.
