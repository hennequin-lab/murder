# M6 — gradient report

The gate asks three things: how wrong the gradient is away from transitions, how
large it gets at them, and what the reverse pass costs against a forward step.
This is the sheet. Every number here is produced by a command named beside it.

## What is differentiated

`Dynamics.compile m` fixes a model's *structure* — topology, integrator, solver
budget, damping, actuator velocity terms — and `Dynamics.run t m d` is the step
with the model's *parameters* read as tensors. A loss is then

```ocaml
let dyn = Dynamics.compile m0 in
let loss params = Dynamics.horizon dyn { m0 with params } d0 10 |> …
```

and `Rune.value_and_grad` differentiates with respect to `params` or with
respect to the state, whichever the caller tracks.

Two things are deliberately *not* differentiable and are documented as such in
`lib/dynamics.mli`: the timestep and the solver's iteration budget. Both are
structure. A caller who differentiates them gets a zero, and the interface says
so rather than leaving it to be discovered.

## Accuracy away from transitions

`Rune.check_grads` passes on `chain6` and `finger` at `eps = 1e-6` — the
acceptance item is the library's checker, not one written here, so that this
codebase's notion of a correct derivative cannot drift from the library's
without failing.

Compared against central differences (`test/test_grad.ml`, "gradient matches
central differences") and, for model parameters, against Richardson-extrapolated
central differences ("gradient with respect to model parameters"): agreement
within `1e-6` relative on `chain6`, `pendulum`, `finger`, `body_mass`,
`jnt_stiffness`, `geom_size` and `actuator_gainprm`.

The extrapolation is not decoration. On a coordinate whose derivative is `2e-5`
next to a loss of order one, a plain `1e-6` difference resolves the reference
only to `8e-6` relative; a test written against it would blame the gradient for
the difference's error.

## The solver's derivative

The rows minimise a convex quadratic over a box, so the derivative is taken at
the KKT point with the active set frozen: `dλ_F = -A_FF⁻¹ (dA λ + db)_F` on the
free rows and zero on the pinned ones. `Rune.custom_jvp` carries that map and a
`Rune.custom_transpose` carries its transpose.

Three checks, all in `test/test_grad.ml`:

| Test | What it pins |
| --- | --- |
| "the row solve's minimiser" | a 3-row QP with both bounds binding, against a hand-checked KKT solution |
| "the implicit rule's gradient" | the rule against Richardson differences of a converged solve |
| "the implicit rule's transpose" | `<w, Jv> = <Jᵀw, v>` on a non-degenerate direction |

The transpose test earned its keep immediately. Its first version perturbed the
matrix and the linear term *together*, and at the optimum a free row's gradient
is zero — so the solution does not move, the identity held for the empty map,
and the test passed while testing nothing. Independent perturbations of `a`, `b`
and `lo` are what make it a test. The minimiser test did the same thing to the
author twice: both hand-derived expected solutions were wrong (the first put a
row at the wrong bound, the second put it at its lower bound instead of its
upper), and the solver was right both times. A small case with a hand-checked
answer is worth the embarrassment.

## Cost

`dune exec packages/mjc/bench/bench_grad.exe`, one process, one thread, host
float64, `30` repetitions after `3` warm-up calls, on states the engine reaches
by stepping from rest:

```
model      rows   fwd/ms  grad/ms     g/f   max|g| | solve_i/ms solve_u/ms     u/i
chain6        0    0.738    1.638    2.22    0.012 |     0.678     7.046    10.4
looped       11    0.170    0.455    2.68      1.9 |     0.592     2.254    3.81
finger        0    0.318    0.654    2.05   0.0013 |     0.640     5.063    7.91
blocks        5    0.154    0.282    1.83     0.53 |     0.573     2.240    3.91
boxes2        0   20.236   47.731    2.36      3.8 |    19.000    75.900    3.99
```

- **`g/f`, the number the gate is about, is 2.0–2.7×.** The budget is a reverse
  pass for a forward step and a bit; the engine is inside it.
- `max|g|` is the largest entry of the gradient of one step's `Σ q²`, i.e. how
  large a contact model's derivative gets at a state the engine reached. It is
  finite and O(1), which is the second thing the gate asks.
- `solve_i` and `solve_u` time the *solver's* backward pass alone with the
  implicit rule and with the unrolled derivative, both eager: `u/i` is
  **3.9–10.4×**, so the rule pays for itself in the solver's own backward pass.
  An earlier version of this sheet said 1.1–2.2×, and the gap was a defect in
  the rule's implementation rather than a property of the rule: a stray
  `ignore (iterate …)` left in to silence an unused-value warning ran the
  iteration *twice* in the forward pass. The rule computes the value and the
  tangent map in one visit; it must not also be run for its value outside.

The entry that is not a performance result: `boxes2`'s forward step costs 20 ms
against `chain6`'s 0.7 ms for a five-box model with no contacts at this state.
That is the box-box tensor path being evaluated for every candidate pair, and it
belongs to M7.

## How many iterations the solver actually runs

From `bench/spike_solver_iter.ml`: one iteration on every corpus model but
`looped` (two) and `blocks` (two). A quadratic with a Newton free-block step
should behave like that. Two consequences are worth recording: `max_iterations =
8` is not a close call, which is what makes its unroll affordable; and the
unrolled gradient equals the implicit one on a converged solve, which is why the
implicit rule is a *cost* and *robustness* decision rather than an accuracy one.
At budget one, `blocks`'s force is off by `6.5`, and a caller who sets that gets
the gradient of an iterate — with the rule, of a KKT point.

## Why the hand rule and not `Rune.root`

`Rune.root` expresses an implicit derivative by stating a square residual and
letting the library apply the implicit function theorem. It works here: with the
residual "the gradient on the free rows and the distance from its bound on a
pinned one" — ten lines, with the active set as comparisons, whose derivatives
are zero — it reproduces the row solve's value *and* its gradient exactly on the
3-row QP, against the hand-written rule.

It is also slower, and the reason is structural. The hand rule already holds the
active-set matrix `A_FF`, so the tangent costs one factorization and one solve.
`root` does not have the matrix; its default path builds the Jacobian by
applying the residual to every basis vector, which is `nb` residual evaluations
where the hand rule does one factorization. Measured on synthetic row sets,
`value_and_grad` through `root` costs 2.3–3.1× the hand rule (2.71–3.48 ms
against 0.87–1.53 ms from 3 to 60 rows), and those numbers are
tracing-dominated, so the arithmetic gap is wider than they show.

`root`'s `linear_solve` hook exists to avoid exactly that — it is given an
abstract operator instead of a matrix. Passing conjugate gradients over that
operator did not reproduce the gradient in this comparison: the answer wandered
with the iteration count (`-0.2986`, `-0.3633`, `-0.3038`, `-0.3925` at 3, 6, 12
and 30 iterations where the answer is `-0.3333`). That is either a limitation of
how the hook is meant to be used or a bug somewhere; either way it is not a path
to put the engine's only derivative on. The hand rule stays, with
`bench/spike_root.ml` as the evidence, and it is worth revisiting if the hook
turns out to be usable: ten lines of residual are easier to defend than fifty
lines of transpose.

## Memory over a long horizon

The gate asks for a 1000-step horizon's peak reverse-pass memory with
rematerialization to be at most a third of the un-rematerialized peak. Measured
with `bench/bench_grad.exe memory <mode> chain6 <steps>`, one process per row,
peak resident bytes from the kernel's high-water mark:

| steps | plain | `remat` in a `for` loop | `scan` + `remat` |
| --- | --- | --- | --- |
| 200 | 3.1 MB, — | 7.1 MB, — | **0.15 MB, —** |
| 1000 | 15.5 MB, 46.3 s | 35.3 MB, 102.4 s | **0.21 MB, 32.4 s** |
| 2000 | 30.1 MB, — | 49.1 MB, — | **0.33 MB, —** |

Two things here are worth more than the number. The first is that the plan's
item — "`remat` annotations on the stage boundaries so a long horizon does not
keep every intermediate" — is **wrong as written**. A plain `for` loop *is* the
graph: every step's data is an argument of a call in the unrolled program, and
`remat` keeps its arguments, so annotating the body made peak memory 2.3× worse,
not better. The second is that the fix is free: expressing the horizon as
`Rune.scan` with a rematerialized body gives **74×** less memory than plain at
1000 steps *and* runs 30% faster, because the reverse pass walks the steps one at
a time instead of a thousand-node-wide graph. `Dynamics.horizon_scan` is that
form, and it is what a long horizon should use.

## Not yet measured

- The `dilqr` demonstration. Not written yet.
- A contact *transition*: the bounded-magnitude sweep runs 1000 states
  (`test/test_grad.ml`, "a thousand contact states stay finite") and reports a
  worst case of order one, but it steps *through* no event deliberately — it
  kicks and steps, so transitions happen, they are not sampled at a boundary.
