# M0 linear-algebra spike

Run with `dune exec packages/mjc/bench/spike_la.exe` on the development
machine (AMD Ryzen 5 7640U, 6 cores / 12 threads, OCaml 5.5.2, Raven
`4a3bc831`). float64 on the host, one process. The full table is the output of
the program; this note records what it means and what it does not measure.

## Method

For each `n`, a symmetric positive definite matrix and one right-hand side.
A "solve" is what a step does: one `Nx.cholesky` factor followed by three
`Nx.solve_triangular` calls (one for `qacc_smooth`, two for the impulse
solve). Eager timings are 20 calls after 3 warm-up calls. Compiled timings use
`Rune.jit'` and time the first call separately (compile) from the steady
state. The batched column stacks 256 matrices and maps the same function with
`Rune.vmap`; the right-hand sides are arguments, not captures, because a
captured matrix is shared across lanes and reports a number the engine cannot
reach.

## Results

| n | eager factor | eager solve | jit factor | jit solve | vmap 256 solve, per env | compile, factor | compile, solve |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.9 us | 7.2 us | 2.3 us | 5.5 us | 0.3 us | 0.02 s | 0.01 s |
| 27 | 2.4 us | 10.4 us | 11.4 us | 19.9 us | 3.1 us | 0.01 s | 0.01 s |
| 39 | 5.2 us | 11.4 us | 25.0 us | 39.6 us | 4.8 us | 0.00 s | 0.01 s |
| 60 | 7.0 us | 19.8 us | 63.8 us | 94.8 us | 5.9 us | 0.00 s | 0.01 s |

LU comparison: `Nx.solve` costs 7.7 us eagerly and 18.5 us compiled at
`n = 27`, and 10.3 / 30.4 us at `n = 39`, so a Cholesky factor plus three
triangular solves is the cheaper path at every size measured. A compiled scan
of 1000 elementwise steps takes 0.20 ms, 151 ns per iteration, which is the
loop floor `MILESTONES.md` §3 gives.

## What changed since the design note

`PLAN.md` §3.3 measured the same operations at pin `3984341` and found two
obstacles: compiled factorizations 3–7× slower than the eager C kernels, and
a first call that took seconds (0.63 s at 20×20, 5.6 s at 40×40, 111 s at
80×80), which it treated as the practical limit on model size.

At pin `4a3bc831` the first call of the same programs takes **10 ms**, not
seconds, at every size up to 60×60. The steady-state costs are the same shape
as before (compiled is 2–5× the eager kernel), and the batched gain is larger
than the note measured (8× at 39×39 rather than 3–4×). The compile-time risk
R1 and the open question §13 Q7 are therefore much smaller than the note says,
and the note's conclusion that `nv ≈ 100` is out of reach no longer follows
from compile time. It may still follow from steady-state cost.

This is one machine and one pin. The benchmark should be re-run when the Raven
pin moves; the two numbers that matter are the first-call time at 39×39 and
the per-environment batched solve.

## Gate G0

| Question | Pass | Conditional | Fail | Measured |
| --- | --- | --- | --- | --- |
| first call of a jitted 39×39 factor+solve | ≤ 1.5 s | ≤ 5 s | > 5 s | **0.01 s** |
| 39×39 factor + 3 triangular solves, `vmap` 256 | ≤ 5 µs/env | ≤ 8 µs/env | > 8 µs/env | **4.8 µs/env** |
| composed step-shaped program, `vmap` 256 | ≤ 100 µs/env | ≤ 150 µs/env | > 150 µs/env | **not measured** |

**G0 passes on the first two rows.** The third is not measured by this spike:
the composed program is M2's `Step.compile` and is measured there (G2). The
conditional threshold applies to it if it misses.

## Not measured, and why

- **A hand-written factorization as a `scan`.** A Cholesky column sweep needs
  a traced column index, which `Nx.slice` does not take; expressing it needs
  `take`-based dynamic slices. With the compile time at 10 ms and the
  batched solve inside the gate, this no longer buys anything and was not
  written. Revisit only if M2 misses G2.
- **A blocked (panel) variant.** Same reason: it was a workaround for the
  compile time the new pin does not have.
- **The composed step-shaped program.** M2 measures the real one.
- **GPU.** Not exercised here; that is M11.
