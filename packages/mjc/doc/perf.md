# Wall time: `mjc` against MuJoCo

A recap of where the OCaml/Raven engine stands against the C engine on the
corpus models, and what to do about it. Measured on one machine, one thread,
warm-up then a timed loop, both engines starting from the same state.

**The headline, said plainly:** `mjc` is between fifty and two thousand times
slower per step than MuJoCo on these models, and the gap is not in the physics
— the rows, contact sets and trajectories agree to 1e-10 or better — but in how
much tensor work a step does, almost all of it fixed-cost and most of it
solver schedule. Sections 3 and 4 size that.

## 1. Method

| | |
| --- | --- |
| Machine | AMD Ryzen AI MAX+ 395, one thread, Linux |
| `mjc` | `dune exec packages/mjc/bench/perf.exe <models>`, library code, host float64 |
| MuJoCo | `python packages/mjc/tools/perf_mujoco.py packages/mjc/test/fixtures <models>`, MuJoCo 3.17 |
| State | the reference pose `qpos0`, `qvel = 0`, `ctrl = 0`, `act = 0` |
| Timing | 50 warm-up steps, then 500 timed; wall clock around the loop |

Both engines step from the same recorded model files, so the comparison is of
stepping code and not of parsing. The MuJoCo models are the unmodified
dm_control and MuJoCo XMLs the corpus already carries.

```sh
dune exec packages/mjc/bench/perf.exe pendulum chain6 boxes2 hopper walker
/tmp/mjcenv/bin/python packages/mjc/tools/perf_mujoco.py \
  packages/mjc/test/fixtures pendulum chain6 boxes2 hopper walker
```

## 2. The numbers

Per step, microseconds. `compiled` is `Step.compile` + `Step.run`; `eager` is
`Step.step`, the same pipeline evaluated through the library's interpreter;
`per env` is the compiled step under `Rune.vmap` over 8 lanes, divided by 8.
`rows` is `nefc` after one step from the reference pose.

| model | nv | rows | nefcmax | dt | mjc eager | mjc compiled | mjc per env (vmap 8) | MuJoCo `mj_step` | compiled / MuJoCo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pendulum | 1 | 0 | 4 | 0.001 | 474 us | **54 us** | 12 us | **1 us** | 54x |
| chain6 | 6 | 0 | 48 | 0.001 | 9195 us | **4856 us** | 9941 us | **3 us** | 1619x |
| boxes2 | 30 | 16 | 176 | 0.002 | 38173 us | **152877 us** | 87737 us | **11 us** | 13898x |
| hopper | 7 | 1 | 64 | 0.005 | 10320 us | **10151 us** | 7882 us | **5 us** | 2030x |
| walker | 9 | 0 | 76 | 0.0025 | 9242 us | **9060 us** | 20225 us | **5 us** | 1812x |

Three things in that table are facts about this engine rather than about the
physics, and each is worth its own measurement before anyone plans work:

- **`pendulum` is fast and everything with geometry is not.** One hinge, no
  rows, 54 us compiled — the per-step floor of the library, within 54× of C.
  Every model with capsules or boxes is a thousand times off, which says the
  cost is in collision and the solver, not in the integrator.
- **`boxes2`'s compiled step is *slower* than its eager one** (153 ms against
  38 ms). The box-box tensor path is evaluated for every candidate pair and
  the compiler has made it worse, not better. This is the one number on the
  sheet that looks like a defect rather than a cost.
- **Batching pays on `pendulum` and loses on everything else.** Per
  environment `pendulum` is 4.5x cheaper batched (12 us against 54 us) and
  `walker` is 2.2x dearer (20 ms against 9 ms). Section 5 shows why: it is the
  batched factorisation, and the two sizes that decide it are on this table.

## 3. Where the time goes: the solver's schedule

The row solve runs a **fixed** schedule: `max_iterations = 8` outer passes of
`inner_rounds = 8` active-set rounds each, as one `Rune.scan` — 64 rounds per
solve whether the problem needs one or sixty. The measured cost of that
schedule, by timing the same forward pass at `iters = 1` and at `iters = 8`:

| model | forward, iters 1 | forward, iters 8 | share the schedule costs |
| --- | ---: | ---: | ---: |
| pendulum | 447 us | 448 us | 0% |
| chain6 | 6041 us | 9181 us | 34% |
| hopper | 6775 us | 10711 us | 37% |
| walker | 5130 us | 9709 us | 47% |
| boxes2 | 58498 us | 130735 us | 55% |

So a third to a half of a step is rounds that were never needed. The solver's
own measurement (`bench/spike_solver_iter.ml`, `doc/grad-report.md`) says the
corpus models settle in **one iteration**, two for `looped` and `blocks`. The
schedule is 64 rounds against a need of 2–16 depending on the model and its
`iters` option — the worst kind of waste, a fixed cost for a variable job.

The scan made this worse before it made anything else better: it exists
because the *unrolled* loop made compile time grow with the budget (four
rounds: 8 s, eight rounds: nearly seven minutes), so the budget had to be
small. The scan bought a flat compile cost and an exact budget, and the price
is that run time is now proportional to the schedule — which is only
affordable if the schedule stops early.

## 4. Where the rest of it goes

Not yet measured to the level of section 3; named with what is known.

- **Per-operation overhead in the eager path.** Eager is 2× compiled on
  `chain6` and the eager path is what the demos, the viewer and the tests use.
  Each library operation is a call with shape checks; a step is thousands of
  them. Not on the critical path for the gate, but it is the path users take.
- **Contact work over every candidate pair.** `boxes2` (5 boxes) costs 38–153 ms
  where `chain6` (6 capsules) costs 5–9 ms, and it has no more bodies. The
  box-box manifold path evaluates all pairs with tensor code; the candidate
  filter is not paying for itself. `bench_step.ml` records a second symptom:
  a model whose geoms can touch *traces* for minutes, because each candidate
  walks its dof chain with one traced `set` per degree of freedom.
- **A fixed-size row buffer.** The solver factorises an `[nefcmax; nefcmax]`
  matrix every round, and `nefcmax` is a per-model bound rather than the
  number of rows this step produced. A step with zero rows still pays the
  whole schedule: `chain6` and `walker` show `rows = 0` at the reference pose
  and still cost 4.8 and 9 ms.
- **Not measured at all:** the humanoid. `perf.exe humanoid` did not finish
  inside a 55-minute budget at these step costs (≈10 ms/step × 550 steps ×
  8 vmap lanes, plus 200 forward passes at `iters = 8`), so it is left out
  rather than estimated. Its single-step cost from `bench_grad`'s table
  (`doc/grad-report.md`) is in the same 10 ms range as `hopper` and `walker`.

## 5. Batching: does `vmap` cost an unrolled loop?

The question worth answering before planning anything about batching, because
the answer decides whether the loss is structural or fixable. The guess is that
`Rune.vmap` over a `Rune.scan` unrolls the loop per lane, so batching a solver
would multiply its rounds by the lane count.

**It does not unroll.** `Rune.scan` lowers to a real loop (`Ops.range` with
carried buffers), and a map around it keeps one loop whose buffers carry a lane
axis. A loop of 64 cheap steps costs 137-154 us on one lane and 166-191 us for
eight: **0.13-0.17x per lane**, six times cheaper to batch than to repeat, and
flat in the buffer size (`nb = 8` and `nb = 64` measure the same).

What the loss is instead is **the solver, and it grows with the row buffer**.
Per lane divided by single lane, eight lanes, three runs each, reproducible to
about 0.03 (`bench/spike_batch.exe`):

| model | `nefcmax` | solve stage | whole step, interpreted | whole step, compiled |
| --- | ---: | ---: | ---: | ---: |
| pendulum | 4 | **0.18x** | **0.33x** | **0.20x** |
| chain6 | 48 | 0.88x | **0.55x** | **2.02x** |
| walker | 76 | **2.51x** | 1.55x | **2.24x** |

Every other stage batches well and consistently: kinematics 0.31-0.37x, mass
0.36-0.37x, contacts 0.30-0.33x, passive and bias 0.30-0.40x per lane. The
solver is the only stage that does not, and it is 40-60% of the step.

Two separate effects are visible in that table, and the second one is the
surprise:

- **The batched factorisation stops paying at about 48 rows.** A loop whose
  step is a matmul and a Cholesky solve costs, per lane: 0.19-0.23x at
  `nb = 8`, 0.27-0.32x at 16-32, 0.84-0.97x at 48, and **0.98-1.04x** at 64 —
  while the matmul alone reaches **1.30x**, a loss. Elementwise work batches
  at every size. So a solver round, which is a small matmul and a Cholesky on
  an `[nefcmax; nefcmax]` matrix, inherits that cliff exactly where the corpus
  models sit (48, 64, 76, 176, 202).
- **Compiling the batched step is much worse than interpreting it.**
  `Rune.vmap f` and `Rune.vmap (Rune.jit f)` are the same program when `f` is a
  small loop (measured equal to the microsecond), but not when `f` is the
  engine's step: on `chain6` the interpreted batched step is **0.55x** per lane
  and the compiled one **2.02x** — the compiled form is 3.7x slower per lane
  than the form the jit is supposed to improve. `perf.exe`'s per-env column
  measures the compiled form, which is why section 2's batching numbers look
  uniformly bad. They are the numbers for the worst of the two compositions.

So: batching is not at odds with the scan, it is at odds with (a) a
factorisation whose batched kernels stop paying at our buffer sizes and (b) a
jit codegen path that does worse on a batched step than the interpreter does.
The first is ours to fix; the second should be reported and possibly avoided.

## 6. The plan

Ordered by measured payoff per unit of work, with how each would be checked.
None of these changes a physics result; the G4 parity tests
(`test_rows_of`, `g4.exe`) are the guard, and they must stay at 1e-12 / 1e-8
across all of them.

1. **Stop the schedule when the KKT test holds.** `Rune.iterate` is exactly
   this: the body is traced once like `scan`'s, and the loop runs until a
   traced predicate says the solve is done. Needs a scalar KKT residual to
   test — the maximum absolute gradient on free rows and the worst inward
   gradient at a bound are both reductions the solver already computes
   pieces of. Expected: removes most of the 34–55% in section 3, because the
   measured need is 1–2 outer passes. Check: `g4.exe` unchanged to 1e-8,
   `u/i` in `bench_grad` unchanged, `perf.exe` before/after.
2. **Skip the solve when there is nothing to solve.** `nefc = 0` is a scalar
   the step knows; a `Rune.switch` around the schedule would make a
   contact-free pose free. `chain6` and `walker` sit at `rows = 0` at the
   reference pose and pay the full 64 rounds. Expected: those two models drop
   to the non-constraint floor (tens of microseconds to low hundreds).
   Check: same as above, plus a test that a model whose rows appear later in
   a run still solves.
3. **Make the row buffer follow the rows.** Section 5 puts a number on this:
   the solver's matrix is `[nefcmax; nefcmax]` (4 for `pendulum`, 48 for
   `chain6`, 76 for `walker`, 202 for `humanoid`) while a step usually has
   under sixteen rows, and the batched factorisation stops paying at 48. Sizing
   the solve to the live rows — host-known in the eager path, bucketed and
   padded in the compiled one — moves most steps under that cliff and shrinks
   every round as well.

4. **Fix `boxes2`.** 153 ms compiled against 38 ms eager is a defect, not a
   cost: find what the compiler did to the box-box path, then reduce the work
   to the pairs that can actually touch (the broadphase result is already a
   host-known set at these sizes). This is the M7 box-box item; the target is
   "no worse than the capsule path per body", i.e. single-digit milliseconds.
5. **Batch the solver the way the measurements say.** Three concrete levers,
   in order of measured payoff:
   - *Composition*: for batched runs, use `Rune.vmap (fun d -> Step.step m d)`
     rather than `Rune.vmap (Step.run (Step.compile m))`. Measured on
     `chain6`: 0.55x per lane interpreted against 2.02x compiled. This is a
     call-site change plus a benchmark that keeps both compositions honest, and
     it makes section 2's batching column look like section 5's interpreted one.
   - *Factorisation*: write the Cholesky as its `n` public steps of elementwise
     operations and one reduction each, which batch at 0.15x per lane, instead
     of calling the batched kernel, which reaches 1.30x on the matmul alone.
     Kept behind the same interface so `test_grad`'s QP oracle and the G4 parity
     tests choose between the two.
   - *Report the jit gap*: a compiled batched step 3.7x slower per lane than the
     interpreted one is a codegen finding, not an engine one. It goes upstream
     with `spike_batch.exe`, and until it is fixed the batched path does not go
     through the jit.
6. **Run the solver on the rows in play, not the bound.** `nefcmax` is the
   compile-time worst case (48-202 on these models); a typical step has fewer
   than sixteen rows and some have none. Sizing the solve to the live rows —
   host-known in the eager path, bucketed and padded in the compiled one —
   would put most steps under the cliff of section 5 as well as shrinking the
   schedule's per-round cost. This is item 2's neighbour: skipping the solve
   when there are no rows removes the worst cases, this one shrinks the rest.
7. **Eager path: only if users feel it.** The demos run at 4 steps a frame;
   the viewer is not the bottleneck. Revisit after 1–4, and only with a
   measurement showing the viewer's frame time is dominated by the step.

Two things are explicitly *not* on this list. Parallelism across bodies and
rows is a larger change than any of the above and should follow, not precede,
the algorithmic wins. And nothing here should be allowed to touch the
differentiable path's guarantees: the implicit rule freezes the KKT active
set, so an early-exiting solver must still return the same `λ` the full
schedule would — which is what `doc/grad-report.md`'s budget-invariance test
checks, and it stays.

## 7. What this sheet is not

It is not a claim about MuJoCo's or `mjc`'s *physics* cost. It is one state per
model, chosen because both engines can be started there from the corpus, and
a state at rest asks the solver less than a state mid-flight: `hopper` and
`walker` have 1 and 0 rows at that pose and both have more in the middle of a
run. A state-by-state distribution would be a better sheet, and `g4.exe`'s
recording is the data for it — running `perf.exe` against the recorded
trajectory states is the obvious next version of this measurement, and would
change the absolute numbers, though not the ranking: the fixed schedule and
the pair-wise contact work are paid at every step regardless.
