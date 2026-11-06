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

| model | nv | rows | dt | mjc eager | mjc compiled | mjc per env (vmap 8) | MuJoCo `mj_step` | compiled / MuJoCo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pendulum | 1 | 0 | 0.001 | 497 us | **54 us** | 12 us | **1 us** | 54× |
| chain6 | 6 | 0 | 0.001 | 9266 us | **4778 us** | 9957 us | **3 us** | 1593× |
| boxes2 | 30 | 16 | 0.002 | 38173 us | **152877 us** | 87737 us | **11 us** | 13898× |
| hopper | 7 | 1 | 0.005 | 10320 us | **10151 us** | 7882 us | **5 us** | 2030× |
| walker | 9 | 0 | 0.0025 | 9242 us | **9060 us** | 20225 us | **5 us** | 1812× |

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
- **Batching at 8 lanes does not pay.** Per environment, `walker` costs 20 ms
  batched against 9 ms single. The `vmap` path is not faster per environment
  at this lane count — it has not been tuned, and the earlier `bench_step`
  numbers at 256 lanes are not reproduced here.

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

## 5. The plan

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
3. **Make the row buffer follow the rows.** Even without the schedule change,
   the factorisation could be sized to the rows in play rather than the
   model-wide bound, or the bound could be tightened when recording. Cheap to
   investigate with `spike_solver_iter`-style scaffolding; matters most for
   models that are mostly contact-free.
4. **Fix `boxes2`.** 153 ms compiled against 38 ms eager is a defect, not a
   cost: find what the compiler did to the box-box path, then reduce the work
   to the pairs that can actually touch (the broadphase result is already a
   host-known set at these sizes). This is the M7 box-box item; the target is
   "no worse than the capsule path per body", i.e. single-digit milliseconds.
5. **Batch properly or say it does not help.** Measure `vmap` at 32, 128, 256
   lanes and per environment; either the batching pays at some lane count and
   the number is quoted, or the sheet says the vectorised path is not faster
   and the claim comes off the README.
6. **Eager path: only if users feel it.** The demos run at 4 steps a frame;
   the viewer is not the bottleneck. Revisit after 1–4, and only with a
   measurement showing the viewer's frame time is dominated by the step.

Two things are explicitly *not* on this list. Parallelism across bodies and
rows is a larger change than any of the above and should follow, not precede,
the algorithmic wins. And nothing here should be allowed to touch the
differentiable path's guarantees: the implicit rule freezes the KKT active
set, so an early-exiting solver must still return the same `λ` the full
schedule would — which is what `doc/grad-report.md`'s budget-invariance test
checks, and it stays.

## 6. What this sheet is not

It is not a claim about MuJoCo's or `mjc`'s *physics* cost. It is one state per
model, chosen because both engines can be started there from the corpus, and
a state at rest asks the solver less than a state mid-flight: `hopper` and
`walker` have 1 and 0 rows at that pose and both have more in the middle of a
run. A state-by-state distribution would be a better sheet, and `g4.exe`'s
recording is the data for it — running `perf.exe` against the recorded
trajectory states is the obvious next version of this measurement, and would
change the absolute numbers, though not the ranking: the fixed schedule and
the pair-wise contact work are paid at every step regardless.
