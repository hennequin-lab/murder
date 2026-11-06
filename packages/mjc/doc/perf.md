# Wall time: `mjc` against MuJoCo

A recap of where the OCaml/Raven engine stands against the C engine on the
corpus models, and what to do about it. Measured on one machine, one thread,
warm-up then a timed loop, both engines starting from the same state.

*See also `doc/research-mjx.md`: how DeepMind's MJX solved the same problem (a
compiled physics engine on a GPU), what it gave up for it, and which of its
answers we should and should not copy.*

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

### What happened when it was built

`lib/linalg.ml` is that factorisation: the Cholesky recurrence and the two
triangular solves written out of broadcast products and reductions, batched or
not, differentiable, and pinned in `test/test_linalg.ml` against the kernels at
1e-12 (`L Lᵀ = A` and `A x = b` as well as against `Nx.cholesky`, for `n` up to
40 and with a lane axis). It is reachable as `Solver.forward ~linear:\`Batched`.

It fixes the scaling and does not pay for itself yet:

| model (`nefcmax`) | linear | single lane | 8 lanes, per lane | 32 lanes, per lane |
| --- | --- | ---: | ---: | ---: |
| pendulum (4) | kernel | 2815 us | 514 us (0.18x) | 150 us (0.05x) |
| | batched | 5496 us | 936 us (0.17x) | 254 us (0.05x) |
| chain6 (48) | kernel | 3798 us | 3100 us (0.82x) | 2828 us (0.77x) |
| | batched | 40971 us | **8994 us (0.22x)** | **4759 us (0.12x)** |
| walker (76) | kernel | 5993 us | 15357 us (**2.56x**) | 14457 us (**2.60x**) |
| | batched | 62653 us | 22348 us (**0.36x**) | 15052 us (0.22x) |

Read the parentheses first, then the absolutes. The batched factorisation is the
only one whose per-lane cost falls with the lane count on every model — the
kernel's *rises* on `walker` (2.56x at 8 lanes, 2.60x at 32), which is the defect
section 5 measured. But its constant is ten times the kernel's, so in absolute
terms it loses at 8 lanes on every model and only breaks even on `walker` at 32.

That constant is a function of the round count, which is the thing section 3
says is wrong. A round is 76 tensor operations for `walker`; the batched
factorisation does them all at 10x the linear-algebra cost and wins the moment
it is not doing them 64 times per step. With item 1 of the plan (two outer
passes instead of eight) the batched solve's 62 ms single-lane becomes about
8 ms, and its per-lane cost at 8 lanes becomes about 2.8 ms against the
kernel's 15 ms. So the factorisation is not a substitute for fixing the
schedule; it is what becomes affordable afterwards.

The accuracy caveat that keeps `\`Batched` off the default path: on the corpus's
stiffest contact state (`limited`, state 9) it moves the solved force by 9e-5
relative where the kernel reproduces MuJoCo's to 1e-9. Both factors are accurate
to 1e-12 on well-conditioned matrices, so this is the system's conditioning
being resolved differently — but a parity gate is not the place to argue about
that, and the kernel stays the default.

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
5. **Batch the solver the way the measurements say.** Two of the three levers
   are done and one is a finding:
   - *Factorisation*: `lib/linalg.ml` implements the batch-friendly factor and
     solve, `test/test_linalg.ml` pins it to the kernels at 1e-12, and
     `Solver.forward ~linear:\`Batched` selects it. It fixes the per-lane
     scaling (walker 2.56x -> 0.36x at 8 lanes) and does not yet pay in absolute
     terms (its constant is 10x the kernel's). It becomes a win once the round
     count is sane — the previous subsection carries the arithmetic — so it
     lands with item 1, not before it.
   - *Composition*: for batched runs, `Rune.vmap (fun d -> Step.step m d)` beats
     `Rune.vmap (Step.run (Step.compile m))` by 3.7x per lane on `chain6`
     (0.55x against 2.02x). Call-site change; `bench/perf.ml` keeps both
     compositions honest until the jit's batched codegen is fixed.
   - *Report the jit gap upstream*: a compiled batched step 3.7x slower per lane
     than the interpreted one is a codegen finding, not an engine one, and
     `spike_batch.exe` is the reproduction.
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

## 8. Kernel search: `Rune.jit ~beam:2`

The beam search times the candidate optimisations of every kernel on the
compiling device and keeps the fastest. It costs compilation, once, and buys
steady-state time, on every call after:

| model | beam 0, compiled | beam 2, compiled | beam 0, per env | beam 2, per env | beam 0, first call | beam 2, first call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pendulum | 58 us | 55 us | 14 us | 11 us | 0.09 s | 4.87 s |
| chain6 | 4910 us | **2203 us** | 9955 us | **2273 us** | 0.79 s | 215 s |

`BEAM=2` is the switch; `perf.exe` now reports the first call, which is where
the search's cost goes. On `chain6` the search pays for itself after about
eighty steps and halves the step; on a model with one body it is not worth the
five seconds. The numbers to remember are the ratio (2.2x on the step, 4.4x per
environment) and the price tag (215 s of compilation for a model that took
0.8 s to compile without it).

## 9. The machine's GPU

This host has an AMD Ryzen AI MAX+ 395 with its integrated Radeon 8060S, so the
question of running the engine there is a fair one. `rig` ships the driver for
it (`rig_amd_amdgpu`, path `/dev/kfd` + `/dev/dri/renderD128`, both present),
and `tolk` compiles kernels for it, so **jit'd work can target it in this
build**. What cannot target it is `mjc`: every signature in the engine is
`Nx.host` (`Model.host`, `Data.host`, `Constraint.f = Nx.host Nx.float64_t`),
so a `Data.t` cannot be placed on a device, and `Rune.jit` of the step is a host
program by construction. There are also no *eager* GPU kernels in the
dependency set (`nx` links `nx_cpu` alone), which does not matter for a jit'd
step but does for the demos, the viewer and the tests, which run eager.

So placing the step on the GPU is not a benchmark away: it is the M7 item the
plan already carries — make the engine's tensors device-polymorphic, then
`Rune.jit` compiles the same program for the device its `Data` lies on.
The measurement that should precede the device-polymorphism work now runs:
`bench/gpu.ml` places a batch of 64 active-set rounds — 76x76 Cholesky plus the
two triangular solves, `[64; 76; 76]` — on the host and on the iGPU.

| | one round over 64 environments |
| --- | ---: |
| host (12 Zen 5 cores, one thread) | 4.02 ms |
| iGPU (gfx1151, via `Rig_amd_amdgpu`) | **0.11 ms** |
| | **38x the host** |

The two devices agree bit for bit: worst relative difference 0. This is the
number the device-polymorphism refactor was gated on, and it clears the gate.
Read it as one round's dense algebra, not a whole step: the solve is 40-60% of a
step from the batching measurements in §5, and transfers are outside the timing
(one copy each way, as a resident training loop would have).

The GPU column needs AMD's *code object manager*: tolk's AMD backend compiles HIP
through `libamd_comgr`, and without it the device still opens and the compile
stops with `comgr not available`. `pacman -S comgr` (in `extra`) is the whole
fix; no ROCm install is needed for the driver path.

Two OCaml obstacles were in the way and both are solved, worth recording because
the engine refactor will meet them again:

- A `Rune.jit` cannot be built inside a functor or a plain function over a
  device set: its brand stays a fresh variable and `The type constructor Set.d
  would escape its scope`. The idiom rune itself uses (`rune/bench/bench_rune.ml`,
  `rune/test/test_jit_devices.ml`) is a **record with a polymorphic field**,
  `type run = { run : 'd. 'd Nx.Placement.t -> ... }`, instantiated per set at
  `Nx.place`ment time — the same source compiles for the host's brand and for a
  device set minted at run time.
- The `Nx.Ptree` signature must be written **inline** in the `Rune.jit` call. A
  top-level `let sig = Nx.Ptree.(pair tensor tensor @-> returns tensor)` is an
  application, so value restriction makes its brand a weak variable, and the
  field can no longer generalise over the device (`less general than 'd0. ...`).



## 10. A whole step on the GPU

`bench/device.ml` runs `Step.step_with` on a placed model and placed data, jit'd
for the device, and compares against the same program on the host:

| model | steps | worst difference from the host |
| --- | ---: | ---: |
| hopper | 100 | 4.4e-15 |
| walker | 50 | 3.6e-16 |

So the engine's own step -- contacts, equalities, the active-set solve, the
integrator -- now executes on the iGPU with answers the host reproduces to
rounding. That is the refactor's purpose met, and it is a *parity* result, not a
throughput one: the loop is 100 steps with one copy in and one out, which is too
short to measure against MuJoCo's microseconds and is not attempted here.

What it costs to get there, which the same bench shows: `humanoid` did not finish
in the time a person will wait for 20 steps, so the compiled program's cost on
this class of model is a real question and the next thing to measure. The
schedule is still 64 rounds regardless of the problem (§6), so a device program
carries that same fixed cost -- now on a device.

The pieces it needed, all of them in the library rather than the bench:
`Model.place`/`Data.place` (a whole pytree at a placement, `Nx.Ptree.place`), a
description as a function (`Model.ptree ()`) because value restriction would
otherwise fix it to one device, and `Step.config` taken from a **host** model --
structure is host data, identical on every device, which is why `Step.step_with`
is the device entry point and `Step.step` stays a host convenience.
