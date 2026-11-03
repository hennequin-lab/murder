# mjc — milestones

Status: **M0–M2 complete**; the number sheet for each is in
[`doc/log.md`](doc/log.md). This document cuts the design note
[`PLAN.md`](PLAN.md) into increments that can be built, tested and stopped one
at a time. `PLAN.md` carries the measurements and the reasoning; this file
carries the order of work, the acceptance tests and the decision points.

The viewer has its own milestones (`V0`–`V5`) and its own note,
[`VIZ.md`](VIZ.md). They run in parallel with the engine milestones below.

## How to read this

Each milestone has:

| Field | Meaning |
| --- | --- |
| Goal | one sentence; what exists at the end that did not before |
| Work | the modules and tests to write |
| Acceptance | numbers that fail today and pass afterwards |
| Gate | a question that can end or reshape the project, with a decision rule |
| Demo | a command that shows the milestone working |
| Deferred | what is deliberately not built yet, and where it went |
| Size | one experienced person, full time, writing at the repo's standards |

A milestone is done when the acceptance tests pass in `./dev check`, the demo
runs, the log entry is written (`doc/log.md`), and the fixtures it added are
committed.

## 1. Principles

1. **A walking skeleton first.** Model loads, forward kinematics, a window.
   Nothing is debugged through a stack of unexercised stages. This is why the
   viewer is not a late milestone.
2. **The C engine is the specification.** Every stage is checked against
   arrays recorded from MuJoCo, not against the values this implementation
   produced last week. A number learned from our own code is a baseline, not a
   test.
3. **Re-synchronised parity, not free-running parity.** Rigid contact is
   chaotic: two correct implementations diverge in a few hundred steps, and a
   1e-9 difference in a contact normal can flip an active set. The gate is
   `f(x_t, u_t)` against the C engine from *its* `x_t`, plus a separately
   reported shadowing horizon for free runs (§9.2).
4. **Retire the two structural risks before they can hide.** The compiled
   factorization (M0) and contact (M4) each get a written gate with a decision
   rule, not a milestone that can silently run over.
5. **One shape-key budget.** Compile time is a property of the *set* of
   tensor shapes a model produces. It is designed at M0 and policed at every
   milestone (§9.5).
6. **The engine is useful before it is finished.** M2 gives forward dynamics,
   M3 gives limits and equality, M4 gives the standard contact benchmarks, M6
   gives gradients. Each is a usable stopping point.

## 2. Cross-cutting threads

Four pieces of infrastructure are built once and extended at every milestone.
They are not milestones because they never finish.

| Thread | Started | Extended at | Artifact |
| --- | --- | --- | --- |
| T1 fixture recorder and parity differ | M0 | every milestone | `tools/record_fixture.py`, `test/fixtures/`, `test/support/parity.ml` |
| T2 benchmarks | M0 (spike), M2 | every milestone | `bench/`, `doc/perf.md` |
| T3 viewer | V0 at M0, V1 at M2 | V2–V5, see `VIZ.md` | `viz/`, `mjc-view` |
| T4 decision log | M0 | every milestone | `doc/log.md`, `doc/adr/NNNN-*.md` |

Two rules keep them honest:

- A milestone is not done until its new stage appears in the parity differ and
  its cost appears in `doc/perf.md`.
- A change that alters a documented decision updates the ADR or the design
  note in the same change. `PLAN.md` is a living design note.

## 3. Dependency graph

Two lanes. The engine lane is sequential. The viewer lane hangs off it at named
points and never blocks it.

```
engine   M0 ---> M1 ---> M2 ---> M3 ---> M4 ---> M5 ---> M6 ---> M7 --> M8 ... M11

viewer   V0 ---> V1 ---> V2 ---> V3 ---> V4 ---> V5
```

| Viewer milestone | Needs | Why | Can start |
| --- | --- | --- | --- |
| V0 stills | M0 | the model reader | with M0 |
| V1 window | M1 | body poses | once `Model.t` exists, drawing identity poses |
| V2 overlays | M2 | inertia, COM, velocities | with M2 |
| V3 contacts and log | M4 | contact rows and forces | with M4 |
| V4 browser | V3 | the log format | after M7 |
| V5 batch | M7 | a vmapped `Data` | after M7 |

The `V0 -> V1` arrow is a build order, because V0's scene description is V1's
input. It is not a hard dependency: a second person can start V1 once the model
reader lands, drawing every geom at the identity pose.

## 4. Milestone index

| Id | Milestone | Size | Gate | After it you can |
| --- | --- | --- | --- | --- |
| M0 | Foundations, fixtures, linear-algebra gate | 2 wk | **G0** | load a model, print its arrays, replay C-engine fixtures |
| M1 | Rigid-body math, forward kinematics | 2 wk | G1 | compute every pose in a model and draw it |
| M2 | Inertia, bias, passive, integration | 2.5 wk | G2 | forward-simulate unconstrained bodies and chains |
| M3 | Constraints without contacts, solvers | 3 wk | **G3** | step a model with limits, welds and motors |
| M4 | Contact | 4.5 wk | **G4** | step hopper, walker, ant, humanoid |
| M5 | Actuators, transmissions, sensors, tendons | 2.5 wk | G5 | load the lab's actuated and tendon models |
| M6 | Differentiability | 2.5 wk | **G6** | differentiate the step, run iLQR/MPC on it |
| M7 | Parity, performance, dtype consolidation | 2 wk | **G7** | quote throughput numbers and ship the package |
| M8 | Convex meshes and hfield (optional) | 4+ wk | G8 | use arbitrary robot meshes |
| M9 | Spatial tendons with wrapping (optional) | 2.5 wk | — | hand and leg models with routed tendons |
| M10 | Native MJCF reader (conditional) | 3.5 wk | — | run MJCF with no MuJoCo installed |
| M11 | GPU and throughput scale-out (optional) | 3 wk | — | batch 8192 environments on one accelerator |

Engine total to the end of M7: **21 weeks**. M8–M11 are 13+ further weeks and
each can be dropped without touching v1. With a 25% buffer, v1 is about six
months of one person. A dash in the gate column means the milestone has an
abandon criterion in its own section rather than a gate in §8.

Viewer: `V0` 0.5 wk, `V1` 1 wk, `V2` 0.5 wk, `V3` 1 wk, `V4` 1.5 wk,
`V5` 0.5 wk. `V1`–`V3` are scheduled inside M2–M5 and add about two weeks of
person-time to the critical path if done by the same person.

## 5. The engine milestones

### M0 — Foundations, fixtures, and the linear-algebra gate (done)

**2 weeks · gate G0 · blocks everything**

**Goal.** A model survives a round trip from MJCF through the converter into
OCaml, the arrays are checked against the C engine, and the factorization cost
is measured well enough to decide whether the rest is worth building.

**Work.**

- `packages/mjc/` with `lib/`, `test/`, `bench/`, `tools/`, `doc/`, plus the
  `(package mjc ...)` stanza in the root `dune-project`. Libraries: `mjc`
  (core). Dependencies: `base`, `stdio`, `nx`, `rune`, `ppx_ptree`, all
  already in the lock, so this milestone adds no dependency and should not
  relock. The `mjc.viz` sublibraries are V0's business and land beside this
  work.
- `lib/model.ml`, `lib/data.ml` as `Ptree` records (`D1`, §9.5):

  ```
  Model:  shape key fields, then topology as host int arrays and
          parameters as tensors (masses, inertias, geom poses and sizes,
          joint axes and ranges, actuator gains, ...)

  Data:   state:    time qpos qvel qacc act ctrl qfrc_applied
          results:  xpos xquat geom_xpos geom_xmat site_xpos xipos
                    qfrc_bias qfrc_passive qfrc_actuator qfrc_smooth
                    efc_force ncon nefc
  ```

  `Model` is what a compiled step captures; `Data` is what `jit`, `vmap`,
  `grad` and checkpointing walk. Everything is in `Data` because a user may
  want it: the states, the results a viewer or a test reads, and the
  warmstart (`efc_force`). The rest of the `efc_*` set and all other work
  buffers are local to the step and never appear in `Data`.
- `lib/mjcb.ml`: the `.mjcb` reader plus a format spec in `doc/mjcb.md`.
  Versioned header, offset table, little-endian, float64 payload with an
  optional float32 section. Host arrays only; no `Nx` in this module, so it
  is testable without tensors.
- `tools/convert_mjcf.py` (dev-time, not linked, not in CI): loads MJCF with
  MuJoCo's Python package, calls `mj_setConst`, and writes `.mjcb`. It records
  the MuJoCo version in the header. The converter is the public model format
  in v1 (`D8`).
- `tools/record_fixture.py`: given a `.mjcb`, an initial state and a control
  sequence, records per-step `qpos qvel qacc act ctrl xpos xquat geom_xpos
  cvel cdof qfrc_bias qfrc_passive qfrc_actuator qfrc_smooth M efc_* ncon
  nefc` into `.mjbin` plus `manifest.json` (§9.3). Subsampling keeps the
  corpus under a few megabytes.
- `test/support/parity.ml`: compares a named array against a fixture with a
  per-quantity tolerance and prints the worst index, the two values and the
  relative error. One implementation, used by every later milestone.
- `tools/inspect.ml`, built as `tools/inspect.exe`: prints a model's arrays, its
  shape keys, and a parity report against a fixture. It is the tool every
  later milestone's demo reuses, so it comes first.
- **The linear-algebra spike** (`bench/spike_la.ml`, report in
  `doc/spike-la.md`): the four measurements of `PLAN.md` §3.3 re-run on this
  machine, at `nv` 8/27/39/60, eager and jitted, alone and at `vmap` 64/256:
  1. `Nx.cholesky` + `Nx.solve_triangular`;
  2. the same factorization written as a `scan` over `[B; n; n]`;
  3. a blocked (panel) variant of the same;
  4. `Nx.solve` (LU) for the comparison the plan warns about.
  Also measure the first-call cost of each, and the composed "step-shaped
  program" from `PLAN.md` appendix A.
- One ADR per decision this milestone settles, numbered `0001`–`0009`.
- `doc/raven-bottlenecks.md`: the measurement-backed request to the Raven
  project that `PLAN.md` §4 and §13 Q7 point at. Two questions, with the
  spike's numbers: can a compiled program call `nx.cpu`'s factorization
  kernels, and can the lowering keep a factorization as a loop instead of an
  unrolled graph? This is the only deliverable in this milestone that another
  project consumes, so it is written before the gate.

**Acceptance.**

1. `dune build @packages/mjc/runtest` is green, and `packages/mjc/` has no
   dependency the lock does not already contain.
2. Three models round-trip: `.mjcb` written by the converter reads back with
   the same shapes, dtypes and byte counts as the manifest; `tools/inspect.exe`
   prints every array's shape and checksum.
3. `mjbin` integrity: every recorded array reads back bit-identical to the
   file, and the manifest's `nstep` matches the data length.
4. The `.mjcb` reader rejects wrong magic, wrong version, truncated offsets
   and a float32 payload read as float64, each with an error naming the field.
5. `doc/spike-la.md` contains the table and answers G0 explicitly.
6. `Model.shape_key` is a record, printed by `mjc-inspect --shape-keys`, and
   every field that can retrace a compiled program appears in it.

**Gate G0.** The engine is worth building if, for the largest shape key in the
benchmark corpus (`nv = 27`, `nefc = 39`):

| Question | Pass | Conditional | Fail |
| --- | --- | --- | --- |
| first call of a jitted 39×39 factor+solve | ≤ 1.5 s | ≤ 5 s, with shape keys kept under 8 per model | > 5 s |
| 39×39 factor + 3 triangular solves, `vmap` 256 | ≤ 5 µs/env | ≤ 8 µs/env | > 8 µs/env |
| composed step-shaped program, `vmap` 256 | ≤ 100 µs/env | ≤ 150 µs/env | > 150 µs/env |

Conditional means: v1 ships as an analysis and MPC engine, and the pitch to
the lab says so. Fail means stop and write the report; the design note is
still a useful document and the converter and fixtures are reusable.

**Demo.** `dune exec packages/mjc/tools/inspect.exe -- humanoid.mjcb` prints
the model, then replays a fixture and prints the worst parity error for the
arrays that exist at this milestone (the raw recorded ones).

**Deferred.** No kinematics, no dynamics. The viewer's V0 (`VIZ.md`) starts
here and uses the model's geom parameters with identity poses.

---

### M1 — Rigid-body math and forward kinematics (done)

**2 weeks · gate G1**

**Goal.** Every pose, axis and frame in a model, for all four joint types,
matching the C engine to double precision.

**Work.**

- `lib/rigid.ml`: quaternion conventions ([w; x; y; z], body-frame
  composition, Hamilton product), `quat_mul`, `quat_rot_vec`,
  `quat_exp`/`quat_log`, `quat_to_mat`/`mat_to_quat`, `axis_angle`, and the
  `vec3` helpers. Written once, float-generic (`D3`).
- `lib/kinematics.ml`: the body scan in topological order. MuJoCo guarantees
  `body_parentid[i] < i`, so one `scan` over bodies, `Nx.take` the parent row,
  compose, `Nx.scatter` the own row. `xpos`, `xquat`, `xanchor`, `xaxis`,
  `geom_xpos`, `geom_xmat`, `site_xpos`, `site_xmat`, `xipos`, `ximat`.
- Joint-type dispatch is `Nx.where` along the joint axis, selecting the
  coordinate layout for each joint. The joint type is host-static, but the
  per-joint array is a tensor, so the dispatch stays inside the traced
  program. Hinge and slide need no dispatch; ball reads a quaternion out of
  `qpos`; free reads a position and a quaternion.
- `lib/pose.ml`: `qpos` ↔ joint coordinates (`jnt_qposadr`, `jnt_dofadr`),
  including the free-joint layout and the ball-joint quaternion
  normalization. `mj_integratePos` lives here.
- `test/test_rigid.ml`: quaternion laws (`prop`), exp/log round trip, matrix
  agreement, normalization.
- `test/test_fk.ml`: fixture parity, plus the batched and jitted agreement
  checks below.

**Acceptance.**

1. `xpos xquat xanchor xaxis geom_xpos geom_xmat site_xpos xipos` match the
   fixtures within `1e-12` absolute for three models (a 6-DOF chain, a model
   with a free joint, a model with a ball joint) at 32 random states each.
2. FK under `vmap` over 8 stacked states equals the OCaml loop over the same
   states to `1e-14`.
3. `Rune.jit` FK equals eager FK to `1e-14`, and the compiled program's shape
   key is the one printed by `tools/inspect.exe`.
4. A ball joint rotated by `q` then `q⁻¹` returns the identity pose; a free
   joint's `xpos` is its `qpos` translation exactly.
5. `quat_to_mat` and `quat_rot_vec` agree with the composed rotation matrix
   for 1000 random unit quaternions to `1e-14`.

**Gate G1.** None that can stop the project. If FK does not match to `1e-12`,
the difference is a convention (quaternion order, body-frame vs world-frame
axis, `xanchor` definition) and must be resolved before M2, because the mass
matrix inherits it.

**Demo.** `mjc-inspect --fk model.mjcb --qpos q.json` prints every frame, and
`mjc-view` (if V1 has landed) draws the same state.

**Deferred.** `ten_length` and tendon geometry (M5). Mesh and hfield frames
(M8). `mj_kinematics`'s `mesh_pos`/`mesh_quat` output.

---

### M2 — Inertia, bias forces, passive forces, integration (done)

**2.5 weeks · gate G2 · first dynamics**

**Goal.** `qacc = M⁻¹ (f_passive + f_actuator − c)` and one integration step,
for unconstrained models.

**Work.**

- `lib/inertia.ml`: composite rigid body. Reverse `scan` over bodies with
  `Nx.scatter ~combine:\`Add` of each body's 6×6 inertia into its parent, then a
  second pass that builds `M` by projecting each body's `crb` onto the dofs
  between it and the root. Assemble with `einsum` into `[nv; nv]`.
  `factorM` and `solveM` are separate entry points and are the only place a
  factorization is allowed (`D7`).
- `lib/bias.ml`: `mj_rne` equivalent. Forward pass for `cvel` (spatial
  velocity) and `cdof_dot`, reverse pass for the force. Gravity enters as the
  root acceleration, so `qfrc_bias` contains it; the free-fall test pins the
  sign.
- `lib/passive.ml`: joint stiffness and `springref`, joint damping,
  `qfrc_passive`. Fluid forces are deferred (`D12`).
- `lib/integrate.ml`: `mj_integratePos` (quaternion exponential for ball and
  free joints) and `mj_integrateVel`; `euler` and `rk4`. The order of the two
  updates inside `mj_Euler` is copied from `engine_forward.c` and pinned by
  fixture parity, not from memory.
- `lib/forward.ml`: the velocity and acceleration stages, composed from the
  modules above and exposed stage by stage (`D4`).
- `lib/step.ml`: `Step.compile : Model.t -> limits:limits -> step`, where
  `step : Data.t -> Data.t` closes over the model (`D5`).
- `bench/bench_stage.ml`: FK, `M`, `factorM`, `solveM`, RNE, one step, at
  `vmap` 1/64/256; appended to `doc/perf.md`.

**Acceptance.**

1. `M` matches the fixture within `1e-12` relative for three models, and is
   symmetric to `1e-14`.
2. `M` also matches an independent naive assembly (explicit per-joint spatial
   transforms, `O(n²)`, written in the test, not in `lib/`) to `1e-10`
   relative. Two implementations, one answer.
3. `M` is positive definite for 1000 random states: `Nx.cholesky` succeeds
   and `min eigenvalue > 0` (`Nx.eigvalsh`).
4. `qfrc_bias qfrc_passive com cvel cdof` match the fixture within `1e-12`
   relative.
5. Free fall: a body with a free joint has velocity `v(t) = v₀ − gt` to
   `1e-12` after 1000 steps at `dt = 1e-3` (exact for the Euler update, which
   pins the gravity sign and magnitude), and position that matches the
   discrete recurrence `z_n = z₀ + v₀t − ½gt² + ½g·dt·t` to `1e-12`. The
   continuous parabola is not the target: Euler's position update is
   first-order, and a test against the parabola would either fail or hide an
   ordering bug. The quaternion stays unit to `1e-14`.
6. Energy: a frictionless double pendulum conserves total energy to the
   integrator's order over 10 s (`Euler`: linear drift per step, reported;
   `RK4`: `O(dt⁴)` confirmed at two step sizes).
7. `solveM`: `‖M x − b‖ / ‖b‖ < 1e-12` for random symmetric positive
   definite `M` at `nv` 8, 27, 60.
8. The compiled step's first-call time and steady-state time are in
   `doc/perf.md`, and G0's numbers still hold for the real (not synthetic)
   step.

**Gate G2: pass.** With `Step.compile`, the batched step costs 1.5–4.9
us per environment at `vmap` 256 on the corpus (nv ≤ 6), against a 100 us pass
threshold; a compiled single step is 26–32 us. The humanoid-sized measurement
arrives with the M4 corpus.

**Demo.** `mjc-view --model pendulum.mjcb --sim` drops a double pendulum and
lets the viewer drive the loop.

**Deferred.** Implicit integration (M3, needs `M + dt·D`), fluid forces,
`qacc_warmstart`, island discovery.

---

### M3 — Constraints without contacts, and the solvers

**3 weeks · gate G3**

**Goal.** Limits, equality constraints and friction loss assembled as rows,
solved by Newton and CG, and integrated implicitly.

**Work.**

- `lib/constraint.ml`: row assembly for joint limits, equality
  (`connect`, `weld`, `joint`, `tendon`), and friction loss. Fixed-size row
  blocks, `Nx.scatter` into `efc_J`, `efc_pos`, `efc_margin`, `efc_solref`,
  `efc_solimp`, `efc_type`, `efc_id`. The `solref`/`solimp` to impedance
  arithmetic (`mj_solref`, `mj_solimp`, `mj_impedance`) lives here with its
  own unit tests.
- `lib/solver.ml`: `qacc_smooth`, the reduced system as an operator
  (`D6`), Newton (factor `A = J M⁻¹ Jᵀ`, line search on the cost) and CG
  (matrix-free `A x` from `solveM` and `J`). Cone projection is a separate
  function so M4 can extend it. Iteration budgets are compile-time.
- `lib/implicit.ml`: the implicit-in-velocity integrator, factoring
  `M + dt·D`, and `implicitfast`'s diagonal approximation.
- `lib/forward.ml`: `qacc = M⁻¹ (f + Jᵀ λ)` and `efc_force`.
- `doc/adr/0010-solver-policy.md`: iteration counts, line search, warmstart,
  and where this deliberately differs from the C engine.

**Acceptance.**

1. Per-row parity: `efc_pos efc_J efc_R efc_aref efc_b` within `1e-10`
   relative for a model with a weld, a connect, a joint limit and a friction
   loss, at 32 random states.
2. Newton and CG agree on `λ` within `1e-8` on the same problem, from a cold
   start, in at most 10 iterations.
3. A model pinned by welds and pushed by `qfrc_applied` tracks the fixture for
   1000 re-synchronised steps within `1e-9` relative on `qpos qvel qacc`.
   (`qfrc_applied` rather than an actuator, because actuators are M5.)
4. Limit and friction-loss rows: `λ` has the sign the row requires, the row is
   active only at its bound, and complementarity holds to `1e-8`.
5. Implicit integration: a stiff spring-damper chain is stable at
   `dt = 1e-2` where explicit Euler diverges within 10 steps; total energy of
   a damped system decreases monotonically over 1000 steps.
6. Newton's cost decreases monotonically across iterations on 1000 random
   constraint problems (no line-search regressions).
7. `Rune.grad` of the (unrolled) step against central differences away from
   active-set changes: relative error under `1e-6`.

**Gate G3.** Solver semantics, recorded in the ADR: does v1 reproduce
MuJoCo's Newton with line search, or use a fixed number of undamped Newton
steps and document the difference? The rule: if acceptance 3 and 6 pass with
the simpler policy, keep it and document; if not, implement the line search
before M4, because contact makes the same question ten times harder.

**Demo.** `mjc-view --sim --model welded_arm.mjcb` shows a model held by
constraints and pushed by an applied force, with `λ` printed per constraint.

**Deferred.** Contact rows (M4), elliptic cone QCQP (M4), tendon equality
rows (M5), islands (not planned), `efc_state` bit packing (we use a plain
tensor and document it).

---

### M4 — Contact

**4.5 weeks · gate G4 · the risky milestone**

**Goal.** The standard contact benchmarks step and track the C engine.

**Work.**

- `lib/collision/pairs.ml`: the candidate pair list, built at compile time
  from `contype`/`conaffinity`, explicit `<pair>` elements and MuJoCo's
  default rules, capped by `max_geom_pairs`. Per step, a bounding-sphere mask
  per pair (`Nx.where`), giving a fixed `[nconmax]` candidate array.
- `lib/collision/primitive.ml`: the analytic pairs MuJoCo supports for
  sphere, capsule, box, cylinder and plane, including box-face contact
  generation at the per-pair contact cap (`mjMAXCONPAIR`, confirm the value
  in `mjmodel.h`). Each pair writes a fixed-size block with a validity mask.
- `lib/collision/frame.ml`: contact frames, `dist`, `pos`, `frame`, `friction`,
  `solref`, `solimp`, `condim`, `margin`, `gap`, and the pair-wise
  combination rules (`solmix`, friction combination, `priority`, `condim`).
- `lib/constraint.ml`: contact rows. `condim` 1/3/4/6, pyramidal cones,
  `impratio` scaling, contact Jacobians through `mj_jac`-equivalent
  body+point transforms, warmstart carried in `Data`. The elliptic cone's
  QCQP projection is added only if the benchmarks need it (`PLAN.md` §8
  leaves that to this milestone).
- `doc/adr/0011-contact-budgets.md`: how `nconmax`/`nefcmax` are chosen per
  model, what overflow does (`Nx.check`, never a silent drop), and the
  semantic difference from the C engine's growing arena.
- `doc/contact-report.md`: the gate's number sheet.

**Acceptance.**

1. `ncon` and `nefc` match the fixture at every step for four models
   (hopper, walker2d, ant, humanoid) driven by the same controls.
2. Contact set parity: for each step, the multiset of (geom pair, contact
   position within `1e-6`, normal within `1e-6`) matches the fixture.
3. Re-synchronised per-step parity: `qpos qvel qacc` within `1e-8` relative
   on at least 99% of 1000 steps, and the outlier steps are named with their
   active-set difference.
4. Friction cone: normal force ≥ 0, tangential force inside the cone to
   `1e-8`, complementarity to `1e-8`.
5. `λ` from Newton matches the fixture within `1e-6` relative on the same
   state.
6. Free-run shadowing: report the number of steps until relative state error
   reaches `1e-3` for each model. No gate on the number; a number under 50
   for a model is a finding to explain.
7. Energy: total energy drift over 10 s is not worse than the C engine's on
   the same model and controls, measured the same way.
8. V3 draws contacts, and its scene test — not its picture — reproduces the
   fixture's contact list for one state.

**Gate G4.** Go/no-go on the hard geometry and on long-horizon claims:

| Outcome | Decision |
| --- | --- |
| 1–5 pass | v1 is a usable engine; continue to M5 |
| 1–5 pass with ≤ 2 named semantic gaps | continue, list the gaps in `PLAN.md` §13, and do not claim exact parity in the docs |
| a benchmark diverges within 20 steps for reasons other than active sets | stop; do not build M5–M6 on an unstable core |
| box-box or cylinder contacts are the only failures | ship v1 without them, move them to M8, and say so |

**Demo.** `mjc-view --sim --model humanoid.mjcb --perturb` with contact points
and normals drawn. `mjc-ant.exe` steps ant for 10 s and prints the shadowing
horizon.

**Deferred.** Convex mesh, hfield, SDF (M8). Elliptic cone QCQP if the
pyramidal path is enough for the benchmarks; it is listed in `PLAN.md` §8 as
M4, so decide here and record it.

---

### M5 — Actuators, transmissions, sensors, tendons

**2.5 weeks · gate G5**

**Goal.** The rest of the model surface the lab's models use, so that a model
written for MuJoCo loads and behaves.

**Work.**

- `lib/transmission.ml`: moment arms for joint, site and fixed-tendon
  transmission. Dense `[nu; nv]` moment matrix (`nu` and `nv` are small);
  sparse patterns are not worth the complexity at these sizes.
- `lib/actuator.ml`: motor, position, velocity, intvelocity, damper, general
  gain/bias; `act` dynamics for `intvelocity`; `gear`, `ctrlrange`,
  `forcerange`, `ctrllimited`. Cylinder and muscle are deferred.
- `lib/tendon.ml`: fixed tendons only. `ten_length`, `ten_velocity`, `ten_J`,
  tendon limits and tendon equality rows (wired into `constraint.ml`).
- `lib/sensor.ml`: the state-projection family and subtree reductions
  (`Nx.reduce_segments`): jointpos/vel, framepos/quat/vel, gyro,
  accelerometer, touch, contact distance/force, actuator and tendon force.
- `tools/convert_mjcf.py`: carry the actuator, sensor and tendon tables.

**Acceptance.**

1. `qfrc_actuator` matches the fixture within `1e-12` relative for every v1
   actuator type on a single-joint model.
2. `act` trajectories match the fixture within `1e-12` relative over 1000
   steps for `intvelocity`.
3. `ten_length ten_velocity ten_J` match within `1e-12` relative; a
   tendon-driven finger tracks the fixture over 1000 re-synchronised steps
   within `1e-9`.
4. Every sensor in the v1 list matches within `1e-12` relative, including a
   subtree reduction on a 17-body model.
5. A full humanoid with position actuators tracks the C engine over 1000
   re-synchronised steps within `1e-8` relative.

**Gate G5.** Whether MJCF reading is now on the critical path. If the lab's
own models are only available as MJCF and the converter loses information,
promote M10 and run it before M6, adding three to four weeks to the schedule.

**Demo.** `mjc-view --sim --model finger.mjcb` with a slider for the tendon
actuator (V3's controls).

**Deferred.** Muscle, adhesion, cylinder, PID, dcmotor, orientation and
general actuators beyond the gain/bias polynomial (M10 or never). Spatial
tendons (M9). 49 sensor types; v1 ships the list in `PLAN.md` §8.

---

### M6 — Differentiability

**2.5 weeks · gate G6 · the reason for the project**

**Goal.** A gradient of the step, of the answer rather than of the last
iterate, at a cost independent of the solver iteration count.

**Work.**

- `lib/solver.ml`: the solver's fixed point written as a residual
  `F(λ) = 0`, and a `Rune.custom_vjp` whose pullback solves the KKT system
  using the `M` factorization the step already holds. Alternative, tested in
  the same milestone: `Rune.root ~residual` with `linear_solve` set to the
  reduced-system solver.
- `lib/dynamics.ml`: `dynamics : Data.t -> Data.t`, the compiled closure
  from `Step.compile`, with the `M`-weighted inner product that `dilqr`
  expects. `Model.t` is captured, not passed.
- `lib/forward.ml`: `remat` annotations on the stage boundaries so a long
  horizon does not keep every intermediate.
- `test/test_grad.ml`: `Rune.check_grads` on the step, central differences of
  the whole step, and `Rune.check_transpose` on the custom rule.
- `doc/grad-report.md`: the gate's number sheet — gradient error by model,
  cost ratio of the reverse pass, peak memory with and without `remat`, and
  the bounded-magnitude table at contact transitions.
- `bench/bench_grad.ml`: reverse-pass time and peak memory versus horizon,
  with and without `remat`.

**Acceptance.**

1. `Rune.check_grads` on a contact-free step (float64, `eps = 1e-6`) passes
   with relative error under `1e-7`.
2. The solver gradient equals central differences of the converged solve
   within `1e-6`, and is unchanged (within `1e-5`) when the iteration budget
   goes from 3 to 10. That is the check that the gradient is implicit, not
   unrolled.
3. Gradients with respect to model parameters (masses, `geom_size`, gains)
   match finite differences within `1e-7`.
4. With `remat`, a 1000-step horizon's peak reverse-pass memory is at most
   one third of the un-rematted peak, measured and recorded in
   `doc/perf.md`.
5. On light contact, the gradient is finite and bounded: no NaN or Inf across
   1000 random states, and the maximum absolute value is reported.
6. `dilqr` runs an iLQR solve on a two-link arm through 20 steps of the mjc
   dynamics in the same process, and its gradient matches `Rune.grad` of the
   same objective.

**Gate G6.** Gradient quality, not existence. Report: gradient error away
from transitions, max magnitude at transitions, and the cost ratio (reverse
pass / forward step). If the reverse pass is more than 10× the forward step,
the engine is not trainable in the loop and the docs must say so.

**Demo.** `mjc-ilqr.exe` swings a double pendulum to a target by differentiating
through contact-free dynamics, printing the loss per iteration.
`mjc-view` shows the iterates (V5).

**Deferred.** Second-order derivatives. Gradients through meshes and hfield
(M8). Gradients through the integrator's `Nx.where` at joint limits, which are
piecewise and reported, not fixed.

---

### M7 — Parity, performance, dtype consolidation

**2 weeks · gate G7**

**Goal.** The package is finished as a library: numbers quoted, float32
tested, CI split, docs complete.

**Work.**

- `bench/bench_step.ml`: the table of `PLAN.md` §6, re-measured on the
  finished engine: per-env step times at `vmap` 1/64/256/1024/8192 for six
  models, plus compile time per shape key, plus MuJoCo C on the same machine
  as the reference row.
- Kernel-count audit: count the dispatches in one compiled step (`tolk`
  tracing or a counter in a debug build) and list the top five costs.
- One attempt at the open performance question: a fused or blocked
  factorization, or compiled access to the host kernels if Raven gains it
  (`PLAN.md` §13 Q7). Keep or drop with numbers in `doc/perf.md`.
- Float32: `Mjc.Make` instantiated at `float32` (`D3`), the parity suite re-run
  at looser tolerance (`1e-4` relative), and the dtype recorded in the model.
- CI: `@packages/mjc/runtest` stays under two minutes warm; the full parity
  and benchmark suites move to `packages/mjc/test/slow/` and a nightly
  workflow, with a documented command to run them locally.
- `README.md` with a working example, `doc/mjcb.md` finalized, `PLAN.md` §13
  updated with the measured answers, and `CHANGES.md` entries.

**Acceptance.**

1. The benchmark table exists with CPU and (if available) one GPU column,
   and the MuJoCo C row on the same machine.
2. Steady-state step time meets G2's conditional threshold (150 µs/env at
   `vmap` 256, `nv = 27`), or the miss is explained in `doc/perf.md`.
3. float32 passes the same acceptance tests as float64 at `1e-4` relative,
   including the contact benchmarks.
4. `dune build @packages/mjc/runtest` under 2 minutes warm;
   `dune build @packages/mjc/runtest-slow` under 20 minutes.
5. `odoc` builds with warnings as errors, and every public value in `lib/*.mli`
   has a doc comment.
6. A user can write a model, load it, step it, differentiate it and save a
   checkpoint using only the README.

**Gate G7.** Whether to continue into M8–M11. The rule: continue with meshes
only if a lab model needs them; continue with MJCF only if a user writes
MJCF; continue with the GPU only if there is a named consumer and the CPU
path is within 5× of MuJoCo on the same machine.

**Demo.** `dune exec packages/mjc/bench/bench_step.exe -- --json` prints the
table; the README embeds it.

---

### M8 — Convex meshes and hfield (optional, gated)

**4+ weeks · optional, after G7**

`PLAN.md` records the reasoning: GJK/EPA is iterative and data-dependent;
MJX uses a branchless SAT and documents the vertex-count cliff. Work: the
branchless SAT for convex-convex, mesh assets with convex hulls computed at
conversion time, `geoms` of type mesh, hfield sampling, and the contact row
path for both. Acceptance: a mesh-collision scene steps and matches the C
engine's contact count and positions within `1e-5`. **Abandon criterion:** if
a 32-vertex pair costs more than 20 µs/env or the contact set matches on
fewer than 95% of steps, publish the measurement and stop; the v1 primitive
path stays.

### M9 — Spatial tendons with wrapping (optional)

**2.5 weeks · optional, after G7**

Cylinder and sphere wrapping, path-dependent wrapping points, `ten_J` for
spatial paths. Acceptance: a routed tendon model (`PLAN.md` names a finger)
matches `ten_length` and `ten_velocity` within `1e-8` and tracks the C engine
for 500 steps. Deferred unless a lab model needs it.

### M10 — Native MJCF reader (conditional)

**3.5 weeks · conditional, after G5**

XML parsing, defaults and class inheritance, `include`, asset loading, model
build, and an `mj_setConst` equivalent. A new XML dependency must be approved
first; the alternative is a small parser in-package. Acceptance: the fixture
corpus loads from MJCF and compiles to byte-identical `.mjcb` arrays.
Decision point: G5, because the lab's model format decides whether this is on
the critical path.

### M11 — GPU and throughput scale-out (optional)

**3 weeks · optional, after G7**

Measure the step on the available accelerators at batch 1k–16k in float32
(`Raven`'s CUDA/Metal/AMD sets), fix the placement boundary, and report
per-env cost and the CPU crossover. Acceptance: a table of per-env step times
at batch 1, 64, 1024, 8192 for three models on CPU and one accelerator, or a
written explanation of which Raven capability is missing.

## 6. The viewer thread

Full design and options in [`VIZ.md`](VIZ.md). The schedule, for reference:

| Id | Milestone | Size | Lands at | Needs |
| --- | --- | --- | --- | --- |
| V0 | Still frames and movies through `geom.viz` | 0.5 wk | M0 | model reader |
| V1 | Native window (`raylib`), orbit camera, joint sliders | 1 wk | M2 | FK |
| V2 | Dynamics overlays: COM, inertia boxes, velocity, trail | 0.5 wk | M2 | dynamics |
| V3 | Contacts, forces, replay log, pause/step | 1 wk | M4 | contact |
| V4 | Browser backend over WebSocket (optional) | 1.5 wk | M7 | nothing new |
| V5 | Batch grid, ghost trajectories, screenshots in docs | 0.5 wk | M7 | batch |

`V1` is the milestone that makes the rest of the work checkable by eye. It is
small on purpose: the scene conversion is a pure function of `Model.t` and
`Data.t`, and each backend is a few hundred lines.

## 7. Schedule and parallel tracks

One person, whole weeks (the bar for each milestone is its size rounded up):

```
            1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
M0        ######
M1              ######
M2                    #########
M3                             #########
M4                                      ###############
M5                                                     #########
M6                                                              #########
M7                                                                       ######

V0        ###
V1                    ######
V2                          ###
V3                                         ######
V4                                                                             ######
V5                                                                             ###
```

Two people, with the second on the viewer and the harness:

| Person | M0–M2 | M3–M4 | M5–M6 | M7+ |
| --- | --- | --- | --- | --- |
| Engine | M0, M1, M2 | M3, M4 | M5, M6 | M7 |
| Tooling | V0, V1, V2, converter, fixture corpus | V3, log writer | V4 (if approved) | V5, docs, benches |

The split works because the converter, the fixture corpus and the viewer share
one interface (`Model.t`, `Data.t`, `mjcb`) and nothing else.

## 8. Gates and stop-losses

| Gate | At | Question | Artifact | If it fails |
| --- | --- | --- | --- | --- |
| G0 | M0 | Can a compiled step afford its factorizations? | `doc/spike-la.md` | stop; the design note stands as a report |
| G1 | M1 | Do conventions match the C engine? | parity report | fix conventions; do not proceed |
| G2 | M2 | Is the composed step within budget? | `doc/perf.md` | return to the factorization; do not build constraints |
| G3 | M3 | Solver policy: line search or not? | `doc/adr/0010` | implement line search before contact |
| G4 | M4 | Do the standard benchmarks track? | `doc/contact-report.md` | ship without the failing primitive pairs, or stop |
| G5 | M5 | Is MJCF on the critical path? | `doc/log.md` | promote M10 before M6 |
| G6 | M6 | Is the gradient usable and affordable? | `doc/grad-report.md` | document the limit; the engine is an analysis tool |
| G7 | M7 | Continue into M8–M11? | `doc/log.md` | stop at a finished v1 |
| G8 | M8 | Is convex collision usable? | measurement | abandon meshes |

Every gate is written down before the milestone starts, with its numbers, so
that the decision is not made by the person who just spent four weeks on the
code.

## 9. Parity and test strategy

### 9.1 Layers

| Layer | What it pins | Tool | Runs |
| --- | --- | --- | --- |
| laws | quaternion algebra, `M` symmetry and positive definiteness, cone residuals, `ten_J` consistency | `windtrap` `prop` | every `runtest` |
| fixtures | each stage's arrays against the C engine | `test/support/parity.ml` | every `runtest` for small models |
| re-synchronised steps | the composed stage, from the C engine's state | parity differ | `runtest` for 200 steps; nightly for 1000 |
| free-run shadowing | long-horizon behaviour | report only | nightly, recorded in `doc/perf.md` |
| gradients | custom rules and implicit derivatives | `Rune.check_grads`, central differences | `runtest` |
| performance | no regressions | `bench/` baselines | nightly; thresholds in `doc/perf.md` |
| viewer | scene construction, log round trip | `windtrap` | `runtest`, no window |

### 9.2 Re-synchronised parity, and why

For each step `t`, the fixture holds `x_t` as the C engine produced it. The
test computes `f(x_t, u_t)` in mjc and compares with `x_{t+1}`. This
attributes every error to one step and cannot be masked or amplified by
chaos. Free runs are still measured, but the number reported is the
*shadowing horizon*: the first step at which relative state error exceeds
`1e-3`. Two correct implementations of rigid contact have a finite shadowing
horizon; quoting a 1000-step tolerance instead would be quoting luck.

### 9.3 The fixture corpus

| Fixture | Model | `nv` | Used from | Purpose |
| --- | --- | --- | --- | --- |
| `pendulum` | single hinge | 1 | M1 | every array, by hand-checkable derivatives |
| `chain6` | 6 hinges | 6 | M1 | FK and CRB on a tree |
| `floating` | one free joint | 6 | M1 | free-joint layout, integration |
| `ball` | one ball joint | 3 | M1 | quaternion `qpos` |
| `welded` | two bodies, weld + connect | 6 | M3 | constraint rows, `λ` |
| `stiff` | spring-damper chain | 12 | M3 | implicit integrator |
| `hopper` | MuJoCo benchmark | 6 | M4 | contact, simple |
| `walker2d` | MuJoCo benchmark | 6 | M4 | contact, feet |
| `ant` | MuJoCo benchmark | 8 | M4 | contact, many pairs |
| `humanoid` | MuJoCo benchmark | 27 | M4, M5 | the size and cost target |
| `finger` | tendon-driven | 3 | M5 | fixed tendon, sensor |
| `arm` | actuated 6-DOF | 6 | M5 | actuator and sensor parity |

Rules: fixtures are generated, never hand-edited; each manifest names the
MuJoCo version, the model file hash, the controls and the sampling; the
corpus stays under 5 MB by storing every step for the first 200 and every
tenth after that. A fixture whose arrays disagree with a freshly recorded run
of the pinned MuJoCo version is regenerated with `tools/record_fixture.py`,
not patched.

### 9.4 Tolerances

| Quantity | Tolerance | Why |
| --- | --- | --- |
| smooth stages (`xpos`, `M`, `c`, `efc_J`) | `1e-12` relative | same arithmetic, same dtype, different summation order |
| composed contact step, re-synchronised | `1e-8` relative | nonlinear projection amplifies the stage error |
| `λ` | `1e-6` relative | active-set conditioning |
| gradients | `1e-6` (central differences), `1e-7` (`check_grads`) | difference truncation dominates |
| free-run, until `1e-3` | reported, not gated | chaotic divergence is a property of the model |

A tolerance is stated with the quantity it applies to, in `parity.ml`, once.
A test that needs a different number argues for it in a comment.

### 9.5 Shape keys and compile time

`Model.shape_key` must contain every value that changes the traced program:
`nv nbody ngeom njnt na nsensor ntendon nsite nconmax nefcmax` and the dtype.
Nothing else. Contact counts are budgets, so a model has one key, not one per
step. The test suite asserts that two models with equal shape keys get equal
keys, and that a second `Step.compile` of the same model does not pay a
first-call cost (the `jit` cache is hit, measured).

The budget: at most 8 keys per model in the parity corpus, each under 1.5 s to
first call, and the whole `runtest` under two minutes warm. Adding a key is a
documented cost, not an accident.

## 10. Decisions

`D#` are settled by this plan unless the user says otherwise; each gets an
ADR in `doc/adr/` when its milestone starts. `Q#` are open and have a deadline.

| Id | Decision | Why | Recorded |
| --- | --- | --- | --- |
| D1 | `packages/mjc/`, libraries `mjc` and `mjc.viz`, docs at the package root | matches `geom`; the viewer can depend on graphics without the core doing so | M0 |
| D2 | The core carries its own rigid-body math; it does not depend on `geom` | `geom` is float32-only today and `PLAN.md` §13 Q5 leans this way; a physics engine's quaternion code has different requirements from a GA library's | M0 |
| D3 | dtype by functor: `Mjc.Make (Config)`, instantiated at float64 and float32 | one dtype per model; `cholesky` and `solve_triangular` require matching dtypes, so a per-value dtype parameter buys type errors, not flexibility | M0, used M7 |
| D4 | Forward stages are public (`Kinematics`, `Inertia`, `Bias`, `Constraint`, `Solver`, `Integrate`), `Forward.t` and `Step.t` compose them | answers `PLAN.md` §13 Q6: parity debugging needs the stages, and it costs nothing | M0 |
| D5 | Budgets (`nconmax`, `nefcmax`, `max_geom_pairs`) are per world and compile-time; overflow is `Nx.check` with the model name and the count | static shapes; a silent drop is unacceptable | M0 |
| D6 | Dense everything; documented cap `nv ≤ ~100` | no sparse tensors in `Nx`; MuJoCo is dense below `nv = 60` too | M0 |
| D7 | `Nx.solve` (LU) is not used in the hot path; symmetric positive definite systems use `cholesky` + `solve_triangular`, factored once and reused | measured 3–7× and no batching gain for LU (`PLAN.md` §3.3) | M0, audited M7 |
| D8 | `.mjcb` is the public model format in v1; the converter is dev-time Python, not linked | removes `mj_setConst` and the MJCF parser from the critical path | M0 |
| D9 | `Model.shape_key` is an explicit record, printed by `tools/inspect.exe` | compile time is a first-class cost | M0 |
| D10 | The solver's iteration count is a model parameter; its budget is a compile-time constant | static shapes; MuJoCo's default of 100 iterations is unaffordable at ~170 ns per `scan` step | M3 |
| D11 | `Mjc_viz.Frame` is the backend-neutral boundary; each backend is a separate library chosen by the executable | keeps `mjc` free of graphics dependencies and lets a backend be added without touching the engine | V0 |
| D12 | Fluid forces, `qacc_warmstart`, islands, sleep and CCD are out of v1 | `PLAN.md` §8; they do not appear in the benchmark corpus | M2 |
| Q1 | Does the lab have the models in `PLAN.md` §8, or a mesh robot? | decides M8's priority | before M4 |
| Q2 | Is GPU in year one? | decides M11 | before M7 |
| Q3 | Is MJCF reading required from day one? | decides M10's position | G5 |
| Q4 | Trajectory equivalence or "same physics, different rounding"? | decides tolerances and how much of MuJoCo's solver policy to copy | before M3 |
| Q5 | Which viewer dependency is approved? | `raylib` is approved for V1; the browser backend for V4 is not | V1 (M2) |

## 11. Deliberate omissions

| Omitted | Where it went | Cost of omitting |
| --- | --- | --- |
| sparse `M`, sparse `efc_J` | never | model size capped near `nv = 100` |
| PGS solver | not in v1 | v1 ships Newton and CG; a sequential PGS is `PLAN.md` §4's "possible, medium", and MuJoCo's PGS timings cannot be reproduced without it |
| islands | never | no per-island parallelism; MuJoCo 3.x's island discovery is not reproduced |
| flex, skin, composites, plugins | never | separate engines |
| IPC, continuous collision, sleep | never | some scenes tunnel where MuJoCo 3.17 would not |
| rendering in the engine | never | the viewer is a separate library (`VIZ.md`) |
| `mj_forward`'s `mj_step1/2` split | M2 (`Forward`), by stage | callers that want warmstart control use `forward` then `integrate` |

## 12. Log

`packages/mjc/doc/log.md` holds one entry per milestone, appended, never
rewritten:

```
## M4 — contact (2026-??-??)
Gate G4: pass, 4 models, 99.6% of 1000 steps within 1e-8.
Shadowing horizon: hopper 412, walker2d 238, ant 187, humanoid 121.
Known gaps: box-box point count differs when the boxes are coplanar.
Shape keys added: 3 (humanoid 27/39, ant 8/24, walker 6/18).
Measured: humanoid step 96 us at batch 1, 41 us/env at 256 (CPU).
Decisions: elliptic cone deferred (pyramidal passes the benchmarks).
```

The log is a record, not a measurement: a number in it is quoted from the
milestone's gate artifact, and the artifact is where the measurement lives.
