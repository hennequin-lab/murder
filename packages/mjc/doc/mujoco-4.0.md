# MuJoCo 4.0 against `mjc`

Status: **audit and plan, no code.** Companion to `PLAN.md` (design) and
`MILESTONES.md` (execution). Every number below was measured on this machine
with the two engines installed side by side; the commands are in §2.3 so the
audit can be repeated when the next release lands.

The short answer: 4.0 changes nothing that `mjc` already computes, and adds
four things it does not have. Two of them are cheap and belong in v1's
neighbourhood (contact surface velocity, elasto-plastic spring references), one
is a data-surface catch-up, and one — speculative contacts — is the first
cheap form of continuous collision DeepMind has shipped, which moves a standing
"never" in `MILESTONES.md` §11 to "priced, gated". The port is fifty to two
thousand times slower per step than the C engine (`doc/perf.md` §2), so
anything that adds per-step work has to argue against the solver's schedule
first: that is why the order below is not the order of the changelog.

---

## 1. What was compared

The corpus records its own provenance: `test/fixtures/manifest.json` says
`"mujoco": "3.17.0"`, and 3.17.0 is the release *before* 4.0.0 (22 October and
6 November 2026, adjacent tags, no release between). So the audit is
3.17.0 → 4.0.0 and nothing else, which is why it could be exhaustive.

Three sources:

| Source | What it gave |
| --- | --- |
| `doc/changelog.rst` at 4.0.0, plus the "Upcoming" section | the intent: 61 numbered changes — 3 significant, 5 engine, 19 deformables, 4 compiler, 7 rendering, 2 Python, 1 samples, 2 models, 2 documentation, 16 bug fixes |
| `git diff 3.17.0 4.0.0 -- src/engine src/xml` | where the work actually is: 80 files, +12 616 / −5 926 lines. Most of it is flex (`engine_core_util.c` +2 193, all interpolated-flex weights and rods), the discrete-integrator metric (`engine_metric.c` 2 491), and the sparse Newton for flex (`engine_solver.c` 1 195) |
| the corpus, stepped under both | what a v1 model actually sees |

## 2. The measurement

### 2.1 Two engines, one corpus

Each of the 29 fixture models is loaded from its committed `.xml`, stepped 200
times under both engines from the same perturbed state with the same controls,
and the trajectories are compared. Three seeds per model.

| Models | Result |
| --- | --- |
| 27 of 29 | `max abs Δqacc` ≤ 2.4e-8, relative `Δ(sum qpos)` ≤ 1e-12 |
| `quadruped` | `Δqacc` 8.1e-3, relative `Δqpos` 1.3e-6 — roundoff amplified, see below |
| `cyl` | not comparable — see §2.2 |

Every model without contacts is bit-identical. The differences that do appear
are last-bit reassociation, not algorithm: 4.0 adds AVX kernels for quaternions
and 3/4-vectors (`engine_util_spatial_avx.h`, `engine_util_blas_avx.h`) and even
the scalar fallback changed associativity — `mju_normalize4` now sums
`(v0²+v1²) + (v2²+v3²)` where 3.17 summed left to right.

**The control experiment.** `quadruped` is the largest entry and it is the
model's own sensitivity, not the release. Advancing `qpos[0]` by one ulp
(2e-17) at the same seed, inside 4.0 alone, moves the trajectory by 3.4e-7
relative over 200 steps — the same order as the 1.3e-6 the two versions
produce. The same one-ulp perturbation moves `hopper` by 3.5e-15, `boxes2` by
2.0e-16 and `humanoid` by 2.3e-14. A bouncing quadruped under 200 steps of
random torque amplifies rounding on its own; the two engines differ by exactly
that much and no more.

Contact geometry is bit-identical everywhere, including `cyl`, which is what
`test/test_contact.ml` compares at 1e-12. The corpus does not have to be
re-recorded to stay faithful to the engine it names.

### 2.2 The `cyl` fixture is degenerate

`test/fixtures/cyl.xml` is four bodies, each on a hinge about `z`, in
frictionless (`condim="1"`) contact, with `gravity="0 0 0"`. Every geom is a
body of revolution about its own hinge axis, so **every constraint Jacobian row
is exactly zero**: a spin about the axis moves the contact point perpendicular
to every contact normal, and translation is not a degree of freedom.

With `J = 0`, `mj_diagApprox` measures `J M⁻¹ Jᵀ ≈ 1e-15`, so the row impedance
`efc_D` is 1e15 and the row solve returns `efc_force ≈ 9e16` — five rows pushing
against each other with nothing to push against, decided by cancellation. Over
the script's 200 steps and three seeds the two engines land `qacc` 6.1e2 apart
while the contact geometry stays bit-identical: the same physical answer, a
different rounding path through a singular problem. Under `test/test_contact.ml`
this is harmless, because that test compares `contact_dist`, `contact_pos` and
`contact_normal`, which agree bit for bit. But
the model cannot see anything about cylinder *dynamics*, and after 4.0 that is
exactly where the changes are: item 27 ("numerical stability of
`nativeccd`/`multiccd`, can slightly change contacts for convex models") and the
two post-4.0.0 fixes for cylinder edge multicontact (issues #3748, #3752). A
cylinder fixture with gravity, `condim="6"` and one dof that is not a spin is
needed before the cylinder path can be claimed on a 4.0 reference.

### 2.3 Repeat it

```sh
git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco /tmp/mujoco
cd /tmp/mujoco && git diff --stat 3.17.0 4.0.0 -- src/engine src/xml

python3 -m venv /tmp/mjcenv40 && /tmp/mjcenv40/bin/pip install "mujoco-core>=4.0"
/tmp/mjcenv/bin/python  tools/cmp_engines.py record test/fixtures > /tmp/a.json  # 3.17.0
/tmp/mjcenv40/bin/python tools/cmp_engines.py record test/fixtures > /tmp/b.json  # 4.0.x
python3 tools/cmp_engines.py diff /tmp/a.json /tmp/b.json
```

`tools/cmp_engines.py` is the script the tables above came from: `record` under
each interpreter, then `diff`, which needs neither MuJoCo nor numpy. Note that
as of 4.0 the Python distribution is split, and
`mujoco-core` is enough for the recorder: `tools/mjcb.py` and
`tools/record_fixture.py` import only `MjModel`, `MjData`, `mj_setConst` and
`mj_forward`, none of which needs Studio or the Filament renderer.

MuJoCo's own per-step cost is unchanged (`perf_mujoco.py`: pendulum 1 µs,
chain6 3 µs, hopper 5→6 µs, walker 5 µs, boxes2 11→10 µs). The gap in
`doc/perf.md` §2 is a fact about `mjc`, not something 4.0 moved.

## 3. What 4.0 changes, against v1's surface

Verdicts: **take** (belongs in the engine), **catch up** (data/format only),
**n/a** (with the reason, so the next audit does not re-derive it).

| # | 4.0 change | v1 has? | Verdict |
| --- | --- | --- | --- |
| 1 | `speculative` flag: speculative contacts, conservative advancement | no CCD at all | **take, gated** — §4.3 |
| 2 | Elasto-plasticity, `mjData.plastic` (`mjSTATE_PLASTIC`): joint `springyield`, tendon `springyield`, flex `yield` | no | **take** the joint and fixed-tendon parts — §4.2; flex is n/a |
| 3 | Python split into `mujoco-core` / `mujoco-studio` | — | **catch up** (dev-time only) |
| 4 | `sidesite` is now a homotopy, not a half-plane test — a tendon can jump | n/a | spatial tendons are M9 |
| 5 | `RK4` is now fourth order for ball and free joint orientations (Runge-Kutta-Munthe-Kaas correction) | no: loader rejects `integrator=1` | **take with RK4** — §4.4 |
| 6 | `mjContact.surfacevel[3]`, populated from `geom_surfacevel`; `mj_addSurfaceVel` rewritten | no | **take** — §4.1 |
| 7 | `xiquat`, `geom_xquat`, `site_xquat` in `mjData` | matrices only | **catch up** |
| 8 | `mjModel.tendon_type`, `mjtTendon`: the fixed/spatial distinction is now a field | inferred from the first wrap | **catch up** |
| 9–17 | Flex: cable/composite removal, per-edge twist bodies, corrected elastic constants, `flexvert` equality removal, per-element materials, 10-value bending records | n/a | flex is out of v1 and out of the next milestone; if flex ever lands, `MILESTONES.md` §11's "separate engines" already says so |
| 18–22 | Sparse `Newton` for flex: fill-reducing ordering, incremental active-set Hessian updates, elliptic cone folding, pre-ordered Cholesky; Newton for flex on articulated bodies | no elliptic cones, dense only | n/a as written; the *idea* is priced in §5 |
| 23–26 | Interpolated-flex contact, passive contact, Jacobian fixes | n/a | flex |
| 27 | `nativeccd`/`multiccd` numerical stability; "can slightly change contacts for convex or mesh models" | cylinder pairs analytic, no convex | measured: no change for the corpus; see §2.2 for why the corpus cannot see it either |
| 28–30 | Compiler: duplicate `include` rejected, invalid models rejected, `fullinertia` bounded by `boundinertia` | n/a | no MJCF reader (M10) |
| 31 | `mj_encode` writes binary MJB for `mjSpec` | — | `.mjcb` is ours and name-keyed; no action |
| 32 | `castshadow` per geom; `mjrf_setRenderableCastShadow` | viewer draws contact shadows | **take** in the viewer — §4.5 |
| 33 | Filament horizon haze and skybox visibility for infinite planes | floor is a finite quad | **take** in the viewer |
| 34 | Light fallback intensity scaled by diffuse brightness, so the brightest light casts the shadows and is deterministic | — | **take** in the viewer |
| 35–38 | Filament material flag, `mjrf_flush`, post-processing flag, `headlight="auto"` | — | Phase 2 of `VIZ.md` at the earliest |
| 39–40 | Python 3.10 dropped; docs and agent skills bundled in the wheel | — | n/a |
| 41 | `sample/record.cc` replaced by `render --duration --fps` | — | n/a (our clips come from `viz`) |
| 42–43 | Example models: `model/plastic/`, `model/speculative/` | — | **use as fixtures** when §4.2 and §4.3 land |
| 44–45 | Studio docs, model-tuning guide, `mujoco.sysid` | — | read `CTuning` before re-picking any solver defaults; no code |
| 46 | `accelerometer`/`framelinacc` report proper acceleration on static and mocap bodies | sensor not implemented | n/a; note it when the acceleration-stage sensors land |
| 47 | Actuators saturated by `joint`/`tendon` `actuatorfrcrange` no longer contribute implicit derivatives | `actfrcrange` not implemented | n/a. `Actuator.damping` already zeroes the coefficient of an actuator whose force sits on its own `forcerange`, which is 3.17's rule and survives 4.0 |
| 48–49 | Sleep filtering fixes; `mj_setConst` wakes trees and stops overwriting camera/light poses | no sleep, no `setConst` | n/a |
| 50 | `<pair>`-defined contacts mix `solref` with `solmix`; element-wise minimum before | no `<pair>` support | n/a |
| 53 | `dcmotor` parameter handling | not in v1 | n/a |
| 54 | Sphere/cylinder *spatial tendon* wrap stability | fixed tendons only | n/a (M9) |
| 55–56 | `mj_ray` SDF scaling; thread pool on reset | — | n/a |
| 57 | `mju_mat2Rot`: normalises, escapes 180° saddles | flex polar decomposition only | n/a |
| 58–61 | Flex cotangent weights, `mjSpec` edits, MJCF save/load, legacy `.msh` winding | — | n/a |

"Upcoming" (post-4.0.0, read as a warning): the sparse Cholesky introduced in
4.0.0 merged rows with different descendants and gave `Newton` a wrong Hessian
factor with a sparse Jacobian (issue #3767). v1 builds a dense Jacobian below
`nv = 60`, so the corpus is unaffected, but **the reference version for any
re-record must be 4.0.1 or later**, never 4.0.0.

## 4. The items worth taking

### 4.1 Contact surface velocity (item 6)

`geom_surfacevel` is six numbers per geom (linear and angular, local frame) and
models a conveyor, a treadmill or a roller. It is not new in 4.0, but 4.0
rewrites it into the form worth porting: collision detection now resolves the
pair's relative surface velocity **once per contact**, in the contact frame,
as `mjContact.surfacevel[3]` (`(tangent1, tangent2, spin)`), and
`mj_addSurfaceVel` shrinks from 60 lines that call `mj_geomSurfaceVelocity` per
side per contact to a 20-line loop that adds the three numbers to `efc_vel`:

```c
// MuJoCo's pyramid rows are (J_n + mu J_k) and (J_n - mu J_k), so the mu
// factor is already in the row and the offset is +/- mu * surfacevel[k]
d->efc_vel[adr + 2*(k-1)]     += mu*sv[k-1];
d->efc_vel[adr + 2*(k-1) + 1] -= mu*sv[k-1];
```

Why it is cheap here: `mjc` already has `Data.efc_vel` and already builds the
pyramidal rows from the contact frame (`lib/constraint.ml`, `lib/collision.ml`).
The work is one tensor of shape `[nconmax;3]` computed beside `contact_frame`,
six numbers per geom in `Params`, and an offset added to `efc_vel`. No new
stage, no new iteration, no change to the solver.

Why it is worth it: a legged robot trained on a treadmill is the lab's most
obvious use, and today the only way to model one is to move the whole world or
to fake the contact force by hand. It is also the only item in this section
that costs a few flops rather than a percentage of the step.

### 4.2 Elasto-plastic spring references (item 2)

4.0 adds ideal elasto-plasticity to joint and tendon springs. The state is one
reference configuration per yielded spring, kept in `mjData.plastic`; the
"return mapping" is a clamp applied after the step, and the spring force reads
the plastic reference instead of `qpos_spring`:

```c
// hinge or slide, in advanceFinish
mjtNum x = d->qpos[padr] - qspring[0];
if (x > yield)       qspring[0] = d->qpos[padr] - yield;
else if (x < -yield) qspring[0] = d->qpos[padr] + yield;
```

and for a ball joint, the same clamp on the rotation vector
(`mju_quatIntegrate(qspring, dif, (r - yield)/r)`). A fixed tendon clamps its
`(lower, upper)` length pair. That is the whole mechanism for the parts v1 has;
everything else in item 2 is flex (`yield` on 2D/3D elements, von Mises return
mapping, per-edge twist) and stays out.

Cost and fit: a new `plastic` state tensor and a `nplastic` topology field, two
new model tensors (`jnt_springyield`, `ten_springyield`), one clamp in
`advance` and a reference swap in `Passive.forward`. It is piecewise linear, so
it differentiates (the plastic branch has zero stiffness, which is the physics,
not a discontinuity to smooth over). It is also the one 4.0 feature that adds a
*state* rather than a *force*, which the ptree checkpoint already handles.

### 4.3 Speculative contacts (item 1) — gated

The headline feature. Its purpose is to stop a fast geom from tunnelling
through a thin one inside one step, which `MILESTONES.md` §11 currently lists
as never: *"IPC, continuous collision, sleep | never | some scenes tunnel where
MuJoCo 3.17 would not"*.

4.0 makes it affordable by refusing to be a general CCD:

1. **Cheap because it is unilateral.** The sweep advances in one direction only
   and always stops on the separated side, so the distance query is GJK's
   *distance* (witness points and a separating normal) and never EPA's
   penetration depth. `lib/gjk.ml` already answers "do these touch"; the
   distance form is the other half of the same iteration.
2. **Cheap because it is bounded.** The per-step travel of a geom is
   `s = |Δt v| + |Δt ω| r_bound`. A geom whose travel is below its own smallest
   AABB half-extent cannot tunnel in one step, so it keeps the static bounds and
   pays nothing; only the ones that can tunnel enter the swept path, where the
   pair filter is a swept bounding-sphere test with a closed-form minimiser
   `τ = clip(-p·v/|v|², 0, 1)`.
3. **Cheap because the contact is ordinary.** The solver needs no change: the
   synthetic `dist = margin − δ` inverts the reference-acceleration equation
   (`δ = (1 − t − b₀)/k₀ · h|v_n|` under Euler) so that the same spring-damper
   arrests `(1−t)` of the closing speed and lands the geom on the surface at
   `t = 1`.

What it needs that v1 does not have: a distance query (rather than a touch
test) over the same primitive pairs, a padded fixed-length conservative
advancement loop (8–10 iterations, host-decided like the solver's schedule), a
swept broadphase filter, and a line-of-sight ray test so a falling box does not
register a speculative contact with the floor through a tabletop. Ray-versus-
primitive for every contact against every geom is the part that scales worst
(`ncon × ngeom`), and it is what the "single-shot" flag disables.

The honest difficulty: it makes collision a function of `(qpos, qvel)` rather
than `qpos` alone, which is a structural change to `Forward`'s stage boundary,
and the port is 50-700× slower than the C engine with collision and the row
buffer as the two remaining costs (`doc/perf.md` §2–§4). A swept filter that
costs +20% of collision on a contact-rich model, to buy a timestep 2-3× larger,
is a good trade on that balance; the same filter on top of a solve that ran
rounds it never needed was not.

**The gate is cleared.** This item was gated on the row solve stopping on its
own KKT condition instead of running a fixed schedule, because until then the
engine's cost was rounds that no problem asked for. That landed (`doc/perf.md`
§3): the loop is a `Rune.iterate` whose carry holds the residual, the compiled
step is 9-14x faster on `boxes2`, `hopper` and `walker`, the answer agrees with
the fixed schedule's to 2.2e-12 relative over the recorded corpus, and `iters`
no longer changes a result.
So P5 below is no longer blocked, and the order of work is a choice rather than
a consequence: it is worth doing on the primitive pairs first (plane, sphere,
capsule, box, cylinder) where distances are analytic, with the convex path left
to M8.

### 4.4 RK4, when RK4 lands (item 5)

`integrator="RK4"` is rejected at load (`lib/topo.mli`), so nothing to fix —
but 4.0 corrects a second-order error that a naive port would reproduce: the
stage angular velocities of ball and free joints must go through
`dexp⁻¹_{-rot}`, not be summed directly (Runge-Kutta-Munthe-Kaas). The kernel
is 20 lines plus a minimax polynomial for the Bortz coefficient
`(1 − (x/2)cot(x/2))/x²` (`mju_derivExpmap`, `engine_util_spatial.c`). Record it
in `MILESTONES.md`'s M2 note ("`RK4`: `O(dt⁴)` confirmed at two step sizes")
now, so that the acceptance test is run on the corrected scheme and not on the
one 3.17 shipped.

### 4.5 Viewer items (items 32–34)

Three small wins for `viz`, none of which touches the engine: per-geom
`castshadow` (a floor should not cast), infinite planes centred under the
camera and blended into the horizon (our demo draws a 10 m quad, and a plane's
`size` is a rendering hint, not a bound — `doc/log.md` says so), and the light
intensity fallback scaled by the diffuse colour so the brightest light is the
one that casts the shadow and the choice is reproducible between runs.

## 5. What we are not taking

- **The sparse-Newton work (items 18–22).** It is 1.4–3× on flex models and
  written for a sparse `J` over `nv`, which v1 does not have (dense below
  `nv = 60`, and `MILESTONES.md` §11 caps the model size there). The one idea
  that transfers is the *decision rule* between refactoring and a rank-1
  update — and it transfers only if `mjc` first stops paying 64 rounds per
  solve (`doc/perf.md` §3). If a Cholesky rank-1 update is ever wanted, it is
  ours to write: Rune has no `cholUpdate`, and the round's free block is a
  principal submatrix of a constant matrix, so a Givens-based update in
  elementwise ops is the shape of it.
- **Elliptic cones.** `lib/constraint.ml` refuses `cone="elliptic"` because the
  row formulation it uses *is* the pyramidal one. 4.0's rewritten
  `ellipticCostDif` (componentwise `N` and `T` before squaring, Lagrange
  identity for `‖Ŵ × V‖²`) is the reference if that ever changes, but it
  changes the solver from a convex QP over a box to a nonlinear cost, which is
  a milestone, not an item.
- **Everything in items 9–17, 23–26, 58** (flex and rods). Out of v1 and out of
  the next milestone; the omission table already says "separate engines".
- **The compiler, Python, rendering-backend and MJCF items.** No reader, no
  wheel, no Filament.

## 6. Defects found on the way, not from 4.0

Recorded here because the audit is the only place they have been written down.

| Defect | Consequence | Fix |
| --- | --- | --- |
| `jnt_stiffnesspoly`, `dof_dampingpoly`, `tendon_*poly`, `actuator_dampingpoly` exist in 3.17 (`<joint stiffness="a b c">` takes 1 + `mjNPOLY` values) and `tools/mjcb.py` writes only the linear coefficient | a model with a quadratic or cubic spring or damper is **silently** simulated linear. The same silent-drop risk applies to the tendon and actuator variants | carry the poly arrays in `.mjcb` and add them to `Passive`/`dof_damping`, or reject nonzero coefficients at load the way unsupported pairs are rejected |
| `README.md` says v1 has "Euler, RK4 and implicit integrators" | the loader rejects both `RK4` and `implicit` (`lib/topo.mli`: Euler and `Implicit_fast`); `MILESTONES.md` §11 does not list them as omissions | correct the README, and add the two rows to the omissions table |
| `lib/collision.mli` says "v1 computes plane-sphere, plane-capsule, sphere-sphere and sphere-capsule" | `Collision.supported` has ten pairs including `Box, Box`, `Plane, Cylinder` and `Sphere, Cylinder`, and `test_contact.ml` exercises them | correct the doc comment |
| `MILESTONES.md` §11 lists continuous collision as "never" | 4.0 ships a cheap form (§4.3), and its gate — the fixed solver schedule — is now gone | change the row to "not in v1; priced in `doc/mujoco-4.0.md` §4.3, unblocked" |
| `test/fixtures/cyl.xml` is rank-deficient (§2.2) | the cylinder path is only covered where it cannot change: frictionless, gravity-free, zero Jacobian | add a `condim="6"`, gravity-on cylinder fixture; keep `cyl` for the contact geometry it does test |

## 7. Plan

Costs are calendar time for one person on the worktree pattern in
`CONTRIBUTING.md`. Order is by value per unit of per-step cost, not by release
order.

### P0 — Parity hygiene (1 day)

- Land `tools/cmp_engines.py`, added with this note: `record` under each
  interpreter, `diff` under any, three seeds of two hundred steps, and the
  degenerate-force flag that found §2.2. It is the one command the next audit
  needs.
- Record in `test/fixtures/manifest.json`'s doc (or `doc/mjcb.md`) that the
  corpus is a 3.17.0 recording, that 4.0 agrees to last-bit reassociation, and
  that any re-record must be ≥ 4.0.1 (issue #3767).
- The five doc corrections in §6.

**Acceptance:** `dune build @packages/mjc/runtest` unchanged; the audit table of
§2.1 reproduces from the committed script.

### P1 — Contact surface velocity (4 days)

- `.mjcb` v15: `geom_surfacevel [ngeom;6]`, `flg_surfacevel` scalar.
- `Collision.contacts`: `contact_surfacevel [nconmax;3]`, the pair's relative
  velocity in the contact frame, computed from the two geoms' local-frame
  surface velocities, their world poses and the contact point.
- `Constraint.impedance`: add the three numbers to `efc_vel` per cone branch
  (pyramidal only).
- Fixture: a `condim="6"` box resting on a plane with `surfacevel`, recorded
  under 4.0.1, plus a treadmill variant of the quadruped.

**Acceptance:** the new fixture matches MuJoCo's `efc_vel`, `efc_force` and 500
steps of trajectory within the existing tolerances (`1e-12` for a state,
`1e-8` for a step); a box on a `surfacevel` plane accelerates exactly as
recorded. **Abandon criterion:** if the velocity has to be applied outside
`efc_vel` — i.e. it needs a term the solver's `b` cannot carry — stop and
write down why; the point of the item is that it is an offset, not a stage.

### P2 — Elasto-plastic spring references (5 days)

- State: `Data.plastic [nplastic]`, `Topo.nplastic`, `jnt_plasticadr`,
  `ten_plasticadr`.
- Model: `jnt_springyield [njnt]`, `ten_springyield [ntendon]`.
- `Passive.forward`: read the plastic reference when the address is
  non-negative (hinge, slide, ball; fixed tendon).
- `Step.advance`: the return mapping, after the position update, gated on a
  `spring` config flag so a model without yield does not pay for it.
- Fixture: `model/plastic/joint.xml` from 4.0.1 (or the minimal two-joint
  reference pair the changelog describes).

**Acceptance:** the elastic and plastic pairs from the 4.0 example models step
identically to MuJoCo for 1000 steps including the plastic branch
(relative `1e-8`); a gradient through a plastic step is finite and matches
central differences away from the yield clamp. **Abandon criterion:** if the
clamp's derivative at exactly `x = yield` makes a training loop's gradient
unstable in the demo, ship the forward model only and document it, the way
`doc/adr/0013` documents the contact-decision gradients.

### P3 — Data-surface catch-up (2 days)

`tendon_type` (replacing the loader's "first wrap is a joint" inference),
`xiquat` / `geom_xquat` / `site_xquat` for the viewer and the sensors, and the
`mujoco-core`-only recorder. No behaviour change; it makes `.mjcb` a faithful
picture of a 4.0 model and makes a future re-record a diff of numbers rather
than of meaning.

**Acceptance:** `.mjcb` round trip bit-exact; the corpus re-records on 4.0.1 and
`runtest-slow` passes at the tolerances of `MILESTONES.md` §9.4.

### P4 — Cylinder coverage (2 days, independent of the rest)

Replace §2.2's fixture gap: a cylinder with gravity, `condim="6"`, at least one
translational dof, recorded under 4.0.1, checked against `contact_*` and
`efc_force`. This is the only item that tests something 4.0 actually changed.

### P5 — Speculative contacts (3 weeks, unblocked)

**Gate: cleared** — the row solve stopped on its KKT condition
(`doc/perf.md` §3), which was the condition this item was waiting on, so the
feature's value is no longer offset by a fixed schedule. Order within the
milestone:

1. Distance queries for the ten supported primitive pairs, and a distance form
   of `Gjk` (witness points, normal, no EPA).
2. Sweep distance per geom from `qvel`, swept sphere filter, `speculative`
   config flag; `dist` inversion for the Euler and `implicitfast` reference
   accelerations.
3. Conservative advancement as a padded scan, per pair, target clearance
   `margin + 0.02 (dist − margin)`.
4. Line-of-sight ray test, behind its own flag (4.0's "single-shot").

**Acceptance:** the three `model/speculative/` scenes (thin-plate Newton's
cradle, interlocking chain, hinged flipper) do not tunnel where 3.17's
discrete detection does; every model in the existing corpus is unchanged with
the flag off; contact sets with the flag on match 4.0.1 within `1e-6`.
**Abandon criterion:** if the swept filter costs more than 35% of the step on
`quadruped` and the timestep cannot be raised by at least 2× on the same
trajectory, publish the measurement and leave CCD out, as §11 says today.

### P6 — Viewer items (3 days, when `VIZ.md` Phase 2 opens)

`castshadow` per geom, infinite-plane horizon, deterministic light choice.

### Documentation to update with P0

- `README.md`: the integrator list (§6), and "Out" gains nothing from 4.0.
- `MILESTONES.md` §11: the CCD row; and a line in §12 pointing here.
- `lib/collision.mli`: the pair list.

## 8. The one question this audit cannot answer

Whether the lab wants a treadmill that MuJoCo agrees with (P1), a spring that
yields (P2), or a contact that cannot tunnel (P5). All three are 4.0 features
this port does not have and each is independently useful; none is on the path
to making `mjc` fast, and speed is what decides whether it is used for training
at all. If the next milestone is chosen for the engine, `doc/perf.md` §3 is the
argument, not this document.
