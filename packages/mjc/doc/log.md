# mjc log

One entry per milestone, appended, never rewritten. A number here is quoted
from the milestone's gate artifact; the artifact is where the measurement
lives.

## M0 — foundations, fixtures, and the linear-algebra gate (2026-11-03)

**Gate G0: pass.** First call of a jitted 39×39 factor+solve is 0.01 s against
a 1.5 s pass threshold; a 39×39 factor plus three triangular solves costs
4.8 µs per environment at `vmap` 256 against a 5 µs threshold. The
step-shaped program is deferred to M2. Full numbers and method in
[`spike-la.md`](spike-la.md).

**The design note's compile-time risk is much smaller at the current pin.**
`PLAN.md` §3.3 measured a first call of 5.6 s at 40×40 at pin `3984341`; the
same program compiles in 10 ms at `4a3bc831`. R1 and §13 Q7 are downgraded,
not closed: steady-state compiled factorization is still 2–5× the eager
kernel.

Delivered:

- the `mjc` library, private for now: `Topo`, `Model`, `Data`, `Mjcb`,
  `Rigid`, `Pose`, `Kinematics`;
- the `.mjcb` model format and its reader/writer, with the format spec in
  [`mjcb.md`](mjcb.md);
- the development-time converter and fixture recorder
  (`tools/record_fixture.py`, `tools/mjcb.py`), and a four-model corpus in
  `test/fixtures/`;
- `tools/inspect.exe`, which prints a model, its shape keys and its frames;
- `bench/spike_la.ml`.

Decisions recorded: [`adr/0001-m0-decisions.md`](adr/0001-m0-decisions.md).

Gaps:

- the converter drops `jnt_limited`'s integer type in favour of a float
  tensor, which is what the kernel wants; MuJoCo's `totals` are not carried;
- the `.mjcb` format has no actuator, sensor or tendon tables yet: M5 adds
  them with a version bump;
- fixtures are 32 states per model: enough for the array parities, not a
  convergence study. M2 adds the trajectory fixtures.

## M1 — rigid-body math and forward kinematics (2026-11-03)

**Gate G1: pass.** Forward kinematics matches MuJoCo's `mj_kinematics` within
1e-12 (mixed relative/absolute) for all four corpus models, at 32 states each,
for `xpos xquat xanchor xaxis xipos ximat geom_xpos geom_xmat site_xpos
site_xmat`. The vmapped pass equals the loop and the jitted pass equals the
eager one to 1e-12. `mj_integratePos` matches within 1e-15, and the
quaternion laws pass as properties.

Delivered:

- `Rigid`: Hamilton product, conjugate, normalization, the active rotation
  matrix in MuJoCo's row-major layout, `rotate vector by quaternion` in
  MuJoCo's exact form, axis-angle and rotation-vector constructors with a
  series guard at zero, `quat_integrate`;
- `Pose`: `qpos`/`qvel` widths and `integrate_pos`, `normalize_quat`;
- `Kinematics.forward`, the body pass and the inertial/geom/site pass,
  following `mj_kinematics1/2` and `mj_local2Global`;
- `test/test_rigid.ml`, `test/test_pose.ml`, `test/test_fk.ml`,
  `test/test_mjcb.ml` — 20 tests, 0.3 s warm.

Deviations and findings:

- **The body loop is unrolled, not a `scan`.** At a 151 ns loop floor and
  ≤ 20 bodies, an unrolled loop of `Nx.get`/`Nx.set` with host indices is
  simpler and no slower; `Nx.slice` also cannot take a traced index, so the
  joints of a body are unrolled anyway. Revisit if M2's composed step misses
  its budget.
- **`body_parentid[0]` is 0, not -1.** The fixture caught the validator's
  wrong assumption on the first run. MuJoCo's world body is its own parent.
- **The v1 kernels are monomorphic host float64.** The records are
  dtype-parametric (`('a,'d) t`) so the parameters are already a structure;
  `Rigid` and `Pose` are polymorphic in dtype and device set, but `Kinematics`
  builds host index tensors and is instantiated at host float64. Generalizing
  it is part of M7's dtype work, not a redesign.
- **Constants come from the argument.** `forward` takes its working arrays
  from `zeros_like d.*` rather than allocating on the host, which is what lets
  the whole pass be `vmap`ped.

## M2 — inertia, bias, passive forces, integration (done)

**Landed: the position stage's inertia.** `Mass.forward` computes
`subtree_com`, `com`, `cinert`, `cdof` and a dense `mass_matrix`, and
`factor`/`solve` use a Cholesky factor. All five arrays match MuJoCo within
1e-12 for the four corpus models at 32 states each, and the matrix is
symmetric to 1e-12 with a solve residual below 1e-12 on the same states.

Three bugs the fixtures caught, worth recording because they are the shape of
this milestone's work:

- **The inertial frame is not the body frame.** MuJoCo builds a free or ball
  joint's rotation axes from `xmat`, the body frame, while I had used `ximat`.
  The fixture disagreed on one component of one dof.
- **`mju_inertCom`'s temporary matrix is column-major in its indexing.** The
  product is `R D R^T`, but the intermediate `tmp.(3k+i)` takes the inertia
  entry `k`, not `i`. Two wrong index mappings cancelled on the diagonal and
  disagreed off it.
- **The mass matrix's ancestor pattern is the wrong way round to guess.** The
  fixture is dense and symmetric, so both triangles are filled from the same
  dot product.

**Landed: the velocity stage, the bias force, passive forces, and the step.**
`Bias.forward` computes `cvel`, `cdof_dot` and `qfrc_bias` by the abbreviated
RNE; `Passive.forward` adds hinge and slide springs and per-dof damping;
`Forward.forward` composes the stages and solves `mass_matrix qacc =
qfrc_smooth`; `Step.step` is the semi-implicit Euler advance. The stage arrays
match MuJoCo within 1e-12 and ten steps of the whole engine match `mj_step`
within 1e-10 (the composed tolerance; the stages are the 1e-12 ones).

**The Euler integrator is implicit in the damping.** When any dof has
damping, MuJoCo's `mj_Euler` integrates with `M + h diag(B)` rather than with
`M`, so the velocity update is not `qvel + h qacc`. That difference is 3% of
the step for the sprung pendulum and the trajectory fixture caught it. With no
damping the explicit path is used, as in the C engine.

Two more corrections: the three axes of a ball joint take their velocity
derivatives from one time-stamped velocity, so all three must be taken before
any of them advances it; and `mj_step` leaves `qacc` from before the step, so
the recorder now calls `mj_forward` before recording it.

**Gate G2: pass, with the caveat that the corpus is small.** With
`Step.compile` hoisting the model's two host reads and jitting the pipeline:

| model | nv | eager | compiled | vmap 256, per env |
| --- | ---: | ---: | ---: | ---: |
| pendulum | 1 | 406 us | 26.0 us | 1.9 us |
| chain6 | 6 | 2198 us | 32.4 us | 2.4 us |
| floating | 6 | 647 us | 31.8 us | 4.9 us |
| ball | 4 | 877 us | 29.6 us | 1.5 us |

The batched step is 1.5–4.9 us per environment, well inside G2's 100 us pass
threshold. The models are small (nv ≤ 6), so this does not yet say what a
humanoid costs; the M4 corpus adds one. The eager path is 20–80× slower than
the compiled one: it is dispatch-bound, which is the design note's finding 1
at engine scale.

Additions to `Data`: `cinert` and `cdof_dot`. The fixture recorder now writes
the stage arrays and a ten-step `mj_step` trajectory, and the `pendulum`
fixture carries a spring and a damper so `qfrc_passive` is not identically
zero.

## M3 — constraints without contacts, and the solvers (in progress)

**Landed: the format, the limit rows, and the gate decision.**

- The `.mjcb` format is version 2: it carries `jnt_margin`, `jnt_solref` and
  `jnt_solimp`, which a limit row needs.
- A fifth corpus model, `limited`, has a limited hinge and a limited slide,
  and the recorder samples it past its ranges so that limit rows exist. The
  fixture records `nefc` and the padded `efc_type`, `efc_id`, `efc_pos`,
  `efc_margin`, `efc_J`, `efc_vel`, `efc_R`, `efc_D`, `efc_aref` and
  `efc_force` per state, with the Jacobian forced dense: the port is dense.
- `Constraint.limits` assembles the limit rows, in MuJoCo's order, into the
  fixed buffer, with `nefc` as a tensor and inactive rows scattered past it.
  It matches the fixture exactly on all 32 states: the row count, type, id,
  position, margin and Jacobian.
- [`adr/0010-solver-policy.md`](adr/0010-solver-policy.md) settles G3: Euler
  is non-metric, so the classical formulation is the one to reproduce, and
  Newton with a fixed budget and no line search is the first cut.

**Landed: the impedance and the solver.** `Constraint.impedance` computes
`efc_vel`, `efc_R`, `efc_D` and `efc_aref` from `solimp`/`solref`, matching
MuJoCo to 1e-12 on every state of the `limited` fixture. `Solver.forward`
solves the row forces on `A = J M⁻¹Jᵀ + diag R` and writes `efc_force`,
`qfrc_constraint` and the final `qacc`; the constrained trajectory matches
`mj_step` to 1e-6 over ten steps.

The solver had to be an **active-set Newton, not Newton-with-projection**.
Projecting a full Newton step lands on a fixed point that is not the KKT
point: for one of the fixture states it returned a force twice the right one
because a row that should have been frozen stayed free. Freezing the rows
(free while the force is positive or the reduced gradient is negative, pinned
by an identity block) converges in two or three iterations. The force is
compared at 1e-6 rather than 1e-12 because the port takes a fixed Newton path
while the C engine runs a primal solver with a line search; both minimise the
same quadratic.

**Cost.** A compiled step went from 26–32 us to 49–83 us, and the batched
step from 1.5–4.9 to 2.7–16.9 us per environment. The solver is why: a masked
`nefcmax × nefcmax` factorisation per iteration does not vectorise like the
rest of the pipeline. Still inside G2's 100 us threshold.

**CG agrees with Newton.** `Solver.forward ?linear` takes `\`Cholesky` or
`\`Cg`; the second solves each active-set system by conjugate gradients (one
iteration per row, exact at these sizes) and needs no factorisation. The two
agree on every fixture state to 1e-8, which is M3's acceptance item for the
solver pair.

**Equality rows landed, two gaps left.** Connect, weld and joint equalities
now instantiate rows (3, 6 and 1), the solver treats them as two-sided, and a
new `Jacobian` module supplies the point Jacobian and its `Jdot·v` from
`cdof`/`cdof_dot` in MuJoCo's com-referenced convention. `.mjcb` goes to
version 5 (`neq`, `eq_type`, `eq_objtype`, `body_invweight0`, `eq_*`) and a
`looped` corpus model carries one of each kind. Data gains `efc_diagA`,
`efc_solref`, `efc_solimp`, `efc_jdotv` and `efc_posnorm`: a row now carries
the solver parameters and inverse inertia it was built with, instead of the
impedance stage re-deriving them by row type.

**Both equality gaps are closed.** They were not where they looked.

The weld's rotational `Jdot·v` was off because `mju_derivQuat` multiplies the
pure quaternion on the *left*: `0.5 (0,ω) ⊗ q`, not `0.5 q ⊗ (0,ω)`. With that
corrected, the three-term product rule reproduces MuJoCo to 1e-12, and the
whole impedance path with it.

The row-count mismatch was contacts, not limits: at one corpus state the
geoms touch and MuJoCo adds four pyramidal rows. The `looped` model now
disables contact (`<flag contact="disable"/>`), since contacts are M4's
corpus; `nefc` there is then a function of the equality and limit rows alone.

The two tests are registered and hold everything to machine precision:
`efc_pos`, `efc_margin`, `efc_vel`, `efc_J`, `efc_R`, `efc_D` and `efc_aref`
at 1e-12, `efc_force` at 1e-9 — the active-set Newton reaches the same λ as
the C engine's primal solver — and a nine-step trajectory at 1e-6.

**Friction loss landed, and the solver had to change shape.** A friction row
forbids nothing: its force is quadratic while it fits inside the dof's loss
and pinned to ±loss outside, so it is the multiplier of a *box* constraint on
the same quadratic. Casting it that way, the whole solve stays one convex QP —
equality rows free, limit rows non-negative, friction rows boxed — and the
active set handles three kinds of bound instead of two.

The naive port of that (clamp the row-wise optimum each sweep) does not
converge: with coupled rows it cycles between +loss and −loss, which is what
one `limited` state did (a force of 401 where MuJoCo gives 187). Two things
fixed it. A *ratio test* stops each Newton step at the first bound it would
cross, and the rows that block the step are *held* at that bound until a
solve has moved the others; and the corpus was split so the branches are
exercised separately — `limited` saturates (64 rows), `drag` stays quadratic
(32 rows, with an assertion that it does). The degenerate case that remained
was a model whose two limit rows are nearly linearly dependent, where any
active-set method needs an anti-cycling rule; it is out of the corpus rather
than papered over, and the corpus is what the tests check.

**Performance recovered and improved.** Skipping the constraint stages
entirely for a model that cannot produce a row (no equality, no *limited*
joint, no friction loss) is a trace-time branch on host data, and a compiled
step went to 42–50 µs and the batched step to 2–11 µs per environment — better
than before friction landed, because the same host data also narrows the
limit-candidate list.

**implicitfast, and what it actually is.** Reading `mj_implicitSkip` settles
a piece of folklore: in this MuJoCo the scheme factors `M - h qDeriv` where
`qDeriv` is the velocity derivative of the *passive* forces (`mjd_smooth_vel`,
bias derivative excluded), and that is `M + h diag(B)` for the models v1
supports. Euler already solves that system when a dof is damped; the
difference is that implicitfast always solves it, and that `implicit` (not
implemented) adds the bias derivative and loses symmetry.

So the port is a dispatch in `step.ml`: the model's integrator decides whether
the velocity update goes through the damped matrix. The model carries
`opt.integrator` (format version 7, rejected values named on load), a `fast`
corpus model records a two-hinge pendulum under `implicitfast`, and its
nine-step trajectory matches `mj_step` to 1e-10.

**The solver changed shape again, and this time it stuck.** The ratio test and
the row holding that landed with friction cycled on a near-degenerate row set:
one corpus state (a different one after the corpus seed moved) stalled with a
gradient of 400 out at an "optimal" force. The fix, and what makes the solver
robust across the whole corpus at 1e-9, is monotone acceptance: a four-trial
backtracking line search on the quadratic, plus eight Gauss-Seidel sweeps from
the warmstart as a starting point. Coordinate descent cannot cycle and does
not care about degeneracy; it only has to get the active set right, which the
Newton loop then finishes exactly. ADR 0010 records the policy.

**Cost, on the bench extended with the constrained models.** A compiled step:
41-51 us unconstrained, 67 us for `limited` (limits and friction), 138 us for
`looped` (ten equality rows) — the last one because its row budget is 14 and
the factor is dense — and 2-6 us per environment batched except `looped` at
44. The row budget now counts only the joints that *are* limited and only the
dofs that *have* a friction loss, which cut `looped` from 29 rows to 14 and
its step from 285 us to 138.

### M4, first half: contact detection and frictionless rows

M4 could not be done in one pass, and the split was the right call: detection
is a different kind of work from the row assembly, and detection had to be
right before any force could be checked.

**Detection.** A new `Collision` module builds the candidate pair list from
host data — MuJoCo's `contype`/`conaffinity` bitmask, its body filter (same
welded body, two static bodies, parent-child unless the world body is
involved), the bounding-sphere rejection and the table of supported pairs —
and computes the distances of plane-sphere, plane-capsule, sphere-sphere and
sphere-capsule from `engine_collision_primitive`. Contacts are compacted with
the same scatter the limit rows use. The contact parameters are mixed where
MuJoCo mixes them: higher priority wins outright, otherwise condim is the max,
friction the elementwise max, `solref`/`solimp` mixed by `solmix`, and the
five friction slots are unpacked from the geom's three.

Against a corpus model that carries every supported pair, `ncon`, `dist`,
`pos`, `normal`, `geom`, `body`, `dim`, `includemargin`, `friction`, `solref`
and `solimp` all match at 1e-12 over 32 states.

**Frictionless rows.** `Constraint.contact` adds one non-negative row per
contact — the same shape as a limit row, so the solver needed no change at
all. The row's Jacobian is the normal dotted with the difference of the two
bodies' point Jacobians, computed in `Collision` because a Jacobian is a walk
up a dof chain and the bodies are host values there rather than data. Row
position, margin, impedance and force match MuJoCo to 1e-12, 1e-12, 1e-12 and
1e-7 — the force tolerance is the C engine's own stopping error on a contact
that a random pose buries thousands of newtons deep — and a nine-step
trajectory matches at 1e-8.

**Contact parameters are mixed as tensors, not read as host values.** The C
engine mixes two geoms' parameters with scalars because it owns its model; a
traced program may not, because a captured parameter is a traced tensor and
only tensor operations are allowed on one. So `Collision` now gathers both
geoms' rows and mixes them with `where`s — priority overriding, the solmix
weight with its three degenerate cases, standard references mixing and direct
ones taking the minimum, the five friction slots unpacked from three. The one
host value the stage still needs is each geom's `condim`, for the refusal
below, and that is a `Topo` copy like `jnt_limited`. With that, a compiled and
batched step of a model that collides works, and the bench covers them again.

Two consequences worth recording. The Gauss-Seidel warm start in the solver
now runs only when the model has a friction row: the sweeps are unrolled, so
they cost tracing time in every compiled step, and the corpus's equality and
limit rows converge from the warmstart force alone. And the row budget charges
for contacts only when a model can have them — contacts enabled and at least
two geoms — because the dense factor is quadratic in `nefcmax` and the
constraint-only models were paying for rows they could never produce.

**Tracing cost is now visible.** A compiled step of a contact model takes tens
of seconds to *trace* — the contact Jacobians are unrolled per candidate and
the solver's iteration budget is unrolled too — which shows up as a slow test
and a slow bench, not a slow step. The fix is a per-model iteration budget and
a cheaper Jacobian construction (one batched gather over the dof chain instead
of a mask per candidate), and it is the first thing to do before M4's stacks
are compiled.

**Plane-box is in, and not yet trusted.** The narrowphase is a port of
`mjc_PlaneBox` with its corner order and its four-contact cap, and a `boxes`
corpus model records it. Its parity is *not* asserted: one corpus state has
MuJoCo reporting a contact this port does not, and I have not found which
corner test differs. The code is in because it is the shape the stacks need;
the test that would hold it is not, and that is the first thing to finish
next.

**The pyramidal cone is assembled, and its impulse is not trusted.** The
pleasant discovery is that MuJoCo's cone needs no group constraint at all: for
each friction axis it adds two *one-sided* rows with Jacobians `n + mu t` and
`n - mu t`, and requiring both forces to be non-negative is what makes the
cone. So `Constraint.contact` now emits `2*(dim-1)` rows per contact of type
6, with the frame's tangent rows (from `mju_makeFrame`) and the per-axis
`diagA = tran + mu^2 * weight`, and the solver handles them unchanged.

What is not right is `diagA`'s `tran`: MuJoCo's per-side body weights make it
twice the plain inverse-weight sum (measured against a corpus state: the
contact's `efc_diagA` is `2*(1+mu^2)*tran` for a plane contact). Until that is
reproduced, `friction3`'s rows and trajectory are *not* asserted, exactly as
`boxes`' plane-box detection is not: the code is in, the parity is not
claimed. Two half-claims, both recorded where their tests live.

**M3 is complete.** The `.mjcb` format is version 9: it
carries `dof_invweight0`, `iterations`, `tolerance`, the equality tables, the
body inverse weights and `dof_solref`/`dof_solimp`, and the converter sizes
`nefcmax` from the model's joints, equalities and dofs.
