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

**Remaining in M3:** equality rows (connect, weld, joint, tendon), friction
loss, the matriz-free CG solver, and `implicitfast`. The `.mjcb` format is
version 4: it carries `dof_invweight0`, `iterations` and `tolerance`, and the
converter now sizes `nefcmax` from the model's joints rather than a padding
constant.
