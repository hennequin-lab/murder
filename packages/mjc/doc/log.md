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

## M2 — inertia, bias, passive forces, integration (in progress)

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

**Remaining:** the velocity stage (`cvel`, `cdof_dot`), the RNE bias, springs
and dampers, the Euler advance, and `Forward`/`Step`; then the trajectory
fixture (`traj_qpos/qvel/qacc`, recorded and not yet read) and gate G2. The
fixture files already carry the arrays for all of it.

Additions to `Data`: `cinert` and `cdof_dot`. The fixture recorder now writes
the stage arrays and a ten-step `mj_step` trajectory, and the `pendulum`
fixture carries a spring and a damper so `qfrc_passive` is not identically
zero.
