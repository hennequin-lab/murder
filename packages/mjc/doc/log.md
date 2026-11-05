## Where box-box stands, in one place

Three artefacts, in the order they were built, each verified before the next:

| Artefact | Verified against | Result |
| --- | --- | --- |
| `lib/collision_box.ml`, host port of `mjc_BoxBox` | the C engine, over the 32 states of `boxes2` | 151 pairs at 1e-12, 314 pairs correctly empty |
| `lib/collision_box_tensor.ml`, stage 1 (15 axes) | the host port, over 400 random poses | same code and separation on all 400 |
| the same, stage 2a (edge-edge) | the host port, same poses | 48 edge-code pairs at 1e-12 |
| stage 2b (face clip) | — | **not written**: face pairs produce no live slot |

Not wired into `Collision`: `slots` for a box pair has to become 12, the
`nconmax`/`nefcmax` budget in `tools/mjcb.py` and `Topo` has to agree with it,
and the narrow phase needs the per-pair gather. Until then the engine refuses
box-box by name, so no model that needs it can run: the gap is loud.

## The tensor box-box: stage 1 and the edge path match, the face clip is next

Ported so far, and checked against the host kernel on 400 random poses rather
than on the corpus (the corpus cannot show whether a straight-line version of a
branching search is the same search): the fifteen-axis separating-axis stage,
including the edge bias and the face substitution, agrees on the winning code
and the separation for every pair; the edge-edge generator agrees on distance,
point and normal to 1e-12 for all 48 pairs that take that branch.

Two things made the tensor form tractable. Every candidate axis is a *value*
with a mask, so "first axis wins a tie" is a minimum over the indices that
achieve the maximum rather than a loop with an early `if`: that reproduces
MuJoCo's tie rule exactly, which a tidier "argmax" would not. And the edge
generator's supporting corner depends on a *data-dependent* axis component, so
the four sign combinations are computed and the closest witness pair is chosen
elementwise — the same enumeration the C engine does with its ambiguity loop.

The face clip is not written. Face-code pairs currently produce no live slot,
the interface says so in the type's documentation, and nothing calls the tensor
kernel yet: `Collision` still refuses box-box by name. The remaining work is a
fixed-slot clip and the assembly that gathers each pair's slot to the candidate
list; ADR 0011 records both options and what each costs.

## M4 box-box: the kernel matches the engine, and the step does not use it yet

MuJoCo 3.17's `mjc_BoxBox` is not the classic clip-based algorithm. It is a
separating-axis search over fifteen axes (three face normals per box, nine
edge crosses) with three rules that exist only so stacks sit still: a slack on
the separation tests proportional to the box scale, an edge bias so a face wins
near-ties (a face gives a multi-point manifold), and a *face substitution* that
replaces a winning edge axis within eight degrees of the best face axis unless
the edge is five percent better. Then two different generators: an edge axis
gives one point from the closest approach of two segments, a face axis gives
every surviving vertex of a Sutherland-Hodgman clip of one face against the
other's side planes — up to twelve contacts, and MuJoCo's comment is explicit
that cutting the patch to four costs two to three orders of magnitude in
residual motion at rest.

I ported it as host code (`lib/collision_box.ml`), statement for statement,
rather than as traced tensor code, because the traced version would be a
different program and so could not be checked against the C engine line by line.
The port is checked as an oracle: `test/test_boxbox.ml` runs the new `boxes2`
corpus — a slab, a squarely stacked box, a box turned 15°, a box tipped 45°, and
a box overhanging the slab's edge and turned 30° (whose clip yields five
vertices) — and reports **151 pairs matching the C engine at 1e-12 and 314 pairs
correctly producing no contact**. That includes the other direction, which is
where an over-eager kernel would show.

It is not wired into the step. The narrow phase is tensor code over the whole
candidate list, and box-box needs twelve slots per pair, a `where`-min over the
fifteen axes, and a fixed-slot clip; `doc/adr/0011-box-box.md` records the rules
and the three pieces of plumbing (slot budget, tensor clip, and the oracle
property test that will keep a tensor port honest on poses the corpus does not
contain). Until then the engine still refuses box-box by name rather than
approximating it, and `VIZ.md`'s two-box acceptance stays unmet.

The corpus is committed and reproducible from `tools/record_fixture.py`, so the
next step starts from a recorded target rather than from MuJoCo on disk.

## "Diverged at step N" is a jump, and one test that only needed a file

The viewer's stated purpose includes "did the run diverge, and when?". With the
log that is a scan and a jump: `Log.divergence` returns the first frame whose
geom poses are non-finite or more than a kilometre from the origin, the window
computes it once when a log opens, and `d` moves the cursor there. The HUD says
`diverged at 100 (d)` before you press anything.

Testing it needed no simulation that actually blows up: the test writes a clean
log, patches one record's first geom position to 1e6 *in the file*, and expects
the index back. That is worth more than it looks — it can only work if the
documented layout is the layout the writer wrote, so the test doubles as a check
of the format description in `log.mli`.

## V3's contact detail needed a field the engine did not have

Drawing a contact the way the solver sees it — force along the normal, friction
along the tangent, the pyramid as the cone it is — needs the contact's *frame*.
MuJoCo keeps it in `mjData.contact_frame`; our `Data` did not have the field at
all, so the viewer could only have guessed a tangent from the normal, and a
guessed basis decomposes the pyramid's four edge forces differently than the
engine did. The fix is the honest one: `contact_frame` is now part of `Data`,
written by the collision pass from the same `mju_makeFrame` that builds the
pyramid rows, and the fixture records it, so `test_contact.ml` holds the
tangents to the C engine at 1e-12 next to `contact_pos` and `contact_normal`.

With the frame, the decomposition is the solver's own arithmetic: the normal
force is the sum of the contact's rows, and the friction is what the edges leave
after cancelling along the normal. The test then bites: `blocks` settled has
four contacts at 78.48 N each against a 313.9 N crate, the ball's one at
29.96 N against 29.9 N, penetration 0.11 mm and 0.37 mm — and the sum of forces
equals the total weight to 1e-3 N. The frame invariants (unit rows, mutually
orthogonal) and Coulomb's bound are checked on every contact of the model, and
depth-as-colour is a pure function with a monotonicity test.

Two mistakes of mine on the way. The first version read the frame's *rows* off
by one — taking row 0 as the first tangent — and the test caught it as
"n·t1 = 1". The second was reading the depth colour off a two-step screenshot
and concluding the colour mapping was broken, when the model starts with the
crate one centimetre into the floor: red was right, and green arrived when the
solver had pushed it out. A log frame settled it in one command — which is what
the log is for.

## V3's log: fixed-stride, seekable, and one wrong assumption about List.init

The log came before the replay UI because it is the half that can be tested
without a display, and because M4 is where "it diverged at step 40 000" becomes
a question someone asks. A record holds geom poses rather than joint positions:
a replay then needs no kinematics, so a reader cannot disagree with the writer
about a model it may not have — which is also why the header carries the shape
(`ngeom`, `nconmax`) and the name, and why opening a log against the wrong
model is an error rather than a picture that looks subtly wrong.

The tests are the acceptance table: a round trip compares every recorded number
with the state it came from, bit for bit; a replay frame is compared with the
live state at the same step; 500 pseudo-random seeks out of an 800-frame log
must fit in half a second, which a scanning reader cannot do. The write costs
0.08 ms/frame and a replay frame's read 0.002 ms, so replay is draw-bound and
the log is not the bottleneck at any size.

Two things went wrong, both worth the ink. The reader checked the header's
version at offset 4 instead of 8 and reported "log version 71" — the tail of
the magic string. And the round-trip test wrote records whose `step` counted
*backwards*: `List.init` in Base applies its function from `n - 1` down to `0`,
so a run's side effects happened in reverse while the list it built was in
order. The library was right and the test was wrong, but only because the
symptom (record 0 holding step 39) looked like an offset bug in the reader.

Still missing from V3: the contact overlay's detail — per-row λ bars, the
friction-cone fan, depth as colour — and the two-box acceptance, which waits on
box-box collision in M4.

## A `ref` angle does not bend a model at rest

The demo's chain hung perfectly straight however much I tilted it, and the
reason is the definition of the joint, not a bug: MuJoCo's kinematics rotates a
hinge by `qpos - qpos0`, and a MJCF `ref` sets `qpos0`. So at the reference
state — which is exactly what `mj_resetData` and our `Data.create` produce — a
reference angle contributes nothing, and a model whose only "bend" is `ref`
opens straight. Our `Kinematics.forward` matches MuJoCo here, statement for
statement (`engine_core_smooth.c`: `axisAngle2Quat(qloc, axis, qpos[qadr] -
qpos0[qadr])`), which is why the parity tests never saw it: every fixture has
`qpos0 = 0` for its joints, so the two conventions are identical there.

The demo now tilts the chain's root *body* instead, which is geometry and does
bend the pose. Anything that needs a bent *initial joint configuration* needs a
keyframe, and keyframes are not in the `.mjcb` format yet — that is the honest
gap, not the demo.

## A demo found a sharp edge in the collision guard

Assembling `viz/demo/demo.mjcb` — a swinging chain, three falling spheres and a
box on the floor — failed on the second frame with
`Invalid_argument("collision between geom types 3 and 6 is not implemented")`.
Types 3 and 6 are capsule and box, and the pair was forty centimetres away from
anything it could touch: `Collision.candidates` raises for every pair the
geometry *could* collide with, before any distance test, because that is the
only place a host-side decision is available (the narrow phase is traced, and a
tensor value cannot raise).

MuJoCo gates the same decision on the bounding-sphere test inside
`mj_collideGeom`, so a capsule fifty metres from a box is simply not a contact.
Ours is stricter than the engine it is meant to match, and it makes any model
that contains both a capsule and a box unloadable — which is most arms with a
gripper. Two ways to fix it when box collisions land in M4: a host copy of
`geom_rbound` in `Topo` plus a distance test on the initial state (cheap,
approximate, and honest only if it warns), or classify unsupported pairs as
dropped candidates and report them once at build time, keeping the row budget
and the parity tests out of it.

For now the demo takes the chain out of collision with `contype="0"
conaffinity="0"`, which is what those masks are for, and says so where it
happens.

## V2: overlays as pure functions, and one wrong assumption

The overlays are five pure functions in `Scene` and a flag each in `of_state`.
What is worth recording is the two things that were wrong first.

The velocity arrow started from `cvel`, whose linear part is com-based over a
body's *subtree* — MuJoCo's own convention. Composing it down a chain needs the
shift `w x (xipos - subtree_com)`, and I wrote that, and the test passed on the
two-link model. Then the energy test failed by 25% on the same model, and the
numbers showed the hand body's velocity as 0.158 where the physics says 0.14:
the shift was being applied to a reference point `cvel` does not actually use
that way. Rather than reverse-engineer the composition, both overlays now take
the velocity of the point from `Jacobian.point` — `J qvel`, exact, tested, no
subtlety — and the energy is a sum of `1/2 m |v|^2 + 1/2 w'Iw` over bodies.

That sum exists to avoid building the mass matrix at display rate: the matrix
route measured **2.6 ms per frame** on `bench64`, which is a HUD nobody would
keep on. The sum measured nothing at all — the scene build stayed at 0.008 ms
with every layer off and 0.234 ms with all of them on a 5-body model (0.648 ms
on the 65-body `bench64`, where the cost is per-body Jacobian overhead). The
test that keeps it honest is the equality with `1/2 qvel'M qvel` on four models
at 1e-9, because otherwise the HUD would be a plausible-looking number.

The inertia box is the formula from MuJoCo's viewer rather than something
invented: `h_x^2 = 3/(2m)(I_y + I_z - I_x)`. Its test is that a box geom's
inertia box comes back as itself — the crate's `0.2 x 0.2 x 0.1` to 1e-12 — and
a sphere's as `r sqrt(3/5)`.

The trail is not in. A trail is a function of history rather than of a state, so
it belongs to whoever owns the loop; V3's log is where that history will exist.

## V1 finished: joint control, screenshots, and the frame budget

The window was drawing nothing at the origin, and the still backend was right:
`Data.create` leaves `geom_xpos` and `geom_xmat` at zero, `mjc-view` runs
`Forward.forward` before building its scene, and the window did not. Every
primitive was therefore at the origin with a zero rotation matrix, which
collapses a box's eight corners into one point — so a box drew twelve
zero-length lines and a sphere drew one sphere in the middle of the frame. The
fix is one line in `open_` and the same line in the reset key, and it is
recorded here because "the picture is empty" reads like a renderer problem and
was a state problem.

Joint control moved out of the window into `Mjc_viz.Controls` before it was
wired to keys. The reason is V2's rule applied one milestone early: anything
that decides what the picture shows should be a pure function of a model and a
state, and tested as such. Six tests now cover the model's ranges, the clamp at
a limit, the links following a hinge (compared against a rotation done by
hand), one keystroke, the reference position, and that a free joint is refused
by kind — `describe` says "free joint, not driven", which is how a user learns
it instead of watching a key do nothing.

The screenshots are how the two backends are now compared rather than asserted
to agree. `mjc-window --shot` writes the last frame from inside the frame, so
the window and `mjc-view` can be asked for the same camera and the same state;
the sphere's centroid agrees to **0.4 px** at 1280×720 and its radius to about
a pixel. That is a real number standing where "looks the same" used to be, and
`doc/figures/viewer-v0-vs-v1.png` is the pair.

Two measurements worth keeping. The frame budget is missed: 65 primitives and
195 lines at 960×540 take 14.6 ms on a Radeon 8060S, against §9's 10 ms
llvmpipe line, and the scaling says it is per-primitive CPU cost — raylib
re-tessellates spheres on every `DrawSphereEx` — not fill, since 960×540 →
1280×720 costs 0.2 ms and 3 → 65 primitives costs 12 ms. The scene build is
0.008 ms, so the IR is not implicated. The way out, when 64-geom models are
normal, is a cached sphere and capsule mesh drawn with matrices, or instanced
drawing per colour; `viz/window/bench64.{xml,mjcb}` is committed so the number
can be re-taken rather than remembered.

What V1 does not have, said plainly: the print says 69 fps at the budget shape
and 68 fps at 1280×720, so the 30 fps requirement is met with room, but "no
allocation per frame" is unmeasured, and the still-versus-window agreement is a
centroid plus a picture, not a per-pixel diff.

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

**Plane-box is in and asserted.** The narrowphase is a port of `mjc_PlaneBox`
with its corner order and its four-contact cap, and a `boxes` model records
it. Its apparent disagreement was not in the corner test at all: the corpus
had two boxes, and the contact MuJoCo reported that this port did not was a
*box-box* pair — a type the port does not compute, dropped silently by the
pair table. So the corpus is now one box on a plane, and, more importantly,
a pair that may touch but whose distance the port cannot compute is now
*refused by name* instead of disappearing.

**The pyramidal cone is assembled, and its impulse is not trusted.** The
pleasant discovery is that MuJoCo's cone needs no group constraint at all: for
each friction axis it adds two *one-sided* rows with Jacobians `n + mu t` and
`n - mu t`, and requiring both forces to be non-negative is what makes the
cone. So `Constraint.contact` now emits `2*(dim-1)` rows per contact of type
6, with the frame's tangent rows (from `mju_makeFrame`) and the per-axis
`diagA = tran + mu^2 * weight`, and the solver handles them unchanged.

What did *not* come out of the source is the pyramid's diagonal. MuJoCo's
`mj_diagApprox` reads `tran + mu^2 * weight`, but the C engine's own
`efc_diagA` for a pyramid edge — recovered from `efc_R` and the impedance
since the field is overwritten downstream — is `tran * 2 mu^2 (1 + mu^2)`
over six friction coefficients from 0.1 to 2, where `(1+mu^2)` is the edge
row's squared norm. The port reproduces the *result*, and says so where the
formula sits. Both `friction3` tests are registered and pass: rows at 1e-12,
trajectory at 1e-8.

**Two more primitives, and a guard that finally bites.** Sphere-box and
capsule-capsule landed. Sphere-box is `mjraw_SphereBox` with both branches
selected by masks, including the inside case where the nearest face sets the
normal — MuJoCo's sign there is the *opposite* of the face that wins, which a
fixture state caught. Capsule-capsule is `mjc_segWeights` (closest points of
two segments, the nearly parallel case yielding two candidate pairs)
followed by a sphere-sphere test on each.

The guard that refuses pairs the port cannot compute was claimed one commit
earlier and *was not in the tree* — a `str.replace` that silently did not
match, which is the second time this session. It is in now, and it proved its
worth at once: several corpus models (chain6 among them) had been quietly
dropping capsule-capsule contacts that MuJoCo computes.

**The trace cost is the blocker now.** With capsule-capsule in, chain6 builds
thirty contact candidates and each walks its dof chain with one traced `set`
per dof, so compiling it takes minutes and the bench no longer finishes. The
fix is a host mask matrix (`nbody × nv`) so that a point Jacobian is two
`take`s instead of a walk, and it should land before anything else touches
contacts.

**The viewer starts, at V0.** `packages/mjc/viz` now has the scene IR and a
still backend. The IR is host floats in world coordinates with the shapes
resolved (`Capsule` of two endpoints, `Box` of centre/axes/half-extents), which
is what makes it testable with no display and keeps the backend replaceable.
The backend is `geom.viz`, whose painter's algorithm needs no window, no GPU and
no dependency the lock does not already pin; `mjc-view` writes PNG or SVG with
body-frame, contact and force overlays.

Three things came out of drawing it. A plane's declared patch is infinite
enough to cross the camera, so the patch is clipped to a few scene widths and
centred under the target. A floor drawn opaque hides whatever stands on it
under a painter's algorithm, so it is translucent. And a capsule drawn as an
axis plus two end spheres looks like a wire, so it is sampled as a short chain
of spheres. Measured cost: 105.8 ms per frame at 1280×720 with PNG encoding, so
this backend is for stills and the window is V1's job.

**V1 runs.** Getting there was a repository problem before it was a code
problem: the workspace takes its libraries from `dune.lock`, and `dune pkg
lock` resolves only what a package `depends` on, so raylib could not enter the
lock while `mjc` was not a package at all. Declaring the package (the library
stays private until M7) with `base`, `nx` and `raylib` fixed that; the relock
moved 22 lock files by one line each — dune's build-rule encoding, no version
changes — and added `raylib.2.4.0`. An earlier trial *without* the package
declaration had modified six lock files and added nothing, which is why I had
stopped; that was the wrong diagnosis of the same fact.

The code then met the binding's shape: raylib splits into `raylib.core`
(window, input, camera, colour), `raylib.models` (3D shapes) and `raylib.text`,
and `Camera3D` is a Ctypes structure read through accessors, so orbit and pan
write a fresh `Vector3` rather than assigning fields. Two primitives do not
exist as such — an oriented cylinder and an ellipsoid — so a cylinder is drawn
as a capsule over its axis and an ellipsoid as a sphere of its largest radius,
both with a comment saying so.

`--check` runs three frames headless and exits 0 on this machine, so the
automatic acceptance is met. Frames per second with a human at the keyboard is
the part no test can report.

**V1 is written but not buildable, and that is a repository fact.** The window
loop exists (`viz/window`): fixed step, orbit/pan/zoom from the mouse, the
pause/step/reset keys, a HUD, and every primitive drawn from `Mjc_viz.Scene`,
so the window and the stills cannot drift. `--check` runs three frames headless.

It cannot build, because the workspace resolves libraries from `dune.lock` and
raylib is not in it. `dune pkg lock` resolves only what a package `depends` on,
so the dependency has to be declared first; a trial run *without* declaring it
modified `base`, `hugin`, `jera`, `lz`, `nx` and `lock.dune` and added nothing,
and was reverted. The path forward is in `viz/window/raylib.md` with that
warning attached, since a lock diff of that size deserves a review rather than
a reflex.

**M3 is complete.** The `.mjcb` format is version 9: it
carries `dof_invweight0`, `iterations`, `tolerance`, the equality tables, the
body inverse weights and `dof_solref`/`dof_solimp`, and the converter sizes
`nefcmax` from the model's joints, equalities and dofs.

**M4's contacts are complete: box-box is in the solver.** The face branch of
`mjc_BoxBox` is written as a candidate set rather than as the engine's
Sutherland-Hodgman clip — the intersection of two convex polygons has its
vertices among the incident corners inside the reference rectangle, the
reference corners inside the incident rectangle, and the crossings of an
incident edge with a reference side plane — twenty-four candidates, filtered,
deduped and compacted into twelve slots. Four things it took to get right, all
of them caught by comparing against the host port rather than by reading:

- **The reference corners are not optional.** Without them a large box resting
  on a small one has no vertices at all, because every incident corner is
  outside the reference face and no incident edge crosses a side plane. That
  was the "four contacts expected, none produced" of the two failed attempts.
- **The depth test is one-sided.** A contact sits *below* the margin, so the
  absolute value that the four side planes want drops every penetration. The
  geometry was already exact when this was still emptying the manifold: the
  corners' depths were −0.03…−0.06 and the filter asked for |depth| ≤ 0.
- **A crossing lies exactly on the plane that made it,** and rounding can put it
  a few ulps outside; the engine never notices because it does not re-test a
  vertex against its own plane. The boundary needs a guard of 1e-12 of the face
  size, or the vertex that *is* the contact vanishes.
- **Two frames, one permutation.** The candidates live in [(ax, ay, a)]; the
  position has to go back into the parent box's slots before the rotation. And
  the edge branch's dead-slot filler has to be masked by the branch, not added:
  a face pair's slots 1..11 came out at −1e30.

Measured: `test/test_boxbox_tensor.ml` compares against the host port on 800
random poses, with and without a margin — 477 touching pairs, 278 of them
face-code, 1206 contacts, at most eight per pair, every contact equal as a set
at 1e-12. `test/test_contact.ml`'s `box-box contacts, as a set` takes `boxes2`
through `Collision.contacts`: nineteen contacts at one state, count, distance,
point, normal and frame equal to the fixture at 1e-12.

Two decisions recorded with the code. **Row order differs from MuJoCo's** for a
face pair: the engine emits vertices in clip order, the kernel takes the set.
That reaches the solver only through the warm start, so the parity test matches
box rows as a set and says so. **The contact budget is now taken from the
recording** — twice the largest `ncon` a corpus saw — because a static budget
guessed from the geom count cannot know that a box pair fills a manifold: the
guess allowed twelve where `boxes2` needs nineteen, and the compaction would
have sent the other seven to a sink slot without a word. The corpus test asserts
`ncon ≤ nconmax` where the values are concrete, since the compiled path cannot.

**M5 is in: actuators, transmissions, fixed tendons, sensors.** One expression
covers the actuator kinds the lab's models use — `force = gain * input + bias`,
both polynomials in the transmission's length and velocity, then
`qfrc_actuator = momentᵀ force` — so a motor, a position or velocity servo, an
integrated setpoint and a damper are the same kernel with different parameters.
Fixed tendons are the same shape: `ten_length = Σ coef qpos`, a dense Jacobian,
and a tendon actuator's moment arm is its gear times that row. Six findings,
all from trajectories or from a fixture that disagreed by a percent:

- **The implicit integrator needs the actuators' velocity derivative.** A
  velocity servo's `kv` is damping by another name, and `implicitfast` puts
  `mjd_actuator_vel`'s term in the matrix: `M + h diag(B) − h Σ k_a m_a m_aᵀ`.
  Without it a servo with a forcerange rings instead of tracking, and the first
  step is off by a milliradian.
- **A saturated actuator contributes nothing to it.** Its force does not move
  when its velocity does, to first order; the engine skips exactly the actuators
  sitting on a forcerange bound.
- **A body's velocity is measured at its inertial position.** `mj_objectVelocity`
  transforms `cvel` to `xipos` from the subtree's centre of mass, not to `xpos`;
  using `xpos` gave a pendulum's world momentum a component it does not have.
- **Angular momentum accumulates twice per body** — once about its own subtree's
  centre and once as the whole subtree's orbit about the parent's. Without the
  second term every total is short by exactly the subtree mass ratio.
- **`mjOBJ_BODY` is the inertial frame** (`xipos`, `ximat`) and `mjOBJ_XBODY` the
  frame (`xpos`, `xquat`). Collapsing them made a body frame position a percent
  and a half wrong on a chain — the kind of number that reads as a rounding bug.
- **A quaternion is a representative.** `mju_mat2Quat` picks the branch by the
  largest component, so a frame within rounding of the boundary can come back
  negated. `Rigid.quat_of_mat` ports the same chain as four candidates and a
  select (traceable, and the same representative), and the sensor test compares
  quaternions up to sign, which is the same orientation.

Refused by name rather than approximated: tendon transmissions need the fixed
tendon's Jacobian (they have it) but a *limited* tendon is a constraint row v1
does not have, and the sensor kinds that need the acceleration stage (gyro,
accelerometer, velocimeter), a ray-geometry query (touch, rangefinder, contact,
tactile), or a reference frame are refused where a model asks for them. Also
refused: ball and free joints in actuators and joint sensors, site wraps, muscle
gain and bias, PID and DC motors, and `actearly`.

**A demo found a constraint bug: the refsafe clamp was applied to the direct
solref form.** MuJoCo's `mj_impedance` clamps `solref[0]` up to `2 dt` — but
only when `solref[0] > 0`, the time-constant form, because the direct form is a
*negative stiffness* and `max(-1e5, 0.004)` is `0.004`: a joint that asked for
a stiff limit got a four-millisecond time constant instead and could be driven
past its range by hand. The corpus never noticed because every model in it uses
the default positive form. It took a demo with light links — where a soft limit
is visibly violated rather than merely wrong in the seventh digit — to see it,
and `test/fixtures/stiff.xml` now pins it: a joint pushed against a direct-form
limit, with `efc_aref` in the fixture, which the bug turns from `+4958` into
`-563`.

## M6 — differentiability, first increment (2026)

**A masked-out branch does not mask an infinite derivative.** `Nx.sqrt`'s reverse
rule divides by the root, so at a zero argument its cotangent is infinite, and
`Nx.where` multiplies the *unselected* branch's cotangent by zero — `0 * inf` is
a NaN. Every `max (sqrt x) tiny` idiom in the codebase therefore handed a NaN to
the reverse pass of any model that reached it, even though the forward value was
right and no test noticed. In practice the idioms were the normalizers
(`make_frame`, `fallback_normal`, `sphere_sphere`) and the quaternion branch
chain (`quat_of_mat`, whose three candidates each take a root of a matrix
combination that is exactly zero at a 180-degree rotation), plus
`norm_along_1`/`quat_of_rotvec` (which reach a zero root when a tangent vector
or a rotation vector is zero). `boxes2` was the model that showed it: the
reverse mode of the step was NaN at every recorded state, in exactly the
coordinates of every box's quaternion, and the first stage that was not finite
was `contact_dist`.

The fix is `Rigid.safe_sqrt` and `Rigid.safe_sqrt_exact`: a root whose argument
has a floor under it, and — for the exact one — a mask that multiplies the floor
root back to the value it would have had, so **the value is unchanged** and only
the derivative at zero differs. The first attempt used the floored root alone,
and `test_rigid`'s "zero rotvec" case failed: a quaternion that must be exactly
`[1; 0; 0; 0]` came back `[1 - 1.3e-16, ...]`. That is the honesty invariant
working — the derivative fix is not allowed to move a value.

The floor is `1e-200` on a *squared* quantity, i.e. a length below `1e-100`:
below anything a model's geometry produces, and large enough that the derivative
at the floor, `5e99`, does not overflow when a cotangent multiplies it. The
floor on a squared argument is the subtlety of the floor: `1e-15` there would
have made every vector shorter than `3e-8` normalize to zero, which is a
different engine.

`test/test_grad.ml` is the harness this bought: the reverse mode of the step is
finite at every recorded state of thirteen models (including `boxes2`'s 22
contacts and `bounce`'s and `blocks`'s); it matches central differences within
`1e-6` on the smooth models; and it does not change when the solver's iteration
budget changes from 3 to 8. What it does **not** yet do is take the gradient
through the solver as an implicit function — the unrolled answer happens to
agree on these models, which is why the budget test passes, but that agreement
is a property of the models, not of the implementation, and the implicit rule is
the next increment.
