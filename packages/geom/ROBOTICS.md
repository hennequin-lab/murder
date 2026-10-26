# Robotics with geom — design note

Status: **design note, R0–R1 landed in first form**. This document asks what a
robotics layer on top of `geom` should be, using
[GAFRO](https://geometric-algebra.tobiloew.ch/gafro/) as a reference. It is a
working document: it records the proposed scope, the decisions, and an order of
work. The model, forward kinematics and differential kinematics are
implemented in [`robot/lib`](robot/lib) (`geom_robot`, module
`Geom_robot.Robot`) with the test suite in [`robot/test`](robot/test) — task
residuals and inverse kinematics followed, in `Task` and `Ik`; §4.0 says exactly
what landed, §5.4 and §5.7 record the two numerical decisions that matter for the
rest of the stack, and §10 tracks every open question. [`robot/README.md`](robot/README.md)
is the user's guide to all of it.

The short recommendation:

1. Build a small, validated **robot model** (links, joints, frames, chains)
   and **forward kinematics** first.
2. Add **differential kinematics** (geometric, analytic and frame Jacobians,
   plus their time derivatives).
3. Add **geometric task residuals** and **inverse kinematics** next; this is
   where geometric algebra pays for itself.
4. Add **model I/O and a robot zoo** so results can be checked against
   Pinocchio/KDL and real robots.
5. Only then add **dynamics** and **optimal control**, preferably through the
   existing `dilqr`/`ilqr_vae` stack.
6. Everything should be `Nx`-native, batched and `Rune`-traceable. Supporting
   `float64` is a hard requirement for dynamics and iLQR.

Steps 1–2 are in the tree (§4.0). Two decisions shape everything after them: the
model layer is a functor over its motion algebra, so `float64` arrives without a
rewrite (§5.2), and derivatives are automatic differentiation at the top and
closed forms at the leaves, glued by `Rune.custom_jvp`/`custom_vjp` (§5.4).

---

## 1. What GAFRO establishes

GAFRO is a C++20 library for conformal geometric algebra (CGA) targeted at
robotics. Its stack includes `gafro`, Python bindings (`pygafro`), a ROS
interface (`gafro_ros`), examples, and benchmark code. The main ideas worth
taking seriously:

- **One representation for objects and motions.** Points, spheres, planes,
  lines, circles and point pairs are CGA blades; motors are the even
  subalgebra. A task can move from a point to a line, plane, circle or sphere
  without changing the structure of the cost function.
- **Robot modeling is part of the library.** GAFRO models links, joints
  (fixed, revolute, prismatic, continuous, free), kinematic chains, serial
  manipulators and larger systems (dual-arm, hands, humanoids, quadrupeds).
  Models can be constructed in code or loaded from YAML, with URDF support
  through the ROS package.
- **Forward and differential kinematics.** End-effector motors, analytic,
  geometric and frame Jacobians, and the time derivative of the geometric
  Jacobian.
- **Inverse kinematics as manifold optimization.** The pose error is
  `log(M_target⁻¹ M(q))`; Gauss–Newton or related solvers operate on the motor
  manifold.
- **Generic geometric task objectives.** GAFRO uses the outer-product
  nullspace error `X_d ∧ M(q) X M(q)~` for a reference primitive `X` and a
  target primitive `X_d`, then embeds the nonzero blades into an ordinary
  vector and Jacobian. The same construction covers point, line, plane,
  circle, sphere and point-pair targets.
- **Dynamics.** Recursive Newton–Euler and articulated-body algorithms in CGA,
  with a nontrivial inertia tensor, mass matrix, Coriolis/centrifugal terms
  and gravity.
- **Control and optimization.** iLQR/MPC on the motor manifold, computed
  torque/inverse-dynamics control, admittance and impedance control,
  cooperative dual-arm task spaces, and probabilistic multivector Gaussians.
- **Benchmarks.** The GAFRO paper reports competitive forward-kinematics and
  Jacobian performance against KDL, Pinocchio and Raisim; it also notes that
  its dynamics implementation is currently slower than the established
  libraries. That is a useful warning: GA-native dynamics is elegant, but it
  must earn its performance.

GAFRO’s implementation exploits compile-time sparse multivectors and
expression templates. We should not copy that machinery: `geom` is dense,
batched and tensor-based. We should copy the **robot modeling ideas**, the
**definitions**, and the **task formulation**, and then implement them in the
`Nx`/`Rune` style.

**Why not just dual quaternions?** A `Motor` already is a dual quaternion; for
pure pose tracking, dual-quaternion kinematics is enough and is cheaper than
dense CGA. CGA earns its place when the task itself is geometric: a point on
a plane, a tool line through a point, a circle to be grasped, a point pair
encoding two possible contacts. In those cases the same algebra carries the
motion, the objects and the incidence relation, so the cost and its Jacobian
have one shape instead of a special case per object type. If most tasks are
pose targets, the GA layer is optional; if tasks are object-relative, it is
the reason to use `geom` at all.

References:

- T. Löw, P. Abbet, S. Calinon, *gafro: Geometric Algebra for Robotics*,
  [arXiv:2310.19090](https://arxiv.org/abs/2310.19090).
- T. Löw, S. Calinon, *Geometric Algebra for Optimal Control with Applications
  in Manipulation Tasks*, IEEE T-RO 2023,
  [arXiv:2212.07237](https://arxiv.org/abs/2212.07237).
- [GAFRO website](https://geometric-algebra.tobiloew.ch/gafro/) and
  [repository](https://gitlab.com/gafro).

---

## 2. What `geom` already has

`geom` is already a good algebra layer for robotics:

| Need | Current state | Relevance |
|---|---|---|
| Poses and rigid motions | `Motor.t`: packed dual quaternion, product, inverse, `exp`/`log`, `slerp`, 6×6 adjoint, `translation`, `rotation` | This is the workhorse for kinematics; no need for a second pose type |
| Robot model, FK, Jacobians | `geom_robot` (module `Geom_robot.Robot`): links, joints, frames, validation, batched FK, geometric and analytic Jacobians, `custom_jvp` bridge | R0–R1; see §4.0 |
| CGA objects | points, spheres, planes, lines, circles, point pairs, incidence, distance, meet, conformal generators | Enough for task-space geometry and object-centric costs |
| PGA objects | points, planes, lines, meet/join, incidence, complement | Useful for plane/line tasks and camera-style constructions |
| Batched tensors | `Nx` arrays with a trailing blade axis; batching is part of the representation | Essential for sampling, trajectory optimization and learning |
| Differentiability | The algebraic core avoids host reads and value-dependent branches | FK, the Jacobians and the pose rule are `Rune.jit`/`jacfwd`/`grad`-clean; the test suite differentiates `Motor.exp` and compares |
| Visualization | `geom-viz` draws points, segments, circles, spheres, planes, frames, movies | Enough to inspect robot kinematics and tasks, though not yet link meshes |
| Tests | Strong algebra/CGA/PGA/motor tests | A good base for robotics regression tests |

The gaps are equally clear:

- no joint-space solvers: task residuals, IK, dynamics, control;
- no velocity propagation, `Jdot` or manipulability yet (the Jacobians are
  in, the time derivatives are not);
- no task residual abstraction beyond primitive objects;
- no model loading, collision or visualization of a model;
- no standard robot models beyond a 2R arm and a 6R chain, and no reference
  fixtures from Pinocchio/KDL/GAFRO yet;
- the algebra modules (`Motor`, `Cga`, `Pga`, `E3`) are hard-coded to
  `float32`, while `dilqr` and research optimization generally use `float64`.
  The robot layer is already a functor over its motion algebra so that it does
  not have to change when they do (§5.2).

---

## 3. What “research-grade” means here

A robotics library for research is not only a production controller. It needs:

1. **Correct conventions.** Frame composition order, active versus passive
   transforms, twist ordering, Jacobian reference points, motor sign
   ambiguity, and the meaning of a joint coordinate must be documented and
   tested. Most robotics bugs are convention bugs, not algebra bugs.
2. **Batch-first evaluation.** A robot model is small and fixed; experiments
   draw thousands of configurations, targets or trajectories. Functions
   should take `q` with leading batch axes and return batched motors,
   Jacobians and residuals. No Python-style per-sample loops.
3. **Traceability and differentiability.** Research code wants
   `Rune.grad`, `Rune.jvp`/`vjp`, `Rune.jit` and `Rune.scan`. The robotics
   layer must avoid host reads on tensor values, use `Nx.where` for
   value-dependent branches, and keep control flow static over joints.
4. **A float64 path.** Kinematics can live in float32 for visualization and
   learning, but dynamics, long-horizon iLQR and finite-difference validation
   need float64. This is a design decision to make early, not after the model
   layer exists.
5. **Reference validation.** Every algorithm needs a check against something
   independent: hand-derived 2R/3R formulas, a 4×4-matrix implementation,
   finite differences, Pinocchio/KDL, or GAFRO-generated reference data.
6. **Solver independence.** Expose residuals, Jacobians, mass matrices and
   dynamics as values. Do not bury them inside one optimizer. Research users
   will want to plug them into Gauss–Newton, iLQR, a QP, a sampling planner or
   a learned model.
7. **Ergonomics.** Named frames, a readable model builder and a standard robot
   zoo matter. A correct library that takes an hour to describe a 7-DOF arm
   will not be used.

---

## 4. Proposed robotics layer

The layer should be small and orthogonal. A possible module map:

| Layer | Responsibility |
|---|---|
| `Robot` / `Model` | links, joints, frames, topology, validation, model builders |
| `Kinematics` | forward kinematics, frame queries, relative poses |
| `Jacobian` | geometric/analytic/frame Jacobians, time derivatives, manipulability |
| `Spatial` | twists, wrenches, inertias, adjoint/coadjoint, spatial algebra |
| `Task` | pose and primitive residuals, weights, constraints, compact embeddings |
| `Ik` | damped least squares/Gauss–Newton, limits, multi-task/nullspace |
| `Dynamics` | RNEA, mass matrix, Coriolis, gravity, forward dynamics |
| `Control` | computed torque, PD+gravity, impedance/admittance, MPC hooks |
| `Model_io` | programmatic models, YAML/URDF import, standard robots |
| `Collision` | primitive proxies, distances, broadphase, constraints |
| `Trajectory` | motor interpolation, joint/Cartesian trajectories, time scaling |
| `Robot_viz` | robot frames/links and task primitives in `geom-viz` |

The names are provisional; the separation is the point.

### 4.0 The first slice: what is in the tree

`geom/robot/lib` is a library (`geom_robot`, module `Geom_robot.Robot`) with
`geom` and `rune` as its only dependencies, and `geom/robot/test` holds the
tests. It covers R0 and the part of R1 that does not need time derivatives:

| Piece | Status |
|---|---|
| Model: links, joints, frames, validation, limits, name lookup, printing | done (`create`, `kind_of_joint`, `limits`, `frame`, `pp`) |
| Joint kinds: fixed, revolute, prismatic — one coordinate each, any axis, axis offset handled by the joint origin | done; helical, floating and multi-DOF joints are the extension point |
| FK: every link, every frame, `[batch; dof]` in, `[batch; …; 8]` out | done (`link_motors`, `frame_motors`, `pose`, `relative`) |
| Analytic Jacobian: `dM/dq`, `[batch; 8; dof]` | done (`motion_jacobian`) |
| Geometric Jacobian: `[batch; 6; dof]`, world or body, referenced to the frame's origin | done (`jacobian`) |
| Differentiable FK: the closed-form tangent map, attached as a `Rune.custom_jvp` rule | done (`pose_tangent`, `pose_diff`) |
| Task residuals: pose and primitive (point/line/plane/circle/sphere/point pair), weights, stacking, custom | done (`Geom_robot.Task`); see §4.4 |
| Inverse kinematics: batched damped Gauss–Newton with a trust region, joint limits, per-sample damping and convergence | done (`Geom_robot.Ik`); no nullspace, hierarchy or restarts yet |
| Velocities: per-link twists by propagation, frame twists, accelerations (`bias` = `J̇ q̇`), `J̇` | done (`link_twists`, `twist`, `bias`, `jacobian_dot`) |
| Conditioning: singular values of the frame Jacobian and Yoshikawa's manipulability | done (`singular_values`, `manipulability`) |
| Standard models: 2R planar, 6R chain | done (`planar2`, `six_dof`); UR5/Panda wait for model I/O |

Two small additions to `geom` came out of it: `Motor.translation` and
`Motor.rotation`, the position and orientation of a motion, which every
consumer of a pose needs and which only the internal helpers could compute up
to now.

The tests are the point of the slice, because every one of them pins a
convention rather than a number: the 2R arm against its hand-derived closed
form; the 6R arm against an independent 4×4 homogeneous-matrix chain; the world
Jacobian against a float64 finite difference of the pose (which fixes
`J qdot = (omega, dp/dt)`); the *body* Jacobian against
`log(M⁻¹ M(q + h qdot)) / h`, that is, against the definition of the
right-trivialised derivative rather than against the world one; and the analytic
Jacobian against `Rune.jacfwd` through `exp`, which is an independent
implementation of the same derivative and a check that the level-0 algebra
really is traceable. Reverse mode through the `custom_jvp` rule is compared with
reverse mode through the branches it replaces.

Two demos exercise the layer end to end: `robot/demo/reach.ml` (damped
Gauss–Newton to a pose, with the pairing above) and `robot/demo/tour.ml` (a
tour: batched workspace clouds, a camera frame on a link, `J qdot` against a
finite difference, statics as `Jᵀ w`, `sigma_min` over a batch, batched IK with
limits, and `jit`/`jacfwd` against the closed form).

### 4.1 The robot model

A robot is a small directed tree:

- **Link**: name, parent, fixed transform from the parent link, mass, centre of
  mass, rotational inertia, and optional visual/collision geometry.
- **Joint**: name, parent link, child link, joint type, joint frame, axis or
  screw generator, limits and zero offset.
- **Frame**: a named fixed transform on a link. End-effector, camera, tool and
  sensor frames are all frames, not special cases.
- **Robot/System**: arrays of links and joints in topological order, a root,
  and named frames. Validation is part of construction: one parent per link,
  no cycles, consistent indices, finite inertias.
- **Kinematic chain**: a path from the root to a frame. It is a view over the
  tree, not a separate model.

Joint types to support, in order:

1. fixed;
2. revolute/continuous (one angle);
3. prismatic (one displacement);
4. free/floating (a full motor, six degrees of freedom);
5. general screw/helical joints later.

For a revolute joint, the natural GA object is a line (the joint axis) plus a
scalar angle; for a prismatic joint, a direction plus a scalar displacement.
Internally, store a joint generator as a twist or bivector so that a general
screw joint is not a special case. Floating-base robots also need a base
motor and a body twist in the state; this is a later extension, but the model
should not make a fixed base impossible to lift. The model should not be tied to DH
parameters: DH is a convenient way to build models, not a representation
format.

Model construction should start as an OCaml builder/record API. A YAML or
URDF loader comes later (Section 4.8). A 2R planar arm and a 6R arm with
hand-computed matrices are the first reference models.

### 4.2 Forward kinematics

For a chain of joints, each joint contributes a local motor
`M_i(q_i) = M_fixed_i · exp(q_i G_i)`, and the world pose of a link is the
ordered product of the local motors. A frame is then
`M_world_frame = M_world_link · M_link_frame`.

Requirements:

- Evaluate **all** links/frames in one pass, not only the end-effector.
- Support a batch of configurations `[B; n]` and return motors `[B; 8]`, or
  `[B; F; 8]` for all frames.
- Precompute fixed transforms and joint generators. The per-call work should
  be products and exponentials only.
- Normalize motors periodically; a long chain in float32 accumulates drift.
- Expose relative poses (`M_a⁻¹ M_b`) because task frames and Jacobians need
  them.
- Keep the implementation pure and traceable: a static loop over joints is
  fine; a dynamic loop over batch entries is not.
- Support a time axis through `Rune.scan` for trajectory evaluation, without
  materializing a Python-style list of configurations.

### 4.3 Differential kinematics

Three Jacobians are useful, and they should be named unambiguously:

- **Geometric Jacobian**: maps joint velocities to the spatial twist (linear
  and angular velocity) of a frame, expressed in a chosen frame. Columns are
  transformed joint screw axes.
- **Analytic Jacobian**: the derivative of the 8-component motor with respect
  to the joint coordinates. Useful for pose tasks and for composing with the
  motor logarithm.
- **Frame Jacobian**: the geometric Jacobian expressed at a named frame,
  obtained by the adjoint of the frame motor.

Also needed for dynamics:

- the **time derivative of the geometric Jacobian**, or equivalently the
  derivative of each transformed screw axis;
- the commutator product of joint screws (GAFRO’s `J×`), or an automatic
  differentiation alternative;
- a manipulability measure and a rank/singular-value interface, at least for
  diagnostics.

Design recommendation: implement analytic derivatives from the start for
speed and clarity, but validate them against `Rune` autodiff and finite
differences. A research user should be able to ask for the Jacobian without
knowing which implementation is used.

Conventions that must be pinned before this is implemented:

- twist ordering: `[ω; v]` or `[v; ω]` (`Motor.log` currently orders `ω`
  first);
- twist reference point: origin of the world frame, origin of the body frame,
  or a named frame;
- active versus passive interpretation of a motor;
- left versus right trivialization for pose errors and Jacobians;
- whether the geometric Jacobian is expressed in world or body coordinates.
  The API should make the choice explicit, probably as a frame argument.

### 4.4 Geometric task residuals

This is the distinctive part. A task maps a robot configuration to a residual
vector and a Jacobian. Two families cover most of the research use cases:

**Pose tasks.** The residual is the Lie-algebra error
`log(M_target⁻¹ M_current(q))`, a six-vector (or a subset for position-only
or orientation-only tasks). This is the natural analogue of a pose target in
classical robotics and is the right first task.

**Primitive tasks.** Choose a reference primitive `X` attached to a tool
frame (a point, line, plane, circle, sphere or point pair) and a target
primitive `X_d`. GAFRO's error is

`E(q) = X_d ∧ (M(q) X M(q)~)`,

zero exactly when the transformed reference primitive lies in the target
primitive’s outer-product nullspace. `geom` stores objects in the *other*
representative — points, spheres and planes as grade-1 vectors, lines, circles
and point pairs as grade-2 blades — and there incidence is the contraction
`X·O = 0`, so the same constraint reads

`E(q) = X_d · (M(q) X M(q)~)`,

with the contraction taken left-to-right in the order that keeps its grade
(higher-grade factor on the left). The two forms are the same constraint in dual
representations; using the representation's own operator is what makes the
residual one line and the Jacobian polynomial. It also decides the residual
dimension: a point on a plane is a scalar — the signed distance — a point on a
sphere is the power, a point on a line or a circle is a grade-1 vector (five
blades, three of them meaningful), a line through a point likewise.

Compact the blades of the resulting grade into a dense vector with a static
mask; the mask determines the residual dimension and the row layout of the
Jacobian. Normalize the objects at construction, with the model's own
normalizer, or the residual scales with the representative and the weights lose
their meaning.

Useful primitive pairs:

| Reference `X` | Target `X_d` | Task |
|---|---|---|
| point | point | position tracking |
| point | line | reach a line, line following |
| point | plane | reach a plane, surface contact |
| point | circle | reach a circle, circular grasping |
| point | sphere | reach a sphere |
| point | point pair | choose one of two targets (dual-arm, options) |
| line | point | pointing: a tool line passes through a point |
| line | line | line alignment |
| plane | plane | surface alignment |

The point-pair case is worth highlighting: it encodes a discrete choice
without a branch in the code. The solver reaches whichever point is closer
from its initial condition, and the Jacobian is valid in both cases. This is a
good example of GA making a task easier to state, not just prettier.

Practical requirements:

- **Normalize primitives.** A CGA point has a scale; a plane, sphere, line or
  circle has a natural normalization. The residual must be invariant to
  rescaling `X` and `X_d`, or the weights become meaningless.
- **Weights and units.** The components of `E` have different units and
  geometric meanings. Expose a per-task weight vector or matrix and document
  the default. For pose errors, weight in the Lie algebra; for primitive
  errors, consider normalizing by a length scale.
- **Constraints.** Equality constraints (a point on a plane), inequality
  constraints (distance to an obstacle, joint limits) and box constraints
  should be separate residual types, not forced into the least-squares sum.
- **Composition.** A task list with weights is the simplest interface. A
  hierarchical/nullspace interface and a constraint-based interface can be
  built on top. Do not hide the residuals behind one objective function.
- **Residuals are values, not just a loss.** Return `r(q)` as a vector and let
  the caller reduce it: exact Gauss–Newton wants `J` and `r`, a sketched solver
  wants `r` marked as a little loss, and a constraint solver wants the entries
  raw. A task that only offers a scalar cost cannot be used by either (§5.7).
- **Jacobians.** Differentiate the residual by the chain rule through FK, or
  use `Rune` autodiff. Compare both. With the analytic Jacobian `J_A` of the
  motor, the primitive-error Jacobian has the same shape as the error:
  `J_E = X_d · (J_A X M~ + M X J_A~)`, followed by the same blade compaction.
  The compact blade mask is static, so the Jacobian rows can be assembled
  without host branches. This is implemented, and the test suite checks it
  against a finite difference of the residual for the point/plane, point/sphere,
  point/line and line/point pairs.

- **A pose error's derivative is not the body Jacobian.** `log(M_d⁻¹ M)` is the
  residual; the body Jacobian is the linearisation of the *pose*, and the two
  agree only to first order (`d log_E` is the identity at the solution). The
  pair still gives Gauss–Newton on the pose — the step solves the linearised
  group equation exactly, so convergence is quadratic — and it is what every
  operational-space controller uses. An exact `d log_E` is a refinement worth
  having for least-squares mixes and for iLQR, and it is on the list.
- **Pose versus primitive.** A pose task constrains full orientation; a
  primitive task often leaves a rotational or translational gauge freedom.
  This is a feature, not a bug: a plane task should not care where on the
  plane the tool point is. Document the gauge freedom of each task.

### 4.5 Inverse kinematics

Start with damped least squares / Gauss–Newton on the task residuals:

`q_{k+1} = q_k − α (Jᵀ W J + λ I)⁻¹ Jᵀ W e(q_k)`.

with `e(q) = log(M_target⁻¹ M(q))` and `J` the **body** Jacobian
(`Robot.jacobian ~space:`Body``). The pairing is not a detail: `log(M_target⁻¹
M)` is expressed in the body coordinates of the current frame, so it is the body
Jacobian that maps a joint step to the change of that error. Pairing it with the
world Jacobian mixes two conventions, and the resulting solver creeps —
linearly, or not at all. It was tried: the first version of `robot/demo/reach.ml`
did exactly that, needed a backtracking line search to converge, and reached
`2e-3` in twenty iterations; with the pairing right it reaches `5e-7` in eight,
with no line search. Use `log(M M_target⁻¹)` if you want a spatial error, but
then the Jacobian is the spatial one referenced to the *world origin*, which is a
third convention and not `Robot.jacobian ~space:`World`` (that one is referenced
to the frame's origin, which is what `J qdot = (omega, dp/dt)` means).

Needed:

- Cholesky or QR solves from `Nx`; a small linear solver is enough;
- Levenberg–Marquardt damping and a line search, plus a **trust region on the
  step**: a pose inside the workspace usually has a solution that winds a joint
  through several turns, and undamped Gauss–Newton finds it happily. It is a
  correct answer and a useless trajectory, which is exactly the sort of thing a
  rendered demo makes obvious and a residual does not;
- joint-limit projection or a barrier/penalty;
- nullspace projection for secondary tasks;
- deterministic initialization and random restarts for nonconvex targets;
- batch solving over many targets, with per-sample convergence flags;
- an adaptive host-side solver (like `Dilqr.Solver.solve`) and a fixed-iteration
  traceable solver (like `Dilqr.Solver.solve_fixed`) for `Rune.jit`/learning.

Return a structured result: `q`, final residual, iterations, success, and
maybe the final Jacobian singular values. Do not throw on failure; IK is
nonconvex and failures are data.

**What shipped (`Geom_robot.Ik`).** A batched Levenberg–Marquardt loop over
`Task` residuals: per-sample damping with a {e gain ratio} (a step is kept where
the residual falls by a fair fraction of what the linear model promised, and the
damping relaxes there; rejected steps stiffen it), a trust region on the step,
projection onto joint limits, and a result carrying `q`, the residual norm, the
iteration count, the damping and a per-sample convergence flag.

Two things this cost, both worth recording. The gain ratio is not decoration:
comparing the two residual values alone, in float32, lets rounding noise accept
steps that change nothing — which relaxes the damping, which enlarges the next
step, which also changes nothing, and the solver sits there while the damping
falls to `1e-12`. Second, it is a local solver: on the six-joint arm, starts
within about a radian of the goal all converge, and around 1.7 rad about a third
settle into a local minimum of the pose error. Restarts are the caller's
decision, not a hidden policy. The whole batch is one tensor computation per
iteration, and `Ik.iterate` is exposed so a demo — or a research idea — can drive
the loop itself. Deliberately absent: nullspace projection, hierarchies,
restarts, and a convergence test other than the residual norm. Those are the open
questions below, not oversights.

For a large number of coordinates — a whole trajectory of waypoints, dual-arm
or whole-body problems — the same residuals can drive a *sketched* Gauss–Newton
step at one forward pass and no Jacobian at all; §5.7 works out when that is
worth it and what it requires of the residual API.

The first research use cases are:

- pose IK for a 6R/7R arm;
- primitive-reaching IK for point/line/plane/circle/sphere targets;
- point-pair IK as a discrete-choice experiment;
- batch success-rate studies and warm-start studies.

### 4.6 Dynamics

Dynamics is the largest and least certain part. It is not needed to start
research on kinematics, task geometry or differentiable IK, and GAFRO itself
reports that its GA dynamics is slower than established libraries. The
recommendation is staged:

**Stage 1 — spatial algebra with GA poses.** Implement:
- `Twist` and `Wrench` as 6-vectors with explicit frame/point conventions;
- `Inertia` as a 6×6 spatial inertia (mass, COM, rotational inertia);
- adjoint/coadjoint transforms from `Motor`;
- recursive Newton–Euler for inverse dynamics;
- composite-rigid-body algorithm for the mass matrix;
- gravity and Coriolis via RNEA with selected terms;
- forward dynamics via a Cholesky solve.

This is conventional robotics dynamics, but with `Motor` poses and the option
to express the same quantities as GA bivectors/wrenches. It is easier to
validate and optimize than a full multivector-matrix dynamics implementation.

**Stage 2 — GA-native dynamics.** Once the spatial version is trusted,
implement the CGA formulation:
- inertia as a grade-preserving map on bivectors (GAFRO’s `Inertia`);
- RNEA and articulated-body algorithms with CGA twists/wrenches;
- mass matrix, Coriolis and gravity as multivector matrices;
- compare numerically and in performance with Stage 1.

The gravity vector is a world-frame quantity; the base is fixed or floating.
A discrete-time integrator (semi-implicit Euler or RK4) belongs outside the
continuous dynamics core, because iLQR needs to differentiate through it and
because different experiments use different integrators.

Validation is non-negotiable:

- RNEA versus finite differences of a Lagrangian;
- mass matrix symmetry and positive definiteness;
- forward dynamics versus `M⁻¹`;
- energy conservation for a passive chain;
- momentum conservation for a free-floating system;
- comparison with Pinocchio/KDL/GAFRO on the same model.

Differentiability: the dynamics functions should be pure `Nx`, with static
loops over joints, so `Rune` can differentiate them for iLQR/MPC. If the
algorithm has data-dependent branches (e.g. a singularity fallback), use
`Nx.where` or a fixed-branch formulation. A hand-derived derivative is a
fallback, not the first implementation.

### 4.7 Control and optimal control

Do not build a general control stack before the model, kinematics and tasks
are stable. Then:

- **Computed torque / inverse-dynamics control**:
  `τ = ID(q, q̇, q̈_d) + Kp(q_d − q) + Kd(q̇_d − q̇)`.
- **PD + gravity** for a first closed-loop test.
- **Task-space / operational-space control** using the geometric Jacobian and
  the task residuals.
- **Impedance/admittance control** using CGA primitives for contact geometry.
  GAFRO’s controllers are a reference; start with one simple Cartesian
  impedance controller, not a zoo.
- **iLQR/MPC**. This is where the Raven ecosystem is unusually strong: the
  workspace already has `dilqr`, a differentiable iLQR implementation on
  `Nx`/`Rune` with `Rune.scan`, adaptive and fixed solvers, and implicit
  gradients through the solve. The robotics layer should expose:
  - a state `x = (q, q̇)` (or a motor/bivector state for kinematic MPC);
  - a control `u = τ` (torque-level) or `q̈` (kinematic-level);
  - a discrete-time dynamics function (with the integrator chosen explicitly);
  - running and terminal costs built from GA task residuals;
  - a `Dilqr.Problem.t`-compatible interface.

  This is the most compelling research direction for this codebase:
  differentiable geometric task costs inside a differentiable optimal-control
  solver. It also forces the float64 and traceability decisions early.

  iLQR wants the Hessian blocks of the stage cost in `(x, u)`, so a sketched
  Gauss–Newton (§5.7) does not drop in — but the *cost* layer is shared, and a
  trajectory-level sketched solve is a useful non-Riccati baseline.
- **Kinematic MPC** on the motor manifold is a useful intermediate step: a
  double integrator in bivector space, with pose/primitive targets, is much
  simpler than torque-level MPC and still exercises the GA task layer.

### 4.8 Model I/O and a robot zoo

Start with programmatic models. Then add:

- a minimal, versioned YAML or JSON robot schema (links, inertias, joints,
  frames, visual/collision proxies);
- a URDF importer, or a conversion tool from URDF to the schema. URDF is the
  standard in robotics; YAML is GAFRO’s choice. Support one, not both, at
  first;
- a small set of reference robots: a 2R planar arm, a 6R arm, Franka Emika
  Panda, UR5. These are enough for regression tests and demos;
- shipped reference data: FK, Jacobians, mass matrices and inverse dynamics
  for a few configurations, generated from Pinocchio/KDL/GAFRO and stored as
  small fixtures;
- asset handling for visualization: mesh paths, link colours, collision
  proxies. Do not require meshes for the algorithms.

A separate `geom_robot_io` package is preferable if parsing brings in XML or
YAML dependencies: the core `geom` package should stay free of file-format
and robotics-stack dependencies.

### 4.9 Collision, distance and constraints

A full collision engine is out of scope. A useful research subset is:

- link collision proxies as a union of spheres, capsules, boxes and planes;
- CGA distance and incidence queries between primitives;
- a broadphase over proxy bounding volumes;
- signed-distance and separating-vector queries for inequality constraints;
- self-collision checks for adjacent/non-adjacent links;
- integration as task constraints in IK/MPC, not as a separate planner.

The GA advantage here is that the narrowphase is exact and uniform for
primitives: point–sphere, point–plane, sphere–plane, line–sphere, etc. The
broadphase and mesh handling are ordinary computational geometry and should
not be forced into GA.

### 4.10 Trajectories and interpolation

Needed for demos, MPC warm starts and learning:

- motor `slerp` for two waypoints;
- log-weighted multi-waypoint interpolation (GAFRO’s
  `exp(Σ w_i(t) log(M_i))`) for viapoints;
- joint-space cubic/quintic/minimum-jerk trajectories with velocity and
  acceleration limits;
- Cartesian trajectories as motor curves, with a time parameterization;
- batch evaluation for plotting, MPC and dataset generation.

`Motor.slerp` exists; the rest is small, but the conventions (which frame,
which endpoint, left or right trivialization) must match the kinematics layer.

### 4.11 Visualization

`geom-viz` already draws frames, points, lines, planes, circles and spheres, and
the bridge from a model to it is short enough to live in a demo:
[`robot/demo/arm.ml`](robot/demo/arm.ml) draws the 6R arm — links as segments
between `Robot.link_motors` positions, a sphere per joint, the tool triad, a
dashed target marker, the tool's path, and an MP4 of the reach — in about forty
lines that only ever hand `Primitive.t` values to the renderer. That is the
shape the `geom_robot_viz` library now has: `entities` and `animated` take a
model and a configuration (or a path) and return `geom-viz` entities, and
[`robot/viz/demo`](robot/viz/demo) has four uses of it — a still and a movie of
the arm reaching a pose, a workspace cloud coloured by `sigma_min`, ten IK starts
converging to one target, and a tool point driven onto a `Cga` circle with the
plane of the circle drawn.

For robotics, add:

- link visualizations: capsules/boxes first, meshes later;
- robot state rendering at a configuration or along a trajectory;
- task primitives and residuals overlaid on the robot;
- collision proxies and contact points;
- end-effector trajectories and frames.

This is low priority for the algorithms but high value for debugging and for
papers. It should remain in `geom-viz`/`geom_robot_viz`, not in the core
algebra.

---

## 5. Representation and numerical decisions

### 5.1 Motor for poses, CGA for tasks

Use `Motor` for every link, joint and frame pose. It is eight floats, already
has the operations kinematics needs, and is the packed dual quaternion that
GAFRO’s CGA motors reduce to. Do not store every link pose as a dense
`Cga.t`; that pays 32 coefficients for a quantity that needs eight.

Use `Cga.t` for task objects and geometric reasoning: points, lines, planes,
circles, spheres, point pairs, incidence and distances. Use `Pga.t` for
plane/line incidence when the projective formulation is clearer. The shared
`Motor` type is what ties the layers together.

### 5.2 Float precision

Current model modules (`Motor`, `Cga`, `Pga`, `E3`) are fixed to
`Nx.float32`. `dilqr` and `ilqr_vae` use `Nx.float64`, and dynamics/IK
validation benefits from float64. The options are:

1. **Functorize the model modules over the scalar kind.** This is the
   cleanest long-term answer: `Algebra.Make` is already scalar-generic; the
   wrappers need the same treatment. It is a substantial refactor.
2. **Add float64 instantiations** (`Motor64`, `Cga64`, …) generated from the
   same functors. Less invasive, but doubles the API surface and risks
   duplication.
3. **Keep GA in float32 and do the surrounding linear algebra in float64.**
   Cheap, but mixed precision is a source of subtle errors in IK/dynamics.
4. **Use float32 for learning/visualization and float64 for reference
   validation only.** Acceptable for early kinematics, not for a serious
   dynamics/MPC layer.

**Decision (taken).** Option 1 eventually, and a functor seam now, so that the
eventual refactor is local to `geom` and touches no robotics code:

- `Geom_robot.Robot.Make (Mo : MOTION)` builds the whole model, FK and Jacobian
  layer over an abstract motion algebra; `Robot.Float32` is the instance the
  library ships, and `Robot` is `Make (Float32)` applied. `MOTION` is the
  eight-component packed dual quaternion interface at some scalar kind:
  product, inverse, `exp`/`log`, adjoint, `translation`, `rotation`.
- A `float64` instance is therefore a second implementation of `MOTION` plus
  `module Robot64 = Robot.Make (Float64)`, not a change to this layer. The
  same code serves float32 for batching and learning and float64 for dynamics,
  IK validation and iLQR.
- What remains open is the algebra side: `Motor` itself is still float32, so a
  `Float64` motion module does not exist yet. The honest options there are
  still the four above, and functorizing `Motor` over `Algebra.SCALAR` (option
  1) is the one to try first, because `Algebra.Make` is already scalar-generic
  and `Motor` is a thin packed layer over it. The one obstacle is
  `Motor.of_rotation`, which reads its input on the host — it needs a
  tensor-native formulation before it can be instantiated at a second kind.

Nothing in `geom_robot` may introduce a `float32` assumption that the `MOTION`
signature does not carry. That is the rule that keeps this decision cheap.

### 5.3 Dense versus sparse

GAFRO stores only the nonzero blades of each specialized multivector at
compile time. `geom` stores all 32 (CGA) or 16 (PGA) coefficients in a dense
`Nx` tensor. For robotics:

- dense 32-float CGA is fine for task objects; batching and `Nx`/`Rune` are
  worth more than a sparse representation;
- use static blade masks and `Nx.take` to compact task residuals and
  Jacobians; do not build a second sparse GA implementation;
- for motor-heavy loops, use `Motor` (8 floats), not dense CGA motors;
- if a hot path is dominated by zero coefficients, optimize the algebra or
  the mask, not the model API.

### 5.4 Automatic differentiation versus analytic derivatives

Neither "AD" nor "hand-derived": they are the same derivative, written at two
levels, and the right answer is to write each leaf once and let AD compose.

**Decision (taken). AD is the composition; analytic derivatives are the
leaves.** `Rune.custom_jvp` and `Rune.custom_vjp` make this exact rather than
aspirational: a closed form is not a parallel implementation that has to be
kept in sync, it is the derivative {e that AD reports} for that function.

Concretely:

- **Compose with AD.** `Rune.jacfwd`, `Rune.jvp`, `Rune.vjp`, `Rune.grad`,
  `Rune.vmap` and `Rune.jit` see ordinary pure-`Nx` code and differentiate it
  in place. No robotics function needs a hand-written "Jacobian of the whole".
- **Rule the leaves.** Where a closed form is stable and cheap, attach it with
  `Rune.custom_jvp p q (fun x -> f x, fun dx -> tangent x dx)`, which serves
  forward mode directly and reverse mode by transposing the same code, and
  keeps higher orders working (the rule's own code gets differentiated).
  `Rune.custom_vjp` is the same arrangement when only the pullback is known or
  when the forward tangent map is expensive to write.
- **Where that pays.** `Motor.exp`/`log` (branchy closed forms with small-angle
  and π limits), the joint motion `q ↦ exp (q g)`, FK, and the task residuals
  built on them. R0–R1 already do this for the pose:
  `Robot.pose_tangent` is the closed form from `Robot.motion_jacobian` and
  `Robot.pose_diff` attaches it to `Robot.pose` as a rule. A model built on
  `pose_diff` therefore differentiates the analytic derivative, in both modes,
  at every order, while `pose` remains the readable definition that the rule is
  checked against — `Rune.jacfwd` through `pose` differentiates `exp` itself,
  and the two agree to float32 in the test suite.
- **Cross-check** the rule against the branches it replaces on random
  configurations. A discrepancy is a bug or a branch-cut issue, and it is a
  test, not a hope.
- **`custom_jvp` versus `custom_vjp`.** Prefer `jvp` when the forward tangent is
  cheap (it is, for kinematics: the tangent map is a matmul against the
  analytic Jacobian) and `vjp` when the output is large and only a scalar or a
  few cotangents come back. Never attach both to the same function at different
  places in a graph and expect agreement for free — the check is the test.
- **What is still hand-derived.** The remaining closed forms are the task
  residuals' Jacobians (§4.4) and the dynamics quantities (§4.6). They follow
  the same rule: write the value, attach the derivative as a rule, and keep an
  AD path as the reference oracle.

One caveat that the R0 slice makes concrete: a rule must not branch on values
that its own differentiation tracks, so anything a rule needs must be computed
inside it, from the arguments, and not captured from a surrounding task. That is
also why the `MOTION` signature has to stay value-free.

### 5.5 Batching and JIT

Choose a layout and keep it everywhere:

- configurations: `[B; n]`;
- motors: `[B; 8]`;
- all frames: `[B; F; 8]`;
- geometric Jacobians: `[B; 6; n]`;
- task residuals: `[B; d]`;
- task Jacobians: `[B; d; n]`;
- time series: a leading time axis or `Rune.scan` over time.

The joint count and frame count are static, so the loops over them are
static. The batch and time axes are dynamic, so they should be tensor axes.
No `Array.map` over a batch of configurations and no host `for` over time in
a traced function.

### 5.6 Conventions

Write a short conventions section in the robotics `.mli` or a companion
note, and test every sign against a 2R arm. At minimum:

- a motor `M_ab` maps coordinates from frame `b` to frame `a`;
- `M_ac = M_ab · M_bc`, and `M_ba = M_ab⁻¹`;
- an active transform of a point is the sandwich `M p M~`;
- a joint motor is `M_fixed · exp(q G)` with `G` in the joint frame;
- twist ordering and reference point (match `Motor.log`);
- geometric Jacobian expressed in world or body, explicitly;
- pose error `log(M_target⁻¹ M_current)`, and its pairing with the body
  Jacobian — or `log(M_current M_target⁻¹)` with the spatial Jacobian at the
  world origin, which is a different object from the world-referenced Jacobian
  at the frame's origin;
- primitive residual `X_d ∧ (M X M~)` and the blade mask;
- normalization of each primitive;
- left/right trivialization for interpolation and error derivatives.

Most published robotics formulas assume a specific convention. GAFRO’s paper
and our notes use similar but not identical conventions. Do not mix them
silently.

### 5.7 Sketched Gauss–Newton: where SOFO fits

The workspace already contains `sofo`, a second-order optimizer that never
backpropagates: it measures the loss and its curvature along a random
`k`-dimensional subspace of parameter space with one batched forward-mode pass,
solves the resulting `k×k` damped system, and steps along the result. Its
curvature is the **generalized Gauss–Newton matrix** `ΘᵀJᵀHJΘ`, accumulated from
little losses the user marks with `Sofo.observe` / `Sofo.mse` / `Sofo.sse`.

That is the same approximation IK and iLQR make anyway. The consequence is a
decision about *shape*, not about algorithms:

- **No approximation is lost, only the search space is restricted.** A damped
  Gauss–Newton step is the exact inverse of the GGN in the full space; a SOFO
  step is the exact inverse of the GGN inside a random `k`-subspace. When
  `k = dof` it *is* Gauss–Newton, up to the choice of subspace. So the same task
  residual layer can serve an exact solver and a sketched one.
- **The premise is `k ≪ P`,** where `P` is the number of numbers being
  optimized. For single-arm IK (`P = 6…7`) the exact Jacobian is a handful of
  motor products — buy the exact one. The sketch earns its place when `P` is
  large: the whole trajectory of a waypoint IK (`P = dof × T`), system
  identification and task-parameter learning (task frames, object parameters,
  motor primitives), dual-arm and whole-body problems, and any setting where the
  residual itself is expensive (many primitives, CGA object distances,
  collision proxies). There, one forward pass through all `k` directions
  replaces both the Jacobian assembly and the normal-equation solve, and the
  memory stays constant in the residual dimension.

**What this asks of the layers above R1.** Three design rules, all cheap now and
expensive to retrofit:

1. **Residuals are values, and costs are sums of little losses.** A task must be
   able to return its residual vector `r(q)` (`[batch; d]`) and, separately, a
   cost built as `Σ_i w_i r_i²` with each block marked by its own curvature.
   `Sofo.mse ~w ~target:zeros` is exactly that block for a squared-error task;
   `Sofo.Curv.hvp` covers a curvature that has no closed form. A loss that
   pre-mixes the residual into a scalar and hides the weights cannot be
   sketched: SOFO's law 6 says curvature enters with weight one, so a mean or a
   scale applied *outside* a mark reaches the loss and the sketched gradient but
   not the curvature. Put the reduction inside the little loss.
2. **Non-least-squares terms stay outside the marks.** Joint-limit barriers,
   hinge penalties and other terms whose Hessian is not `JᵀJ` contribute to the
   loss but must not be marked, or the sketched curvature is that of a loss
   nobody is minimizing. This is the same distinction §4.4 draws between
   residuals and constraints, and it is the reason to keep them separate in the
   API rather than summing them into one objective.
3. **One residual function, three drivers.** The same `loss` must run under
   `Rune.value_and_grad`, under a bare `Rune.vmap` around `Rune.jvp`, and under
   `Sofo.sketch`. That is free if §5.4 is followed: SOFO drives forward-mode AD
   through `k` tangent lanes, so the `custom_jvp` rules on `exp`, the joint
   motion and `pose` are what the sketch differentiates — the sketch is cheap
   exactly because the leaves are closed forms.

**Inverse kinematics, sketched.** For a batch of targets and a task list, with
`q` as the parameters:

```
r(q)          (* the task residuals, [batch; d] *)
loss q        (* Σ blocks, each marked with Sofo.sse / Sofo.mse *)
step          (* Sofo.Optim.sketch_jit ~k Params.ptree Aux.ptree loss *)
st, out       (* one batched forward pass: C = Θᵀ Jᵀ W r, G̃ = Θᵀ Jᵀ W J Θ *)
z             (* Sofo.Optim.coordinates (damped k×k solve, host, O(k³)) *)
q := q - η Θ z
```

Note what never gets built: no `6 × dof` Jacobian, no `dof × dof` normal
matrix, no reverse pass. What does get built is the `k × k` sketched GGN on the
host and a `k`-lane forward evaluation. Damping is relative to the sketched
spectrum, so it transfers between tasks (`Sofo.Optim.damping`). Convergence
flags, restarts and joint limits stay the caller's business, exactly as in
§4.5 — a sketched step is still a step, and IK is still nonconvex.

**In MPC and iLQR, more speculatively.** The iLQR backward pass needs the
Hessian blocks of the stage cost in `(x, u)`, not a `k`-subspace of the whole
parameter vector, so a sketch does not drop into `dilqr` unchanged: it changes
the algorithm (a sketched Riccati recursion, or a Gauss–Newton QP solved on a
subspace), and that is research, not integration. Two uses are more immediate:

- **trajectory-level optimization** — treat all `T` configurations as the
  parameter vector and sketch over them; the residual is the whole task
  trajectory and `k ≪ dof × T` is where the method shines. This is a useful
  baseline to compare against iLQR, not a replacement for it;
- **outer-loop learning** — the parameters that a differentiable iLQR solve is
  differentiated *with respect to* (task frames, weights, object parameters) are
  usually a large, unconstrained vector and a GGN sketch over them is exactly
  SOFO's setting.

Caveats to keep honest: sketches are float64 (`Sofo` says so: the accumulation
and the `k×k` solve are not float32 work), the `k×k` eigendecomposition is
host-side and does not compile, and a sketch's step is a random variable — seed
it, and report the seed in any experiment.

---

## 6. What to borrow from GAFRO, what not to borrow

**Borrow:**

- the decomposition into links, joints, frames, chains and systems;
- the joint-type taxonomy and model description ideas;
- forward-kinematics and Jacobian definitions;
- the motor-log pose error and the generic primitive nullspace error;
- compact embedding of multivector residuals into vectors/Jacobians;
- the task examples: pose, point, line, plane, circle, sphere, point pair,
  pointing;
- the dynamics formulation as a reference and validation target;
- the idea of compositional task spaces (single-arm, cooperative, dual-arm);
- admittance/impedance control as a later, concrete use case.

**Do not borrow:**

- compile-time sparse multivector templates; our cost model is dense `Nx` +
  batching;
- ROS as a core dependency; a separate package can provide it later;
- a full mesh/physics stack;
- the entire controller zoo before the model and task layers are solid;
- GA-native dynamics as the first dynamics implementation;
- GAFRO’s exact benchmarks or performance claims without re-measuring in
  OCaml;
- a general-purpose nonlinear solver; expose residuals and use the ecosystem.

---

## 7. Milestones

A dependency-ordered plan. Each milestone should end with tests and a small
example, not only an API.

| Milestone | Scope | Status |
|---|---|---|
| **R0 — Model + FK** | link/joint/frame/robot types, validation, 2R and 6R builders, FK for all frames, batching, convention note | **done** — 2R hand formulas, 6R matrix FK, batching, precision decision recorded (§5.2) |
| **R1 — Differential kinematics** | geometric/analytic/frame Jacobians, `Jdot`, manipulability, twist/wrench types | **done** — both Jacobians, world/body, the `custom_jvp` rule, velocity propagation, `J̇ q̇` and `J̇`, singular values and manipulability, all checked against finite differences and `Rune.jacfwd`; wrenches wait for dynamics |
| **R2 — Tasks + IK** | pose task, primitive tasks (point/line/plane/circle/sphere/point pair), compact residuals/Jacobians, damped Gauss–Newton, limits, batch IK | **done** — `Task` and `Ik`; Jacobians checked against finite differences, batched IK, limits and trust region tested; demos in `robot/viz/demo` |
| **R3 — Model I/O + zoo** | minimal YAML/URDF import, 2R/6R/Panda/UR5, reference fixtures, `geom-viz` robot rendering | acceptance: load and compare FK/Jacobians with Pinocchio/KDL/GAFRO fixtures; visual demos |
| **R4 — Dynamics** | `Spatial` types, RNEA, mass matrix, gravity/Coriolis, forward dynamics, differentiability | needs the float64 motion module first (§5.2); acceptance: energy/momentum checks, RNEA vs Lagrangian, ABA/`M⁻¹`, Pinocchio/KDL comparison, `Rune` trace |
| **R5 — Control + MPC** | computed torque, PD+gravity, task-space control, `dilqr` integration, kinematic and torque MPC examples | acceptance: stabilize a 2R/6R arm, track a pose/primitive, iLQR converges, gradients match finite differences |
| **R6 — Research extensions** | collision constraints, dual-arm/point-pair tasks, uncertainty/probabilistic models, learning/system ID, benchmarks | task-specific; compare against reference implementations |

R0–R2 are the minimum viable research stack. R3 makes it credible on real
robots. R4–R5 are the larger investment. R6 is open-ended.

---

## 8. Research directions this unlocks

The point of the layer is not to reimplement a production robot library. It is
to make geometric-algebra ideas cheap to test. Some concrete directions:

- **Differentiable geometric optimal control.** Combine R2/R5 with `dilqr`:
  GA task costs, GA pose errors and `Rune`-traced dynamics inside
  differentiable iLQR/MPC. Gradients through the solver can be used for
  system identification, learning task parameters, or inference in
  iLQR-VAE-style models.
- **Task-and-motion research with primitives.** Pose, line, plane, circle,
  sphere and point-pair tasks share one residual/Jacobian structure. This
  makes it cheap to compare task parameterizations, study gauge freedoms, and
  learn task frames.
- **Discrete choices without branches.** Point-pair targets and
  cooperative/dual-arm task spaces encode options in the algebra. This is a
  natural fit for manipulation with two targets, two arms, or two contact
  modes.
- **Uncertainty on the motor manifold.** Gaussians on bivectors/motors, as in
  GAFRO’s probabilistic layer, for filtering, imitation learning and
  uncertainty-aware control.
- **Object-centric learning and manipulation.** CGA objects are a natural
  representation for object-relative tasks; differentiable task residuals
  make object parameters learnable.
- **Benchmarking GA against matrix robotics.** FK, Jacobian, IK and dynamics
  benchmarks against KDL/Pinocchio, with the same models and hardware. This is
  needed to make claims about GA performance credible.
- **Neuroscience/behavior modeling.** If the target is movement science,
  motor primitives, screw trajectories and geometric task variables are
  natural low-dimensional descriptors; the batched/differentiable stack makes
  model fitting practical.

---

## 9. Non-goals

For the first robotics layer:

- a full physics engine (contact, friction, constraints, integration);
- mesh-based collision detection or mesh asset management;
- ROS integration, real-time middleware or hardware drivers;
- a general nonlinear programming or trajectory-optimization solver;
- a production controller zoo;
- arbitrary-dimension or symbolic geometric algebra;
- GPU kernels of our own (use `Nx`/`Rune`);
- reproducing GAFRO’s compile-time sparse multivector design;
- a full URDF standard implementation before the model/kinematics layer is
  stable.

---

## 10. Open questions

Status of each. "Decided" means the decision is recorded in the section named
and the code follows it; the rest are genuinely open and block nothing yet.

1. **Scalar genericity.** *Decided for this layer, open for the algebra.*
   `geom_robot` is a functor over a motion module and ships a float32 instance;
   float64 needs a `MOTION` implementation, which needs `Motor` itself
   functorized over `Algebra.SCALAR` (§5.2). Do that before R4, not before R2.
2. **Dynamics strategy.** Spatial algebra first (§4.6), still the recommendation
   — GA-native RNEA/ABA has to earn its performance, and GAFRO's own numbers are
   the warning. Open: what performance target, and on which hardware.
3. **Task residual representation.** *Decided:* compact the blades of the
   residual's grade with a static mask, one residual vector per task, weights as
   row scales, tasks stacked by concatenation. §5.7 adds a constraint the
   implementation follows — a residual is a value that a caller can mark as a
   little loss, so the compaction happens before any reduction.
4. **Model format.** Open. YAML first (small, no XML dependency), URDF through
   a separate `geom_robot_io` package when a real robot is needed; do not put an
   XML parser in `geom` or `geom_robot`.
5. **Analytic vs AD derivatives.** *Decided* (§5.4): AD composes, closed forms
   are `custom_jvp`/`custom_vjp` leaves, and each closed form is cross-checked
   against the AD path through the definition it replaces.
6. **The differential of the pose logarithm.** Open. `Task.pose` pairs
   `log(M_d⁻¹ M)` with the body Jacobian: exact at the solution, first order
   near it, enough for Gauss–Newton and not the derivative of the residual. An
   exact `d log_E` (the inverse right Jacobian of `SE(3)`) would make pose tasks
   exact least-squares terms, which matters when they are mixed with weights or
   used as iLQR costs. The closed form is known and branchy; a `Rune.custom_jvp`
   rule on `Motor.log` is the other way to get it, and the better one if it
   works.
7. **Frame conventions.** *Decided for the layer* (§4.0 and the `Robot` module
   docs): a motor maps coordinates of its right operand into its left;
   [M_child = M_parent M_fixed exp(q g)]; twists are [(omega, v)] with the
   linear part referred to the origin of the frame the twist is expressed in;
   the geometric Jacobian is referenced to the frame's origin and is world by
   default, body on request. The tests are the specification; conversions to
   Pinocchio's and GAFRO's conventions still need writing down when their
   fixtures arrive.
8. **Control/task-space scope.** Open, but ordered: single-arm pose and
   primitive tasks (R2) before dual-arm, before hands and humanoids. The
   point-pair task is the cheapest dual-arm experiment and should come first.
9. **Visualization.** Open. Capsules and boxes are enough for R0–R2 debugging;
   mesh loading is a question for papers, and belongs in `geom_robot_viz`.
10. **Benchmark policy.** Open. Proposal: 2R and 6R analytic, UR5 and Panda from
   Pinocchio, and a fixed set of configurations stored as float64 fixtures
   checked into the repo, with FK, Jacobians and (later) mass matrices and
   inverse dynamics. GAFRO-generated fixtures are worth having as a second
   opinion on the algebra, not as the primary reference.
11. **Package boundary.** *Decided for now:* `geom` stays the algebra;
    `geom_robot` is a library in the same dune project with `geom` and `rune`
    as its dependencies; I/O and visualization become separate libraries
    (`geom_robot_io`, `geom_robot_viz`) exactly as `geom_viz` is separate from
    `geom`. If `geom_robot` grows to need a file format or a rendering stack in
    its core, it splits.

---

## References and pointers

- [GAFRO website](https://geometric-algebra.tobiloew.ch/gafro/)
- [gafro: Geometric Algebra for Robotics (arXiv:2310.19090)](https://arxiv.org/abs/2310.19090)
- [Geometric Algebra for Optimal Control with Applications in Manipulation Tasks (arXiv:2212.07237)](https://arxiv.org/abs/2212.07237)
- [GAFRO repository](https://gitlab.com/gafro)
- [`PLAN.md`](PLAN.md) — the core algebra design and implementation plan
- [`VIZ.md`](VIZ.md) and [`viz/README.md`](viz/README.md) — the renderer
- [`doc/notes.typ`](doc/notes.typ) — conventions, derivations and the motor algebra
- `../dilqr` — differentiable iLQR on `Nx`/`Rune`
- `../ilqr_vae` — iLQR-VAE on the same stack
- `../sofo` — sketched second-order optimization (§5.7)
- [`robot/lib/robot.mli`](robot/lib/robot.mli) — the layer this note produced
- [`robot/test/test_robot.ml`](robot/test/test_robot.ml) — the conventions, as
  executable statements
