# MuJoCo on Raven — feasibility and plan

Status: **design note, no code**. The execution plan is
[`MILESTONES.md`](MILESTONES.md); the viewer design is [`VIZ.md`](VIZ.md).

Written after reading MuJoCo 3.17's engine and Raven at pin `3984341`, and
after measuring both on one machine. The numbers in §6 and appendix A are from
this machine: AMD Ryzen 5 7640U, 6 cores, 12 threads, OCaml 5.5.2, float64,
warm `~/.cache/dune`.

The short answer: a native port is feasible for the subset that MuJoCo XLA
(MJX) already implements, and it costs a few months of focused work. It buys
full automatic differentiation through the step, batching and compilation in
the same language as the rest of the lab's stack. It does not buy speed: on the
measurements in §6, expect 3–5× slower per environment than MuJoCo on CPU. The
binding constraints are the compiled dense factorization (3–7× slower than the
same operation on the host) and its compile time (seconds per shape key, and
minutes at `nv` near 100). The parts that are genuinely hard are convex mesh
collision, tendon wrapping and the MJCF compiler — not the dynamics, and not the
solver.

---

## 1. The question

Can we have a physics engine that computes what MuJoCo computes, written in
OCaml on Nx and Rune, with no C library underneath?

Two things make this plausible now and did not exist when MuJoCo was written:

- Raven has the transformations a physics engine needs — reverse and forward
  mode, `vmap`, `jit`, `scan`, `iterate`, `root` — and they compose with Nx's
  tensor operations, `scatter` with `Add`, `reduce_segments`, `einsum`, dense
  linear algebra.
- MJX is a working proof that MuJoCo's algorithms survive the move to a tensor
  language: DeepMind reimplemented the same pipeline in JAX, kept the soft
  constraint formulation, and shipped it.

What follows is what the port has to contain, which parts Nx/Rune support
directly, which parts they do not, what it costs to run, and an order of work.

---

## 2. What MuJoCo is

### 2.1 The pipeline

`mj_step` is four position/velocity stages and an integrator. Each stage is a
handful of functions over `mjModel` (constants) and `mjData` (state and
work arrays):

```
  mjModel: constants.   mjData: state (qpos, qvel, qacc, act, ctrl) and
  masses, geoms, trees   work arrays (xpos, xquat, M, efc_*, qfrc_*)

  position stage    mj_fwdKinematics     body poses xpos, xquat, geom frames
                    mj_makeM, factorM    body inertia, CRB, dense or sparse L'DL
                    mj_collision         broadphase, narrowphase, contacts
                    mj_makeConstraint    equality, limit, friction, contact rows
                    mj_island            constraint islands (3.x)
                    mj_projectConstraint impedance scaling of the rows
                    mj_transmission      actuator and tendon transmission

  velocity stage    mj_fwdVelocity       com, RNE bias forces c, passive forces
                    mj_fwdActuation      references, resistance, actuators

  acceleration      mj_fwdAcceleration   qacc_smooth = M⁻¹ f
                    mj_fwdConstraint     solve for λ; qacc = M⁻¹ (f + Jᵀ λ)

  integration       mj_Euler, mj_RungeKutta, mj_implicit, mj_discrete
```

The interesting properties, and they are what a port must reproduce:

- **Soft constraints.** Every constraint carries `solref` and `solimp`, which
  turn the complementarity problem into a smooth-ish nonlinear one where
  constraint force is a function of violation and velocity. This is why
  MuJoCo steps stably at 1–5 ms timesteps and why it is differentiable in
  practice (§7).
- **Constraints are rows.** All of equality, limits, friction loss and contact
  become rows of one `efc_*` set, solved together by one of three solvers.
- **The mass matrix is factored once per step** and reused by the solver, the
  acceleration solve and the implicit integrator.

### 2.2 Feature surface

MuJoCo 3.17, as installed here:

| Area | What exists | Notes for a port |
| --- | --- | --- |
| Joints | hinge, slide, ball, free | all four are cheap tensor ops |
| Transmissions | joint, jointinparent, site, tendon, slider-crank, body, so3 | joint/site/tendon are the ones in use |
| Actuators | 12 types (motor, position, velocity, intvelocity, damper, cylinder, muscle, adhesion, pid, orientation, dcmotor, general) | general gain/bias polynomials cover most |
| Sensors | 49 types | mostly projections of state; subtree ones need segment reductions |
| Geoms | plane, hfield, sphere, capsule, ellipsoid, cylinder, box, mesh, SDF, flex | primitives are analytic; mesh/SDF are the hard part |
| Constraint types | 8 (equality, limit joint/tendon, friction dof/tendon, contact pyramidal/elliptic/frictionless) | one row builder each |
| Solvers | PGS, CG, Newton | PGS is sequential; CG and Newton vectorize |
| Cones | pyramidal, elliptic | elliptic needs a QCQP projection |
| Integrators | Euler, RK4, implicit, implicitfast, discrete | RK4 = 4 forward evaluations |
| Defaults | solver Newton, iterations 100, ls_iterations 50, integrator Euler, cone pyramidal, jacobian auto (dense below `nv = 60`), `solref = (0.02, 1)`, `solimp = (0.9, 0.95, 0.001, 0.5, 2)` | these are the semantics to match |
| Out of scope | flex, skin, composites, plugins, rendering, UI, inverse/derivative APIs, IPC, CCD, sleep, SDF geoms | each is its own project |

### 2.3 Size of the thing being replaced

Lines of C in `src/engine` (the physics) and `src/xml` (the MJCF compiler):

| File | Lines | Subsystem |
| --- | --- | --- |
| `engine_core_smooth.c` | 2 624 | kinematics, com, transmission, tendon, CRB, factorM, RNE |
| `engine_core_constraint.c` | 3 437 | constraint rows, Jacobians, solref/solimp, makeConstraint |
| `engine_solver.c` | 3 227 | PGS, CG, Newton, cone projections |
| `engine_forward.c` | 2 446 | stage drivers, integrators, `mj_step` |
| `engine_collision_driver.c` | 2 667 | broadphase, pair filtering, contact output |
| `engine_collision_convex.c` | 1 916 | convex-convex via GJK/EPA, primitive dispatch |
| `engine_collision_gjk.c` | 2 749 | GJK/EPA implementation |
| `engine_collision_primitive.c` | 604 | 8 analytic primitive pairs |
| `engine_collision_box.c` | 1 069 | box-specific contact generation |
| `engine_passive.c` | 1 430 | springs, dampers, gravity, fluid |
| `engine_sensor.c` | 1 870 | 49 sensor types |
| `engine_util_solve.c` | 2 465 | Cholesky, QR, LCP, QCQP, projections |
| `engine_util_sparse.c` | 1 598 | sparse L'DL factor and sparse products |
| `engine_core_util.c` | 2 684 | object indexing, Jacobian assembly, utilities |
| `engine_setconst.c` | 1 945 | reference configuration, `mj_setConst` |
| `engine_util_misc.c` | 2 395 | quaternions, rotations, misc math |
| **`src/engine` total** | **73 029** | of which about 35k is the subset above |
| `src/xml` | 11 811 | MJCF/URDF reader and compiler |
| `src/user` | 40 877 | high-level API, callbacks, scenes — not ported |

A tensor implementation of the same subset is smaller than the C, because the
scalar loops over dofs, bodies and constraint rows collapse into array
operations. The estimate in §9 is 6–9k lines of library OCaml plus a test suite
of comparable size. MJCF compilation is not smaller: it is a parser, a
defaults/class inheritance resolver, a model builder and a lot of validation,
and it carries the most tedium per unit of risk in the whole project.

---

## 3. What Raven gives us

### 3.1 Transformations

Everything in `Rune` that a physics engine needs exists, and the semantics are
the ones physics wants:

| Need | Rune |
| --- | --- |
| per-body / per-iteration loops | `scan` (compiled as a loop under `jit`), `iterate` with per-lane stopping |
| branch on a value | OCaml `if` on `Nx.item` eager; `Nx.where` under `jit`/`vmap` |
| implicit differentiation of the solver | `root ~residual` (implicit function theorem, `linear_solve` hook), `custom_vjp` |
| gradient through the whole step | `grad`, `value_and_grad`, `vjp`, `jvp`, `remat` |
| batched environments | `vmap` over axis 0 of a `Data` structure; `lanes`/`lane_index` |
| domain randomization | `Nx.Rng` with splittable keys, `fold_in`, `split_batch` |
| model parameters as one structure | `Nx.Ptree` records; captures of `jit` are bound once |
| assertions with data | `Nx.check` (raises with index and values, even compiled) |

The `Ptree` design matters here: a `Data` record of `qpos`, `qvel`, `qacc`,
`act`, `ctrl`, `time`, warmstart `λ` is a structure that `vmap`, `jit`, `grad`
and `encode`/`decode` all walk without glue code. Gradients with respect to
`Model` parameters come for free by putting the parameters in a structure and
closing over them.

### 3.2 Array operations

The operations the algorithms need, checked against `Nx`'s interface:

- **assembly and accumulation**: `scatter ~combine:\`Add` (writes into a
  tensor at tensor indices, accumulating), `reduce_segments`/`reduce_ranges`
  (per-body reductions along ids), `set`, `slice`, `take`, `one_hot`.
- **linear algebra**: `cholesky`, `solve_triangular`, `solve` (LU),
  `einsum`, `tensordot`, `matmul`, `pinv`/`lstsq` if a solver wants them.
- **geometry**: `cross`, `dot`, `norm`, `arange`, `meshgrid`, `stack`,
  `concatenate`, `maximum`, `sign`, `where`, `clamp`.
- **sorting and searching** for broadphase and for `argsort` of constraint
  rows: `sort`, `argsort`, `top_k`, `searchsorted`, `unique`.
- **float64** everywhere, natively on the host and in compiled programs.

### 3.3 The measured cost model

This is the part that decides the design, so it was measured rather than
assumed. All numbers: float64, one process, jitted with `Rune.jit'`, warmed.

| Operation | Eager | Jit, 1 env | Jit, `vmap` 256, per env |
| --- | --- | --- | --- |
| trivial elementwise on 36 elements | — | 2–3 µs | 0.09 µs |
| 1000 elementwise ops written out, 100 elements | — | 2.7 µs total | — |
| `scan`, 1000 iterations, one elementwise op | — | 184 µs (170 ns/iteration) | — |
| 27×27 matvec | — | 3.8 µs | 0.07–0.33 µs |
| 27×27 triangular solve | — | 5.0 µs | 0.9–1.2 µs |
| 27×27 Cholesky | 1.5–1.9 µs | 5.1 µs | — |
| 39×39 Cholesky | 4.4 µs | 9.4 µs | — |
| 60×60 Cholesky | 14.8 µs | 46.5 µs | — |
| 27×27 LU solve | 4.6–7.8 µs | 24–36 µs | 11–14 µs |
| 60×60 LU solve | 20–23 µs | 132–140 µs | 102–145 µs |
| 60×60 Cholesky + 3 triangular solves | — | 62.9 µs | 15.9 µs |
| 39×39 Cholesky + 3 triangular solves | — | 22.3 µs | 8.3 µs |
| 27×27 Cholesky + 2 triangular solves | — | 11.1 µs | 3.0 µs |
| three 60×60 LU solves in a `scan` | — | 164 µs | 114 µs |
| 100 diagonal sweeps, 39 constraints | — | 26.4 µs | 5.4 µs |
| 27 matvecs of `[27;27]` in a `scan` | — | 22.9 µs | 16.8 µs |
| FK-shaped `scan`, 6 bodies: gather, matmul, scatter | — | 6.4 µs | 0.69 µs |
| first call of a jitted `cholesky`, 20×20 | — | 0.63 s | — |
| first call of a jitted `cholesky`, 40×40 | — | 5.6 s | — |
| first call of a jitted `cholesky`, 50×50 | — | 12.9 s | — |
| first call of a jitted `cholesky`, 80×80 | — | 111 s | — |

Four measurements shape the design.

**Finding 1: an elementwise chain is free, a loop iteration is not.** 1000
`add_s` operations written out cost 2.7 µs in total (about 2.7 ns each: they
fuse into one kernel). The same 1000 operations inside a `scan` cost 184 µs
(184 ns per iteration), and the per-iteration cost is flat in the tensor size:
166 ns for a 1-element tensor, 205 ns for a 1000-element tensor. A compiled
`scan` therefore costs about 170 ns per iteration for dispatch, whatever the
body does.

**Finding 2: compiled factorization is the expensive primitive, and its
compile time is worse than its run time.** `Nx` keeps hand-written C kernels
for the host path, and they are fast: 1.5 µs for a 27×27 Cholesky, 14.8 µs for
60×60, 20 µs for a 60×60 LU solve. `Rune.jit` cannot call them: it lowers the
factorization into tensor operations, unrolled over the columns at trace time.
The compiled result is 3× slower for Cholesky and 6–7× slower for the LU-based
`solve`, and the unrolled graph makes the *first* call scale badly with the
matrix size: 0.63 s at 20, 5.6 s at 40, 12.9 s at 50, 111 s at 80, for one
factorization in an otherwise empty program. A physics engine factors once or
twice per step at sizes between 27 and 60, so it pays seconds of compilation
per shape key, and models near `nv = 100` are out of reach. Two consequences
for the design: use `cholesky` + `solve_triangular` rather than `solve` (27×27:
5.1 + 5.0 µs against 25–36 µs; 60×60 with three right-hand sides: 63 µs against
three LU solves at 164–553 µs), and count the shape keys a model needs.

**Finding 3: batching pays for the pieces and less for the whole.** Matvec
batches 12–50× (3.8 µs → 0.07–0.33 µs), triangular solve 4–5×, and Cholesky
followed by two or three triangular solves 3–4×: 27×27 from 11.1 to 3.0 µs,
39×39 from 22.3 to 8.3, 60×60 from 62.9 to 15.9. The LU-based `solve` does not:
27×27 is 24–36 µs alone and 11–14 µs per env, 60×60 is 132–140 µs alone and
102–145 µs per env. A whole step-shaped program batches by only 1.5× (83 µs
alone, 54 µs per env at 256) even though its pieces batch by 3–4×: the fused
batched program does not keep the per-operation gains, and the batched stages
sum to 34 µs, below what the composed program measures. Whoever profiles the
composition will find more there than in any individual op.

**Finding 4: a capture is not a constant, except when it is.** Three 60×60
solves in a `scan` take 166 µs whether the matrix is captured or passed as an
argument, so nothing is folded at trace time. Under `vmap`, though, a captured
matrix shares one factorization across the lanes: 17 µs per env against 127 µs
when the matrix is an argument. The physics matrices are state-dependent, so a
benchmark or a design that captures them reports a number that the engine
cannot reach; it is the difference between a 7× optimistic and a correct
estimate of the same program (appendix A).

Compilation cost, for the same programs: 20–60 ms for a micro-benchmark,
35–260 ms for a program with a 27-body scan, a 100-iteration loop and four
solves. A full engine step is larger than any of these, so expect seconds per
(model, contact budget, batch shape) key, once, cached in `~/.cache/dune`.

### 3.4 Limits that shape the design

| Limit | Consequence |
| --- | --- |
| No sparse tensors. `Nx_ragged` is for variable-length 1-D segments, not matrices. | Dense `M`, dense `efc_J`, dense `A = J M⁻¹ Jᵀ`. Fine to `nv ≈ 100`–`200`; MuJoCo switches to sparse at `nv ≥ 60`. |
| Shapes are fixed at trace time; `jit` recompiles per shape key. | Contact and constraint counts must be compile-time budgets (`nconmax`, `nefcmax`), with masks, as in MJX. A model whose contact count exceeds the budget silently loses contacts unless checked. |
| A branch on a traced value raises under `jit`. | Every discontinuity becomes `Nx.where`, or a mask on a fixed-size buffer. Collision detection cannot early-exit. |
| `scan` iteration ≈ 170 ns, and MuJoCo's defaults would put ~130 iterations in a step. | Target solver iteration counts in the single digits (Newton converges in 1–3), not the C defaults. |
| Compiled factorization is 3–7× slower than the host kernels, and its first call costs seconds at the sizes a model needs | `Cholesky` of a 39×39 system takes 9.4 µs compiled against 4.4 µs eager, and compiles in ~2–6 s; at 80×80 the first call takes 111 s. Each distinct matrix size is its own compiled program. |
| float64 works on the host; GPU float64 is slow on consumer cards and absent on Metal. | float64 default on CPU; float32 opt-in for GPU training, with the stability caveat MJX documents. |
| GPU backends exist in Raven (CUDA, Metal, AMD) but were not exercised here. | The GPU story is unverified; the plan treats it as a milestone, not an assumption. |

---

## 4. Feasibility, subsystem by subsystem

| Subsystem | Verdict | Difficulty | Why |
| --- | --- | --- | --- |
| Model/state containers | direct | low | `Ptree` records of tensors plus host-side topology arrays |
| Forward kinematics | direct | low | one `scan` over bodies in topological order; gather parent row, `matmul`, scatter own row. Measured: 6.4 µs for 6 bodies, 0.69 µs/env at 256 |
| Geom frames, com, mass | direct | low | elementwise over `[ngeom]`, `[nbody]` |
| Composite rigid body / `M` | direct | medium | reverse scan with `scatter ~combine:\`Add` into parents, then `einsum` to assemble `[nv;nv]` |
| RNE bias forces | direct | medium | reverse scan for velocities, forward scan for forces; same shape as FK |
| Passive forces, gravity, fluid | direct | low | elementwise |
| Transmission (joint/site/tendon) | direct | low–medium | `matmul` with sparse-pattern moment arms, built once |
| Actuator dynamics, muscles | direct | low | small ODE per actuator, one `scan` over time or explicit formula |
| Sensors | direct | low | most of the 49 are projections of state; subtree ones are `reduce_segments` |
| Constraint rows and Jacobians | direct | medium | fixed-size row blocks, `scatter` into `efc_J`, `solref`/`solimp` formulas |
| Dense factorization of `M` and of the reduced system | direct, but slow to compile | medium | `cholesky` + `solve_triangular`; 5.1 µs at 27×27, 9.4 µs at 39×39, 3–4× batching gain; first call 0.6 s at 20×20, 5.6 s at 40×40 |
| Newton / CG solver | direct | medium | `iterate` with a budget; Newton pays one Cholesky per iteration, CG avoids them with matrix-free matvecs |
| PGS solver | possible | medium | sequential; only as a `scan` over constraint blocks, ~170 ns per sweep is too slow at 100 sweeps |
| Integrators | direct | low–medium | Euler/RK4 trivial; implicit needs the `M + dt D` factorization |
| Collision: broadphase | possible | medium | static candidate pair list built at compile time, per-step sphere-bound mask; MuJoCo's BVH is not reproducible without dynamic loops |
| Collision: primitives | direct | medium | 8 analytic pairs; `Nx` has the vector algebra they need |
| Collision: convex mesh | hard | high | GJK/EPA is iterative and data-dependent; MJX uses a branchless SAT and documents the cost blowup |
| Collision: hfield, SDF | hard | high | defer |
| Tendons: fixed | direct | low | mass-matrix contributions and moment arms |
| Tendons: spatial with wrapping | hard | medium–high | path-dependent wrapping around cylinders/spheres; the branchiness is the problem |
| Flex, skin, composites | out of scope | — | separate project |
| MJCF compilation | possible | high (tedious) | 11.8k lines of C: parser, defaults inheritance, model build, validation |
| Differentiability of the step | direct | medium | AD through the solver, or `root` on the solver residual; §7 |

Four of these deserve a paragraph.

**Convex collision is the main risk.** MuJoCo's `engine_collision_gjk.c` is 2 749
lines of GJK and EPA with iterative support functions, termination tests and
polytope bookkeeping. A branchless version that works at a fixed iteration
count is possible — MJX's SAT is exactly that — but the documentation is
explicit that it wants fewer than ~32 vertices per convex mesh and that it
degrades badly with more, and that large-mesh collision is why MJX-Warp exists.
A port should support primitives and planes first, boxes and meshes later, and
should treat "the standard humanoid/ant/hopper benchmark suite" as the target
rather than arbitrary scenes.

**MJCF is the tedium, and it is skippable at first.** MJX does not parse MJCF:
it receives an already-compiled `mjModel` from the C library and copies its
arrays onto the device. A native port has no such luxury. Two paths:

1. *Native reader.* XML parser, defaults/class inheritance, include handling,
   asset loading, model build, `mj_setConst`. 3–4k lines, several weeks, and a
   long tail of validation quirks.
2. *Converter.* A dev-time tool (outside the library, not linked at runtime)
   loads MJCF with MuJoCo's Python package and writes the compiled model —
   masses, inertias, joint ranges, geom parameters, pair lists, transmission
   moment arms — into a compact binary. The OCaml side reads that. This makes
   the runtime fully native while borrowing the compiler's semantics, which is
   exactly what MJX does, and it removes `mj_setConst` from the critical path.

Recommendation: build (2) first, define the interchange format as the public
model format, and implement (1) later for models the converter cannot reach.
The exception: if "users write MJCF and run it with no MuJoCo installed" is a
requirement from day one, the order reverses and the schedule grows by a month.

**The factorizations set the speed and the build time.** The compiled dense
factorization (§3.3) is the single number that decides what this engine can be.
A 39×39 Cholesky is 9.4 µs compiled against 4.4 µs eager, and its first call
costs seconds; at 80×80 the first call takes 111 s. A step needs one
factorization of `M`, one per Newton iteration (the engine should aim at three),
and one more for the implicit integrator, so factorization is both the largest
run-time term and a per-model compilation cost. The algorithmic answers are
design choices rather than research questions: use `cholesky` and triangular
solves rather than `solve`, factor `M` once and reuse it, prefer CG on the
reduced system when its iteration count is small, and keep the number of
distinct matrix sizes in a model small. The engineering answer — a fused
batched Cholesky in `tolk`, or a way for a compiled program to call `nx.cpu`'s
C kernels — is worth 3× on the run time and most of the compile time. This is
the one place where the project depends on a Raven capability that does not
exist yet, and `doc/mujoco-raven-bottlenecks.md` is the request. M0 settles it.

**The solver carries the semantics.** Everything else is bookkeeping; the
solver is where MuJoCo's behaviour lives: friction cones, impedance,
warmstart, and the way a slip becomes a stick. It is also where this port can
differ from the C engine at the same step count, because `Rune.root` can
differentiate the solver's own fixed point instead of unrolling it (§7). It
should be built early and tested against a model whose answer is known, not at
the end.

---

## 5. Prior art: MJX

MJX-JAX, in the MuJoCo repository, is the same project in JAX. Its documented
choices are a checklist for us:

| MJX choice | Reason | Our equivalent |
| --- | --- | --- |
| Fixed `naconmax` (contacts, all worlds) and `njmax` (constraints per world) | XLA needs static shapes | fixed `nconmax`, `nefcmax` in the compiled step |
| Dense Jacobians only | sparse is not faster on vector hardware for these sizes | dense `efc_J` |
| CG and Newton only, no PGS | PGS is sequential | same, plus `iterate` |
| Branchless SAT for convex collision | GJK's branching kills throughput | later milestone |
| float32 default | GPU double precision is slow | float64 default on CPU, float32 opt-in |
| `max_contact_points`, `max_geom_pairs` | bounding the branchless broadphase | candidate pair list per model |
| Explicit `<pair>` lists recommended | fewer candidates is the biggest win | same, and we can precompute pairs offline |
| Single scene: **10×** slower than MuJoCo | dispatch, small-op and factorization overhead | measured 4–5× before collision detection (§6) |
| Batched, one A100, 8192 worlds: 950k steps/s | vector hardware | unverified for us; ~8.6 µs per env |

MJX-Warp (the CUDA-graph, NVIDIA-specific implementation) reaches 3.35M
steps/s on the humanoid, but does not support automatic differentiation. That
is the shape of the trade: the branchless, monolithic, GPU-native version is
fast and not differentiable; the vectorized, functional version is
differentiable and slower.

The conclusion to draw is not "MJX did it, so it is easy". It is: MuJoCo's own
authors judged the tensor rewrite worth doing, took the "borrow the compiler,
reimplement the engine" shortcut, and had to change the algorithms in three
places (broadphase, convex narrowphase, solver) to fit static shapes.

---

## 6. Performance estimate

Baseline, MuJoCo 3.17 C on this machine, `mj_step`:

| Model | Bodies | `nv` | Geoms | Contacts | Step |
| --- | --- | --- | --- | --- | --- |
| `car.xml` | 4 | 8 | 5 | 5 | 5.4 µs |
| `humanoid.xml` | 17 | 27 | 20 | 13 | 18.7 µs (Newton), 27.1 µs (PGS), 20.7 µs (CG) |
| `balloons.xml` | 6 | 30 | 11 | 6 | 38.9 µs |
| humanoid, 64 copies stepped in a row | — | — | — | — | 18.4 µs each |

MuJoCo's cost is flat in the number of copies: it is already at one step per
18 µs and stays there.

Raven, on a synthetic program shaped like a step, with every state-dependent
matrix passed as an argument and the factorization done with `cholesky` +
`solve_triangular`: a 27×27 mass solve, 27 body matvecs in a `scan`, 100
diagonal constraint sweeps in a `scan`, and a 39×39 Cholesky followed by three
triangular solves in a `scan` (the shape of a three-iteration Newton solver).

| | Single env | Per env at `vmap` 64 | 256 | 1024 |
| --- | --- | --- | --- | --- |
| whole program | 83 µs | 52 µs | 54 µs | 63 µs |
| mass Cholesky + 2 triangular solves, 27×27 | 11.1 µs | — | 3.0 µs | — |
| 27 body matvecs | 22.9 µs | — | 16.8 µs | — |
| 100 constraint sweeps | 26.4 µs | — | 5.4 µs | — |
| 39×39 Cholesky + 3 triangular solves | 22.3 µs | — | 8.3 µs | — |
| compile + first call | 3.3 s | 18 s | 17 s | 17 s |

The stages sum to 83 µs alone and 34 µs per environment batched, while the
composed program measures 83 µs and 54 µs. So the composition is free at one
environment and costs 60% at 256: the batched program does not keep the
per-operation batching gains. That, not any single operation, is the first
thing a profiler would look at.

What this means, with the uncertainty stated:

- **Single environment: 4–5× slower than MuJoCo** (83 µs against 18.7 µs)
  before collision detection, constraint assembly and actuators are added.
  Those stages are elementwise and scan-shaped, so they will batch well but
  cost something on top. Same order as MJX's documented 10×, better than the
  first version of this estimate because the factorization is done with
  `cholesky` and because the input tensors are not rebuilt per call.
- **Batched on CPU: about 3× slower per environment** (54 µs against 18.7 µs).
  Batching buys 1.5×, not the 3–4× that the individual stages buy.
- **Compile time is the practical obstacle, not throughput.** The step
  compiles in 3.3 s for one environment and 17 s per batched shape, and the
  Cholesky inside it is what dominates: one factorization in an otherwise empty
  program takes 0.63 s at 20, 5.6 s at 40, 12.9 s at 50 and 111 s at 80. A
  model with `nv = 60` and a 39-row constraint set therefore costs tens of
  seconds of compilation per shape key, and a model near `nv = 100` is out of
  reach. This is a Raven-side property, not a property of the port, and §13
  records it as a question for that project.
- **GPU: unmeasured here.** MJX's own numbers put an A100 at 8192 worlds at
  8.6 µs per environment, twice MuJoCo's per-env cost on a 64-core machine, and
  a 64-core machine running MuJoCo, stepping one world per thread, reaches
  1.8M steps/s against the A100's 950k. The reason to go to a GPU is not
  throughput per step but throughput per machine, memory, and the absence of a
  thread pool; float64 on consumer cards will not deliver it.

Four levers matter, in this order:

1. **Never use `solve` where `cholesky` will do.** The LU-based path is 3–7×
   slower compiled and does not batch: three 60×60 LU solves cost 164 µs alone
   and 114 µs per environment, where one Cholesky and three triangular solves
   cost 63 µs and 16 µs. Every symmetric positive-definite system in the engine
   (`M`, `M + dt D`, the Newton Hessian) is in this category.
2. **Keep the factorization count and the matrix sizes down.** Each distinct
   size is a separate compiled program with its own seconds of compilation, and
   the compile time grows steeply with the size. Two factorizations at 27 and
   39 are affordable; a per-island factorization at a dozen sizes is not.
3. **Fewer kernels per step, and scans for the repeated ones.** The difference
   between 2.7 ns and 170 ns per operation is the difference between a fused
   chain and a loop boundary, and repeated small linear algebra is measurably
   cheaper in a `scan` than written out (three 60×60 LU solves: 164 µs in a
   `scan`, 553 µs written out). Treat the step as a graph of large fused
   regions rather than a line-by-line port of `mj_fwdAcceleration`.
4. **Profile the composed batched program, not its stages.** The stage
   measurements overstate what the batched engine gets, by about 60% here. The
   benchmark harness should time the composed step and report the gap.

---

## 7. Differentiability

This is the reason to do the project, and the part that has no equivalent in
the C engine. MuJoCo offers `mjd_transitionFD` (finite differences of the
step) and analytic derivatives of a few quantities; it has no reverse-mode
transformation.

What happens under `Rune.grad` on `step`:

- **Kinematics, mass matrix, RNE, passive forces, actuation, integration** are
  smooth compositions of arithmetic. AD is exact, at machine precision, for
  any parameter in the structure — including masses, inertias, geom sizes,
  gains and `solref`/`solimp`. This alone makes system identification and
  policy optimization in the same language with the same model possible.
- **Constraint rows** are smooth in the state except at the point and normal
  of contact, which come from collision detection; the `solref`/`solimp`
  force law is a smooth function of penetration once the contact exists.
- **The solver** is where the discontinuities are. Its output is a projection
  onto a convex set: the map is piecewise smooth, and its derivative jumps at
  every stick/slip and contact-on/off transition. There are three ways to get
  a gradient, and they differ in cost and in accuracy:

| Approach | Cost | Accuracy |
| --- | --- | --- |
| Unroll the solver, `Rune.grad` through it | `iterations` × the reverse pass, and memory for the replay | exact for the unrolled program, which is a perturbation of the true solver |
| `Rune.root` on the solver residual: λ such that `F(λ) = 0` | one linear solve with the reduced Hessian at the solution | derivative of the *converged* solver, by the implicit function theorem |
| `custom_vjp` around the solver, with the KKT system solved by hand | one solve, no unrolling | same as `root`, cheaper to shape by hand |

`dilqr` already does the third thing for iLQR (`implicit_grad.ml`): the solve
is idealized as the argmin of a QP, the KKT system is solved with the Riccati
machinery, and the gradient is assembled from the solution perturbation. The
same pattern applies here with the constraint solver's KKT system, and it is
the recommended route: it decouples gradient cost from iteration count, it
gives the derivative of the answer rather than of the last iterate, and it
reuses the factorization of `M` that the step already has.

What differentiability does *not* buy: correct gradients across contact mode
changes. Where a contact appears, disappears or starts slipping, the true map
is discontinuous and every method above returns one side's derivative. This is
a property of rigid contact, not of the AD, and it is why MuJoCo's soft
constraints matter: they make the map continuous, so the gradient is large
but finite. The tests should check gradients against finite differences of
the whole step at points away from transitions, and assert boundedness at
transitions.

There is a second-order consequence worth planning for: `grad` of `step` is
what an iLQR/MPC loop consumes, and `dilqr` wants `(x, u) → f` as a
differentiable dynamics function with a `M`-weighted inner product. The
port should expose exactly that: `Model.step : Data.t -> Data.t` and
`Model.dynamics : Data.t -> Data.t -> Data.t`, both differentiable, batched
and jittable, with a `custom_vjp` on the solver stage.

---

## 8. Scope for a first version

In:

- Joints: hinge, slide, ball, free. Joint limits and joint armature.
- Geoms: sphere, capsule, box, cylinder, plane; contact pairs between them.
  Ellipsoid if the primitive math is ported with it.
- Contact model: `condim` 1/3/4/6, pyramidal cones, `solref`, `solimp`,
  `margin`, `gap`, warmstart, friction loss, `impratio`.
- Constraints: joint limits, equality (connect, weld, joint, tendon),
  friction loss.
- Actuators: motor, position, velocity, intvelocity, damper, general
  gain/bias; joint, site and tendon transmission.
- Sensors: the state-projection family (jointpos/vel, framepos/quat/vel,
  gyro, accelerometer, touch, contact distance/force, actuator/tendon
  force).
- Tendons: fixed only.
- Integrators: Euler, RK4, implicitfast.
- Solvers: Newton and CG, with a fixed iteration budget.
- Dtype: float64 on the host; float32 supported and tested, not default.
- Differentiability: `grad`/`vjp` through the step, implicit solver gradient.

Out, with a note on when they could come:

| Out | Why | Later? |
| --- | --- | --- |
| Convex mesh, hfield, SDF geoms | iterative narrowphase, order-of-magnitude cost | M8 |
| Spatial tendons with wrapping | path-dependent branching | M9 |
| Elliptic cone QCQP projection (if not in v1) | an extra solver inside the solver | M4 |
| Muscle and adhesion actuators | ODE + length-velocity curve, niche | M5 |
| Flex, skin, composites, plugins | separate engines | no |
| Continuous collision, sleep, islands, IPC | correctness/performance features of the C engine | islands: maybe; the rest: no |
| MJCF reader (v1 uses the converter) | tedium, not risk | M10 (conditional) |
| Rendering, UI, viewer | a separate library, not a stage of the engine | V0–V5, see [`VIZ.md`](VIZ.md) |
| GPU | unverified in this stack | M11 |

---

## 9. Plan

The milestone set lives in [`MILESTONES.md`](MILESTONES.md): eleven engine
milestones with acceptance tests, gates and stop-losses, plus the viewer
thread in [`VIZ.md`](VIZ.md). This section keeps only the mapping from the
milestone ids used above to the ones in that document.

| Design-note milestone | Now |
| --- | --- |
| M0 skeleton, fixtures, linear-algebra spike | M0 |
| M1 kinematics, mass, bias, integration | M1 (kinematics) and M2 (inertia, bias, integration) |
| M2 constraints without contacts, solvers | M3 |
| M3 contact | M4 |
| M4 actuators, transmissions, sensors, tendons | M5 |
| M5 differentiability | M6 |
| M6 native MJCF reader | M10 (conditional) |
| M7 the hard geometry | M8 (convex, hfield) and M9 (spatial tendons) |
| M8 performance and the GPU | M7 (consolidation) and M11 (GPU) |

The numbering in the rest of this note has been updated to the new ids. The one
structural change is that the old M1 is split: kinematics (M1) is where
conventions are pinned against fixtures, and inertia, bias and integration (M2)
is where the first dynamics exists. The viewer is not in that list because it
is not a phase of the engine: it starts at M0 with still frames and reaches a
live window at M2.

---

## 10. Package design

One package, `packages/mjc/`, with modules that can become libraries later if
a second user appears:

```
mjc/lib/
  mjc.ml                 library root
  model.ml/.mli          constants: topology arrays + parameter tensors
  data.ml/.mli           state: qpos, qvel, qacc, act, ctrl, time, warmstart
  mjcb.ml                interchange-format reader (and later MJCF)
  rigid.ml               quaternions, exp/log, frames (float32 and float64)
  kinematics.ml          fwdPosition
  inertia.ml             CRB, factorM, solveM
  bias.ml                RNE, passive, gravity
  transmission.ml        actuators, tendons, sensors
  constraint.ml          row assembly, solref/solimp, impedance
  solver.ml              Newton, CG, projections
  collision.ml           candidate pairs, primitive narrowphase
  integrate.ml           Euler, RK4, implicit
  step.ml                the pipeline, and the compiled closure
```

Decisions:

- **Compile the model into a closure.** `Step.compile : Model.t -> step` where
  `step : Data.t -> Data.t` closes over the model tensors; `Rune.jit` binds
  them once as captures. Model changes make a new closure; the compiled
  programs are keyed by the data shapes only.
- **`Data.t` is a record of tensors, not an array of work buffers.** Work
  buffers are local to the step. That keeps `vmap`, `jit`, `grad` and saving
  working without glue.
- **Contact and constraint budgets are compile-time.** `limits : { nconmax;
  nefcmax }` in the model; overflowing is an `Nx.check`, not a silent drop.
  The C engine's arena grows; ours does not, which is a semantic difference
  to document once and test.
- **float64 by default, float32 by functor or by a dtype parameter.** Physics
  needs the range; training on a GPU may want the speed. `geom`'s `Motor` is
  float32-only today, so either `geom` is generalized over the dtype or `mjc`
  carries its own rigid-body math. The second is less work and fewer
  dependencies for v1.
- **Naming.** `mjc` is the prefix MuJoCo itself uses for its C++-free API and
  for its collision functions; it reads as "MuJoCo-compatible" without
  claiming to be MuJoCo. The interchange format is `.mjcb`.
- **The factorization is a first-class decision.** `inertia.ml` exposes its
  factorization and back-substitution separately (`factorM`, `solveM`), and
  `solver.ml` takes the reduced system as an operator rather than as a matrix,
  so the planner can swap the dense `Nx.cholesky` call for a fused kernel, a
  cached factor, or a matrix-free CG without touching the solver logic. The
  strategy is a parameter of `compile`, alongside the constraint budget.

Dependencies: `base`, `stdio`, `nx`, `rune`, and `ppx_ptree` if the records
shrink nicely. No XML library in v1 (the converter is Python); an XML parser
would be a new dependency to ask about at M10.

---

## 11. Risks

Ranked by how much they can change the plan.

| # | Risk | Consequence | Mitigation |
| --- | --- | --- | --- |
| R1 | Compiled factorization stays at 9.4 µs (39×39) and compiles in seconds, minutes at 80×80 | the engine is 3–5× slower than MuJoCo and slow to build; large models are out | M0 spike with a compile-time and a run-time gate; `cholesky` only, one or two factorizations per step, few distinct sizes; raise the compile time with Raven |
| R2 | Dense linear algebra caps the model size at `nv ≈ 100`–`200` | humanoids with many joints and complex models are out | accept for v1; the dense path is MuJoCo's own default below `nv = 60` |
| R3 | Convex mesh collision does not reach usable performance | no arbitrary robot meshes; primitive-approximated models only | M8; treat as a research spike with an abandon criterion |
| R4 | Fixed contact budget changes semantics versus the C engine | long-horizon divergence on contact-rich scenes | budget by measurement per model; `Nx.check` on overflow; document the difference |
| R5 | Gradient quality at contact transitions is poor | learning through contact is brittle | implicit solver gradient, softened contacts, tests that bound rather than assert smoothness |
| R6 | Compilation time per shape key (seconds) hurts iteration | slow development, shape-specialized batch sizes | keep loops compiled rather than unrolled; reuse keys; the persistent cache helps across runs |
| R7 | float64 on GPU is slow or unavailable (Metal) | the throughput story is CPU-only in double | float32 training path, CPU float64 reference |
| R8 | MuJoCo keeps moving (3.17 added IPC, SDF geoms, sleep, islands) | the port chases a target | freeze the semantics at one version, record it in the doc, re-evaluate annually |
| R9 | The lab's real models use features outside the v1 subset | the engine is not used | pick the subset from the models actually in use, not from the manual |
| R10 | Effort is 2× the estimate | the project stalls half-built | M2 and M4 are useful on their own (analysis, MPC on contact-free models); stop-loss at the M0 spike and at the M4 review |

---

## 12. Alternatives, and why not

| Alternative | Why not |
| --- | --- |
| Bindings to `libmujoco` | excluded by the question; also no AD, no `vmap`, no composition with `dilqr` |
| Generate C from MJCF and call it (like MuJoCo's own mjz) | same as above; and the generated code cannot be differentiated |
| Reuse MJX through a Python bridge | a C++/Python boundary in the hot loop, and the model would live on the wrong side |
| Port MuJoCo Warp (CUDA graphs) | fast and not differentiable; NVIDIA-only |
| Write only a solver and reuse MuJoCo's model compilation | model compilation is the tedious part, not the risky one; and it keeps the C dependency |
| Analytic (Riccati/KKT) gradients only, no general AD | narrower than what `Rune.grad` gives for free |
| Brax-style custom engine with its own semantics | the whole point of MuJoCo is the semantics: models, defaults, `solref` |

---

## 13. Open questions

1. Which models does the lab actually need? The subset in §8 is chosen from
   the ML benchmark suite; a robot with meshes changes M4 and forces M8 early.
2. Is GPU a requirement in year one, or is batched CPU enough?
3. Does the plan need to read MJCF, or is a converted format acceptable?
   This is worth one month and should be decided before M10.
4. Does the port aim at long-horizon trajectory equivalence with the C engine,
   or at "same physics, different rounding"? The first requires matching
   MuJoCo's solver defaults exactly, the second only the algorithms.
5. Should `mjc` depend on `geom` for rotations, or carry its own rigid-body
   math (needed if `geom` stays float32)? Leaning to its own, for v1.
6. Does the lab want `mjc` to expose `mj_forward` stage by stage (needed for
   debugging against the C engine) or only `step`?
7. Can Raven compile a factorization faster? The measurements in §3.3 say the
   first call of a program holding one 39×39 `cholesky` takes ~2 s, and 111 s
   at 80×80, because the lowering unrolls the factorization at trace time; the
   run time is 3–7× the eager C kernels. Compiled access to the host kernels,
   or a loop instead of the unroll, would move both numbers. This is the one
   question whose answer belongs to another project, and `doc/mujoco-raven-
   bottlenecks.md` is the asking. *M0 update:* at pin `4a3bc831` the same
   first call takes 10 ms at every size up to 60×60, so the compile-time half
   of this question is answered; the steady-state half is not. See
   `doc/spike-la.md`.

---

## Appendix A — how the numbers were measured

Machine: AMD Ryzen 5 7640U, 6 cores / 12 threads, 32 GB. `mujoco` 3.17.0 from
PyPI (`/tmp/mjenv`), `mj_step` timed over 20 000 steps after 500 warmup steps,
model XML from the MuJoCo repository. Raven at pin `3984341`, OCaml 5.5.2,
float64, `Rune.jit'`, single process, no `beam` search.

The microbenchmarks that matter. The first is shown in full; the others are
described, because their value is in their shape, not their code.

```ocaml
(* 1. dispatch: a scan iteration costs ~170 ns, an unrolled op ~2.7 ns *)
let scan_cost n =
  let xs = Nx.cast Nx.float64 (Nx.arange Nx.int32 0 1000 1) in
  let f q =
    Rune.scan' ~init:q
      ~f:(fun q _ -> (Nx.add_s q 1.0, Nx.scalar Nx.float64 0.0))
      xs
    |> fst
  in
  Rune.jit' f (Nx.ones Nx.float64 [| n |])
```

2. *Factorizations and scans.* `cholesky`, `solve_triangular`, `solve` and
   `matmul` at 8, 27, 39 and 60, eager and jitted, alone and under `vmap` at
   16, 64, 256 and 1024 lanes; and the same 60×60 solve three times, written
   out and in a `scan`, with the matrix captured and with it as an argument.
3. *FK-shaped scan.* Six bodies in topological order; the carry holds `[nb;3]`
   positions and `[nb;3;3]` rotations; each step gathers the parent row with
   `Nx.take ~axis:0`, multiplies by a rotation built from the joint angle, and
   writes its own row with `Nx.scatter ~axis:0`. Also timed under `vmap` over
   256 stacked joint vectors.
4. *Step-shaped program.* All matrices passed as arguments: a 27×27 Cholesky
   mass solve, 27 body matvecs in a `scan`, 100 diagonal constraint sweeps in
   a `scan`, and a 39×39 Cholesky followed by three triangular solves in a
   `scan`, timed single and under `vmap` at 64, 256 and 1024 lanes. Each stage
   was also timed alone with its inputs built once.

| Measurement | Value |
| --- | --- |
| scan, 1000 iterations, `add_s` on `[1]` | 166 µs → 166 ns/iteration |
| scan, 1000 iterations, `add_s` on `[1000]` | 205 µs → 205 ns/iteration |
| 1000 unrolled `add_s` on `[100]` | 2.7 µs → 2.7 ns/op |
| Cholesky 27×27 | eager 1.5 µs; jit 5.1 µs; `vmap` 256 not separable from noise |
| Cholesky 39×39 | eager 4.4 µs; jit 9.4 µs |
| Cholesky 60×60 | eager 14.8 µs; jit 46.5 µs |
| LU solve 27×27 | eager 4.6–7.8 µs; jit 24–36 µs; `vmap` 256 11–14 µs/env |
| LU solve 60×60 | eager 20–23 µs; jit 132–140 µs; `vmap` 256 102–145 µs/env |
| Cholesky + 3 triangular solves, 39×39 | jit 22.3 µs; `vmap` 256 8.3 µs/env |
| Cholesky + 3 triangular solves, 60×60 | jit 62.9 µs; `vmap` 256 15.9 µs/env |
| three LU solves 60×60, `scan` | jit 164 µs; `vmap` 256 114 µs/env |
| three LU solves 60×60, written out | jit 553 µs; `vmap` 256 308 µs/env |
| first call, jitted `cholesky` 20 / 30 / 40 / 50 / 80 | 0.63 s / 2.1 s / 5.6 s / 12.9 s / 111 s |
| whole step-shaped program | 83 µs single; 52–63 µs/env at 64/256/1024; 3.3 s / 17 s to compile |
| FK-shaped scan, 6 bodies, gather + matmul + scatter | 6.4 µs; 0.69 µs/env at `vmap` 256 |
| MuJoCo humanoid, `mj_step` | 18.7 µs (Newton), 27.1 µs (PGS), 20.7 µs (CG) |
| MuJoCo humanoid, 64 worlds in a row | 18.4 µs per step per world |

Caveats, stated plainly: the step-shaped program is not a MuJoCo step. It has
no collision detection, no constraint assembly and no actuator or sensor stage,
so its 83 µs per environment is a floor, not an estimate of the finished
engine. Everything here is one laptop with six cores, and the batched figures
move by up to a factor of two between runs; the eager-versus-compiled ratios,
the compile-time growth and the `scan` floor were stable across runs, the
per-lane batched figures less so.

Three measurement mistakes are worth recording because each one changed a
conclusion.

1. *Timing the input copy.* Pass `vmap` a tensor built once, outside the timed
   call. A `broadcast_to` + `copy` inside the closure added up to 7.4 MB of
   materialisation per call at `vmap` 256, which made factorization look as
   though batching made it slower. With the input built once, Cholesky plus
   triangular solves batches 3–4×: 11.1 → 3.0 µs at 27×27.
2. *Benchmarking with captured matrices.* A jitted solve whose matrix is a
   capture shares one factorization across the lanes (17 µs per env against
   127 with the matrix as an argument), so a benchmark that captures it reports
   a number the engine cannot reach. Pass state-dependent matrices as
   arguments.
3. *Composing stages and assuming the sum is the total.* The batched stage
   measurements sum to 34 µs per environment, the composed program measures
   54. Fusion helps at one environment and costs at 256, so the composed step
   is the only measurement worth quoting.

## Appendix B — what the C engine's lines buy

Per subsystem, so the estimate in §9 can be checked against something:

| Subsystem | C lines | Port estimate (OCaml) |
| --- | --- | --- |
| kinematics, com, transmission, tendon, CRB, RNE, solveM | 2 624 | 900 |
| constraints and Jacobians | 3 437 | 1 100 |
| solver, projections, factorization | 3 227 + 2 465 | 1 300 |
| forward stages and integrators | 2 446 | 700 |
| collision driver and primitive pairs | 2 667 + 604 + 1 069 | 900 |
| convex collision (deferred) | 1 916 + 2 749 | 1 200 |
| passive forces | 1 430 | 250 |
| sensors | 1 870 | 400 |
| utility math, quaternions, sparse helpers | 2 684 + 2 395 + 1 598 | 800 |
| model build (`setconst`) | 1 945 | 400 (or 0 with the converter) |
| MJCF compiler | 11 811 | 3 500 |
| **v1 subset (M1–M6)** | ~30 000 | **~6 500** |
