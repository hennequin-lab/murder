# How MJX departs from the C engine, and what we should borrow

Notes on DeepMind's MuJoCo XLA, read because `mjc` faces the same problem: a
physics engine whose reference implementation branches on data, reads its own
model on the host, and runs loops until a tolerance is met, now has to run as a
compiled program on a GPU. This records what MJX did about that, with sources,
and what it implies for us. It is *not* a claim that MJX is our baseline: our
baseline is `mj_step`, and MJX is deliberately not comparable to it (§4, §5).

Sources, all read for this note:

- the MJX documentation page, `https://mujoco.readthedocs.io/en/stable/mjx.html`
  — feature parity table, performance tuning, "The Sharp Bits";
- the MJX solver source as rendered by autodoc,
  `https://mujoco.readthedocs.io/en/latest/_modules/mujoco/mjx/_src/solver.html`
  — this is where the actual divergence is visible, in the code and its comments;
- Han Wang, *Improving MuJoCo-MJX with first principles*,
  `https://www.hansolowang.com/posts/improving_mujoco_mjx_with_first_principles/`
  — a practitioner's profiling of quaternion multiplication and compile time;
- issue/PR threads cited inline where they bear on a claim.

## 1. It is a port of the same algorithms

The solver source says so in its own comments: `_update_constraint` "Corresponds
to CGupdateConstraint in mujoco/src/engine/engine_solver.c", `_update_gradient`
"Corresponds to CGupdateGradient", `_LSPoint.create` "roughly corresponds to
CGEval". The documentation agrees: MJX-JAX "uses the same algorithms as the
MuJoCo implementation", and the declared divergences are API-level.

The useful reading is therefore: **MJX is a transliteration of the C engine into
a language without unbounded loops, data-dependent branching, or host reads**,
and every compromise is the price of that transliteration. Ours is a
reimplementation in OCaml/Raven, and we hit the same three walls.

## 2. The three forcing functions

### 2.1 Structure must be host data, because a compiled program cannot branch on it

The docs are explicit about the split, and it is exactly ours:

> Numpy arrays in `mjx.Model` and `mjx.Data` are structural fields that control
> the output of JIT compilation. Modifying these arrays will force JAX to
> recompile MJX functions. As an example, `jnt_limited` is a numpy array … If
> `jnt_limited` is modified, JAX will re-compile MJX functions. On the other
> hand, `jnt_range` is a JAX array that can be modified at runtime, and will only
> apply to joints with limits as specified by the `jnt_limited` field.

That is the same conclusion as `Step.config` and `Constraint.config`: what
decides *which* rows exist is host data read once; what their numbers are is a
tensor. We arrived there from the differentiability side (`Nx.item` in a traced
program raises), they from the recompilation side. Same design.

### 2.2 Loops become bounded scans

MuJoCo's line search is a loop that stops when the bracket converges. MJX cannot
express that, so:

```
def _while_loop_scan(cond_fun, body_fun, init_val, max_iter):
  """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""
  ...
  return jax.lax.scan(_fun, init, None, length=max_iter)[0][0]
```

with `_fun` running the body under `lax.cond(cond, _iter, lambda x: (x, False))`
— once converged, the remaining rounds are no-ops. The outer solver loop is a
`lax.while_loop`, except at `iterations == 1`, which is special-cased
(`if m.opt.iterations == 1: ctx = body(ctx)`).

**This is our 64-round scan with its phase-encoded schedule, arrived at
independently, and for the same reason.** The difference is only that their bound
is a tuned `ls_iterations`/`iterations` and ours is a fixed budget documented as
not differentiable. `doc/perf.md` §6's first item — exit early on a KKT residual
— is the same lever they pull by hand-tuning the bound.

### 2.3 Small, branchy work has to become bounded and uniform

`nconmax` and `nefc` are the fixed buffers, as ours are. But MJX also needs an
*approximate* broadphase, exposed as custom numeric parameters:

> While MuJoCo handles broadphase culling out of the box, MJX-JAX requires
> additional parameters. For an approximate version of broadphase, use the
> experimental custom numeric parameters `max_contact_points` and
> `max_geom_pairs`. `max_contact_points` caps the number of contact points sent
> to the solver for each condim type. `max_geom_pairs` caps the total number of
> geom-pairs sent to respective collision functions for each geom-type pair.

We kept exact culling by computing candidates on the host (`Collision.candidates`)
and only ever *placing* the resulting index tables. That is a real advantage in
correctness and a real cost in host work per call — the trade we have not had to
make yet because we do not batch environments.

## 3. The solver: "Newton" is not MuJoCo's Newton

This is the divergence with teeth. In MJX, the two solvers are two
*preconditioners* inside one CG-shaped loop:

- `CG`: `mgrad = smooth.solve_m(m, d, grad)` — precondition by the mass matrix.
- `NEWTON`: form `h = full_m + (Jᵀ * D * active) @ J`, symmetrise it
  (`h_sym = (h + h.T) * 0.5`, "to reduce the chance of numerical issues in
  cholesky"), `jax.scipy.linalg.cho_factor`, then `cho_solve` — a factorisation of
  an `nv × nv` matrix **inside every solver iteration**, used as the
  preconditioner.

MuJoCo's `NEWTON` is an active-set method: solve the free block, walk to the
first bound crossed, pin, re-solve. MJX has none of that; it has a smooth,
unrolled, preconditioned descent. Consequences, in their words:

> The `NEWTON` Solver delivers excellent convergence with very few (often just
> one) solver iterations, and performs well on GPU.

— which follows from the preconditioner being the exact Hessian: one step is
one Newton step. And the accuracy trade is stated just as plainly:

> The `iterations` and `ls_iterations` attributes … should be brought down to
> just low enough that the simulation remains stable. **Accurate solver forces
> are not so important in reinforcement learning** in which domain randomization
> is often used to add noise to physics for sim-to-real.

That is the single clearest difference from `mjc`, which reproduces MuJoCo's
solved forces to 1e-9–1e-12 and whose G4 gate is 1000/1000 steps within 1e-8
over a 1000-step recording. Their solve is smooth (good for gradients, good for
SIMD); ours is exact (good for parity). They cannot have both at that price, and
neither can we.

Warm starting is present and behaves like the C engine's: MJX compares the cost
of starting from `qacc_warmstart` against `qacc_smooth` and takes the better,
guarded by `DisableBit.WARMSTART`.

## 4. Precision: float32 by default

JAX runs in float32 unless `jax_enable_x64` is set, so MJX does too; the solver
source carries f32-specific tolerance handling (`gtol_accept = max(gtol, 8 · eps
· dmag)`, "to avoid exceeding gtol in f32"). Enabling x64 has had its own
friction — int-dtype mismatches between `make_data`/`put_data` and compiled
functions (PRs #3334 and #3409 fix exactly that class), and issue #2932 reports
MJX producing NaNs on a first step where native MuJoCo is stable, with
`jax_enable_x64` making no difference.

`mjc` is float64 everywhere. That is why our rows can be compared to `mj_step`
at 1e-12 at all, and it is a large part of why we are slower.

## 5. Collision: uniformity bought with exactness

- Narrowphase: MuJoCo uses GJK/MPR/EPA; MJX "uses a branchless version of the
  Separating Axis Test". Budgets, from the docs: roughly ≤200 vertices for
  convex-vs-primitive, **fewer than 32** for convex-convex.
- **`BOX` is implemented as a mesh** in MJX (feature-parity table). We wrote an
  analytic box-box with face clipping (`doc/adr/0011-box-box.md`), which was the
  hardest contact work in M4; they routed boxes through the generic convex path
  instead. Uniform code versus exact geometry, stated in one line.
- Coverage is a subset: `ELLIPSOID` and `CYLINDER` collide only with primitives;
  no `SDF`; `DENSE` Jacobian only; no flex; a sensor subset; ray-casting slow for
  meshes and unimplemented for hfield/flex; `IMPLICITFAST` without fluid drag.

We refuse unsupported things by name instead, which is why the dm_control
quadruped is blocked on `plane`–`cylinder` and GJK/EPA rather than quietly
approximated. MJX *does* support the elliptic cone, which we refuse.

## 6. The decision that says the most: they shipped a second engine

The docs recommend MJX-Warp over MJX-JAX for real work, because Warp "resolves
key performance bottlenecks exhibited in MJX-JAX around contacts and
constraints". And:

> unlike MJX-JAX, MJX-Warp does not support automatic differentiation and has no
> immediate plans to support auto-diff.

So after two years DeepMind's own conclusion is that the portable, differentiable
JAX version cannot be fast on contact-rich scenes, and the fast version does not
differentiate. They forked the trade rather than dissolving it. Their published
numbers for a single humanoid:

| engine | hardware | steps/s |
| --- | --- | ---: |
| C MuJoCo | Apple M3 Max (CPU) | 950 K |
| C MuJoCo | 64-core AMD 3995WX, 2×cores threads | **650 K** |
| MJX-JAX | A100, batch 8192 | 1.8 M |
| MJX-JAX | TPU v5e, batch 16384 | 2.7 M |

and the sharp-bits section is candid that throughput "decreases more rapidly than
MuJoCo" as contacts grow, and that for a *single* scene "MJX-JAX can be **10x**
slower than MuJoCo, which has been carefully optimized for CPU".

A second, smaller datapoint on compile-and-codegen cost, which we have also felt
(`BEAM=2`, and the jit's batched-step regression): quaternion multiplication in
MJX is 28 floating-point operations and compiles to **96 HLO operations**,
mostly memory traffic; reformulating it as a 4×4×4 tensor contraction (200+
flops) took it to 8 HLO operations, −30% compile time and −6% runtime for
`mjx.kinematics`, and −18% compile / −8% runtime for a full step. Their author
notes the most expensive part of the step is collision detection "by far".

## 7. What it means for `mjc`

| decision | MJX | `mjc` |
| --- | --- | --- |
| structure vs parameters | numpy (recompile) vs jax arrays | `Step.config`/`Constraint.config`; `Nx.item` only on host — **same** |
| bounded loops | fixed `ls_iterations` scan, no-op rounds | fixed 64-round scan, phase-encoded — **same compromise**; ours is the top perf lever |
| solver | CG, or PCG with the exact Newton Hessian as preconditioner | active-set Newton with ratio test — **ours exact, theirs smooth** |
| warm start | yes, cost-compared against `qacc_smooth` | yes, same rule |
| precision | f32 default, f64 opt-in and historically rocky | f64 throughout |
| box-box | box as mesh, generic SAT | analytic SAT + face clip — **ours exact, theirs uniform** |
| convex | branchless SAT, vertex caps | GJK/EPA planned; refuse by name meanwhile |
| broadphase | approximate, capped by `max_geom_pairs` | exact, on the host |
| elliptic cone | supported | refused by name |
| scale | 10⁴ envs on GPU; 10× slower single scene | one env; batching flat so far |

Three things to take, in order of how much they change our plan:

1. **Our fixed schedule is not a wart, it is the standard answer**, and the way
   out is the same one they took: bound the loop, make converged rounds no-ops,
   and tune the bound. We can do better than "tune it" — exit on a KKT residual
   (`Rune.iterate`) and keep the parity gate as the check that the exit is safe.
2. **Newton-as-preconditioner is a legitimate fallback**, worth having as an
   alternative `~linear:`-style choice if our active set ever proves
   ill-conditioned on device: one `cho_solve` of `M + JᵀDJ` per round, no ratio
   test, branch-free, cheap to differentiate. The price is exactly what MJX pays
   — the solve stops being comparable to MuJoCo's forces — so it must not be the
   default.
3. **They forked; we should not.** Their split exists because JAX's programming
   model could not express the C engine's branching. Raven's `Rune` can express
   it (our solver *is* the scan, our rows *are* the C engine's), which is why our
   parity numbers exist. The thing to protect is that: exactness and a single
   implementation, with speed bought from the schedule and the buffer sizes
   rather than from a second engine.

The standing caveat, so this note cannot be misread: MJX is a different engine
with different goals. Our baseline for correctness is `mj_step` and our baseline
for speed is `mj_step` on one CPU core (`doc/perf.md`). MJX is the right
comparison for *batched training throughput*, and we have not measured that on
either side yet.
