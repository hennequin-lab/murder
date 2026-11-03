# mjc — a differentiable physics engine on the Raven stack

`mjc` computes what MuJoCo computes: soft-constraint contact dynamics from a
compiled model, stepped in a tensor program. It is written in OCaml on `nx`
and `rune`, has no C library underneath, and is differentiable through the
whole step with `Rune.grad`.

## Status

**M3 in progress.** Equality rows (connect, weld, joint) are assembled and
solved; their rows, anchors and Jacobians match MuJoCo to 1e-12, with two
localised gaps left in the weld's rotational reference acceleration and one
row count. Joint limits are solved: the rows and the impedance match
MuJoCo to 1e-12, the solved force to 1e-6, and a constrained trajectory to
1e-6 over ten steps. The solver runs as an active-set Newton, with CG behind
the same loop agreeing to 1e-8. Equality constraints and friction loss are
next.

**M2 complete.** Forward kinematics, the mass matrix, the velocity stage, the
bias force, springs and dampers, and the semi-implicit Euler step match MuJoCo
3.17 on a four-model parity corpus: the stage arrays to 1e-12, and ten steps
of the whole engine to 1e-10. A compiled step costs about 30 us. The library
is private and has no `(package ...)` stanza yet, so it cannot be depended on
from outside this repository. The G0 linear-algebra gate passes with margin:
[`doc/spike-la.md`](doc/spike-la.md). What exists, and what each milestone
measured, is in [`doc/log.md`](doc/log.md).

Try it:

```sh
dune exec packages/mjc/tools/inspect.exe -- packages/mjc/test/fixtures/ball.mjcb --fk
dune build @packages/mjc/runtest
```

| Document | What it is |
| --- | --- |
| [`PLAN.md`](PLAN.md) | the design note: what MuJoCo does, what Raven gives us, measured costs, feasibility per subsystem, risks |
| [`MILESTONES.md`](MILESTONES.md) | the execution plan: milestones, acceptance tests, gates, stop-losses, parity strategy |
| [`VIZ.md`](VIZ.md) | the live viewer: requirements, library survey, scene model, backend milestones |

Read `PLAN.md` §2–§3 for the problem, `MILESTONES.md` §1–§4 for the order of
work, and `VIZ.md` §3–§4 for the viewer decision.

## What v1 is

In: hinge, slide, ball and free joints; sphere, capsule, box, cylinder and
plane geoms with contact; `condim` 1/3/4/6 with pyramidal cones and
`solref`/`solimp`; joint limits, equality and friction-loss constraints;
motor, position, velocity, intvelocity, damper and general actuators; fixed
tendons; the state-projection sensors; Euler, RK4 and implicit integrators;
Newton and CG solvers with a fixed budget; float64 by default; gradients
through the step.

Out: convex meshes, hfield and SDF geoms, spatial tendons with wrapping, flex,
skin, plugins, islands, continuous collision, the MJCF parser (v1 reads a
converted `.mjcb` model), and GPU support. Each has a milestone in
`MILESTONES.md` and can be dropped without touching v1.

## Shape of the package

```
packages/mjc/
  lib/            topo, model, data, mjcb, rigid, pose, kinematics
                  (inertia, bias, constraint, solver, collision, integrate,
                  step arrive in M2–M6)
  test/           windtrap suites and the parity corpus
  bench/          the linear-algebra spike, then stage benchmarks
  tools/          inspect.exe, convert/record scripts
  viz/            scene description and backends (V0 onward)
  doc/            adr/, log.md, mjcb.md, spike-la.md
```

The core `mjc` library depends on `base`, `stdio`, `nx`, `rune` and
`ppx_ptree`. It does not depend on `geom`, on a graphics library or on a file
format. The viewer is a sublibrary and never a dependency of the engine.

## Reading the milestone set

`MILESTONES.md` §1 states the principles the order of work follows. §8 lists
the gates and what happens if one fails, and §9 states how a stage is tested
against the C engine. Start there before opening a pull request: the
acceptance tests are written before the code, and a milestone that changes a
number in §9.4 has to argue for it in the same change.
