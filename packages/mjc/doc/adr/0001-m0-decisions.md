# 0001 — foundations: package, dtype, structure, format

Status: accepted, M0. Records decisions D1–D9 of
[`../../MILESTONES.md`](../../MILESTONES.md) as they were implemented, and the
two places where implementation differs from the plan.

## D1 — Layout

`packages/mjc/` holds the library, its tests, benchmarks, tools and docs.
`lib/` is `mjc`; the viewer's sublibraries (`viz/`) arrive with V0. The
library is **private** until M7: no `public_name`, no `(package ...)` stanza.
That is `geom`'s interface-first convention, it keeps `dune.lock` untouched,
and it means a user cannot accidentally depend on an API that is still
moving.

## D2 — Own rigid-body math

`Rigid` is `mjc`'s own quaternion code. `geom`'s `Motor` is `float32` at this
pin, and a physics engine's quaternion conventions (MuJoCo's `[w;x;y;z]`,
active rotation matrices, `qpos` layouts) are not a GA library's.

## D3 — Dtype, as implemented

The record types are dtype-parametric: `('a, 'd) Model.t`,
`('a, 'd) Data.t`, and `('a, 'd) Model.Params.t`, so a model is already a
structure over whatever tensor type a caller instantiates. `Rigid` and `Pose`
are polymorphic in storage dtype and device set.

The **kernels** are monomorphic at host `float64` in v1: `Kinematics.forward`
takes `Model.host` and `Data.host`, because it builds host index tensors from
the topology. The planned `Mjc.Make (Config)` functor is deferred to M7, when
float32 is required. This is a schedule change, not a design change: the
records and the math already carry the parameters.

## D4 — Stages are public

`Kinematics.forward` is public on its own; `Forward` and `Step` compose it in
M2. Parity debugging needs the stage, and exposing it costs nothing.

## D5 — Budgets in the topology

`nconmax`, `nefcmax` and `max_geom_pairs` are `Topo.t` fields, so they travel
with the model rather than with a compile call. The converter chooses them
from the model size; a caller converting can override. Overflow is an
`Nx.check`, never a silent drop (M4).

## D6 — Dense

No sparse path. `nv ≤ ~100` is the documented cap.

## D7 — Factorization

`Nx.cholesky` + `Nx.solve_triangular`, never `Nx.solve`, in hot paths. The
spike in [`../spike-la.md`](../spike-la.md) measures both: at `n = 27` a
factor plus three triangular solves costs 10.4 us eagerly against 7.7 us for
one LU solve, and at `n = 39` 11.4 us against 10.3 us, so the advantage is in
the compiled path, where the triangular solves cost 39.6 us against 30.4 us
for LU at 39 — comparable. The rule stands for the structural reason (the
factor is reused by the solver and the implicit integrator), not for a
large measured gap on this pin.

## D8 — `.mjcb` is the v1 model format

Spec in [`../mjcb.md`](../mjcb.md). The converter is development-time Python.

## D9 — Shape keys

`Model.shape_key` is a record: dtype, the object counts, and the three
budgets. It is printed by `tools/inspect.exe --shape-key`. `nnz`-style
quantities are not in it, because the budgets are compile-time and no
data-dependent value may be.

## Deviations from the plan

1. **Unrolled body loop instead of a `scan`.** `Nx.slice` cannot take a traced
   index, so each body's joints are unrolled regardless; at a 151 ns loop
   floor and ≤ 20 bodies the unrolled pass is not slower and is simpler. M2
   measures the composed step (G2) and reopens this if the budget misses.
2. **`jnt_limited` is a float tensor, not a byte.** Every consumer selects or
   multiplies with it.

## Not decided here

PGS, elliptic cones, sparse storage and islands remain out of v1 as planned.
The `.mjcb` format has no actuator, sensor or tendon tables until M5.
