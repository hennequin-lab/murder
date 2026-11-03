# 0010 — constraint rows and the solver policy

Status: accepted, M3. Answers gate G3: which of MuJoCo's two constraint
formulations v1 reproduces, and what the solver's policy is.

## Which formulation

MuJoCo 3.17 has two: a classical soft-constraint one, and a discrete-time
"metric" one gated by `mj_isMetric`, which is true only for the discrete
integrator (`engine_core_util.c`). v1 ships Euler and RK4, both non-metric, so
the classical path is the one to reproduce. The metric path (`KBIP` divided by
`implicitFactor`, the `efc_Y` reduced factor, the refsafe rebound adjustment)
is out of scope until an integrator needs it.

Consequently the quantities a row needs are the classical ones:

```
row        dist = side * (bound - value);  active when dist < margin
Jacobian   -side at the row's dof
diagA      dof_invweight0[dofadr]              (mmj_diagApprox, limit case)
R          max(MINVAL, (1 - imp) * diagA / imp)
KBIP       K, B from solref (standard when solref.0 > 0, direct otherwise),
           I = imp, P = imp'
aref       -B * efc_vel - K * I * (efc_pos - efc_margin)
force      -D * jar, projected by the row type (limits are non-negative rows)
```

`imp` and `imp'` come from `solimp` evaluated at the row's position and
margin; `efc_vel` is `J qvel`. `dof_invweight0` is a constant of the model,
`1 / M[i,i]` at the reference configuration, computed once at load rather
than stored in the file.

## Row layout

Rows are ordered equality, friction, limits, contacts, as in the C engine. A
step's row count is data, so the port writes into a fixed `nefcmax` buffer and
keeps the count in `nefc`, scattering inactive rows past the buffer. The limit
rows are assembled; equality, friction loss and contact follow in that order.

## Newton, and not PGS

The reduced system is `A = J M^-1 J^T + diag R`, dense, of size `nefcmax`.
Newton solves `A λ = -b` over the free rows and pins the rest to zero with an
identity block, so the system stays square and positive definite and the
active set is handled exactly.

**Not Newton with projection.** Projecting a full Newton step after each
iteration has fixed points that are not KKT points: the first fixture run
returned twice MuJoCo's force because a row that should have been frozen was
carried in the solve. The free set is therefore explicit: a row is free while
its force is positive or its reduced gradient is negative, and the first
iteration takes every row free so a cold start reaches the unconstrained
optimum before anything is frozen.

CG applies `A` as an operator built from `solveM` and `J`, which needs no
factor of `A` but converges linearly; it is not implemented yet.

Policy for v1:

- **Fixed iteration budget**, from the model's `iterations` option, and no
  line search in the first cut. MuJoCo's Newton does a backtracking line
  search on the cost; if the trajectory parity at M3 demands it, add it with a
  fixed number of trials selected by `Nx.where`, which keeps the program
  static. Gate G3's rule: keep the simpler policy only if the acceptance
  tests pass with it.
- **Newton by default**, CG behind the same interface for models where the
  factor is the bottleneck.
- PGS is not implemented (D3's omission, `MILESTONES.md` §11).

## What this costs

The per-row quantities and the Jacobian are reproduced exactly, so `efc_pos`,
`efc_J`, `efc_R`, `efc_D` and `efc_aref` are comparable array by array at
1e-12. The solved `λ` is comparable only when the iteration policy matches, so
the tests hold it to 1e-6 and hold a ten-step constrained trajectory to the
same 1e-6, which is what the physics depends on.
