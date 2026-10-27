# Changelog

All notable user-visible changes. Internal refactors, tests, and formatting
changes do not need an entry.

## Unreleased

### geom

- Imported from `hennequin-lab/geom`.

### cmdargs

- Imported from `hennequin-lab/cmdargs`; now a local package.

### dilqr

- Imported from `hennequin-lab/dilqr-raven`.
- Forward-mode differentiation: `solve_ilqr_jvp` and `solve_ilqr_fixed_jvp`
  attach an implicit JVP to the solve, sharing one Riccati factorization across
  a whole batch of directions (`Lqr.backward_common`, `Lqr.solve_jvp`,
  `Solver.tangent`). `Lqr` also gains `costate`, `solve` and the two-half
  `backward_common`/`backward_lin` split, and `mu` is a scalar tensor there.
- The forward-mode tangent sweeps compute the tangents of the first-order data
  step by step (`Solver.tangent_solve`, `Lqr.tangent_step`,
  `Lqr.tangent_roll_step`) instead of materializing a tangent tape, so a
  direction costs `O (horizon · (n + m))` rather than
  `O (horizon · (n² + n·m))`.
- The reverse pass no longer unrolls the horizon (its costate recursions are
  scans), and `loss_of_taus` is one scan rather than one loss per step. The
  local model's expected-decrease coefficient `d2` is `-d1`, as it always was
  numerically.

### ilqr-vae

- Imported from `hennequin-lab/ilqr-vae-raven`.

### symo

- Imported from `hennequin-lab/symo-raven`.

### sofo

- Imported from `hennequin-lab/sofo-raven`.
