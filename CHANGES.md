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
- The user-facing solves are `solve_adaptive` and `solve_fixed`, each returning
  a reusable solve closure with the implicit reverse-mode rule. Forward-mode
  forms live in `Dilqr.Jvp`. The flat `solve_ilqr*` names are gone.
- `solve_adaptive` compiles its tensor kernels through `Solver.compile` when
  `~beam` or `~parallel` is supplied, and runs eagerly otherwise; the host
  keeps the regularization and line-search control flow either way.
- `Solver.compile` returns the adaptive solve with its traceable kernels
  (dynamics and cost scans, linearization and Riccati pass, line-search
  rollout) compiled by `Rune.jit`, with `~beam` and `~parallel` for the kernel
  search. The host keeps the regularization retries and line-search control
  flow.
- `Lqr.rollout_tensor` is `Lqr.rollout` with the step size as a scalar tensor,
  so a traced line search can pass it as an argument.
- Forward-mode differentiation: `Jvp.solve_adaptive` and `Jvp.solve_fixed`
  attach an implicit JVP to the same primal solves, sharing one Riccati
  factorization across a whole batch of directions (`Lqr.backward_common`,
  `Lqr.solve_jvp`, `Solver.tangent`). `Lqr` also gains `costate`, `solve` and
  the two-half `backward_common`/`backward_lin` split, and `mu` is a scalar
  tensor there.
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
- Mini-batches are tensors: `Elbo.batch_loss_fixed` takes
  `[batch…; n_steps; n_output]` instead of an array, and the recognitions and
  the bound accept the matching leading trial axes, so a JIT-compiled step can
  take the batch as an argument.
- Trial axes now lead the horizon everywhere: latent inputs are
  `[batch…; n_steps; n_samples; m]`, state paths `[batch…; n_steps + 1; …]`,
  and `Dynamics.integrate` takes `[batch…; n_steps; m]`. The new
  `Dynamics.integrate_samples` is the sample-batched form.

### symo

- Imported from `hennequin-lab/symo-raven`.

### sofo

- Imported from `hennequin-lab/sofo-raven`.
