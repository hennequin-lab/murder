# Changelog

All notable user-visible changes. Internal refactors, tests, and formatting
changes do not need an entry.

## Unreleased

### geom

- Imported from `hennequin-lab/geom`.
- Multivectors and motors carry nx's device set, as `'d Cga.t` and
  `'d Motor.t`: an operation computes where its operands lie. Constructors from
  numbers (`scalar`, `one`, `basis`, `e`, `of_array`, `point3`, `Cga.einf`,
  `Motor.translate3`) give host values.
- `Motor.compose []` raises `Invalid_argument`, where it was `Motor.one`: an
  empty list names no device set to make `one` on.
- `Geom_robot`: configurations, motions and twists are `'d Robot.q`,
  `'d Robot.motion` and `'d Robot.twist`, and `Task.t`, `Ik.state` and
  `Ik.result` take the device set of their targets and configurations. A model
  holds host motions: `joint_spec.origin` and `frame_spec.pose` are `Nx.host`.
- `Geom_viz` and `Geom_robot_viz` take objects, motions and configurations of
  any device set and read them on the host. `Camera.view` is a host motor, and
  `Frame.rgba` a host image.
- `Cga.distance` of two coincident points is `0.`, with derivative `0.`.
  Rounding could leave their `distance2` below zero, and the distance was NaN.

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
- Ported to raven 19a5d88: the problem and every tensor type take the device
  set of their tensors (`('w, 'p, 'd) Problem.t`, `('w, 'd) Problem.value`,
  `(float, 'w, 'd) Lqr.tape`), `inputs problem` describes the inputs at that
  set, and `Lqr.Rows`, `Lqr.Tangent_rows` and `Lqr.Tangent_roll_rows` are
  `('a, _) t` structures, as a parameter record now is.

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
- Ported to raven 19a5d88. The orbit machinery and the optimizer run on the
  parameter tree's device set: a `Model` is an `Nx.Ptree.S` declared
  `('a, _) t` with `dims` and `symmetries` at every set; the set is a new
  parameter of `Orbit.S.t`, both `factors`, `Optim.S.tree`, `State.t`, `Mid.t`,
  `state`, `mid` and `Compiled.t`; `random_transform` takes a `'d Nx.Rng.key`,
  `Compiler.t`'s closures run on any set, and `ptree` describes the tree on
  `Nx.host`.

### sofo

- Imported from `hennequin-lab/sofo-raven`.
- Arrays carry their device set: `Curv.t` is `'d Curv.t`, `sketch` is
  `('p, 'd) sketch` and `Optim.step_info` is `('p, 'd) Optim.step_info`, and
  `Optim.state`'s key is a host `Nx.Rng.key`. `softmax_ce` takes float logits.
- `Optim.gram` raises on directions with no tensor leaf.
- A `Rune.vmap` around `Sofo.sketch` raises at raven 19a5d88: `Rune.lanes`
  in a tangent map that an outer map evaluates does not see the map around
  the call, a rune bug.
- A sketch must be given the uncompiled loss: a `Rune.jit`ed loss first traced
  outside a sketch keeps no marks, and its sketch has a zero GGN.
