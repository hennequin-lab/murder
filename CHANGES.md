# Changelog

All notable user-visible changes. Internal refactors, tests, and formatting
changes do not need an entry.

## Unreleased

### mjc

- New package: a differentiable physics engine on Nx and Rune, private until
  its API settles. Forward kinematics, the mass matrix, the velocity stage and
  the Euler step match MuJoCo 3.17 on a recorded parity corpus, to 1e-12 per
  stage and 1e-10 over ten steps; the `.mjcb` interchange format loads
  compiled models without linking MuJoCo. See `packages/mjc/README.md`.

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
  a reusable solve closure. One custom JVP carries both implicit derivatives:
  forward mode runs its tangent map, and reverse mode applies the implicit VJP
  as that map's transpose. `Rune.grad` and `Rune.jvp` therefore apply to the
  same function; `Dilqr.Jvp` is gone. The flat `solve_ilqr*` names are gone.
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
- Forward-mode differentiation: the same solves have an implicit JVP,
  sharing one Riccati factorization across a whole batch of directions
  (`Lqr.backward_common`, `Lqr.solve_jvp`, `Solver.tangent`). `Lqr` also gains
  `costate`, `solve` and the two-half `backward_common`/`backward_lin` split,
  and `mu` is a scalar tensor there.
- The forward-mode tangent sweeps compute the tangents of the first-order data
  step by step (`Solver.tangent_solve`, `Lqr.tangent_step`,
  `Lqr.tangent_roll_step`) instead of materializing a tangent tape, so a
  direction costs `O (horizon · (n + m))` rather than
  `O (horizon · (n² + n·m))`.
- `Lqr.backward_common` retains the value Hessian at the horizon's start
  (`common.value_hessian`), and `Lqr.backward_lin` and `Lqr.backward` return
  the value gradient there as a fourth result. Both were already the last
  carries of the two recursions and were discarded. The new
  `Lqr.initial_condition` fuses that value function with a Gaussian prior on
  the initial state, returning the mean and precision the local model implies
  for `x_0` — the truncation-point message of the Riccati pass. Its prior is
  `~prior:(mean * precision) option`: a solve whose free initial state is
  priced by the tape's own step-0 cost passes `None`, and then the call is the
  stationary point of the value function itself.
- The reverse pass no longer unrolls the horizon (its costate recursions are
  scans), and `loss_of_taus` is one scan rather than one loss per step. The
  local model's expected-decrease coefficient `d2` is `-d1`, as it always was
  numerically.
- Ported to raven 19a5d88: the problem and every tensor type take the device
  set of their tensors (`('w, 'p, 'd) Problem.t`, `('w, 'd) Problem.value`,
  `(float, 'w, 'd) Lqr.tape`), `inputs problem` describes the inputs at that
  set, and `Lqr.Rows`, `Lqr.Tangent_rows` and `Lqr.Tangent_roll_rows` are
  `('a, _) t` structures, as a parameter record now is.
- `solve_fixed` (and `Solver.solve_fixed`) take `~max_iters` instead of
  `~iters`: the budget is an upper bound, and the solve stops when the batch
  has converged. Because it stops on its own test now, a solve that used to run
  a given number of iterations converges a little less far than the same count
  did; raise the budget or lower `~conv_threshold` where that matters. `Solver.t` gains `iters`, the number of iterations the solve
  ran; a batch spends its slowest member's.
- The fixed solve's line search is recursive: it tries the trial steps in
  order and stops at the first the batch accepts, each level a `Rune.cond`
  branch, so a level nobody reaches computes nothing. It selects the largest
  accepted step, as the adaptive solve does, where it used to take the argmin
  of the accepted trials' costs (and its Armijo test applied `c1` twice).
- Batched fixed solves take one step and one damping schedule for the whole
  batch: a trial is accepted when every member that has not converged
  satisfies Armijo at it, so the batch moves at the step its slowest member
  can take. A converged member is frozen and does not vote, so it cannot veto
  the others. The implicit derivative is still taken member by member.
- `Dilqr.Damping` is how a solve damps: `Off`, `Fixed of float` (one absolute
  strength) or `Adaptive of Regularisation.t`, with `Damping.default` —
  adaptive — the default for both solves. It replaces the fixed solve's
  `~mu`/`~reg` pair, which could be given together, and the adaptive solve's
  `~reg`. An adaptive strength is a fraction of each member's own value-Hessian
  scale, so members whose curvature differs are damped alike; the schedule is
  carried through the traced loops as tensors. `~conv_threshold` sets the fixed
  solve's stopping test.
- `Lqr.backward_common` shifts the damping into the value Hessian
  (`V + μ I`) instead of adding Tassa's three terms one by one: expanding the
  shift gives exactly those terms, so a damped pass costs one diagonal
  addition per step. `mu` may hold one strength per batch member.
  `Regularisation.floor` names the floor of a schedule.

### ilqr-vae

- Imported from `hennequin-lab/ilqr-vae-raven`.
- The fixed inner solve runs at most the recognition's iteration count and
  stops earlier when it has converged, so a fixed recognition is a budget
  rather than an exact count. `Recognition.fixed` takes `?conv_threshold` for
  how tightly it should converge, and `?damping` (`Dilqr.Damping.t`, adaptive
  by default) instead of `?mu`.
- Mini-batches are tensors: `Elbo.batch_loss_fixed` takes
  `[batch…; n_steps; n_output]` instead of an array, and the recognitions and
  the bound accept the matching leading trial axes, so a JIT-compiled step can
  take the batch as an argument.
- Trial axes now lead the horizon everywhere: latent inputs are
  `[batch…; n_steps; n_samples; m]`, state paths `[batch…; n_steps + 1; …]`,
  and `Dynamics.integrate` takes `[batch…; n_steps; m]`. The new
  `Dynamics.integrate_samples` is the sample-batched form.
- Ported to raven 19a5d88: every value, component, recognition and model
  type takes the device set of its tensors, named `'s` (`('w, 's) value`,
  `('w, 'p, 's) Dynamics.t`, `('w, 'd, 'l, 'p, 'e, 'r, 's) Model.t`), keys
  are `'s Nx.Rng.key`, and the parameter records are `('a, _) params`
  structures (`('a, 's) params` where one nests another). The constructors
  make host components, as nx's creations do.
- `Model.structure` and `Recognition.gparams_structure` name their parts by
  field alone: a path that was `0.dynamics.w` is `dynamics.w`.
- A recognition can infer the initial state. `Recognition.Iilqr.make` takes
  `?infer_initial`, and `Recognition.gparams` (and so the encoder's
  `Dilqr.Problem.t`) carries the model's `initial_prior`, which is then the
  step-0 running loss of the encoder's control problem — the boundary
  condition that makes `x_0` free. The recognition's posterior over `x_0` is
  the local model's Laplace posterior: its mean is the joint mode, reached in
  `?rounds` rounds of solve-then-replace, and its covariance is
  `exp (2 initial_log_scale) · P⁻¹`. Both are values of the local model, taken
  stopped in the draws and in the density the bound differentiates, so the
  block's one free parameter — and its one gradient — is the log scale;
  `Recognition.Iilqr.posterior_initial` reports the pair. A
  model with such a recognition must pass `~initial_prior`; one with the
  fixed-zero recognition must not, as before.
- `Elbo.batch_loss_fixed` is gone: `Elbo.loss_fixed` already takes a tensor mini-batch, and the two names differed by a rank check and documentation only.
- `Recognition.solve` selects the iLQR inner solver: `Recognition.adaptive` and
  `Recognition.fixed` replace the `?iters` argument of every recognition call,
  `Iilqr.posterior_mean_fixed` folds into `Iilqr.posterior_mean`, and
  `Ilqr_vae.posterior_mean`, `posterior_draws`, `log_prob` and `loss` take
  `?solve`.
- `Recognition.generative` and `Recognition.gparams` no longer carry the
  generative input prior, which no recognition reads, and `generative` no
  longer carries the initial-state prior, which is a parameter:
  `Recognition.t` takes one type parameter fewer.
- `Ilqr_vae.sample_generative`, `predictions` and `predictions_deterministic`
  return the records `Ilqr_vae.generated` and `Ilqr_vae.prediction` instead of
  tuples of same-typed tensors.
- The bound of a recognition that leaves the inputs at the point mass at zero
  (`Lfads ~with_inputs:false`) no longer carries the input prior's term, which
  moved with the prior's scales although there was no input density to weigh.
  `Recognition.infers_inputs` states the case.
- Shapes are checked where they were broadcast into a wrong answer: the latents
  of `Recognition.log_prob`, the batch axes of `Elbo.compute`'s `log_prob`,
  `Elbo.log_prior`'s two tensors, and the dimension axes of
  `Dynamics.integrate` and `integrate_samples`.
- `Iilqr.problem` takes no gparams, which it never read, and takes `?verbose`;
  `Dynamics.Linear.make` takes `?dt_over_tau`; `Model.make` checks both vectors
  of an `initial_prior`; and the decoders use the horizon of the state path
  they are given instead of the model's.
- `Recognition.Iilqr.posterior_initial` returns `Recognition.initial` — a mean
  and a precision — instead of `Dilqr.Lqr.initial`: a caller reading the
  encoder's initial state no longer names dilqr's record, and the conversion
  happens once, inside the recognition.
- Every component constructor takes `~at`, the device set its parameters live
  at: `~at:Nx.Host.on` is the host, as before, and a device placement makes a
  model whose parameters are there (`Covariance.make`, `Initial.make`, the
  priors, the likelihoods, the dynamics and the recognitions).
- A model component is a description, and its parameters are a value of their
  own: none of the component records has a `params` field, `Dynamics.Linear.make`
  and the rest build only the description, and `init` draws the parameters a
  fresh model starts from — `Model.init ~at ~key model`, one subkey per
  component. `Model.make` no longer returns parameters, and `Model.params` is
  gone: building a model allocates nothing, so building one per device costs
  nothing.
- `place` moves a component's parameter value to another placement —
  `Prior.Gaussian.place ~at p`, and the same for every component and for
  `Initial` — with `Nx.Ptree.place`, so a checkpoint can follow the device it
  runs on.
- `Density` is the library's Gaussian log density, written once for the form
  the covariance is held in — `Isotropic`, `Diagonal`, `Cholesky` of the
  covariance, `Precision_cholesky` — so a diagonal costs `O (d)` and a factor
  one triangular solve. `Initial.logp`, the priors and the two recognitions'
  own densities go through it.
- `Model.make` takes no dimensions. Every one of them is a field of a
  component — `n` and `m` of the dynamics, `n_output` of the likelihood,
  `n_steps` of the recognition — so the model reads them from there and only
  checks that the components agree. `Recognition.dims` is still the source of
  the horizon, which no other component knows.

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

- The subspace solve is documented as the one part of a step that stays on the
  host, and the reason has moved twice. It was that an eigendecomposition could
  not be compiled; then that Raven compiles one (`Nx.eigh` lowers to Jacobi
  rotations, to `1e-15` of the host's eigenvalues) but that compiling it took
  8.5 s for a single `k = 24` program, nearly all of it in one LLVM pass of one
  kernel — LLVM 22's loop-idiom recogniser on a chain of float maxima, which
  Raven 72044867 turns off for host kernels; and now that it compiles in a
  fraction of a second but runs slower than the host's: 8× at `k = 128`, and
  10–60× on an AMD GPU, against a round trip of three `k×k` tensors.
  `doc/jit-solve.md` records the measurements on both sides, the diagnosis, and
  what would make a compiled solve worth having. No API change.
- Imported from `hennequin-lab/sofo-raven`.
- `Optim.directions`, `Optim.step` and `Optim.sketch_jit` take `?custom`, a
  batched set of tangents that becomes the first lanes of the sketch and is
  completed with random lanes up to `k`; `Optim.tangents` stacks a list of
  parameter-shaped tangents on the lane axis. In `sketch_jit` the custom set
  is an input of the compiled program (`Nx.Ptree.option`), so it may change
  between replays; with a fixed lane count only its values need to change.
- The direction draw caps `k` at the number of float parameters, so asking
  for more tangents than the parameter space has dimensions no longer makes
  the Gram matrix singular.
- Arrays carry their device set: `Curv.t` is `'d Curv.t`, `sketch` is
  `('p, 'd) sketch` and `Optim.step_info` is `('p, 'd) Optim.step_info`, and
  `Optim.state`'s key is a host `Nx.Rng.key`. `softmax_ce` takes float logits.
- `Optim.gram` raises on directions with no tensor leaf.
- A `Rune.vmap` around `Sofo.sketch` raises at raven 19a5d88: `Rune.lanes`
  in a tangent map that an outer map evaluates does not see the map around
  the call, a rune bug.
- A sketch must be given the uncompiled loss: a `Rune.jit`ed loss first traced
  outside a sketch keeps no marks, and its sketch has a zero GGN.
- `Optim.update_jit` is the parameter step with its compilation hoisted out of
  the call: `Rune.jit` caches per closure, so the step `Optim.update` builds
  was recompiled — kernel search included — on every iteration of a training
  loop. `Optim.update` keeps its signature and builds one per call; `update`
  and `update_jit` take `?beam ?parallel` and inherit the runtime's settings
  (`BEAM`, `PARALLEL`) instead of forcing `~beam:2 ~parallel:8`, which made a
  step cost a kernel search whose cost grows steeply with the lane count.
- An observed curvature is a captured value of the mark's custom rule, and
  rune refuses a rule that uses one the differentiation tracks. `mse` and
  `sse` now detach their weight, and `Curv.hvp` states the requirement for its
  closure: a curvature computed from the prediction must be passed through
  `Rune.detach`. Without it, `sketch_jit` raises `Rune.custom_jvp: the rule
  uses a value its own differentiation tracks` where an eager sketch quietly
  works.
