# `ilqr-vae` Clean-up Plan

Context: `dilqr` and `ilqr-vae` now support inferring the initial state $x_0$
directly as a free variable of the optimal control problem, priced by an
`Initial.t` Gaussian prior $p(x_0) = \mathcal{N}(\mu_0, \mathrm{diag}(\sigma_0^2))$
at step $0$ and differentiated via `dilqr`'s implicit VJP. Consequently, the
legacy convention from the 2022 paper — fixing $x_0 = 0$ and treating $u_0$ as
an unscaled Gaussian impulse that sets $x_1$ — is obsolete and conflicts with
inferring $x_0$.

## Todo List

- [x] **1. Remove `first_bin` / `first_stds` ($k = 0$ special cases) from `Prior`**
  - [x] `Prior.Gaussian`: remove `log_first_bin` from `params` and `?first_bin`
    from `make`; use `log_spatial_stds` at every step $k \ge 0$ in `logp_t`,
    `sample`, and `kl_to_gaussian`.
  - [x] `Prior.Student`: remove `log_first_stds` from `params` and `?first_std`
    from `make`; evaluate and sample the multivariate Student-t at every step
    $k \ge 0$ (and draw an independent $\chi^2_\nu$ per time step in `sample`
    rather than one scalar shared across the horizon).
  - [x] `Prior.Laplacian`: remove `log_first_stds` from `params` and
    `?first_std` from `make`; evaluate and sample the Laplace distribution at
    every step $k \ge 0$.
  - [x] Remove `first_step`, `gaussian_logp`, and `steps_mask` helpers from
    `prior.ml` and update `prior.mli` documentation.

- [x] **2. Remove the $k = 0$ `dt_over_tau` bypass from `Dynamics`**
  - [x] Remove `first_step` and the `where (first_step k) u scaled` branch in
    `Dynamics.input_term` (`dynamics.ml`) so `Linear_nonlinear` and `Gnode`
    scale $u_k$ by `dt_over_tau` at every step $k \ge 0$, matching `Linear`.
  - [x] Update `dynamics.mli` docstrings for `Linear`, `Linear_nonlinear`, and
    `Gnode` to remove `(k = 0 ? u : dt_over_tau · u)`.

- [x] **3. Always infer the initial state $x_0$ (remove `infer_initial = false`)**
  - [x] `Recognition.Iilqr`:
    - Remove `?infer_initial` from `Recognition.Iilqr.make` (always infer $x_0$).
    - In `Recognition.Iilqr.params`, make `initial_log_scale : 'a` a required
      field of `Kron` rather than `'a option`.
    - In `Recognition.Iilqr.problem`, always set `free_initial = true` and
      price $x_0$ at step $0$ with `theta.initial_prior`.
    - Make `Recognition.Iilqr.posterior_initial` return `('w, 's) initial`
      rather than `('w, 's) initial option`.
  - [x] `Recognition`:
    - Make `initial_prior : (('w, 's) value, 's) Initial.params` non-optional
      in `Recognition.gparams`.
    - Remove `infers_initial` from `Recognition.t` and `Recognition.make`.
  - [x] `Model`, `Elbo`, `Ilqr_vae`, `Initial`:
    - Make `initial_prior` non-optional in `Model.t` (`Initial.t`) and
      `Model.params` (`(('w, 's) value, 's) Initial.params`).
    - In `Model.make`, default `?initial_prior` to
      `Initial.make ~n:(Dynamics.n dynamics) ()` so callers get $\mathcal{N}(0, I_n)$
      by default and can override it when desired.
    - Always include `Initial.logp` in `Elbo.log_prior` and always sample `x0`
      from `Initial.sample` in `Ilqr_vae.sample_generative`.
    - Update docstrings across `initial.mli`, `recognition.mli`, `model.mli`,
      `elbo.mli`, `ilqr_vae.mli`, and `README.md`.

- [x] **4. Fix `Laplace` recognition mean & detachment issues in `Recognition.Iilqr`**
  - [x] Center the `Laplace` Gaussian directly at the iLQR solution
    `nominal = (xs, us)` (zeroing `feedforward` alongside setting
    `initial.mean = xs.(0)` in `local`), so `mean_x0 = xs.(0)` and
    `mean_u = us` under both `Kron` and `Laplace` and 100% of the mean gradient
    comes from `dilqr`'s implicit VJP.
  - [x] Detach the local quadratic model's curvature factors (`tape`,
    `feedback`, `quu_chol`, `initial.precision`) in `local` so that:
    - `infer_one` does not leak gradients through `tape.blocks.a` and
      `tape.blocks.b` during `Laplace.sample` or `Laplace.log_prob` (preserving
      Sticking-the-Landing);
    - `log_prob_one` evaluates the exact live density of $q$ (live mean
      `(xs.(0), us)` and live `log_scale`, stopped curvature), allowing
      finite-difference gradient checks (`Rune.check_grads`) on both `Kron` and
      `Laplace`.

- [x] **5. Update and expand tests & documentation**
  - [x] Update `test/test_ilqr_vae.ml` and `test/mpi/test_mpi.ml` for the
    cleaned-up APIs and parameter tree structures.
  - [x] Add `Rune.check_grads` coverage for `Recognition.Iilqr` with `Laplace`
    covariance and inferred initial state.
  - [x] Run `dune build @packages/ilqr-vae/check @packages/ilqr-vae/fmt @packages/ilqr-vae/runtest`.
