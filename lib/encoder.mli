(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The iLQR encoder.

    The posterior mean of the recognition distribution is the solution of a
    finite-horizon optimal control problem. With a trial's observations [o] of
    shape [[n_steps; n_output]] and the convention that [o_k] is emitted from
    the state reached after the first [k + 1] inputs:

    {[
      min_{u_0 … u_{T−1}}  Σ_k l(x_k, u_k) + φ(x_T)
        l(x_k, u_k) = −log p̃(u_k) − [k ≥ 1] · log p(o_{k−1} | x_k)
        φ(x_T)      = −log p(o_{T−1} | x_T)
        x_0 = 0,  x_{k+1} = f(x_k, u_k)
    ]}

    Every observation is used exactly once, and the terminal cost carries the
    last one. {!problem} is this problem as a {!Dilqr.Problem.t}: its
    parameters are the model's own {!Model.params}, so {!Dilqr} differentiates
    the solve with respect to the whole model and its implicit VJP chains the
    gradient back through the encoder.

    The cost's likelihood terms are indexed by the step tensor [k] the
    {!Dilqr} scans carry: they read [o_{k−1}] with [Nx.take], so the
    encoder is traceable and JIT-compilable under {!posterior_mean_fixed}.

    Two solvers, mirroring {!Dilqr.Solver}: {!posterior_mean} is the adaptive
    iLQR (regularization retries, Armijo backtracking, host-side control
    flow), eager only; {!posterior_mean_fixed} runs a fixed number of
    iterations with fixed regularization and a line search realized as a
    masked argmin, and traces. *)

open Types

(** [problem model params data] is the encoder's control problem for one
    trial: horizon [n_steps], state [n], control [m], dynamics from
    [model.dynamics], running loss the encoder prior minus the likelihood and
    terminal loss the last observation's likelihood. The trial's data is
    captured by the losses. *)
val problem
  :  ('w, 'd, 'l, 'p, 'e) Model.t
  -> ('w, 'd, 'l, 'p, 'e) Model.params
  -> 'w value
  -> ('w, ('w, 'd, 'l, 'p, 'e) Model.params) Dilqr.Problem.t

(** [posterior_mean model params ?u_init ?max_iter ?conv_threshold data] is
    the adaptive iLQR solution's controls, of shape [[n_steps; m]] — the
    recognition mean [μ_u]. [u_init] ([n_steps; m], zeros when omitted) is the
    initial guess and carries no gradient: the idealized solve does not depend
    on it. [max_iter] defaults to [20] and [conv_threshold] to [1e-6], as in
    {!Dilqr.Solver.solve}.

    Differentiable through {!Dilqr.solve_ilqr}'s implicit VJP. Eager only:
    the solver reads scalars and retries the factorization on the host, so do
    not put it inside {!Rune.jit}; use {!posterior_mean_fixed} there. *)
val posterior_mean
  :  ('w, 'd, 'l, 'p, 'e) Model.t
  -> ('w, 'd, 'l, 'p, 'e) Model.params
  -> ?u_init:'w value
  -> ?max_iter:int
  -> ?conv_threshold:float
  -> 'w value
  -> 'w value

(** [posterior_mean_fixed model params ~iters ?u_init ?alphas ?mu ?c1 data] is
    the same mean, computed with [iters] fixed iLQR iterations: tensor-only
    control flow, one {!Rune.scan} per horizon, traceable and JIT-compilable
    together with its implicit VJP, as {!Dilqr.solve_ilqr_fixed}.

    [alphas] is the line-search trial schedule (default: eight halvings from
    [1.]), [mu] the fixed regularization (default [0.]), [c1] the Armijo
    constant (default [1e-2]). *)
val posterior_mean_fixed
  :  ('w, 'd, 'l, 'p, 'e) Model.t
  -> ('w, 'd, 'l, 'p, 'e) Model.params
  -> ?u_init:'w value
  -> iters:int
  -> ?alphas:float list
  -> ?mu:float
  -> ?c1:float
  -> 'w value
  -> 'w value
