(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The recognition distribution.

    [q(u)] is the Gaussian [N(μ_u, Σ_time ⊗ Σ_space)] whose mean is the iLQR
    posterior mean ({!Encoder}) and whose covariance factors are the model's
    {!Covariance.params} values. Sampling is reparameterized — [u = μ + L_time ξ L_spaceᵀ] with [ξ] standard normal — so the samples are differentiable in
    [μ] and in both factors, which is what the ELBO's gradient needs. The
    density is exact: {!log_prob} and {!entropy} use the factors, not a
    Monte-Carlo estimate.

    The leading axis is time and the feature axis is last, so a sample batch is
    [[n_steps; n_samples; m]]: {!sample} draws one batch, and the closed-form
    quantities take the mean's shape, [[n_steps; m]]. *)

open Types

(** [sample model params ~mu ~n_samples ~key] draws [n_samples] inputs from
    [q], of shape [[n_steps; n_samples; m]]. Reparameterized and differentiable
    in [mu] and in the covariance parameters, so it can sit inside a
    {!Rune.grad}. *)
val sample
  :  ('w, 'd, 'l, 'p, 'e) Model.t
  -> ('w, 'd, 'l, 'p, 'e) Model.params
  -> mu:'w value
  -> n_samples:int
  -> key:Nx.Rng.t
  -> 'w value

(** [log_prob model params ~mu u] is log q(u) for an input sequence of shape
    [[n_steps; m]], or one value per sequence for a batch [[n_steps; batch…; m]].
    The quadratic form is evaluated with two triangular solves against the
    Cholesky factors. *)
val log_prob
  :  ('w, 'd, 'l, 'p, 'e) Model.t
  -> ('w, 'd, 'l, 'p, 'e) Model.params
  -> mu:'w value
  -> 'w value
  -> 'w value

(** [entropy model params] is the exact differential entropy of [q],
    [½ (d (1 + log 2π) + log det Σ)] with [d = n_steps · m], a scalar tensor.
    The ELBO uses it to estimate the KL when the prior is not Gaussian. *)
val entropy
  :  ('w, 'd, 'l, 'p, 'e) Model.t
  -> ('w, 'd, 'l, 'p, 'e) Model.params
  -> 'w value
