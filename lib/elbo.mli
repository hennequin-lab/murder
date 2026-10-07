(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The evidence lower bound.

    With [q(u) = N(μ_u, Σ_time ⊗ Σ_space)] and [u_1 … u_N] reparameterized
    draws from it, the bound and its terms are

    {[
      log p(o)         ≈ (1/N) Σ_i log p(o | z(u_i))
      KL(q ‖ p)        = E_q [log q − log p]
      ELBO             = log p(o) − β · KL(q ‖ p)
    ]}

    The likelihood term is estimated by rolling the dynamics out from each
    sample; the KL is exact when the generative prior reports
    [kl_to_gaussian] (a Gaussian prior), and otherwise estimated as
    [−H(q) − (1/N) Σ_i log p(u_i)]: the recognition entropy is analytic, so
    only the prior term is sampled. Both are differentiable in the model
    parameters — including the mean, through {!Encoder}'s implicit VJP — and
    traceable.

    {!loss} is the negated ELBO, which is what a training loop minimizes.
    {!loss_fixed} additionally runs the encoder, so one call is the whole
    per-trial objective; {!batch_loss_fixed} folds it over trials. *)

open Types

(** One ELBO pass's result: the bound and its two terms, from the same
    samples. Use it to monitor training without paying for a second pass. *)
type 'w t =
  { elbo : 'w value (** the bound, a scalar *)
  ; log_likelihood : 'w value (** [(1/N) Σ_i log p(o | z(u_i))], a scalar *)
  ; kl : 'w value (** [KL(q ‖ p)] at [β = 1], a scalar *)
  }

(** [compute model params ~mu_u ~key ~n_samples data] samples [n_samples]
    inputs from the recognition distribution at [mu_u], rolls the dynamics out
    under each, and returns the bound and its terms:
    [elbo = log_likelihood − beta · kl].

    [mu_u] is the encoder's output ([n_steps; m]); it carries gradients, so
    composing [compute] with {!Encoder.posterior_mean_fixed} differentiates
    the whole inference path. [beta] defaults to [1.].

    Raises [Invalid_argument] if [n_samples] is not positive or the data's
    shape is not [[n_steps; n_output]]. *)
val compute
  :  ('w, 'd, 'l, 'p, 'e) Model.t
  -> ('w, 'd, 'l, 'p, 'e) Model.params
  -> ?beta:float
  -> mu_u:'w value
  -> key:Nx.Rng.t
  -> n_samples:int
  -> 'w value
  -> 'w t

(** [loss …] is [Nx.neg (compute …).elbo], the per-trial training objective. *)
val loss
  :  ('w, 'd, 'l, 'p, 'e) Model.t
  -> ('w, 'd, 'l, 'p, 'e) Model.params
  -> ?beta:float
  -> mu_u:'w value
  -> key:Nx.Rng.t
  -> n_samples:int
  -> 'w value
  -> 'w value

(** [compute_fixed model params ~key ~n_samples ~iters data] is {!compute}
    with the mean computed by {!Encoder.posterior_mean_fixed ~iters}. Bound
    and diagnostics come from the same samples, and the whole call —
    fixed-iteration iLQR, its implicit gradient, the samples and the ELBO —
    is a pure tensor computation: this is what goes inside {!Rune.jit}. *)
val compute_fixed
  :  ('w, 'd, 'l, 'p, 'e) Model.t
  -> ('w, 'd, 'l, 'p, 'e) Model.params
  -> ?beta:float
  -> key:Nx.Rng.t
  -> n_samples:int
  -> iters:int
  -> ?u_init:'w value
  -> 'w value
  -> 'w t

(** [loss_fixed model params ~key ~n_samples ~iters data] is
    [Nx.neg (compute_fixed model params ~key ~n_samples ~iters data).elbo]:
    encoder and bound in one scalar call. *)
val loss_fixed
  :  ('w, 'd, 'l, 'p, 'e) Model.t
  -> ('w, 'd, 'l, 'p, 'e) Model.params
  -> ?beta:float
  -> key:Nx.Rng.t
  -> n_samples:int
  -> iters:int
  -> ?u_init:'w value
  -> 'w value
  -> 'w value

(** [batch_loss_fixed model params ~key ~n_samples ~iters trials] is the mean
    of {!loss_fixed} over the trials. The key is split into one subkey per
    trial with [Nx.Rng.split], so the trials' samples are independent and the
    result is deterministic in [key].

    This is the mini-batch objective: [Rune.value_and_grad (Model.structure model) (fun p -> batch_loss_fixed model p …) params] gives the gradient
    over the batch in one pass. *)
val batch_loss_fixed
  :  ('w, 'd, 'l, 'p, 'e) Model.t
  -> ('w, 'd, 'l, 'p, 'e) Model.params
  -> ?beta:float
  -> key:Nx.Rng.t
  -> n_samples:int
  -> iters:int
  -> 'w value array
  -> 'w value
