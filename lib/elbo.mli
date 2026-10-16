(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The evidence lower bound.

    With [q] the model's recognition distribution, [x_0^i, u^i] reparameterized
    draws from it and [z^i] the state paths they drive, the bound is the
    Monte-Carlo estimate

    {[
      ELBO = (1/N) Σ_i [ log p(o | z^i) + log p(x_0^i) + Σ_k log p(u_k^i)
                         − log q(x_0^i, u^i) ]
    ]}

    evaluated in one pass: {!Recognition.infer} returns the draws together
    with [log q] at them, and {!compute} adds the likelihood of the rolled-out
    states and the model's log prior at the same draws. The KL term is thus
    sampled rather than analytic — no Gaussian assumption on [q] — and both
    densities are evaluated at the same reparameterized draws.

    {b Sticking the landing.} {!Recognition.infer} evaluates [log q] with the
    recognition's distribution parameters stopped, so differentiating the
    estimate reaches them only through the sampled latents: the
    path-derivative gradient estimator of Roeder, Wu & Duvenaud (2017). Its
    expectation is the gradient of the bound, it is lower variance than the
    score-function alternative, and its variance goes to zero as [q]
    approaches the exact posterior. The values of the bound and its terms are
    unchanged — the difference is only in the backward pass.

    {!loss} is the negated bound, which is what a training loop minimizes. It
    runs the recognition itself; {!loss_fixed} is the traceable form with a
    fixed number of iLQR iterations, and {!batch_loss_fixed} folds it over
    trials. A caller that already has a posterior — reused across several
    bounds, or built with the adaptive iLQR solve — passes it to {!compute}
    directly. *)

open Types

(** One ELBO pass's result: the bound and its two terms, from the same
    draws. Use it to monitor training without paying for a second pass. *)
type 'w t =
  { elbo : 'w value (** the bound, a scalar *)
  ; log_likelihood : 'w value
    (** the mean over draws of [Σ_k log p(o_k | z_k)], a scalar *)
  ; kl : 'w value (** the sampled [E_q [log q − log p]], a scalar, at [β = 1] *)
  }

(** [compute model params ~posterior data] rolls the dynamics out under the
    posterior's draws and returns the bound and its terms,
    [elbo = log_likelihood − beta · kl].

    The posterior is what {!Recognition.infer} returns; it carries the draws
    and [log q] at them, so all three terms come from the same draws and one
    encoder pass. It carries gradients, so composing {!compute} with an
    inference differentiates the whole path. [beta] defaults to [1.].

    Raises [Invalid_argument] if the posterior batch is empty. *)
val compute
  :  ('w, 'd, 'l, 'p, 'e, 'r) Model.t
  -> ('w, 'd, 'l, 'p, 'e, 'r) Model.params
  -> ?beta:float
  -> posterior:'w Recognition.posterior
  -> 'w value
  -> 'w t

(** [loss model params ?iters ~key ~n_samples data] is the negated bound:
    the recognition's draws and {!compute}, in one call.

    [iters] is passed to the recognition: [Some n] for [n] fixed iLQR
    iterations — tensor-only control flow, traceable and JIT-compilable
    together with the solve's implicit gradient — and [None] for the
    adaptive solve. An LFADS recognition ignores it, since its encoder is a
    plain feedforward computation. *)
val loss
  :  ('w, 'd, 'l, 'p, 'e, 'r) Model.t
  -> ('w, 'd, 'l, 'p, 'e, 'r) Model.params
  -> ?beta:float
  -> ?iters:int
  -> key:Nx.Rng.t
  -> n_samples:int
  -> 'w value
  -> 'w value

(** [loss_fixed model params ~iters ~key ~n_samples data] is {!loss} with
    [iters] required: the traceable and JIT-compilable objective. *)
val loss_fixed
  :  ('w, 'd, 'l, 'p, 'e, 'r) Model.t
  -> ('w, 'd, 'l, 'p, 'e, 'r) Model.params
  -> ?beta:float
  -> iters:int
  -> key:Nx.Rng.t
  -> n_samples:int
  -> 'w value
  -> 'w value

(** [batch_loss_fixed model params ~key ~n_samples ~iters trials] is the mean
    of {!loss_fixed} over the trials. The key is split into one subkey per
    trial with [Nx.Rng.split], so the trials' draws are independent and the
    result is deterministic in [key].

    This is the mini-batch objective: [Rune.value_and_grad (Model.structure model) (fun p -> batch_loss_fixed model p …) params] gives the gradient
    over the batch in one pass. *)
val batch_loss_fixed
  :  ('w, 'd, 'l, 'p, 'e, 'r) Model.t
  -> ('w, 'd, 'l, 'p, 'e, 'r) Model.params
  -> ?beta:float
  -> key:Nx.Rng.t
  -> n_samples:int
  -> iters:int
  -> 'w value array
  -> 'w value

(** [log_prior model params ~x0 ~u] is the model's log prior at a latent
    sample: [log p(x_0) + Σ_k log p(u_k)], one value per member of a batch.
    It is the term the bound contrasts with [log q]. *)
val log_prior
  :  ('w, 'd, 'l, 'p, 'e, 'r) Model.t
  -> ('w, 'd, 'l, 'p, 'e, 'r) Model.params
  -> x0:'w value
  -> u:'w value
  -> 'w value
