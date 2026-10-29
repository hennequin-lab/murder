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
    fixed number of iLQR iterations, and {!batch_loss_fixed} is the same
    objective on a tensor mini-batch of trials. A caller that already has a
    posterior — reused across several bounds, or built with the adaptive iLQR
    solve — passes it to {!compute} directly.

    {b Trial batches.} A single trial's data is [[n_steps; n_output]]; a
    mini-batch stacks trials on leading axes, [[batch…; n_steps; n_output]],
    which is one tensor the caller can pass to a JIT-compiled step. The
    recognition's draws then carry the same leading axes: [x0] is
    [[batch…; n_samples; n]], [u] is
    [[n_steps; batch…; n_samples; m]] (time leads, as everywhere else in this
    library) and [log_prob] is [[batch…; n_samples]]. The bound and its terms
    are the means over every draw of every trial. The batch axes are tensor
    axes, not an OCaml array: their shape is part of the compiled program's
    key, but the number of trials is not frozen into a closure.

    {b The KL weight.} [beta] is a scalar tensor, so a warm-up (β from 0 to 1,
    the recipe that keeps an LFADS-style posterior from collapsing onto the
    prior before the decoder has learned anything) is a value of
    {!Schedule.at} passed in as an argument of the compiled step rather than
    a constant baked into it. *)

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

    [data] is one trial ([[n_steps; n_output]]) or a mini-batch
    ([[batch…; n_steps; n_output]]); [posterior] has the matching leading
    axes, as described in the module documentation. The posterior is what
    {!Recognition.infer} returns; it carries the draws and [log q] at them, so
    all three terms come from the same draws and one encoder pass. It carries
    gradients, so composing {!compute} with an inference differentiates the
    whole path. [beta] is a scalar tensor (default: a one of the model's
    dtype), not a host float, so a training step can take it as an argument
    and follow a {!Schedule} — or any other per-step weight — without
    recompiling under {!Rune.jit}.

    Raises [Invalid_argument] if the data and posterior shapes disagree or if
    the sample batch is empty. *)
val compute
  :  ('w, 'd, 'l, 'p, 'e, 'r) Model.t
  -> ('w, 'd, 'l, 'p, 'e, 'r) Model.params
  -> ?beta:'w value
  -> posterior:'w Recognition.posterior
  -> 'w value
  -> 'w t

(** [loss model params ?iters ~key ~n_samples data] is the negated bound:
    the recognition's draws and {!compute}, in one call.

    [iters] is passed to the recognition: [Some n] for [n] fixed iLQR
    iterations — tensor-only control flow, traceable and JIT-compilable
    together with the solve's implicit gradient — and [None] for the
    adaptive solve. An LFADS recognition ignores it, since its encoder is a
    plain feedforward computation.

    [data] is one trial or a tensor mini-batch, as for {!compute}. On a
    mini-batch the key is split into one subkey per trial, so the draws are
    independent and the result is deterministic in [key]. *)
val loss
  :  ('w, 'd, 'l, 'p, 'e, 'r) Model.t
  -> ('w, 'd, 'l, 'p, 'e, 'r) Model.params
  -> ?beta:'w value
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
  -> ?beta:'w value
  -> iters:int
  -> key:Nx.Rng.t
  -> n_samples:int
  -> 'w value
  -> 'w value

(** [batch_loss_fixed model params ~key ~n_samples ~iters data] is the mean
    of {!loss_fixed} over a tensor mini-batch of trials: [data] is
    [[batch…; n_steps; n_output]]. The key is split into one subkey per trial,
    so the trials' draws are independent and the result is deterministic in
    [key].

    This is the mini-batch objective. [data] is an argument of the compiled
    function rather than an OCaml array captured by it, so a JIT-compiled
    training step takes the mini-batch as a tensor:

    {[
      let step =
        Rune.jit
          Nx.Ptree.(Model.structure model @-> Nx.Rng.ptree @-> tensor @-> returns tensor)
          (fun params key data ->
             Elbo.batch_loss_fixed model params ~key ~n_samples:16 ~iters:4 data)
      in
      let loss = step params key batch
    ]}

    The batch axes are part of the compiled program's key, as every tensor
    shape is. *)
val batch_loss_fixed
  :  ('w, 'd, 'l, 'p, 'e, 'r) Model.t
  -> ('w, 'd, 'l, 'p, 'e, 'r) Model.params
  -> ?beta:'w value
  -> key:Nx.Rng.t
  -> n_samples:int
  -> iters:int
  -> 'w value
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
