(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** iLQR-VAE on Raven: latent input inference with a differentiable iLQR
    encoder.

    {b The model.} A trial is a tensor of observations [[n_steps; n_output]].
    Inputs [u_k] drive a state [z_k] through a dynamics ([z_0 = 0],
    [z_{k+1} = f(z_k, u_k)]); each observation is emitted from the state
    reached by the inputs before it, [o_k ~ p(o | z_{k+1})]. The generative
    model is completed by a factorized prior over inputs, [p(u)], and the
    recognition distribution is a Gaussian [q(u) = N(μ_u, Σ_time ⊗ Σ_space)]
    over input sequences. {!Model} bundles the pieces; {!Model.params} bundles
    their parameters, whose structure {!Model.structure} is what
    {!Rune.grad}, {!Rune.jit} and the optimizers take.

    {b The encoder.} [μ_u] solves a finite-horizon control problem whose cost
    is [−log p̃(u) − log p(o | z(u))] with a Gaussian encoder prior [p̃]
    ({!Encoder}). The solve is iLQR, and {!posterior_mean} differentiates
    through it with {!Dilqr}'s implicit VJP: a perturbed Riccati pass and a
    linearized rollout solve the adjoint KKT system of the local QP, so the
    gradient does not unroll the iterations.

    {b The objective.} The ELBO is estimated with reparameterized draws from
    [q] ({!Elbo}); its KL is exact for Gaussian priors and analytic-entropy
    plus a sampled prior term otherwise. {!loss_fixed} is encoder and ELBO in
    one call, with fixed-iteration iLQR, and is pure tensor computation: put
    it inside {!Rune.jit} with a {!Rune.grad} inside and the solve and its
    implicit gradient compile together, on CPU or GPU.

    {b A training step.}

    {[
    let model = … Model.make …
    let params = Model.init model in
    let step =
      Rune.jit
        Nx.Ptree.(Model.structure model @-> Nx.Rng.ptree @-> returns Nx.Ptree.tensor)
        (fun params key ->
           Rune.value_and_grad
             (Model.structure model)
             (fun p ->
                Elbo.batch_loss_fixed model p ~key ~n_samples:16 ~iters:4 trials)
             params)
    in
    ]}

    {b Sampling and prediction.} {!sample_generative} draws from the prior and
    rolls the dynamics out; {!predictions} draws from the recognition
    distribution and decodes the states; {!integrate} is the deterministic
    rollout.

    {b Everything else.} The pieces are exposed as their own modules —
    {!Types}, {!Covariance}, {!Prior}, {!Likelihood}, {!Dynamics}, {!Model},
    {!Recognition}, {!Encoder}, {!Elbo} — and the functions below are the
    front page. There is no functor and no parameter framework: a model is a
    value, its parameters are a value, and a transformation is a function. *)

(** The tensor type every function here takes: [float] payload, ['w] layout. *)
type 'w value = (float, 'w) Nx.t

(** A model: dimensions, dynamics, likelihood, priors, covariances. See
    {!module:Model}. *)
type ('w, 'd, 'l, 'p, 'e) model = ('w, 'd, 'l, 'p, 'e) Model.t

(** A model's parameters. See {!module:Model}. *)
type ('w, 'd, 'l, 'p, 'e) params = ('w, 'd, 'l, 'p, 'e) Model.params

module Types = Types
module Covariance = Covariance
module Prior = Prior
module Likelihood = Likelihood
module Dynamics = Dynamics
module Model = Model
module Recognition = Recognition
module Encoder = Encoder
module Elbo = Elbo

(** {1:inference Inference} *)

(** The recognition mean by adaptive iLQR; eager only. See
    {!Encoder.posterior_mean}. *)
val posterior_mean
  :  ('w, 'd, 'l, 'p, 'e) model
  -> ('w, 'd, 'l, 'p, 'e) params
  -> ?u_init:'w value
  -> ?max_iter:int
  -> ?conv_threshold:float
  -> 'w value
  -> 'w value

(** The recognition mean by fixed-iteration iLQR; traceable. See
    {!Encoder.posterior_mean_fixed}. *)
val posterior_mean_fixed
  :  ('w, 'd, 'l, 'p, 'e) model
  -> ('w, 'd, 'l, 'p, 'e) params
  -> ?u_init:'w value
  -> iters:int
  -> ?alphas:float list
  -> ?mu:float
  -> ?c1:float
  -> 'w value
  -> 'w value

(** The estimated evidence lower bound, a scalar. See {!Elbo.compute} for the
    bound's terms. *)
val elbo
  :  ('w, 'd, 'l, 'p, 'e) model
  -> ('w, 'd, 'l, 'p, 'e) params
  -> ?beta:float
  -> mu_u:'w value
  -> key:Nx.Rng.t
  -> n_samples:int
  -> 'w value
  -> 'w value

(** The negated bound, the per-trial objective. *)
val loss
  :  ('w, 'd, 'l, 'p, 'e) model
  -> ('w, 'd, 'l, 'p, 'e) params
  -> ?beta:float
  -> mu_u:'w value
  -> key:Nx.Rng.t
  -> n_samples:int
  -> 'w value
  -> 'w value

(** Encoder and bound in one call, fixed-iteration and traceable. See
    {!Elbo.loss_fixed}. *)
val loss_fixed
  :  ('w, 'd, 'l, 'p, 'e) model
  -> ('w, 'd, 'l, 'p, 'e) params
  -> ?beta:float
  -> key:Nx.Rng.t
  -> n_samples:int
  -> iters:int
  -> ?u_init:'w value
  -> 'w value
  -> 'w value

(** The mean of {!loss_fixed} over a batch of trials. See
    {!Elbo.batch_loss_fixed}. *)
val batch_loss_fixed
  :  ('w, 'd, 'l, 'p, 'e) model
  -> ('w, 'd, 'l, 'p, 'e) params
  -> ?beta:float
  -> key:Nx.Rng.t
  -> n_samples:int
  -> iters:int
  -> 'w value array
  -> 'w value

(** {1:sampling Sampling and prediction} *)

(** [integrate model params ~u] rolls the model's dynamics out from the zero
    initial state under the inputs [u] ([[n_steps; m]], or a sample batch
    [[n_steps; n_samples; m]]): states [[n_steps + 1; …; n]]. Differentiable
    in [u] and in the parameters. *)
val integrate
  :  ('w, 'd, 'l, 'p, 'e) model
  -> ('w, 'd, 'l, 'p, 'e) params
  -> u:'w value
  -> 'w value

(** [sample_generative model params ~key] draws a trial from the generative
    model: inputs from the prior ([[n_steps; m]]), their state trajectory
    ([[n_steps + 1; n]]) and observations sampled from the likelihood
    ([[n_steps; n_output]]), from the states the observations belong to
    ([z_1 … z_T]). The key is split for the two draws. *)
val sample_generative
  :  ('w, 'd, 'l, 'p, 'e) model
  -> ('w, 'd, 'l, 'p, 'e) params
  -> key:Nx.Rng.t
  -> 'w value * 'w value * 'w value

(** [predictions model params ~mu ~n_samples ~key] draws [n_samples] inputs
    from the recognition distribution at [mu], rolls them out and decodes
    them: inputs [[n_steps; n_samples; m]], states
    [[n_steps + 1; n_samples; n]] and per-sample readouts
    [[n_steps; n_samples; n_output]], decoded from the states the observations
    belong to ([z_1 … z_T]). The readouts are the likelihood's deterministic
    [mean], not samples, so they are noise-free predictions. *)
val predictions
  :  ('w, 'd, 'l, 'p, 'e) model
  -> ('w, 'd, 'l, 'p, 'e) params
  -> mu:'w value
  -> n_samples:int
  -> key:Nx.Rng.t
  -> 'w value * 'w value * 'w value

(** [predictions_deterministic model params ~mu] rolls the mean input sequence
    out and decodes it: states [[n_steps + 1; n]] and readouts
    [[n_steps; n_output]], decoded from [z_1 … z_T]. *)
val predictions_deterministic
  :  ('w, 'd, 'l, 'p, 'e) model
  -> ('w, 'd, 'l, 'p, 'e) params
  -> mu:'w value
  -> 'w value * 'w value
