(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** iLQR-VAE on Raven: latent-input inference with a swappable recognition
    model.

    {b The generative model.} A trial is a tensor of observations
    [[n_steps; n_output]]. Latents [x_0] (the initial state, [[n]]) and
    [u_0 … u_{T−1}] (the inputs, [[n_steps; m]]) drive a state through the
    dynamics ([z_{k+1} = f(z_k, u_k)]); each observation is emitted from the
    state reached by the inputs before it, [o_k ~ p(o | z_{k+1})]. The
    generative model is completed by a factorized prior over inputs [p(u)],
    an optional prior over the initial state [p(x_0)], and — when the input
    is used — an encoder prior [p̃] that shapes the iLQR cost. {!Model}
    bundles the pieces; {!Model.params} bundles their parameters, whose
    structure {!Model.structure} is what {!Rune.grad}, {!Rune.jit} and the
    optimizers take.

    {b The recognition model.} [q(x_0, u | o)] is a {!Recognition.t}: a value
    with its own parameters. Two are provided and the model takes either as
    [~recognition]:

    - {!Recognition.Iilqr}: the iLQR-VAE recognition. [x_0 = 0], and the
      mean of [u] solves a finite-horizon optimal control problem whose cost
      is [−log p̃(u) − log p(o | z(u))]; the covariance is a Kronecker
      product of a factor over time and a factor over inputs
      ([Σ_time ⊗ Σ_space]). {!Dilqr} solves the problem and differentiates
      it with an implicit VJP, so the gradient does not unroll the
      iterations.
    - {!Recognition.Lfads}: an LFADS-style recognition. Bidirectional
      encoder RNNs infer a Gaussian posterior over [x_0] and a second
      encoder–controller pair produces an autoregressive Gaussian posterior
      over the inputs; every RNN is {!Dynamics.Mgu}.

    Swapping them is changing one argument of {!Model.make}: the objective,
    the sampling, the training step and the JIT compilation are unchanged.

    {b The objective.} The ELBO is estimated with reparameterized draws from
    [q] ({!Elbo}): the likelihood of the rolled-out states, the model's log
    prior and [log q] at the same draws. The KL is sampled rather than
    analytic, which is what lets [q] be any distribution a recognition can
    sample and score — a diagonal Gaussian per step, an autoregressive one,
    a space-time Gaussian. Differentiating the estimate through the draws is
    the pathwise gradient of Roeder, Wu & Duvenaud (2017), "Sticking the
    landing".

    {b A training step.}

    {[
      let model = … Model.make … in
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

    {b Sampling and prediction.} {!sample_generative} draws from the prior
    and rolls the dynamics out; {!predictions} rolls the recognition's own
    draws out and decodes them; {!integrate} is the deterministic rollout.

    {b Everything else.} The pieces are exposed as their own modules —
    {!Types}, {!Covariance}, {!Initial}, {!Prior}, {!Likelihood},
    {!Dynamics}, {!Schedule}, {!Recognition}, {!Model}, {!Elbo} — and the
    functions below are the front page. There is no functor and no parameter framework: a
    model is a value, its parameters are a value, and a transformation is a
    function. *)

(** The tensor type every function here takes: [float] payload, ['w] layout. *)
type 'w value = (float, 'w) Nx.t

(** A model: dimensions, dynamics, likelihood, priors, initial-state prior
    and recognition. See {!module:Model}. *)
type ('w, 'd, 'l, 'p, 'e, 'r) model = ('w, 'd, 'l, 'p, 'e, 'r) Model.t

(** A model's parameters, the recognition's included. See {!module:Model}. *)
type ('w, 'd, 'l, 'p, 'e, 'r) params = ('w, 'd, 'l, 'p, 'e, 'r) Model.params

module Types = Types
module Covariance = Covariance
module Initial = Initial
module Prior = Prior
module Likelihood = Likelihood
module Dynamics = Dynamics
module Schedule = Schedule
module Recognition = Recognition
module Model = Model
module Elbo = Elbo

(** {1:inference Inference} *)

(** [posterior_mean model params ?iters data] is the recognition's posterior
    mean: one deterministic latent trajectory ([x0] of shape [[1; n]] and
    [u] of shape [[n_steps; 1; m]]). [iters] asks for that many fixed iLQR
    iterations; [None] uses the adaptive solve. See {!Recognition.mean}. *)
val posterior_mean
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> ?iters:int
  -> 'w value
  -> 'w Recognition.samples

(** [posterior_draws model params ~n_samples ~key data] draws [n_samples]
    latent trajectories from the recognition distribution and evaluates
    [log q] at each. This is the [q] the ELBO is built from. See
    {!Recognition.infer}. *)
val posterior_draws
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> ?iters:int
  -> n_samples:int
  -> key:Nx.Rng.t
  -> 'w value
  -> 'w Recognition.posterior

(** [log_prob model params data samples] is [log q(samples | data)] for given
    latents, one value per member of a batch. See {!Recognition.log_prob}. *)
val log_prob
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> ?iters:int
  -> 'w value
  -> 'w Recognition.samples
  -> 'w value

(** The estimated evidence lower bound, a scalar, from a posterior's draws.
    See {!Elbo.compute} for the bound's terms. *)
val elbo
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> ?beta:'w value
  -> posterior:'w Recognition.posterior
  -> 'w value
  -> 'w value

(** The negated bound from fresh draws: the per-trial objective. See
    {!Elbo.loss}. *)
val loss
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> ?beta:'w value
  -> ?iters:int
  -> key:Nx.Rng.t
  -> n_samples:int
  -> 'w value
  -> 'w value

(** Encoder and bound with fixed-iteration inference: traceable. See
    {!Elbo.loss_fixed}. *)
val loss_fixed
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> ?beta:'w value
  -> iters:int
  -> key:Nx.Rng.t
  -> n_samples:int
  -> 'w value
  -> 'w value

(** The mean of {!loss_fixed} over a batch of trials. See
    {!Elbo.batch_loss_fixed}. *)
val batch_loss_fixed
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> ?beta:'w value
  -> key:Nx.Rng.t
  -> n_samples:int
  -> iters:int
  -> 'w value array
  -> 'w value

(** {1:sampling Sampling and prediction} *)

(** [integrate model params ~u] rolls the model's dynamics out from
    [x_0 = 0] under the inputs [u] ([[n_steps; m]], or a sample batch
    [[n_steps; n_samples; m]]): states [[n_steps + 1; …; n]]. Differentiable
    in [u] and in the parameters. {!integrate_from} starts from a given
    initial state. *)
val integrate
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> u:'w value
  -> 'w value

(** [integrate_from model params ~x0 ~u] is {!integrate} from [x0]
    ([[n]], or a batch [[n_samples; n]]). *)
val integrate_from
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> x0:'w value
  -> u:'w value
  -> 'w value

(** [sample_generative model params ~key] draws a trial from the generative
    model: the initial state from its prior (or zero, when the model has
    none; [[n]]), inputs from the prior ([[n_steps; m]]), their state
    trajectory ([[n_steps + 1; n]]) and observations sampled from the
    likelihood ([[n_steps; n_output]]), from the states the observations
    belong to. The key is split for the draws. *)
val sample_generative
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> key:Nx.Rng.t
  -> 'w value * 'w value * 'w value * 'w value

(** [predictions model params posterior] rolls the posterior's draws out and
    decodes them: the inputs ([[n_steps; n_samples; m]]), the states
    ([[n_steps + 1; n_samples; n]]) and the per-sample readouts
    ([[n_steps; n_samples; n_output]]), decoded from the states the
    observations belong to. The readouts are the likelihood's deterministic
    [mean], not samples. *)
val predictions
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> 'w Recognition.posterior
  -> 'w value * 'w value * 'w value

(** [predictions_deterministic model params samples] rolls a deterministic
    latent trajectory out and decodes it: states [[n_steps + 1; n]] and
    readouts [[n_steps; n_output]] for a batch of one. *)
val predictions_deterministic
  :  ('w, 'd, 'l, 'p, 'e, 'r) model
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> 'w Recognition.samples
  -> 'w value * 'w value
