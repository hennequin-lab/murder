(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The recognition model — the [q] of the VAE.

    A recognition model is a distribution over a trial's latent variables
    conditioned on the trial's observations. The latents are the initial
    state [x_0] ([[n]]) and the inputs [u_0 … u_{T−1}] ([[n_steps; m]]); a
    model that fixes [x_0 = 0] and a model that infers it are both expressed
    here, and {!Model.t} carries whichever recognition the user built. Trial
    axes lead every tensor, then comes the horizon, then the sample axis.

    Two constructors are provided:

    - {!Iilqr}: the iLQR-VAE recognition. Its mean solves a finite-horizon
      optimal control problem with {!Dilqr} and its covariance is a Kronecker
      product of a factor over time and a factor over input dimensions. It
      fixes [x_0 = 0].
    - {!Lfads}: an LFADS-style recognition. Bidirectional encoder RNNs infer
      a Gaussian posterior over the initial state, and an encoder–controller
      pair produces an autoregressive Gaussian posterior over the inputs.
      Both use {!Dynamics.Mgu} as their recurrent cell.

    A recognition is a value with its own parameters: they are a field of
    {!Model.params} and are learned with the rest of the model. Swapping the
    two is changing the [~recognition] argument of {!Model.make}; nothing
    else in the model or in {!Elbo} knows which one is in use.

    {b The objective.} {!Elbo} draws from [q] with {!val-infer} and evaluates
    both [log q] (the returned [log_prob] field) and the model's
    log prior at the same draws. The bound is the Monte-Carlo estimate

    {[
      ELBO = (1/N) Σ_i [log p(o | z(x_0^i, u^i)) + log p(x_0^i) + Σ_k log p(u_k^i)
        − log q(x_0^i, u^i)]
    ]}

    with the reparameterized draws differentiated end to end. {!val-infer}
    evaluates [log q] with the recognition's distribution parameters stopped,
    which is the path-derivative gradient estimator of Roeder, Wu &
    Duvenaud (2017), "Sticking the landing": the value of the bound is
    unchanged, the gradient is the reparameterization gradient with its
    score-function part removed, and the variance of that gradient goes to
    zero as [q] approaches the exact posterior. *)

open Types

(** The generative model as a recognition sees it: the dimensions, the
    dynamics, the likelihood and the priors. A subset of {!Model.t}, built by
    {!Model.generative}. *)
type ('w, 'd, 'l, 'p, 'e) generative =
  { dtype : (float, 'w) Nx.dtype
  ; n : int
  ; m : int
  ; n_steps : int
  ; n_output : int
  ; dynamics : ('w, 'd) Dynamics.t
  ; likelihood : ('w, 'l) Likelihood.t
  ; prior : ('w, 'p) Prior.t (** the generative prior over inputs *)
  ; encoder_prior : ('w, 'e) Prior.t (** the prior of the iLQR cost, [p̃] *)
  }

(** The corresponding parameters: a subset of {!Model.params}. *)
type ('w, 'd, 'l, 'p, 'e) gparams =
  { dynamics : 'd
  ; likelihood : 'l
  ; prior : 'p
  ; encoder_prior : 'e
  }

(** One reparameterized draw of a trial's latents: the initial state, of shape
    [[n_samples; n]], and the inputs, of shape [[n_steps; n_samples; m]].

    A mini-batch of trials adds leading trial axes to every tensor: [x0] is
    [[batch…; n_samples; n]], [u] is [[batch…; n_steps; n_samples; m]] and
    [log_prob] is [[batch…; n_samples]]. Trial axes lead, then the horizon,
    then the sample axis. *)
type 'w samples =
  { x0 : 'w value
  ; u : 'w value
  }

(** A draw together with the value of [log q] at it: [x0] and [u] as in
    {!type-samples}, and [log_prob] of shape [[n_samples]], one scalar per
    draw, or [[batch…; n_samples]] for a mini-batch. *)
type 'w posterior =
  { x0 : 'w value
  ; u : 'w value
  ; log_prob : 'w value
  }

(** The dimensions a recognition is built for. {!Model.make} checks them
    against the model's. *)
type dims =
  { n : int
  ; m : int
  ; n_steps : int
  ; n_output : int
  }

(** A recognition model over parameters ['r]. The functions take the
    generative model and its parameters as values, and the recognition's own
    parameters explicitly, so a recognition built once is applied at whatever
    parameters a training step reaches. *)
type ('w, 'd, 'l, 'p, 'e, 'r) t =
  { dims : dims
  ; infers_initial : bool
  ; name : string (** a short name, for diagnostics and printing *)
  ; structure : 'r Nx.Ptree.t
  ; params : 'r (** the parameters a fresh model starts from *)
  ; infer :
      ('w, 'd, 'l, 'p, 'e) generative
      -> ('w, 'd, 'l, 'p, 'e) gparams
      -> params:'r
      -> ?iters:int
      -> n_samples:int
      -> key:Nx.Rng.t
      -> 'w value
      -> 'w posterior
  ; mean :
      ('w, 'd, 'l, 'p, 'e) generative
      -> ('w, 'd, 'l, 'p, 'e) gparams
      -> params:'r
      -> ?iters:int
      -> 'w value
      -> 'w samples
  ; log_prob :
      ('w, 'd, 'l, 'p, 'e) generative
      -> ('w, 'd, 'l, 'p, 'e) gparams
      -> params:'r
      -> ?iters:int
      -> 'w value
      -> 'w samples
      -> 'w value
  }

(** [infer rec gen gp ~params ~n_samples ~key data] draws [n_samples] latent
    trajectories from [q(· | data)] and evaluates [log q] at each of them.

    [data] is one trial ([[n_steps; n_output]]) or a mini-batch
    ([[batch…; n_steps; n_output]]). A mini-batch is one tensor argument: the
    recognition maps one trial's encoder over its leading axes, and the key
    is split into one subkey per trial, so their draws are independent. The
    result carries the batch axes as documented at {!type-posterior}.

    The returned density is evaluated with the recognition's own distribution
    parameters detached: its value is [log q], but its gradient reaches the
    recognition only through the sampled latents. That is the path-derivative
    (sticking-the-landing) estimator that {!Elbo} relies on; {!val-log_prob}
    evaluates the same density with live parameters, for objectives that
    differentiate through the density itself. The bound's value is the same
    either way; only the backward pass differs. In particular the gradient is
    an estimator of the bound's gradient rather than the derivative of one
    traced value at one key, so a finite-difference gradient check of the
    library objective will disagree — check a live-density objective, as the
    test suite does. *)
val infer
  :  ('w, 'd, 'l, 'p, 'e, 'r) t
  -> ('w, 'd, 'l, 'p, 'e) generative
  -> ('w, 'd, 'l, 'p, 'e) gparams
  -> params:'r
  -> ?iters:int
  -> n_samples:int
  -> key:Nx.Rng.t
  -> 'w value
  -> 'w posterior

(** [mean rec gen gp ~params data] is the posterior mean: a single
    deterministic latent trajectory per trial, shaped like a sample batch of
    one ([x0] of shape [[1; n]] and [u] of shape [[n_steps; 1; m]] for one
    trial, and [[batch…; 1; n]] and [[batch…; n_steps; 1; m]] for a
    mini-batch). *)
val mean
  :  ('w, 'd, 'l, 'p, 'e, 'r) t
  -> ('w, 'd, 'l, 'p, 'e) generative
  -> ('w, 'd, 'l, 'p, 'e) gparams
  -> params:'r
  -> ?iters:int
  -> 'w value
  -> 'w samples

(** [log_prob rec gen gp ~params data s] is [log q(s | data)], one value per
    member of a sample batch, with live parameters: the same value
    {!val-infer} returns, but differentiable through the density's parameters
    as well as through the latents. [data] and [s] may carry trial axes, as
    for {!val-infer}. {!val-infer} returns [log q] at its draws without a
    second pass over the data, so this is for latents from elsewhere: an
    importance-weighted bound, a fixed set of draws, a diagnostic. *)
val log_prob
  :  ('w, 'd, 'l, 'p, 'e, 'r) t
  -> ('w, 'd, 'l, 'p, 'e) generative
  -> ('w, 'd, 'l, 'p, 'e) gparams
  -> params:'r
  -> ?iters:int
  -> 'w value
  -> 'w samples
  -> 'w value

(** [gparams_structure gen] is the [Nx.Ptree.t] of {!type-gparams} at [gen],
    for a caller that differentiates through a recognition's own use of the
    generative parameters. *)
val gparams_structure
  :  ('w, 'd, 'l, 'p, 'e) generative
  -> ('w, 'd, 'l, 'p, 'e) gparams Nx.Ptree.t

(** {1 iLQR recognition} *)

(** The iLQR-VAE recognition, as its own constructor with its own parameters.

    The posterior mean is the solution of the finite-horizon optimal control
    problem

    {[
      min_{u_0 … u_{T−1}}  Σ_k l(x_k, u_k) + φ(x_T)
        l(x_k, u_k) = −log p̃(u_k) − [k ≥ 1] · log p(o_{k−1} | x_k)
        φ(x_T)      = −log p(o_{T−1} | x_T)
        x_0 = 0,  x_{k+1} = f(x_k, u_k)
    ]}

    and the posterior is [N(μ_u, Σ_time ⊗ Σ_space)]: the two covariance
    factors are the parameters of this module, so the recognition owns them.
    The initial state is deterministic at zero, so the model must be built
    without an initial-state prior.

    The parameters are learned by the ELBO's gradient, and the gradient
    reaches the model's dynamics, likelihood and encoder prior through
    {!Dilqr}'s implicit VJP of the solve. *)
module Iilqr : sig
  (** The recognition's parameters: the Cholesky factors of the two
      covariances, exactly as {!Covariance.params}. *)
  type 'a params =
    { space_cov : 'a Covariance.params (** [[m]]-dimensional factor *)
    ; time_cov : 'a Covariance.params (** [[n_steps]]-dimensional factor *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype. *)
  val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

  (** [init ?space_sigma2 ?time_sigma2 ~n ~m ~n_steps ~n_output dtype] is the
      initial parameters: factors of [space_sigma2 · I] and [time_sigma2 · I],
      both [1.] by default. Raises [Invalid_argument] if a dimension is not
      positive. *)
  val init
    :  ?space_sigma2:float
    -> ?time_sigma2:float
    -> n:int
    -> m:int
    -> n_steps:int
    -> n_output:int
    -> (float, 'w) Nx.dtype
    -> 'w value params

  (** [problem gen gp data] is the encoder's control problem for one trial,
      as a {!Dilqr.Problem.t} whose parameters are [gp]. *)
  val problem
    :  ('w, 'd, 'l, 'p, 'e) generative
    -> ('w, 'd, 'l, 'p, 'e) gparams
    -> 'w value
    -> ('w, ('w, 'd, 'l, 'p, 'e) gparams) Dilqr.Problem.t

  (** [posterior_mean gen gp ?u_init ?max_iter ?conv_threshold data] is the
      adaptive iLQR solution's controls, of shape [[n_steps; m]]. [u_init]
      ([n_steps; m], zeros when omitted) is the initial guess and carries no
      gradient. Differentiable through {!Dilqr.solve_adaptive}'s implicit VJP;
      eager only, so do not put it inside {!Rune.jit}. *)
  val posterior_mean
    :  ('w, 'd, 'l, 'p, 'e) generative
    -> ('w, 'd, 'l, 'p, 'e) gparams
    -> ?u_init:'w value
    -> ?max_iter:int
    -> ?conv_threshold:float
    -> 'w value
    -> 'w value

  (** [posterior_mean_fixed gen gp ?u_init ~iters ?alphas ?mu ?c1 data] is the
      same mean with [iters] fixed iterations: tensor-only control flow,
      traceable and JIT-compilable together with its implicit VJP. *)
  val posterior_mean_fixed
    :  ('w, 'd, 'l, 'p, 'e) generative
    -> ('w, 'd, 'l, 'p, 'e) gparams
    -> ?u_init:'w value
    -> iters:int
    -> ?alphas:float list
    -> ?mu:float
    -> ?c1:float
    -> 'w value
    -> 'w value

  (** [make ?space_sigma2 ?time_sigma2 ~n ~m ~n_steps ~n_output dtype] is the
      recognition, with covariance factors initialized as {!init} does. *)
  val make
    :  ?space_sigma2:float
    -> ?time_sigma2:float
    -> n:int
    -> m:int
    -> n_steps:int
    -> n_output:int
    -> (float, 'w) Nx.dtype
    -> ('w, 'd, 'l, 'p, 'e, 'w value params) t
end

(** {1 LFADS recognition} *)

(** The LFADS recognition (Pandarinath et al., 2018): bidirectional encoder
    RNNs infer the initial state, and an encoder–controller pair infers the
    inputs.

    {b Initial condition.} One encoder RNN reads the observations forward and
    one backward. Their final states are concatenated and a linear readout
    gives the mean and log-variance of a diagonal Gaussian over [x_0], whose
    prior is the model's {!Initial} prior.

    {b Controller.} A second pair of bidirectional encoder RNNs produces a
    feature at every step. A controller RNN runs forward in time over the
    concatenation of that feature with the previous generator state, and a
    linear readout of its state gives the mean and log-variance of a diagonal
    Gaussian over [u_t]. The posterior is therefore autoregressive in time,
    [q(x_0, u_{0:T−1}) = q(x_0) Π_t q(u_t | u_{<t}, x_{<t}, o)], and
    [log q] is exact.

    {b The recurrent cell.} Every RNN is a {!Dynamics.Mgu}, the primitive
    cell of this library, with the activation given by [~phi].

    {b Without inferred inputs.} [~with_inputs:false] leaves the controller
    out of the posterior: the inputs are the point mass at zero and only the
    initial state is inferred, the original LFADS model. *)
module Lfads : sig
  (** The recognition's parameters: the five cells, their learned initial
      states, and the two readouts. All are learned with the rest of the
      model. *)
  type 'a params =
    { ic_fwd : 'a Dynamics.Mgu.params (** initial-condition encoder, forward *)
    ; ic_bwd : 'a Dynamics.Mgu.params (** initial-condition encoder, backward *)
    ; ci_fwd : 'a Dynamics.Mgu.params (** controller-input encoder, forward *)
    ; ci_bwd : 'a Dynamics.Mgu.params (** controller-input encoder, backward *)
    ; con : 'a Dynamics.Mgu.params (** controller RNN *)
    ; ic_fwd_init : 'a (** [[n_ic]] *)
    ; ic_bwd_init : 'a (** [[n_ic]] *)
    ; ci_fwd_init : 'a (** [[n_ci]] *)
    ; ci_bwd_init : 'a (** [[n_ci]] *)
    ; con_init : 'a (** [[n_ctrl]] *)
    ; ic_readout : 'a (** [[2 n_ic; 2 n]]; the mean and log-variance of [x_0] *)
    ; ic_readout_bias : 'a (** [[2 n]] *)
    ; u_readout : 'a (** [[n_ctrl; 2 m]]; the mean and log-variance of [u_t] *)
    ; u_readout_bias : 'a (** [[2 m]] *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype. *)
  val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

  (** [make ?n_ic ?n_ci ?n_ctrl ?with_inputs ?phi ?key ~n ~m ~n_steps ~n_output dtype] is the recognition.

      - [n_ic] is the state size of the two initial-condition encoders
        (default [64]);
      - [n_ci] that of the controller-input encoders (default [n_ic]);
      - [n_ctrl] that of the controller (default [32]);
      - [with_inputs] (default [true]) selects whether the inputs are
        inferred at all;
      - [phi] (default {!Nx.tanh}) is the activation of every cell;
      - [key] seeds the initialization, and defaults to the current random
        scope.

      The encoder cells start from learned initial states at zero, and the
      input readout is initialized small so that the initial posterior is
      close to the model's input prior. Raises [Invalid_argument] if a size
      is not positive. *)
  val make
    :  ?n_ic:int
    -> ?n_ci:int
    -> ?n_ctrl:int
    -> ?with_inputs:bool
    -> ?phi:('w value -> 'w value)
    -> ?key:Nx.Rng.t
    -> n:int
    -> m:int
    -> n_steps:int
    -> n_output:int
    -> (float, 'w) Nx.dtype
    -> ('w, 'd, 'l, 'p, 'e, 'w value params) t
end
