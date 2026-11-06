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

    A recognition is a description with parameters of its own type: the
    constructor returns both, the parameters are a field of {!Model.params}
    and are learned with the rest of the model. Swapping the two is changing
    the [~recognition] argument of {!Model.make}; nothing else in the model or
    in {!Elbo} knows which one is in use.

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
    dynamics, the likelihood and the encoder prior. A subset of {!Model.t},
    built by {!Model.generative}: the components appear as their descriptions,
    whose maps and structures a recognition reads, and the parameters those
    maps take arrive with {!type-gparams}. The generative prior over inputs is
    not part of this view (only {!Elbo} reads it), and neither is the
    initial-state prior, which is a parameter. *)
type ('w, 'd, 'l, 'e, 's) generative =
  { dtype : (float, 'w) Nx.dtype
  ; n : int
  ; m : int
  ; n_steps : int
  ; n_output : int
  ; dynamics : ('w, 'd, 's) Dynamics.t
  ; likelihood : ('w, 'l, 's) Likelihood.t
  ; encoder_prior : ('w, 'e, 's) Prior.t (** the prior of the iLQR cost, [p̃] *)
  }

(** The corresponding parameters: a subset of {!Model.params}. The device set
    ['s] is that of the initial-state prior's tensors, which the encoder
    prices.

    [initial_prior] is present exactly when the encoder infers an initial
    state: the prior is then the boundary condition of the encoder's control
    problem, so it is a term of the cost the solve differentiates. *)
type ('w, 'd, 'l, 'e, 's) gparams =
  { dynamics : 'd
  ; likelihood : 'l
  ; encoder_prior : 'e
  ; initial_prior : (('w, 's) value, 's) Initial.params option
  }

(** One reparameterized draw of a trial's latents: the initial state, of shape
    [[n_samples; n]], and the inputs, of shape [[n_steps; n_samples; m]].

    A mini-batch of trials adds leading trial axes to every tensor: [x0] is
    [[batch…; n_samples; n]], [u] is [[batch…; n_steps; n_samples; m]] and
    [log_prob] is [[batch…; n_samples]]. Trial axes lead, then the horizon,
    then the sample axis. *)
type ('w, 's) samples =
  { x0 : ('w, 's) value
  ; u : ('w, 's) value
  }

(** A draw together with the value of [log q] at it: [x0] and [u] as in
    {!type-samples}, and [log_prob] of shape [[n_samples]], one scalar per
    draw, or [[batch…; n_samples]] for a mini-batch. *)
type ('w, 's) posterior =
  { x0 : ('w, 's) value
  ; u : ('w, 's) value
  ; log_prob : ('w, 's) value
  }

(** The Gaussian the encoder's local quadratic model puts on the initial
    state: a mean and a precision, both [[n]]. Its inverse is the covariance
    the analogous linear model's posterior would have, and it is what an
    encoder that infers the initial state draws from, up to the recognition's
    one learned scale. *)
type ('w, 's) initial =
  { mean : ('w, 's) value
  ; precision : ('w, 's) value
  }

(** The dimensions a recognition is built for. {!Model.make} takes [n_steps]
    from here — no other component knows the horizon — and checks the rest
    against the other components. *)
type dims =
  { n : int
  ; m : int
  ; n_steps : int
  ; n_output : int
  }

(** How a recognition whose mean solves an iterative problem reaches it.

    Build one with {!adaptive} or {!fixed} and pass it as [~solve] wherever a
    recognition is asked for a mean. The choice is a value rather than the
    presence or absence of an iteration count, because it selects behaviour:
    the adaptive solve stops on a host-side test and runs eagerly, the fixed
    one runs a known number of iterations with tensor-only control flow and is
    traceable. A recognition without an iterative inner solve ignores it. *)
type solve

(** [adaptive ?max_iter ?conv_threshold ()] stops when the solve converges,
    with the solver's own defaults. Its stopping rule reads scalars on the
    host, so it cannot be traced or compiled.

    Raises [Invalid_argument] if [max_iter] is not positive. *)
val adaptive : ?max_iter:int -> ?conv_threshold:float -> unit -> solve

(** [fixed ?alphas ?mu ?c1 ?conv_threshold ~iters ()] runs at most [iters]
    iterations with tensor-only control flow: traceable and compilable together
    with its implicit gradient. The inner solve stops earlier when it has
    converged, as tightly as [conv_threshold] asks, and a batch of trials stops
    when its last member has. [alphas] is the line-search step list, [mu] the
    Levenberg-Marquardt damping (adaptive when omitted) and [c1] the line-search
    constant; each defaults to the solver's own.

    Raises [Invalid_argument] if [iters] is not positive. *)
val fixed
  :  ?alphas:float list
  -> ?mu:float
  -> ?c1:float
  -> ?conv_threshold:float
  -> iters:int
  -> unit
  -> solve

(** [default] is {!adaptive} with every default: the solver a recognition uses
    when the caller passes no [~solve]. *)
val default : solve

(** A recognition: a description, read through the accessors below and built by
    {!make} or by one of the recognitions further down. Its three functions take
    the generative model and its parameters as values, and the recognition's own
    parameters explicitly, so a recognition built once is applied at whatever
    parameters a training step reaches. It holds no parameter value and no
    tensor. *)
type ('w, 'd, 'l, 'e, 'r, 's) t

(** [make ~dims ~infers_initial ~infers_inputs ~name ~structure ~init ~infer ~mean ~log_prob ()] is the recognition those describe, and the way to write
    one the library does not ship. [dims] is the shape of model it is built for,
    [infers_initial] and [infers_inputs] say which factors [q] carries (what
    {!Model.make} and {!Elbo} check), [structure] is how its parameters are
    walked, [init] draws them, and the three maps are the posterior. *)
val make
  :  dims:dims
  -> infers_initial:bool
  -> infers_inputs:bool
  -> name:string
  -> structure:('r, 's) Nx.Ptree.t
  -> init:(?key:'s Nx.Rng.key -> (float, 'w) Nx.dtype -> 'r)
  -> infer:
       (('w, 'd, 'l, 'e, 's) generative
        -> ('w, 'd, 'l, 'e, 's) gparams
        -> params:'r
        -> ?solve:solve
        -> n_samples:int
        -> key:'s Nx.Rng.key
        -> ('w, 's) value
        -> ('w, 's) posterior)
  -> mean:
       (('w, 'd, 'l, 'e, 's) generative
        -> ('w, 'd, 'l, 'e, 's) gparams
        -> params:'r
        -> ?solve:solve
        -> ('w, 's) value
        -> ('w, 's) samples)
  -> log_prob:
       (('w, 'd, 'l, 'e, 's) generative
        -> ('w, 'd, 'l, 'e, 's) gparams
        -> params:'r
        -> ?solve:solve
        -> ('w, 's) value
        -> ('w, 's) samples
        -> ('w, 's) value)
  -> unit
  -> ('w, 'd, 'l, 'e, 'r, 's) t

(** {1:reading Reading a recognition} *)

(** [dims r] is the shape of model it is built for. *)
val dims : ('w, 'd, 'l, 'e, 'r, 's) t -> dims

(** [infers_initial r] is [true] when [q] carries an initial-state factor, so
    that {!Model.make} can require a matching initial-state prior. *)
val infers_initial : ('w, 'd, 'l, 'e, 'r, 's) t -> bool

(** [infers_inputs r] is [true] when [q] carries an input factor. A recognition
    that leaves the inputs at the point mass at zero has none: {!Elbo} then drops
    the input prior's term, because pricing a point mass is not part of the
    bound. *)
val infers_inputs : ('w, 'd, 'l, 'e, 'r, 's) t -> bool

(** [name r] is a short name, for diagnostics and printing. *)
val name : ('w, 'd, 'l, 'e, 'r, 's) t -> string

(** [structure r] is the [Nx.Ptree.t] of its parameters. *)
val structure : ('w, 'd, 'l, 'e, 'r, 's) t -> ('r, 's) Nx.Ptree.t

(** [init ?key dtype r] draws the parameters a fresh model starts from, at the
    placement the description was made at. [key] defaults to the current random
    scope. *)
val init : ?key:'s Nx.Rng.key -> (float, 'w) Nx.dtype -> ('w, 'd, 'l, 'e, 'r, 's) t -> 'r

(** [infer rec gen gp ~params ~n_samples ~key data] draws [n_samples] latent
    trajectories from [q(· | data)] and evaluates [log q] at each of them.

    [data] is one trial ([[n_steps; n_output]]) or a mini-batch
    ([[batch…; n_steps; n_output]]). A mini-batch is one tensor argument: the
    recognition applies one trial's encoder to each of its leading axes — one
    {!Rune.vmap} when the solve is fixed, one host-side iteration per trial
    when it is adaptive — and the key is split into one subkey per trial, so
    their draws are independent. The result carries the batch axes as
    documented at {!type-posterior}.

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
    test suite does.

    [solve] selects how a recognition with an iterative inner solver reaches
    its mean; see {!type-solve}. The adaptive solve cannot run inside
    {!Rune.vmap}, so a mini-batch is solved one trial at a time with it and
    with one map of the whole batch when the solve is {!fixed}. *)
val infer
  :  ('w, 'd, 'l, 'e, 'r, 's) t
  -> ('w, 'd, 'l, 'e, 's) generative
  -> ('w, 'd, 'l, 'e, 's) gparams
  -> params:'r
  -> ?solve:solve
  -> n_samples:int
  -> key:'s Nx.Rng.key
  -> ('w, 's) value
  -> ('w, 's) posterior

(** [mean rec gen gp ~params data] is the posterior mean: a single
    deterministic latent trajectory per trial, shaped like a sample batch of
    one ([x0] of shape [[1; n]] and [u] of shape [[n_steps; 1; m]] for one
    trial, and [[batch…; 1; n]] and [[batch…; n_steps; 1; m]] for a
    mini-batch). *)
val mean
  :  ('w, 'd, 'l, 'e, 'r, 's) t
  -> ('w, 'd, 'l, 'e, 's) generative
  -> ('w, 'd, 'l, 'e, 's) gparams
  -> params:'r
  -> ?solve:solve
  -> ('w, 's) value
  -> ('w, 's) samples

(** [log_prob rec gen gp ~params data s] is [log q(s | data)], one value per
    member of a sample batch, with live parameters: the same value
    {!val-infer} returns, but differentiable through the density's parameters
    as well as through the latents. [data] and [s] may carry trial axes, as
    for {!val-infer}. {!val-infer} returns [log q] at its draws without a
    second pass over the data, so this is for latents from elsewhere: an
    importance-weighted bound, a fixed set of draws, a diagnostic. *)
val log_prob
  :  ('w, 'd, 'l, 'e, 'r, 's) t
  -> ('w, 'd, 'l, 'e, 's) generative
  -> ('w, 'd, 'l, 'e, 's) gparams
  -> params:'r
  -> ?solve:solve
  -> ('w, 's) value
  -> ('w, 's) samples
  -> ('w, 's) value

(** [gparams_structure gen] is the [Nx.Ptree.t] of {!type-gparams} at [gen],
    for a caller that differentiates through a recognition's own use of the
    generative parameters. *)
val gparams_structure
  :  ('w, 'd, 'l, 'e, 's) generative
  -> (('w, 'd, 'l, 'e, 's) gparams, 's) Nx.Ptree.t

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

    {b The initial state.} With [infer_initial] (default [false]) the initial
    state is a latent and [x_0 = 0] is replaced by "x_0 is free, priced by the
    model's {!Initial} prior", which the encoder's problem then carries as its
    step-0 running loss. The mean is not a variable of the solve: it is a
    variational parameter whose value is the joint mode of the local quadratic
    model, reached in a fixed number of rounds of "solve from the current
    estimate, then take the Laplace mean the new model implies". The
    covariance is [exp (2 log_scale) · P⁻¹] with [P] the Hessian of the
    encoder's cost in [x_0], which is the exact posterior precision of a linear
    model and the Gauss-Newton one otherwise.

    Both are values rather than functions of the parameters the bound
    follows: the draws and the density take them stopped, as they take the two
    covariance factors, and [log_scale] is the block's one free parameter and
    the only part of it that carries a gradient. The term left out is the
    bound's derivative in the distribution's own mean and precision, which is
    zero where the bound is stationary in [q]; following it would mean
    differentiating the Riccati recursion behind [P], or making [x_0] a
    variable of the solve.

    The parameters are learned by the ELBO's gradient, and the gradient
    reaches the model's dynamics, likelihood, encoder prior and initial-state
    prior through {!Dilqr}'s implicit VJP of the solve. Circularity is
    confined to the rounds: [x_0] is held fixed while the solve is
    differentiated. *)
module Iilqr : sig
  (** The recognition's parameters: the Cholesky factors of the two
      covariances, exactly as {!Covariance.params}. *)
  type ('a, 's) params =
    { space_cov : ('a, 's) Covariance.params (** [[m]]-dimensional factor *)
    ; time_cov : ('a, 's) Covariance.params (** [[n_steps]]-dimensional factor *)
    ; initial_log_scale : 'a option
      (** a scalar when the initial state is inferred: the log scale of its
          covariance, [exp (2 log_scale) · P⁻¹]. Absent when it is fixed. *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype. *)
  val structure : unit -> ((('a, 'b, 's) Nx.t, 's) params, 's) Nx.Ptree.t

  (** [place ~at p] is [p]'s tensors at [at]. *)
  val place
    :  at:'e Nx.Placement.t
    -> (('w, 's) value, 's) params
    -> (('w, 'e) value, 'e) params

  (** [problem gen ?verbose data] is the encoder's control problem for one
      trial, as a {!Dilqr.Problem.t} whose parameters are {!type-gparams}.
      [verbose] is the solver's, off by default. *)
  val problem
    :  ('w, 'd, 'l, 'e, 's) generative
    -> ?verbose:bool
    -> ('w, 's) value
    -> ('w, ('w, 'd, 'l, 'e, 's) gparams, 's) Dilqr.Problem.t

  (** [posterior_mean gen gp ?u_init ?solve ?rounds data] is the iLQR
      solution's controls, of shape [[n_steps; m]] — the recognition mean.
      [u_init] ([n_steps; m], zeros when omitted) is the initial guess and
      carries no gradient: the idealized solve does not depend on it.
      Differentiable through {!Dilqr}'s implicit VJP of whichever solve
      {!type-solve} selects. *)
  val posterior_mean
    :  ('w, 'd, 'l, 'e, 's) generative
    -> ('w, 'd, 'l, 'e, 's) gparams
    -> ?u_init:('w, 's) value
    -> ?solve:solve
    -> ?rounds:int
    -> ('w, 's) value
    -> ('w, 's) value

  (** [posterior_initial gen gp ?u_init ?solve ?rounds data] is the Laplace
      posterior of the initial state — the mean and the precision of the
      Gaussian the local quadratic model implies for it — when the model has
      an initial-state prior, and [None] when it fixes [x_0 = 0].

      The mean is the joint mode of the encoder's cost over [x_0] and the
      inputs, i.e. the value the trajectory starts from once [rounds] rounds
      (default 2) of solve-then-replace have converged; the precision is the
      same cost's Hessian in [x_0], so its inverse is the covariance the
      analogous linear model's posterior would have.

      This is the diagnostic entry point as much as a component: it answers
      "what initial condition does the encoder want, given this trial", which
      is what the [u_0] channel of the fixed-zero encoder stood in for. *)
  val posterior_initial
    :  ('w, 'd, 'l, 'e, 's) generative
    -> ('w, 'd, 'l, 'e, 's) gparams
    -> ?u_init:('w, 's) value
    -> ?solve:solve
    -> ?rounds:int
    -> ('w, 's) value
    -> ('w, 's) initial option

  (** [make ?space_sigma2 ?time_sigma2 ~infer_initial ~n ~m ~n_steps ~n_output dtype] is the recognition, with covariance factors initialized as
      {!init} does. *)
  val make
    :  at:'s Nx.Placement.t
    -> ?space_sigma2:float
    -> ?time_sigma2:float
    -> ?infer_initial:bool
    -> n:int
    -> m:int
    -> n_steps:int
    -> n_output:int
    -> unit
    -> ('w, 'd, 'l, 'e, (('w, 's) value, 's) params, 's) t
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
  type ('a, 's) params =
    { ic_fwd : ('a, 's) Dynamics.Mgu.params (** initial-condition encoder, forward *)
    ; ic_bwd : ('a, 's) Dynamics.Mgu.params (** initial-condition encoder, backward *)
    ; ci_fwd : ('a, 's) Dynamics.Mgu.params (** controller-input encoder, forward *)
    ; ci_bwd : ('a, 's) Dynamics.Mgu.params (** controller-input encoder, backward *)
    ; con : ('a, 's) Dynamics.Mgu.params (** controller RNN *)
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
  val structure : unit -> ((('a, 'b, 's) Nx.t, 's) params, 's) Nx.Ptree.t

  (** [make ~at ?n_ic ?n_ci ?n_ctrl ?with_inputs ?phi ?key ~n ~m ~n_steps ~n_output dtype] is the recognition, its parameters made at [at].

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
    :  at:'s Nx.Placement.t
    -> ?n_ic:int
    -> ?n_ci:int
    -> ?n_ctrl:int
    -> ?with_inputs:bool
    -> ?phi:(('w, 's) value -> ('w, 's) value)
    -> n:int
    -> m:int
    -> n_steps:int
    -> n_output:int
    -> unit
    -> ('w, 'd, 'l, 'e, (('w, 's) value, 's) params, 's) t

  (** [place ~at p] is [p]'s tensors at [at]. *)
  val place
    :  at:'e Nx.Placement.t
    -> (('w, 's) value, 's) params
    -> (('w, 'e) value, 'e) params
end
