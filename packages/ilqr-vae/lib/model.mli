(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The model, as a value.

    A model is everything the iLQR-VAE needs about the problem: its
    dimensions, its dynamics, its likelihood, its generative and encoder
    priors, and the recognition model that infers the latents. It carries no
    parameters at all: the parameters are a separate value
    ({!type-params}) whose structure {!structure} returns, and which is what
    {!Rune.grad}, {!Rune.jit} and the optimizers take. {!make} builds the
    description from its components and {!init} is the one draw.

    {b What is abstract, and why.} A model and its components —
    {!Dynamics.t}, {!Prior.t}, {!Likelihood.t}, {!Recognition.t} — are abstract
    descriptions: their dimensions, their parameter [structure] and their
    [init] are reached through accessors, so a description carries no parameter
    value and no tensor, and the recipe for drawing one travels with the
    component it belongs to. The types that are plain data stay records, because
    their fields {e are} the interface: {!type-params} (the value a training loop
    holds and updates), {!Initial.t} (three numbers), and the recognition's
    {!Recognition.generative} and {!Recognition.gparams} views.

    The components choose their own parameter records, so the model has a
    type parameter for each, the recognition included. Code that only passes
    a model around names none of them: they are inferred from {!make}.

    {[
    let model, params =
      Model.make
        Nx.float64
        ~dynamics:
          (Dynamics.Linear_nonlinear.make
             ~at:Nx.Host.on
             ~dt:0.01
             ~tau:0.04
             ~n:3
             ~nh:128
             ~m:3
             ~phi:Nx.tanh
             Nx.float64)
        ~likelihood:(Likelihood.Gaussian.make ~at:Nx.Host.on ~n:3 ~n_output:3 Nx.float64)
        ~prior:(Prior.Student.make ~at:Nx.Host.on ~m:3 Nx.float64)
        ~encoder_prior:(Prior.Gaussian.make ~at:Nx.Host.on ~m:3 Nx.float64)
        ~recognition:
          (Recognition.Iilqr.make
             ~at:Nx.Host.on
             ~n:3
             ~m:3
             ~n_steps:32
             ~n_output:3
             Nx.float64)
        ()
    in
    Rune.value_and_grad (Model.structure model) objective params
    ]}

    {b Swapping the recognition.} The recognition is a value with its own
    parameters, so replacing iLQR by an LFADS-style encoder is one argument:

    {[
      ~recognition:
        (Recognition.Lfads.make ~at:Nx.Host.on ~n_ic:64 ~n_ctrl:32 ~n:3 ~m:3 ~n_steps:32 ~n_output:3
           Nx.float64)
    ]}

    plus the initial-state prior the LFADS posterior needs:

    {[
      ~initial_prior:(Initial.make ~at:Nx.Host.on ~std:1.0 ~n:3 Nx.float64)
    ]}

    Everything else — the ELBO, the sampling, the training step, the JIT
    compilation — is the same.

    {b Device sets.} A model holds no tensors: its components describe maps and
    structures, and the parameters are the separate value {!init} returns.
    Every component constructor takes [~at], the placement its parameter value
    is made at, so a host model is built with [~at:Nx.Host.on] throughout and a
    model whose parameters live on a device is built with that placement
    instead — the components' structures and maps take the set the placement
    names. *)

open Types

(** A model: a description, read through the accessors below and built by
    {!make}. It holds no parameter value and no tensor — the parameters are the
    separate value {!type-params}, and {!init} is the one draw. *)
type ('w, 'd, 'l, 'p, 'e, 'r, 's) t

(** The parameters of a model: one value per component, in the component's own
    parameter type. {!structure} is the [Nx.Ptree.t] of this record, and this
    is the value a training loop holds and updates. *)
type ('w, 'd, 'l, 'p, 'e, 'r, 's) params =
  { dynamics : 'd
  ; likelihood : 'l
  ; prior : 'p
  ; encoder_prior : 'e
  ; initial_prior : (('w, 's) value, 's) Initial.params option
  ; recognition : 'r
  }

(** [make dtype ~dynamics ~likelihood ~prior ~encoder_prior ~recognition ?initial_prior ()] is the model those components describe. Each component is
    the description its own constructor returned, so a call site reads

    {[
      let model =
        Model.make
          dtype
          ~dynamics:(Dynamics.Linear.make ~at:Nx.Host.on ~n ~m ())
          ~likelihood:(Likelihood.Gaussian.make ~at:Nx.Host.on ~n ~n_output ())
          ~prior:(Prior.Gaussian.make ~at:Nx.Host.on ~m ())
          ~encoder_prior:(Prior.Gaussian.make ~at:Nx.Host.on ~m ())
          ~recognition:(Recognition.Iilqr.make ~at:Nx.Host.on ~n ~m ~n_steps ~n_output ())
          ()
      in
      let params = Model.init ~at:Nx.Host.on ~key model in
    ]}

    [initial_prior] is the description of the prior over [x_0]: it belongs to
    the parameters (the model keeps no tensor at all), it is required exactly
    when the recognition infers the initial state, and {!init} draws its value
    with the rest.

    {b No dimension arguments.} Every dimension the model needs is already
    reported by one of its components: [n] and [m] of the dynamics, [n_output]
    of the likelihood, and [n_steps] of the recognition — the only component
    that knows the horizon, since a dynamics or a likelihood is defined for
    any horizon. [make] reads them from there and checks that the components
    agree with each other: their constructors take the dimensions separately,
    so a mismatch is a mistake the type system does not catch.

    [recognition] says how the latents are inferred ({!Recognition.Iilqr} or
    {!Recognition.Lfads}, or a value built the same way). [initial_prior] must
    be given exactly when the recognition infers the initial state — LFADS
    does, iLQR fixes [x_0 = 0] and must not have one.

    Raises [Invalid_argument] if a dimension is not positive, if the
    components disagree on one, or if the initial-state prior and the
    recognition do not match. *)
val make
  :  (float, 'w) Nx.dtype
  -> dynamics:('w, 'd, 's) Dynamics.t
  -> likelihood:('w, 'l, 's) Likelihood.t
  -> prior:('w, 'p, 's) Prior.t
  -> encoder_prior:('w, 'e, 's) Prior.t
  -> recognition:('w, 'd, 'l, 'e, 'r, 's) Recognition.t
  -> ?initial_prior:Initial.t
  -> unit
  -> ('w, 'd, 'l, 'p, 'e, 'r, 's) t

(** [init ~at ?key model] draws the parameters a fresh model starts from, from
    one subkey of [key] per component; [key] defaults to the current random
    scope. Each component draws at the placement its description was made at,
    and [at] — which the type ties to the same set — places the initial-state
    prior, the one block that is deterministic and so has no key of its own.

    Building the model itself allocates nothing, so building one per device
    costs nothing; this is the one draw. *)
val init
  :  at:'s Nx.Placement.t
  -> ?key:'s Nx.Rng.key
  -> ('w, 'd, 'l, 'p, 'e, 'r, 's) t
  -> ('w, 'd, 'l, 'p, 'e, 'r, 's) params

(** {1:reading Reading a model} *)

(** [dtype m] is the float width every tensor of the model is made in. *)
val dtype : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> (float, 'w) Nx.dtype

(** [n m] is the state dimension. *)
val n : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> int

(** [m m] is the input dimension. *)
val m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> int

(** [n_steps m] is the number of control steps and observations. *)
val n_steps : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> int

(** [n_output m] is the observation dimension. *)
val n_output : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> int

(** [dynamics m] is the model's dynamics. *)
val dynamics : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> ('w, 'd, 's) Dynamics.t

(** [likelihood m] is the model's likelihood. *)
val likelihood : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> ('w, 'l, 's) Likelihood.t

(** [prior m] is the generative prior of the ELBO. *)
val prior : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> ('w, 'p, 's) Prior.t

(** [encoder_prior m] is the prior of the iLQR cost, [p̃]. *)
val encoder_prior : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> ('w, 'e, 's) Prior.t

(** [initial_prior m] is the prior over [x_0], present exactly when the
    recognition infers the initial state. *)
val initial_prior : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> Initial.t option

(** [recognition m] is the model's recognition: how the latents are inferred. *)
val recognition : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> ('w, 'd, 'l, 'e, 'r, 's) Recognition.t

(** [structure model] is the structure of a {!type-params} value, assembled from the
    components' structures: paths read ["dynamics.a1"],
    ["likelihood.log_variances"], ["recognition.ic_fwd.uf"] and so on. This
    is the structure {!Rune.grad}, {!Rune.jit} and {!Vega} take. *)
val structure
  :  ('w, 'd, 'l, 'p, 'e, 'r, 's) t
  -> (('w, 'd, 'l, 'p, 'e, 'r, 's) params, 's) Nx.Ptree.t

(** [generative model] is the view of the model a recognition reads. *)
val generative
  :  ('w, 'd, 'l, 'p, 'e, 'r, 's) t
  -> ('w, 'd, 'l, 'e, 's) Recognition.generative

(** [gparams params] is the matching view of the parameters. *)
val gparams
  :  ('w, 'd, 'l, 'p, 'e, 'r, 's) params
  -> ('w, 'd, 'l, 'e, 's) Recognition.gparams

(** [infer model params ~n_samples ~key data] draws [n_samples] latent
    trajectories per trial from the model's recognition distribution and
    evaluates [log q] at each of them. [data] is one trial or a tensor
    mini-batch ([[batch…; n_steps; n_output]]), and the posterior carries the
    matching batch axes. See {!Recognition.infer}. [solve] picks the
    recognition's inner solver; see {!Recognition.type-solve}. *)
val infer
  :  ('w, 'd, 'l, 'p, 'e, 'r, 's) t
  -> ('w, 'd, 'l, 'p, 'e, 'r, 's) params
  -> ?solve:Recognition.solve
  -> n_samples:int
  -> key:'s Nx.Rng.key
  -> ('w, 's) value
  -> ('w, 's) Recognition.posterior

(** [mean model params data] is the recognition's posterior mean, one
    deterministic latent trajectory per trial. See {!Recognition.mean}. *)
val mean
  :  ('w, 'd, 'l, 'p, 'e, 'r, 's) t
  -> ('w, 'd, 'l, 'p, 'e, 'r, 's) params
  -> ?solve:Recognition.solve
  -> ('w, 's) value
  -> ('w, 's) Recognition.samples

(** [log_prob model params data samples] is [log q(samples | data)] for given
    latents, with the batch axes of [data]. See {!Recognition.log_prob}. *)
val log_prob
  :  ('w, 'd, 'l, 'p, 'e, 'r, 's) t
  -> ('w, 'd, 'l, 'p, 'e, 'r, 's) params
  -> ?solve:Recognition.solve
  -> ('w, 's) value
  -> ('w, 's) Recognition.samples
  -> ('w, 's) value
