(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The model, as a value.

    A model is everything the iLQR-VAE needs about the problem: its
    dimensions, its dynamics, its likelihood, its generative and encoder
    priors, an optional prior over the initial state, and the recognition
    model that infers the latents. It carries no parameters of its own beyond
    the initial ones the components were built with; the parameters are a
    separate value ({!type-params}) whose structure {!structure} returns, and
    which is what {!Rune.grad}, {!Rune.jit} and the optimizers take.

    The components choose their own parameter records, so the model has a
    type parameter for each, the recognition included. Code that only passes
    a model around names none of them: they are inferred from {!make}.

    {[
    let model =
      Model.make
        Nx.float64
        ~n:3
        ~m:3
        ~n_steps:32
        ~n_output:3
        ~dynamics:
          (Dynamics.Linear_nonlinear.make
             ~dt:0.01
             ~tau:0.04
             ~n:3
             ~nh:128
             ~m:3
             ~phi:Nx.tanh
             Nx.float64)
        ~likelihood:(Likelihood.Gaussian.make ~n:3 ~n_output:3 Nx.float64)
        ~prior:(Prior.Student.make ~m:3 Nx.float64)
        ~encoder_prior:(Prior.Gaussian.make ~m:3 Nx.float64)
        ~recognition:(Recognition.Iilqr.make ~n:3 ~m:3 ~n_steps:32 ~n_output:3 Nx.float64)
        ()
    in
    let params = Model.init model in
    Rune.value_and_grad (Model.structure model) objective params
    ]}

    {b Swapping the recognition.} The recognition is a value with its own
    parameters, so replacing iLQR by an LFADS-style encoder is one argument:

    {[
    ~recognition:
      (Recognition.Lfads.make ~n_ic:64 ~n_ctrl:32 ~n:3 ~m:3 ~n_steps:32 ~n_output:3
         Nx.float64)
    ]}

    plus the initial-state prior the LFADS posterior needs:

    {[
    ~initial_prior:(Initial.make ~std:1.0 ~n:3 Nx.float64)
    ]}

    Everything else — the ELBO, the sampling, the training step, the JIT
    compilation — is the same. *)

open Types

(** The model description, all data: dimensions, components, the optional
    initial-state prior and the recognition. Build one with {!make}. *)
type ('w, 'd, 'l, 'p, 'e, 'r) t =
  { dtype : (float, 'w) Nx.dtype
  ; n : int (** state dimension *)
  ; m : int (** input dimension *)
  ; n_steps : int (** number of control steps and observations *)
  ; n_output : int (** observation dimension *)
  ; dynamics : ('w, 'd) Dynamics.t
  ; likelihood : ('w, 'l) Likelihood.t
  ; prior : ('w, 'p) Prior.t (** the generative prior of the ELBO *)
  ; encoder_prior : ('w, 'e) Prior.t (** the prior of the iLQR cost, [p̃] *)
  ; initial_prior : 'w value Initial.params option
    (** the prior over [x_0], present exactly when the recognition infers it *)
  ; recognition : ('w, 'd, 'l, 'p, 'e, 'r) Recognition.t
  }

(** The parameters of a model: one value per component, in the component's own
    parameter type. {!structure} is the [Nx.Ptree.t] of this record, and this
    is the value a training loop holds and updates. *)
type ('w, 'd, 'l, 'p, 'e, 'r) params =
  { dynamics : 'd
  ; likelihood : 'l
  ; prior : 'p
  ; encoder_prior : 'e
  ; initial_prior : 'w value Initial.params option
  ; recognition : 'r
  }

(** [make dtype ~n ~m ~n_steps ~n_output ~dynamics ~likelihood ~prior ~encoder_prior ~recognition ?initial_prior ()] is the model they
    describe.

    [recognition] says how the latents are inferred ({!Recognition.Iilqr} or
    {!Recognition.Lfads}, or a value built the same way); its dimensions
    must agree with the model's. [initial_prior] must be given exactly when
    the recognition infers the initial state — LFADS does, iLQR fixes
    [x_0 = 0] and must not have one.

    Raises [Invalid_argument] if a dimension is not positive, if a
    component's own dimensions disagree with [n], [m], [n_steps] or
    [n_output], if the recognition was built for other dimensions, or if the
    initial-state prior and the recognition do not match. *)
val make
  :  (float, 'w) Nx.dtype
  -> n:int
  -> m:int
  -> n_steps:int
  -> n_output:int
  -> dynamics:('w, 'd) Dynamics.t
  -> likelihood:('w, 'l) Likelihood.t
  -> prior:('w, 'p) Prior.t
  -> encoder_prior:('w, 'e) Prior.t
  -> recognition:('w, 'd, 'l, 'p, 'e, 'r) Recognition.t
  -> ?initial_prior:'w value Initial.params
  -> unit
  -> ('w, 'd, 'l, 'p, 'e, 'r) t

(** [init model] collects the components' initial parameters: the value a
    training run starts from. Deterministic — the randomness was spent when
    the components were built. *)
val init : ('w, 'd, 'l, 'p, 'e, 'r) t -> ('w, 'd, 'l, 'p, 'e, 'r) params

(** [structure model] is the structure of {!init}'s value, assembled from the
    components' structures: paths read ["dynamics.a1"],
    ["likelihood.log_variances"], ["recognition.ic_fwd.uf"] and so on. This
    is the structure {!Rune.grad}, {!Rune.jit} and {!Vega} take. *)
val structure : ('w, 'd, 'l, 'p, 'e, 'r) t -> ('w, 'd, 'l, 'p, 'e, 'r) params Nx.Ptree.t

(** [generative model] is the view of the model a recognition reads. *)
val generative : ('w, 'd, 'l, 'p, 'e, 'r) t -> ('w, 'd, 'l, 'p, 'e) Recognition.generative

(** [gparams params] is the matching view of the parameters. *)
val gparams : ('w, 'd, 'l, 'p, 'e, 'r) params -> ('w, 'd, 'l, 'p, 'e) Recognition.gparams

(** [infer model params ~n_samples ~key data] draws [n_samples] latent
    trajectories from the model's recognition distribution and evaluates
    [log q] at each of them. See {!Recognition.infer}. [iters] is the
    recognition's solver hint: [Some n] for [n] fixed iLQR iterations
    (traceable), [None] for the default complete solve; LFADS ignores it. *)
val infer
  :  ('w, 'd, 'l, 'p, 'e, 'r) t
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> ?iters:int
  -> n_samples:int
  -> key:Nx.Rng.t
  -> 'w value
  -> 'w Recognition.posterior

(** [mean model params data] is the recognition's posterior mean, a single
    deterministic latent trajectory. See {!Recognition.mean}. *)
val mean
  :  ('w, 'd, 'l, 'p, 'e, 'r) t
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> ?iters:int
  -> 'w value
  -> 'w Recognition.samples

(** [log_prob model params data samples] is [log q(samples | data)] for given
    latents. See {!Recognition.log_prob}. *)
val log_prob
  :  ('w, 'd, 'l, 'p, 'e, 'r) t
  -> ('w, 'd, 'l, 'p, 'e, 'r) params
  -> ?iters:int
  -> 'w value
  -> 'w Recognition.samples
  -> 'w value
