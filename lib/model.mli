(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The model, as a value.

    A model is everything the iLQR-VAE needs about the problem: its
    dimensions, its dynamics, its likelihood, its generative and encoder
    priors, and the two factors of the recognition covariance. It carries no
    parameters of its own beyond the initial ones the components were built
    with; the parameters are a separate value ({!type-params}) whose
    structure {!structure} returns, and which is what {!Rune.grad},
    {!Rune.jit} and the optimizers take.

    The components choose their own parameter records, so the model has a type
    parameter for each. Code that only passes a model around names none of
    them: they are inferred from {!make}.

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
        ()
    in
    let params = Model.init model in
    Rune.value_and_grad (Model.structure model) objective params
    ]} *)

open Types

(** The model description, all data: dimensions, components and the initial
    parameters of the recognition covariances. Build one with {!make}. *)
type ('w, 'd, 'l, 'p, 'e) t =
  { dtype : (float, 'w) Nx.dtype
  ; n : int (** state dimension *)
  ; m : int (** input dimension *)
  ; n_steps : int (** number of control steps and observations *)
  ; n_output : int (** observation dimension *)
  ; dynamics : ('w, 'd) Dynamics.t
  ; likelihood : ('w, 'l) Likelihood.t
  ; prior : ('w, 'p) Prior.t (** the generative prior of the ELBO *)
  ; encoder_prior : ('w, 'e) Prior.t (** the prior of the iLQR cost, [p̃] *)
  ; space_cov : 'w value Covariance.params
    (** the recognition covariance over inputs, as its initial parameters; a
        covariance has no behaviour, so the model holds it directly *)
  ; time_cov : 'w value Covariance.params (** the recognition covariance over steps *)
  }

(** The parameters of a model: one value per component, in the component's own
    parameter type. {!structure} is the [Nx.Ptree.t] of this record, and this
    is the value a training loop holds and updates. *)
type ('w, 'd, 'l, 'p, 'e) params =
  { dynamics : 'd
  ; likelihood : 'l
  ; prior : 'p
  ; encoder_prior : 'e
  ; space_cov : 'w value Covariance.params
  ; time_cov : 'w value Covariance.params
  }

(** [make dtype ~n ~m ~n_steps ~n_output ~dynamics ~likelihood ~prior ~encoder_prior ()] is the model they describe.

    [space_cov] and [time_cov] default to factors of an identity covariance of
    dimension [m] and [n_steps] respectively ({!Covariance.make} with
    [sigma2 = 1.]).

    Raises [Invalid_argument] if a dimension is not positive, if a component's
    own dimensions disagree with [n], [m], [n_steps] or [n_output], if
    [dynamics.m <> m] or [prior.m <> m], or if a covariance's size disagrees
    with what it must multiply. *)
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
  -> ?space_cov:'w value Covariance.params
  -> ?time_cov:'w value Covariance.params
  -> unit
  -> ('w, 'd, 'l, 'p, 'e) t

(** [init model] collects the components' initial parameters: the value a
    training run starts from. Deterministic — the randomness was spent when
    the components were built. *)
val init : ('w, 'd, 'l, 'p, 'e) t -> ('w, 'd, 'l, 'p, 'e) params

(** [structure model] is the structure of {!init}'s value, assembled from the
    components' structures: paths read ["dynamics.a1"],
    ["likelihood.log_variances"], ["space_cov.d"] and so on. This is the
    structure {!Rune.grad}, {!Rune.jit} and {!Vega} take. *)
val structure : ('w, 'd, 'l, 'p, 'e) t -> ('w, 'd, 'l, 'p, 'e) params Nx.Ptree.t
