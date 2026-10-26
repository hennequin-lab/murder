(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Priors over input sequences.

    A prior is factorized over time: [p(u) = Π_k p(u_k)], each factor
    depending on the step only through its index [k] (the first step is
    special in every prior here, as in the reference implementation). Two
    priors live in a model: the {e generative} prior of the ELBO, and the
    {e encoder} prior [p̃] that regularizes the iLQR inference problem. They
    may differ — a Student-t generative prior with a Gaussian encoder prior is
    the sparse-input configuration the paper argues for.

    A prior is a value. Its parameters are passed into its functions, so
    everything below is differentiable and traceable; its {!type-t} carries
    the parameters a fresh model starts from, and [structure] is the
    [Nx.Ptree.t] the transformations walk. [k] is a tensor, because the
    encoder's cost runs under {!Rune.scan}; a prior that treats the first step
    differently selects on it with [Nx.where] rather than a host [if]. *)

open Types

type ('w, 'p) t =
  { m : int (** input dimension *)
  ; structure : 'p Nx.Ptree.t (** the structure of the prior's parameters *)
  ; params : 'p (** the parameters a fresh model starts from *)
  ; logp_t : 'p -> k:'w value -> u:'w value -> 'w value
    (** [logp_t p ~k ~u] is log p(u_k). [u] is one input row, of shape [[m]]
        or a batch [[batch…; m]]; the result is a scalar tensor, or one value
        per row. *)
  ; sample : 'p -> key:Nx.Rng.t -> n_steps:int -> 'w value
    (** [sample p ~key ~n_steps] draws one input sequence, of shape
        [[n_steps; m]]. *)
  ; kl_to_gaussian :
      ('p -> mu:'w value -> space_factor:'w value -> time_factor:'w value -> 'w value)
        option
    (** [Some kl] exactly when the prior is Gaussian;
        [kl p ~mu ~space_factor ~time_factor] is the exact KL [KL(q ‖ p)] for
        the recognition distribution [q = N(mu, Σ_time ⊗ Σ_space)], where the
        two factors are the Cholesky factors [L] of the covariances
        ([Σ = L Lᵀ], triangular, as {!Covariance.factor} returns). {!Elbo}
        estimates its KL from samples whatever the prior — that is what lets
        [q] be any distribution a recognition can sample — so this remains
        for diagnostics and for a caller with a Gaussian prior who wants the
        exact value. *)
  }

(** {1:gaussian Gaussian} *)

(** A factorized Gaussian: [u_{k,j} ~ N(0, σ_{k,j}²)] with per-dimension
    scales, and a scalar multiplier at the first step:
    [σ_{0,j} = exp (log_first_bin + log_spatial_stds_j)] and
    [σ_{k,j} = exp log_spatial_stds_j] for [k ≥ 1]. Scales are
    log-parameterized, so they are positive by construction and unconstrained
    for the optimizer.

    A Gaussian prior knows its KL to the recognition Gaussian in closed form,
    so it sets [kl_to_gaussian]. *)
module Gaussian : sig
  (** The parameters of a Gaussian prior over payload ['a]. {!structure}
      gives their [Nx.Ptree.t]. *)
  type 'a params =
    { log_spatial_stds : 'a (** [[m]] *)
    ; log_first_bin : 'a (** a scalar, [()] *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype: call it where the
      structure is used. A top-level [let s = structure ()] would pin it to
      the first dtype, as any application does; write [let s () = structure ()]
      instead. *)
  val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

  (** [make ~spatial_std ~first_bin ~m dtype] is a Gaussian prior over [[m]]
      dimensional inputs whose scales all start at [spatial_std], and whose
      first step's multiplier starts at [first_bin].

      Raises [Invalid_argument] if [m] is not positive or either scale is not
      positive. *)
  val make
    :  ?spatial_std:float
    -> ?first_bin:float
    -> m:int
    -> (float, 'w) Nx.dtype
    -> ('w, (float, 'w) Nx.t params) t

  (** The per-step log density of {!make}; the function the prior's
      [logp_t] is. *)
  val logp_t : (float, 'w) Nx.t params -> k:'w value -> u:'w value -> 'w value

  (** The sampler of {!make}: one sequence, [[n_steps; m]]. *)
  val sample : (float, 'w) Nx.t params -> key:Nx.Rng.t -> n_steps:int -> 'w value

  (** The exact [KL(q ‖ p)]: with [q = N(mu, Σ_time ⊗ Σ_space)] and a
      factorized Gaussian [p],

      {[
        KL = ½ (tr (Σ_p⁻¹ Σ_q) + muᵀ Σ_p⁻¹ mu + log det Σ_p − log det Σ_q − d)
      ]}

      computed from the two Cholesky factors of [Σ_q] and the prior's scales,
      in [O(T·m)] tensor operations. *)
  val kl_to_gaussian
    :  (float, 'w) Nx.t params
    -> mu:'w value
    -> space_factor:'w value
    -> time_factor:'w value
    -> 'w value
end

(** {1:student Student-t} *)

(** A scaled multivariate Student-t over each input row, with a Gaussian first
    step:

    {[
      p(u_k) ∝ (1 + ‖u_k / σ_k‖² / ν) ^ (−(ν + m) / 2),  ν = 2 + exp log_nu_minus_two
    ]}

    with [σ_{k,j} = sqrt ((ν − 2) / ν) · exp log_spatial_stds_j] for [k ≥ 1]
    and [σ_{0,j} = exp log_first_stds_j]. Heavy tails make posterior means
    sparse, which is the paper's recipe for interpretable inferred inputs.

    Not Gaussian: [kl_to_gaussian] is [None] and the ELBO estimates
    the KL from samples. The density is written exactly as in the reference
    implementation (a spherical t in the scaled coordinates, not a product of
    univariate t's). *)
module Student : sig
  (** The parameters of a Student-t prior over payload ['a]. {!structure}
      gives their [Nx.Ptree.t]. *)
  type 'a params =
    { log_spatial_stds : 'a (** [[m]] *)
    ; log_nu_minus_two : 'a (** a scalar, [()] *)
    ; log_first_stds : 'a (** [[m]], the Gaussian first step *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype: call it where the
      structure is used. A top-level [let s = structure ()] would pin it to
      the first dtype, as any application does; write [let s () = structure ()]
      instead. *)
  val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

  (** [make ~spatial_std ~nu ~first_std ~m dtype] is a Student-t prior over
      [[m]] dimensional inputs with degrees of freedom [nu], scales [spatial_std]
      and a first step with scales [first_std].

      [nu] defaults to [10.]. Raises [Invalid_argument] if [m] is not positive,
      [nu ≤ 2], or a scale is not positive. *)
  val make
    :  ?spatial_std:float
    -> ?nu:float
    -> ?first_std:float
    -> m:int
    -> (float, 'w) Nx.dtype
    -> ('w, (float, 'w) Nx.t params) t

  (** The per-step log density and the sampler of {!make}. *)
  val logp_t : (float, 'w) Nx.t params -> k:'w value -> u:'w value -> 'w value

  val sample : (float, 'w) Nx.t params -> key:Nx.Rng.t -> n_steps:int -> 'w value
end

(** {1:laplacian Laplacian} *)

(** A Laplacian over each input row, with a Gaussian first step:

    {[
      log p(u_k) = −‖u_k / σ_k‖₁ + cst,
    ]}

    which promotes sparse inputs like the Student-t and is cheaper. Not
    Gaussian: [kl_to_gaussian] is [None]. *)
module Laplacian : sig
  (** The parameters of a Laplacian prior over payload ['a]. {!structure}
      gives their [Nx.Ptree.t]. *)
  type 'a params =
    { log_spatial_stds : 'a (** [[m]] *)
    ; log_first_stds : 'a (** [[m]], the Gaussian first step *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype: call it where the
      structure is used. A top-level [let s = structure ()] would pin it to
      the first dtype, as any application does; write [let s () = structure ()]
      instead. *)
  val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

  (** [make ~spatial_std ~first_std ~m dtype] is a Laplacian prior over [[m]]
      dimensional inputs with scales [spatial_std] and a first step with
      scales [first_std].

      Raises [Invalid_argument] if [m] is not positive or a scale is not
      positive. *)
  val make
    :  ?spatial_std:float
    -> ?first_std:float
    -> m:int
    -> (float, 'w) Nx.dtype
    -> ('w, (float, 'w) Nx.t params) t

  (** The per-step log density and the sampler of {!make}. *)
  val logp_t : (float, 'w) Nx.t params -> k:'w value -> u:'w value -> 'w value

  val sample : (float, 'w) Nx.t params -> key:Nx.Rng.t -> n_steps:int -> 'w value
end
