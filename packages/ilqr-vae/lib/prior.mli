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

    A prior is a {e description}: the dimension, the [Nx.Ptree.t] the
    transformations walk, the functions that take its parameters, and how to
    draw those parameters. The parameters themselves are a value of their own,
    made by {!val-init}. Everything below is differentiable and traceable. [k] is a tensor, because the
    encoder's cost runs under {!Rune.scan}; a prior that treats the first step
    differently selects on it with [Nx.where] rather than a host [if]. *)

open Types

(** A prior: a description, read through the accessors below and built by
    {!make} or by one of the priors further down. It holds no parameter value and
    no tensor. *)
type ('w, 'p, 's) t

(** [make ~m ~structure ~init ~logp_t ~sample ?kl_to_gaussian ()] is the prior
    those describe, and the way to write one the library does not ship. [m] is
    the input dimension, [structure] is how the parameters are walked, [init]
    draws them, and the maps are the density and the sampler.
    [kl_to_gaussian] is [Some kl] exactly when the density is Gaussian.

    Raises [Invalid_argument] if [m] is not positive. *)
val make
  :  m:int
  -> structure:('p, 's) Nx.Ptree.t
  -> init:(?key:'s Nx.Rng.key -> (float, 'w) Nx.dtype -> 'p)
  -> logp_t:('p -> k:('w, 's) value -> u:('w, 's) value -> ('w, 's) value)
  -> sample:('p -> key:'s Nx.Rng.key -> n_steps:int -> ('w, 's) value)
  -> ?kl_to_gaussian:
       ('p
        -> mu:('w, 's) value
        -> space_factor:('w, 's) value
        -> time_factor:('w, 's) value
        -> ('w, 's) value)
  -> unit
  -> ('w, 'p, 's) t

(** {1:reading Reading a prior} *)

(** [m p] is the input dimension. *)
val m : ('w, 'p, 's) t -> int

(** [structure p] is the [Nx.Ptree.t] of its parameters. *)
val structure : ('w, 'p, 's) t -> ('p, 's) Nx.Ptree.t

(** [init ?key dtype p] draws the parameters a fresh model starts from, at the
    placement the description was made at. [key] defaults to the current random
    scope. *)
val init : ?key:'s Nx.Rng.key -> (float, 'w) Nx.dtype -> ('w, 'p, 's) t -> 'p

(** [logp_t p θ ~k ~u] is log p(u_k). [u] is one input row, of shape [[m]] or a
    batch [[batch…; m]]; the result is a scalar tensor, or one value per row. *)
val logp_t
  :  ('w, 'p, 's) t
  -> 'p
  -> k:('w, 's) value
  -> u:('w, 's) value
  -> ('w, 's) value

(** [sample p θ ~key ~n_steps] draws one input sequence, of shape
    [[n_steps; m]]. *)
val sample : ('w, 'p, 's) t -> 'p -> key:'s Nx.Rng.key -> n_steps:int -> ('w, 's) value

(** [kl_to_gaussian p] is the exact [KL(q ‖ p)] when the prior is Gaussian, and
    [None] otherwise. With [q = N(mu, Σ_time ⊗ Σ_space)] the two factors are the
    Cholesky factors [L] of the covariances ([Σ = L Lᵀ], triangular, as
    {!Covariance.factor} returns). {!Elbo} estimates its KL from samples whatever
    the prior — that is what lets [q] be any distribution a recognition can sample
    — so this remains for diagnostics and for a caller with a Gaussian prior who
    wants the exact value. *)
val kl_to_gaussian
  :  ('w, 'p, 's) t
  -> ('p
      -> mu:('w, 's) value
      -> space_factor:('w, 's) value
      -> time_factor:('w, 's) value
      -> ('w, 's) value)
       option

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
  type ('a, _) params =
    { log_spatial_stds : 'a (** [[m]] *)
    ; log_first_bin : 'a (** a scalar, [()] *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype: call it where the
      structure is used. A top-level [let s = structure ()] would pin it to
      the first dtype, as any application does; write [let s () = structure ()]
      instead. *)
  val structure : unit -> ((('a, 'b, 's) Nx.t, 's) params, 's) Nx.Ptree.t

  (** [make ~at ~spatial_std ~first_bin ~m ()] is a Gaussian prior over [[m]]
      dimensional inputs whose parameters start with scales [spatial_std] and a
      first step's multiplier [first_bin]; {!init} draws them at [at].

      Raises [Invalid_argument] if [m] is not positive or either scale is not
      positive. *)
  val make
    :  at:'s Nx.Placement.t
    -> ?spatial_std:float
    -> ?first_bin:float
    -> m:int
    -> unit
    -> ('w, ((float, 'w, 's) Nx.t, 's) params, 's) t

  (** [place ~at p] is [p]'s tensors at [at]. *)
  val place
    :  at:'e Nx.Placement.t
    -> ((float, 'w, 's) Nx.t, 's) params
    -> ((float, 'w, 'e) Nx.t, 'e) params

  (** The per-step log density of {!make}; the function the prior's
      [logp_t] is. *)
  val logp_t
    :  ((float, 'w, 's) Nx.t, 's) params
    -> k:('w, 's) value
    -> u:('w, 's) value
    -> ('w, 's) value

  (** The sampler of {!make}: one sequence, [[n_steps; m]]. *)
  val sample
    :  ((float, 'w, 's) Nx.t, 's) params
    -> key:'s Nx.Rng.key
    -> n_steps:int
    -> ('w, 's) value

  (** The exact [KL(q ‖ p)]: with [q = N(mu, Σ_time ⊗ Σ_space)] and a
      factorized Gaussian [p],

      {[
        KL = ½ (tr (Σ_p⁻¹ Σ_q) + muᵀ Σ_p⁻¹ mu + log det Σ_p − log det Σ_q − d)
      ]}

      computed from the two Cholesky factors of [Σ_q] and the prior's scales,
      in [O(T·m)] tensor operations. *)
  val kl_to_gaussian
    :  ((float, 'w, 's) Nx.t, 's) params
    -> mu:('w, 's) value
    -> space_factor:('w, 's) value
    -> time_factor:('w, 's) value
    -> ('w, 's) value
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
  type ('a, _) params =
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
  val structure : unit -> ((('a, 'b, 's) Nx.t, 's) params, 's) Nx.Ptree.t

  (** [make ~at ~spatial_std ~nu ~first_std ~m ()] is a Student-t prior over
      [[m]] dimensional inputs whose parameters start with degrees of freedom
      [nu], scales [spatial_std] and a first step with scales [first_std];
      {!init} draws them at [at].

      [nu] defaults to [10.]. Raises [Invalid_argument] if [m] is not positive,
      [nu ≤ 2], or a scale is not positive. *)
  val make
    :  at:'s Nx.Placement.t
    -> ?spatial_std:float
    -> ?nu:float
    -> ?first_std:float
    -> m:int
    -> unit
    -> ('w, ((float, 'w, 's) Nx.t, 's) params, 's) t

  (** [place ~at p] is [p]'s tensors at [at]. *)
  val place
    :  at:'e Nx.Placement.t
    -> ((float, 'w, 's) Nx.t, 's) params
    -> ((float, 'w, 'e) Nx.t, 'e) params

  (** The per-step log density and the sampler of {!make}. *)
  val logp_t
    :  ((float, 'w, 's) Nx.t, 's) params
    -> k:('w, 's) value
    -> u:('w, 's) value
    -> ('w, 's) value

  val sample
    :  ((float, 'w, 's) Nx.t, 's) params
    -> key:'s Nx.Rng.key
    -> n_steps:int
    -> ('w, 's) value
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
  type ('a, _) params =
    { log_spatial_stds : 'a (** [[m]] *)
    ; log_first_stds : 'a (** [[m]], the Gaussian first step *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype: call it where the
      structure is used. A top-level [let s = structure ()] would pin it to
      the first dtype, as any application does; write [let s () = structure ()]
      instead. *)
  val structure : unit -> ((('a, 'b, 's) Nx.t, 's) params, 's) Nx.Ptree.t

  (** [make ~at ~spatial_std ~first_std ~m ()] is a Laplacian prior over [[m]]
      dimensional inputs whose parameters start with scales [spatial_std] and a
      first step with scales [first_std]; {!init} draws them at [at].

      Raises [Invalid_argument] if [m] is not positive or a scale is not
      positive. *)
  val make
    :  at:'s Nx.Placement.t
    -> ?spatial_std:float
    -> ?first_std:float
    -> m:int
    -> unit
    -> ('w, ((float, 'w, 's) Nx.t, 's) params, 's) t

  (** [place ~at p] is [p]'s tensors at [at]. *)
  val place
    :  at:'e Nx.Placement.t
    -> ((float, 'w, 's) Nx.t, 's) params
    -> ((float, 'w, 'e) Nx.t, 'e) params

  (** The per-step log density and the sampler of {!make}. *)
  val logp_t
    :  ((float, 'w, 's) Nx.t, 's) params
    -> k:('w, 's) value
    -> u:('w, 's) value
    -> ('w, 's) value

  val sample
    :  ((float, 'w, 's) Nx.t, 's) params
    -> key:'s Nx.Rng.key
    -> n_steps:int
    -> ('w, 's) value
end
