(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Observation models.

    A likelihood emits one observation row from one state: [o_k ~ p(o | z_k)].
    It is used three ways: [logp_t] is the data-fidelity term of the iLQR
    encoder cost and of the ELBO, [mean] is the deterministic readout used for
    predictions and R², and [sample] generates observations.

    A likelihood is a value whose parameters are passed into its functions;
    nothing here mentions [rune], so every operation differentiates and traces
    by construction. Parameters that must be positive (Gaussian variances,
    Poisson gains) are log-parameterized: the stored parameter is the log of
    the quantity and the functions exponentiate it.

    A {e datum} is one row of a trial, of shape [[n_output]]; a [logp_t] over
    a batch of rows takes [[batch…; n]] states and [[batch…; n_output]]
    data and returns one scalar per row. *)

open Types

type ('w, 'p) t =
  { n : int (** state dimension *)
  ; n_output : int (** observation dimension *)
  ; structure : 'p Nx.Ptree.t (** the structure of the likelihood's parameters *)
  ; params : 'p (** the parameters a fresh model starts from *)
  ; logp_t : 'p -> k:'w value -> x:'w value -> o:'w value -> 'w value
    (** [logp_t p ~k ~x ~o] is log p(o_k | x), summed over the observation
        channels: a scalar for one row, or one value per row for a batch
        ([[batch…; n]] states against [[batch…; n_output]] data). *)
  ; mean : 'p -> 'w value -> 'w value
    (** [mean p z] is the readout of a state or state sequence: the last axis
        ([n]) is replaced by [n_output], every leading axis kept. *)
  ; sample : 'p -> key:Nx.Rng.t -> mean:'w value -> 'w value
    (** [sample p ~key ~mean] draws observations of the shape of [mean]. *)
  }

(** {1:gaussian Gaussian} *)

(** A linear-Gaussian readout: [o_k ~ N(c z_k + bias, diag σ²)] with a
    learned readout matrix, bias and per-channel variances. The state enters
    on the right, [c] is [[n_output; n]], so [mean p z] is [z @ cᵀ + bias] for
    any leading axes of [z]. *)
module Gaussian : sig
  (** The parameters of a Gaussian likelihood over payload ['a].
      {!structure} gives their [Nx.Ptree.t]. *)
  type 'a params =
    { c : 'a (** the readout, shape [[n_output; n]] *)
    ; bias : 'a (** shape [[n_output]] *)
    ; log_variances : 'a (** shape [[n_output]]; the variances are
                             [exp log_variances] *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype: call it where the
      structure is used. A top-level [let s = structure ()] would pin it to
      the first dtype, as any application does; write [let s () = structure ()]
      instead. *)
  val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

  (** [make ~key ~c_mask ~scale ~sigma2 ~bias ~n ~n_output dtype] is a
      Gaussian readout of [[n]] dimensional states to [[n_output]]
      observations.

      - [c] is drawn ([Rng.normal] with standard deviation [scale / sqrt n],
        as in the reference implementation) unless [c_mask] is given, in which
        case the readout is the draw multiplied elementwise by [c_mask]. A
        mask pins the readout to zero where it is zero without a parameter
        constraint; [None], the default, has no mask.
      - [sigma2] is the initial variance of every channel ([1.]).
      - [bias] is the initial bias ([0.]).
      - [key] defaults to the current random scope.

      The likelihood's [logp_t], [mean] and [sample] are the fields of the
      returned record; they capture [c_mask], which is why they are not also
      module-level functions.

      Raises [Invalid_argument] if [n] or [n_output] is not positive, or
      [sigma2] or [scale] is not positive. *)
  val make
    :  ?key:Nx.Rng.t
    -> ?c_mask:'w value
    -> ?scale:float
    -> ?sigma2:float
    -> ?bias:float
    -> n:int
    -> n_output:int
    -> (float, 'w) Nx.dtype
    -> ('w, (float, 'w) Nx.t params) t
end

(** {1:poisson Poisson} *)

(** A Poisson (spike-count) model: [o_k ~ Poisson(rate)], with a learned
    readout, bias, gain and link function. The rate is
    [dt · exp(log_gain) · link (c z_k + bias)]; the default link is
    [x ↦ 1e-3 + exp x], the reference implementation's floored exponential.

    The log density includes the [log o_j!] term, so an ELBO is a normalized
    log-likelihood. That term does not depend on the state or the parameters. *)
module Poisson : sig
  (** The parameters of a Poisson likelihood over payload ['a].
      {!structure} gives their [Nx.Ptree.t]. *)
  type 'a params =
    { c : 'a (** the readout, shape [[n_output; n]] *)
    ; bias : 'a (** shape [[n_output]] *)
    ; log_gain : 'a (** shape [[n_output]]; the gains are [exp log_gain] *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype: call it where the
      structure is used. A top-level [let s = structure ()] would pin it to
      the first dtype, as any application does; write [let s () = structure ()]
      instead. *)
  val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

  (** [make ~key ~c_mask ~link ~dt ~n ~n_output dtype] is a Poisson readout of
      [[n]] dimensional states to [[n_output]] observation channels.

      - [c] is drawn from a standard normal scaled by [1 / sqrt n] (the
        reference initialization) and masked by [c_mask] when given.
      - [link] defaults to [fun x -> 1e-3 + exp x].
      - [dt] is the bin width multiplying the rate, default [1.].
      - The gains start at [1] (that is, [log_gain = 0]) and the bias at [0].
      - [key] defaults to the current random scope.

      As for {!module:Gaussian}, [logp_t], [mean] and [sample] are the fields
      of the returned record, capturing [c_mask], [link] and [dt]. [sample]
      draws counts with [Rng.poisson].

      Raises [Invalid_argument] if [n] or [n_output] is not positive, or [dt]
      is not positive. *)
  val make
    :  ?key:Nx.Rng.t
    -> ?c_mask:'w value
    -> ?link:('w value -> 'w value)
    -> ?dt:float
    -> n:int
    -> n_output:int
    -> (float, 'w) Nx.dtype
    -> ('w, (float, 'w) Nx.t params) t
end
