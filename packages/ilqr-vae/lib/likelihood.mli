(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Observation models.

    A likelihood emits one observation row from one state: [o_k ~ p(o | z_k)].
    It is used three ways: [logp_t] is the data-fidelity term of the iLQR
    encoder cost and of the ELBO, [mean] is the deterministic readout used for
    predictions and R², and [sample] generates observations.

    A likelihood is a {e description} — dimensions, the structure of its
    parameters, the three functions that take them, and how to draw those
    parameters. Its parameters are a value of their own, made by {!val-init}:
    a description holds no tensor, so building one is free, and it can be
    built before the device that will run it is known. Nothing here mentions [rune], so every operation differentiates and
    traces by construction. Parameters that must be positive (Gaussian variances,
    Poisson gains) are log-parameterized: the stored parameter is the log of
    the quantity and the functions exponentiate it.

    A {e datum} is one row of a trial, of shape [[n_output]]; a [logp_t] over
    a batch of rows takes [[batch…; n]] states and [[batch…; n_output]]
    data and returns one scalar per row. [k] is the step of the {e state}, not
    of the datum: the encoder emits [o_{k−1}] from [x_k], and the terminal loss
    is the last observation at [k = n_steps]. A likelihood that ignores [k]
    — both of this module's are stationary — is the usual case. *)

open Types

(** A likelihood: a description, read through the accessors below and built by
    {!make} or by one of the likelihoods further down. It holds no parameter value
    and no tensor. *)
type ('w, 'p, 's) t

(** [make ~n ~n_output ~structure ~init ~logp_t ~mean ~sample ()] is the
    likelihood those describe, and the way to write one the library does not
    ship. [n] and [n_output] are the state and observation dimensions,
    [structure] is how the parameters are walked, [init] draws them, and the three
    maps are the density, the readout and the sampler.

    Raises [Invalid_argument] if [n] or [n_output] is not positive. *)
val make
  :  n:int
  -> n_output:int
  -> structure:('p, 's) Nx.Ptree.t
  -> init:(?key:'s Nx.Rng.key -> (float, 'w) Nx.dtype -> 'p)
  -> logp_t:
       ('p -> k:('w, 's) value -> x:('w, 's) value -> o:('w, 's) value -> ('w, 's) value)
  -> mean:('p -> ('w, 's) value -> ('w, 's) value)
  -> sample:('p -> key:'s Nx.Rng.key -> mean:('w, 's) value -> ('w, 's) value)
  -> unit
  -> ('w, 'p, 's) t

(** {1:reading Reading a likelihood} *)

(** [n l] is the state dimension. *)
val n : ('w, 'p, 's) t -> int

(** [n_output l] is the observation dimension. *)
val n_output : ('w, 'p, 's) t -> int

(** [structure l] is the [Nx.Ptree.t] of its parameters. *)
val structure : ('w, 'p, 's) t -> ('p, 's) Nx.Ptree.t

(** [init ?key dtype l] draws the parameters a fresh model starts from, at the
    placement the description was made at. [key] defaults to the current random
    scope. *)
val init : ?key:'s Nx.Rng.key -> (float, 'w) Nx.dtype -> ('w, 'p, 's) t -> 'p

(** [logp_t l p ~k ~x ~o] is log p(o | x) at step [k], summed over the observation
    channels: a scalar for one row, or one value per row for a batch ([[batch…; n]]
    states against [[batch…; n_output]] data). *)
val logp_t
  :  ('w, 'p, 's) t
  -> 'p
  -> k:('w, 's) value
  -> x:('w, 's) value
  -> o:('w, 's) value
  -> ('w, 's) value

(** [mean l p z] is the readout of a state or state sequence: the last axis ([n])
    is replaced by [n_output], every leading axis kept. *)
val mean : ('w, 'p, 's) t -> 'p -> ('w, 's) value -> ('w, 's) value

(** [sample l p ~key ~mean] draws observations of the shape of [mean]. *)
val sample
  :  ('w, 'p, 's) t
  -> 'p
  -> key:'s Nx.Rng.key
  -> mean:('w, 's) value
  -> ('w, 's) value

(** {1:gaussian Gaussian} *)

(** A linear-Gaussian readout: [o_k ~ N(c z_k + bias, diag σ²)] with a
    learned readout matrix, bias and per-channel variances. The state enters
    on the right, [c] is [[n_output; n]], so [mean p z] is [z @ cᵀ + bias] for
    any leading axes of [z]. *)
module Gaussian : sig
  (** The parameters of a Gaussian likelihood over payload ['a].
      {!structure} gives their [Nx.Ptree.t]. *)
  type ('a, _) params =
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
  val structure : unit -> ((('a, 'b, 's) Nx.t, 's) params, 's) Nx.Ptree.t

  (** [make ~key ~c_mask ~scale ~sigma2 ~bias ~n ~n_output ()] is a Gaussian
      readout of [[n]] dimensional states to [[n_output]] observations. Its
      parameters, which {!init} draws, start as follows.

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
    :  at:'s Nx.Placement.t
    -> ?c_mask:('w, 's) value
    -> ?scale:float
    -> ?sigma2:float
    -> ?bias:float
    -> n:int
    -> n_output:int
    -> unit
    -> ('w, ((float, 'w, 's) Nx.t, 's) params, 's) t

  (** [place ~at p] is [p]'s tensors at [at]. *)
  val place
    :  at:'e Nx.Placement.t
    -> ((float, 'w, 's) Nx.t, 's) params
    -> ((float, 'w, 'e) Nx.t, 'e) params
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
  type ('a, _) params =
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
  val structure : unit -> ((('a, 'b, 's) Nx.t, 's) params, 's) Nx.Ptree.t

  (** [make ~at ~c_mask ~link ~dt ~n ~n_output ()] is a Poisson readout of
      [[n]] dimensional states to [[n_output]] observation channels; {!init}
      draws its parameters:

      - [c] is drawn from a standard normal scaled by [scale / sqrt n] (the
        reference initialization) and masked by [c_mask] when given.
      - [link] defaults to [fun x -> 1e-3 + exp x].
      - [dt] is the bin width multiplying the rate, default [1.].
      - The gains start at [1] (that is, [log_gain = 0]) and the bias at [0].

      Raises [Invalid_argument] if [n] or [n_output] is not positive, or [dt]
      is not positive. *)
  val make
    :  at:'s Nx.Placement.t
    -> ?c_mask:('w, 's) value
    -> ?link:(('w, 's) value -> ('w, 's) value)
    -> ?dt:float
    -> ?bias:float
    -> ?scale:float
    -> ?log_gain:float
    -> n:int
    -> n_output:int
    -> unit
    -> ('w, ((float, 'w, 's) Nx.t, 's) params, 's) t

  (** [place ~at p] is [p]'s tensors at [at]. *)
  val place
    :  at:'e Nx.Placement.t
    -> ((float, 'w, 's) Nx.t, 's) params
    -> ((float, 'w, 'e) Nx.t, 'e) params
end
