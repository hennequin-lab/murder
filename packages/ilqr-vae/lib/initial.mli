(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The prior over the initial state.

    A model built without one keeps the convention that every trial starts at
    [x_0 = 0] and the initial state is not a latent variable. A recognition
    that infers an initial state — {!Recognition.Lfads} — needs a prior to
    weigh its posterior against, and {!Model.make} takes it as
    [~initial_prior].

    The distribution is a factorized Gaussian
    [p(x_0) = N(mu, diag (exp (2 · log_stds)))]. Both [mu] and [log_stds] are
    learned. LFADS initializes this prior at [N(0, κ I)]; the reference
    implementation makes it learnable, and so does this library. *)

open Types

(** A prior over [n] dimensions, starting at [N(mean, std²)] in every one. It
    is a description: it holds no tensor, so [make] needs neither a dtype nor
    a placement, and [init] turns it into the parameter value a model
    optimises. *)
type t =
  { n : int
  ; mean : float
  ; std : float
  }

(** The parameters of the initial-state prior over payload ['a]: the mean and
    the log standard deviations, each of shape [[n]]. *)
type ('a, _) params =
  { mean : 'a
  ; log_stds : 'a
  }
[@@deriving ptree]

(** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
    value, so that it stays polymorphic in the dtype: call it where the
    structure is used. A top-level [let s = structure ()] would pin it to the
    first dtype, as any application does; write [let s () = structure ()]
    instead. *)
val structure : unit -> ((('a, 'b, 's) Nx.t, 's) params, 's) Nx.Ptree.t

(** [make ?mean ?std ~n ()] is the prior [N(mean, std²)] in every dimension.
    [mean] defaults to [0.], [std] to [1.]. Raises [Invalid_argument] if [n]
    is not positive or [std] is not positive. *)
val make : ?mean:float -> ?std:float -> n:int -> unit -> t

(** [init ~at dtype p] is [p]'s parameter value, made at [at]: the mean and
    the log standard deviations, each [[n]]. *)
val init
  :  at:'s Nx.Placement.t
  -> (float, 'w) Nx.dtype
  -> t
  -> (('w, 's) value, 's) params

(** [place ~at p] is [p]'s tensors at [at]: the same prior, living where a run
    will use it. *)
val place
  :  at:'e Nx.Placement.t
  -> (('w, 's) value, 's) params
  -> (('w, 'e) value, 'e) params

(** [logp p x] is [log p(x)] summed over the last axis: one value per row when
    [x] is [[n_samples; n]], a scalar for a single state. Differentiable and
    traceable. *)
val logp : (('w, 's) value, 's) params -> ('w, 's) value -> ('w, 's) value

(** [sample p ~key ~n_samples] draws [n_samples] initial states, of shape
    [[n_samples; n]]. Reparameterized, so it differentiates in [mean] and
    [log_stds]. *)
val sample
  :  (('w, 's) value, 's) params
  -> key:'s Nx.Rng.key
  -> n_samples:int
  -> ('w, 's) value
