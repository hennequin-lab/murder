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

(** The parameters of the initial-state prior over payload ['a]: the mean and
    the log standard deviations, each of shape [[n]]. *)
type 'a params =
  { mean : 'a
  ; log_stds : 'a
  }
[@@deriving ptree]

(** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
    value, so that it stays polymorphic in the dtype: call it where the
    structure is used. A top-level [let s = structure ()] would pin it to the
    first dtype, as any application does; write [let s () = structure ()]
    instead. *)
val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

(** [make ~mean ~std ~n dtype] is the prior [N(mean, std²)] in every
    dimension. [mean] defaults to [0.], [std] to [1.]. Raises
    [Invalid_argument] if [n] is not positive or [std] is not positive. *)
val make : ?mean:float -> ?std:float -> n:int -> (float, 'w) Nx.dtype -> 'w value params

(** [logp p x] is [log p(x)] summed over the last axis: one value per row when
    [x] is [[n_samples; n]], a scalar for a single state. Differentiable and
    traceable. *)
val logp : 'w value params -> 'w value -> 'w value

(** [sample p ~key ~n_samples] draws [n_samples] initial states, of shape
    [[n_samples; n]]. Reparameterized, so it differentiates in [mean] and
    [log_stds]. *)
val sample : 'w value params -> key:Nx.Rng.t -> n_samples:int -> 'w value
