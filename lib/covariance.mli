(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Factors of the recognition covariance.

    The recognition distribution over a sequence of inputs is a Gaussian whose
    covariance is a Kronecker product

    {[
     Σ = Σ_time ⊗ Σ_space
    ]}

    of a covariance over the horizon and a covariance over the input
    dimensions. Each factor is given by its Cholesky factor [L], a triangular
    matrix with [Σ = L Lᵀ], and [L] is parameterized by its diagonal [d] and
    its strictly upper triangle [t]: [L = triu t 1 + diag d]. Nothing else
    about a factor is a parameter, and both are learned together with the rest
    of the model.

    A factor {e is} its parameters here: a covariance has no behaviour of its
    own, so the model holds the two of them directly and the functions below
    act on them. Both are differentiable and traceable, like every other
    tensor computation in this library. *)

open Types

(** The parameters of one covariance factor over payload ['a]. *)
type 'a params =
  { d : 'a (** the diagonal, shape [[n]] *)
  ; t : 'a (** the strictly upper triangle, shape [[n; n]]; read is [triu t 1] *)
  }
[@@deriving ptree]

(** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
    value, so that it stays polymorphic in the dtype: call it where the
    structure is used. A top-level [let s = structure ()] would pin it to the
    first dtype, as any application does; write [let s () = structure ()]
    instead. *)
val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

(** [make ~sigma2 ~n dtype] is the factor of a covariance [sigma2 · I]: its
    diagonal is [sqrt sigma2] and its strictly upper triangle is zero.

    [sigma2] defaults to [1.]. Raises [Invalid_argument] if [n] is not
    positive or [sigma2] is not positive. *)
val make : ?sigma2:float -> n:int -> (float, 'w) Nx.dtype -> 'w value params

(** [factor p] is [L = triu p.t 1 + diag p.d], the triangular factor with
    [Σ = L Lᵀ]. Differentiable and traceable. *)
val factor : 'w value params -> 'w value

(** [log_det p] is [log det Σ = 2 Σ log (diag L)] for [L = factor p], a
    scalar tensor. Only the diagonal of the factor contributes. *)
val log_det : 'w value params -> 'w value
