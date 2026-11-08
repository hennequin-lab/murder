(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gaussian log densities, from whichever form of the covariance the caller
    holds.

    The density is over the last axis: [x] is [[batch…; d]] and the covariance
    is [d]-dimensional, with any leading axes it needs to broadcast against
    [x]'s. A result is one value per row of [x] — a scalar for a single state.
    Every function here differentiates and traces, like the rest of the
    library.

    The point of the {!type-covariance} variant is that the caller says which
    form it has and the computation takes the cheapest route for it: [O (d)]
    work for a scalar or a diagonal, one triangular solve — or one product —
    for a factor. Nothing is ever densified, and no case asks the caller to
    convert.

    {[
    let logp =
      Density.logp
        ~mean:initial.Initial.mean
        ~covariance:(Density.Diagonal { log_stds = initial.Initial.log_stds })
        x
    ]} *)

open Types

(** A [d]-dimensional covariance, in the form its owner holds it. *)
type ('w, 's) covariance =
  | Isotropic of ('w, 's) value
  (** [v · I] for a scalar [v], the variance of every dimension. *)
  | Diagonal of { log_stds : ('w, 's) value }
  (** [diag (exp (2 · log_stds))]. [log_stds] is read against the last axis
      of [x]: [[d]] for one scale per dimension, or a tensor with leading
      axes when every row has its own. Log-parameterized because that is how
      a model stores a positive scale. *)
  | Cholesky of { factor : ('w, 's) value }
  (** The lower triangular [[d; d]] Cholesky factor [L] of the covariance,
      with [Σ = L Lᵀ], as {!Nx.cholesky} returns by default. *)
  | Precision_cholesky of { factor : ('w, 's) value }
  (** The upper triangular [[d; d]] Cholesky factor [U] of the precision,
      with [P = UᵀU], as {!Nx.cholesky} returns under `~triangle:Upper`: the
      covariance is [P⁻¹]. A scaled covariance is a scaled factor —
      [e^{-s} U] is the precision factor of [e^{2s} P⁻¹] — so the scale
      needs no case of its own. *)

(** [logp ?mean ~covariance x] is [log N(x; mean, Σ)], summed over the last
    axis of [x]: one value per row. [mean] is [[d]] or broadcasts against
    [x]; omitted, it is the origin, the zero-mean prior of this library.

    The work is [O (rows · d)] for {!Isotropic} and {!Diagonal}, [O (rows · d²)] for a factor, whose quadratic form is one triangular solve
    ({!Cholesky}) or one product ({!Precision_cholesky}). Raises
    [Invalid_argument] if a factor is not square. *)
val logp
  :  ?mean:('w, 's) value
  -> covariance:('w, 's) covariance
  -> ('w, 's) value
  -> ('w, 's) value
