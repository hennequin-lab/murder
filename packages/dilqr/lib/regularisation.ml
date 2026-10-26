(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Levenberg-Marquardt style regularization schedule for the Riccati backward
   pass. The control-cost Hessian [Quu] of the linearized problem is only
   positive semi-definite in general; adding [mu * B B^T] makes it positive
   definite so the Cholesky solve exists. When the pass succeeds, [mu] decays
   back down; when the factorization fails, it grows geometrically. *)

type t =
  { delta : float
  ; mu : float
  }

(** Start at [mu = 1]: mild enough not to slow convergence on easy problems,
    regularized enough that the first gains are sane on hard ones. *)
let init = { delta = 1.0; mu = 1.0 }

let increase { delta; mu } =
  let delta = Float.max 2.0 (2.0 *. delta) in
  { delta; mu = Float.max 1e-6 (mu *. delta) }

let decrease { delta; mu } =
  let delta = Float.min 0.5 (delta /. 2.0) in
  { delta; mu = Float.max 1e-6 (mu *. delta) }
