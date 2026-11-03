(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The constraint solver.

    The problem, in MuJoCo's classical formulation: find the row forces [λ]
    that minimise [½ λᵀ A λ + bᵀ λ] over the row types, where
    [A = J M⁻¹ Jᵀ + diag R], [b = J qacc_smooth - aref], and the limits are
    non-negative rows. The acceleration is then
    [qacc = qacc_smooth + M⁻¹ Jᵀ λ].

    The solver is an active-set Newton on that quadratic over the fixed
    [nefcmax] buffer: the rows whose force is positive or whose reduced
    gradient is negative are free, the rest are pinned to zero by an identity
    block, and each iteration solves the free block exactly. It converges in a
    handful of iterations for the row counts v1 supports; the budget is the
    model's [iterations] option capped at [max_iterations], because a compiled
    program unrolls it. PGS is not implemented; CG over the same operator is the
    alternative when a model's row count makes the factor the bottleneck
    (`doc/adr/0010`). *)

(** {1:solve Solve} *)

(** [max_iterations] is the largest budget a compiled step will unroll. *)
val max_iterations : int

(** The type for how the free block is solved: by a Cholesky factor, or by
    conjugate gradients, which needs no factor of the reduced system. *)
type linear =
  [ `Cholesky
  | `Cg
  ]

(** [forward ?linear m d] is [d] with [efc_force], [qfrc_constraint] and the
    final [qacc]. It reads [d.qacc] as the smooth acceleration, [d.efc_J],
    [d.efc_R], [d.efc_D], [d.efc_aref] and [d.nefc], and warmstarts from
    [d.efc_force]. [`Cg] takes one conjugate-gradient iteration per row, which
    is exact for the row counts v1 supports and needs no factorisation. *)
val forward : ?linear:linear -> Model.host -> Data.host -> Data.host
