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

(** The type for how the free block is solved: by a Cholesky factor, by
    conjugate gradients, which needs no factor of the reduced system, or by
    {!Linalg}'s factorisation, which is written out of elementwise operations
    so that batching a step keeps one lane's cost instead of paying the batched
    kernel's. [`Batched] matches the kernel's factor to 1e-12 on well-conditioned
    matrices (`test/test_linalg.ml`) but is not the default: on the corpus's
    stiffest contact state it moves the solved force by 1e-4. See `doc/perf.md`
    §5 for the measurement and the decision. *)
type linear =
  [ `Cholesky
  | `Batched
  | `Cg
  ]

(** {1:rows The row solve}

    The rows minimise a convex quadratic over a box. This is the piece the
    implicit derivative is attached to, so it is exposed: a caller can build a
    small row set and check the rule against finite differences of a solve it
    knows the answer to. *)

(** The type for row arrays at [float64] on the device ['d] a row set lives
    on. *)
type ('s, 'd) f = ('s, 'd) Rigid.f

(** The type for a row set: the quadratic, the box, and the warm start.
    Polymorphic in its arrays, as {!Model.Params} is, so the same description
    serves every dtype. *)
type ('a, 'd) rows_ =
  { a : 'a (** [[nb;nb]], the free block's matrix, symmetric positive definite. *)
  ; b : 'a (** [[nb]], the linear term. *)
  ; lo : 'a (** [[nb]], the lower bound of each row's force. *)
  ; hi : 'a (** [[nb]], the upper bound. *)
  ; mask : 'a (** [[nb]], [1] on the rows that exist. *)
  ; bilateral : 'a (** [[nb]], [1] on the rows that pull as well as push. *)
  ; lambda0 : 'a (** [[nb]], the warm start. *)
  }

(** The type for a float64 row set on the device ['d]. *)
type ('s, 'd) rows = (('s, 'd) Rigid.f, 'd) rows_

(** [rows_ptree] describes a row set to the transformations: its arrays are
    leaves at [float64] on the host. *)
val rows_ptree : unit -> (('s, 'd) rows, 'd) Nx.Ptree.t

(** [solve_rows ?implicit ?stop ~linear ~iters ~boxed r] is the minimiser over
    [r]'s box.

    [iters] × {!inner_rounds} is the *budget* of active-set rounds, not the work
    done: the loop stops as soon as the KKT conditions of the box-constrained
    quadratic hold, so a problem that settles in two rounds costs two and a
    problem that does not costs [iters] × {!inner_rounds}. [stop] is the
    residual at which it stops, relative to the driving term's magnitude, and
    defaults to [1e-12]; [0.] disables the test and reproduces the fixed
    schedule that the parity corpus was recorded against, which is what a
    measurement of "what did stopping cost" compares against.

    [implicit] selects the derivative only. It defaults to [true], which
    differentiates the minimiser as the solution of its KKT conditions with the
    active set frozen where it converged to: [dλ_F = -A_FF⁻¹ (dA λ + db)_F] on
    the free rows, zero on the pinned ones. [false] differentiates through the
    iteration instead, which is what the cost sheet measures the rule against
    and what a caller must not use to learn anything but cost. *)
val solve_rows
  :  ?implicit:bool
  -> ?stop:float
  -> linear:linear
  -> iters:int
  -> boxed:bool
  -> ('s, 'd) rows
  -> ('s, 'd) f

(** {1:solve Solve} *)

(** [kkt_tolerance] is the residual, relative to the driving term's magnitude
    at which the row solve stops by default: the largest violation may be
    [kkt_tolerance] times [1 + ‖b‖∞]. It is the residual of the KKT conditions of the
    box-constrained quadratic — a free row whose gradient is not zero, or a row
    held at a bound whose gradient points back into the box — not a change in
    the cost, because a blocked round makes no progress and is not a converged
    one. *)
val kkt_tolerance : float

(** [inner_rounds] is the number of active-set rounds per unit of [iters]: the
    budget is [iters * inner_rounds] rounds, and the block boundary at which
    the working set is rebuilt from the gradient falls on every [inner_rounds]
    of them. *)
val inner_rounds : int

(** [max_iterations] is the largest budget a step will run: [iters] is capped
    here, so the most rounds a solve can take is [max_iterations * inner_rounds]. *)
val max_iterations : int

(** [forward ?linear ?implicit ?iters m d] is [d] with [efc_force],
    [qfrc_constraint] and the final [qacc]. It reads [d.qacc] as the smooth
    acceleration, [d.efc_J], [d.efc_R], [d.efc_D], [d.efc_aref] and [d.nefc],
    and warmstarts from [d.efc_force]. [`Cg] takes one conjugate-gradient
    iteration per row, which is exact for the row counts v1 supports and needs
    no factorisation.

    [implicit] selects the derivative, not the value: [true] (the default)
    differentiates the row solve as the fixed point of its KKT conditions, so
    the gradient does not depend on [iters]; [false] differentiates through the
    iteration, which is what {!solve_rows}'s cost sheet measures the rule
    against. [stop] is {!solve_rows}'s stopping residual. *)
val forward
  :  ?linear:linear
  -> ?implicit:bool
  -> ?iters:int
  -> ?rows:int
  -> ?stop:float
  -> ('s, 'a) Model.m
  -> ('s, 'a) Data.d
  -> ('s, 'a) Data.d
