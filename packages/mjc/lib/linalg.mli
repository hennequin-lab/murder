(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A factorisation that batches.

    {!Nx.cholesky} and {!Nx.matmul} are the right tools on one matrix and the
    wrong ones on a batch of small ones: measured per lane under
    {!Rune.vmap}, a batched matrix product loses to doing the lanes one at a
    time from about 48 rows up, and the batched factorisation follows it.
    Elementwise work and reductions batch at a sixth of a lane's cost at every
    size ({!Spike_batch} carries the numbers, `doc/perf.md` §5 the argument).

    So this module writes the factorisation the same mathematics out of
    operations that batch: the Cholesky recurrence as [n] steps of
    broadcast-multiply-and-reduce, and the two triangular solves as [n] steps
    of the same. It is [O(n)] tensor operations of [O(n²)] work each, which is
    slower than a kernel per matrix by a constant and faster by the lane count.

    Every operation is differentiable and none reads a value as a host number,
    so the same code runs eager, compiled, under a map, and under a gradient
    with respect to the matrix *or* the right-hand side. *)

(** The type for a batched square system: any number of leading axes, then
    [n] by [n]. *)
type 'd t = 'd Nx.float64_t

(** [cholesky a] is the lower factor of the symmetric positive definite
    matrices [a], which must have at least two axes and be square in the last
    two. A matrix that is not positive definite gives NaN where a kernel would,
    which is the same contract as {!Nx.cholesky}. *)
val cholesky : 'a t -> 'a t

(** [solve a b] is [a⁻¹ b] for the factors [a] that {!cholesky} returns and a
    right-hand side whose shape is [a]'s without its last axis, plus a
    trailing axis of extent 1 for a column. *)
val solve : 'a t -> 'a t -> 'a t

(** [solve_spd a b] is [solve (cholesky a) b], the entry point the solver uses. *)
val solve_spd : 'a t -> 'a t -> 'a t
