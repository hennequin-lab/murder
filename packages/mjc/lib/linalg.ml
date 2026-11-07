(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx

type 'd t = 'd Nx.float64_t

(* Every operation below addresses the last axis of its operand and leaves the
   leading ones alone, so each is a batch of the same operation for any number
   of leading axes. The index lists are built at run time because the rank is
   not known when the module is compiled: a batched call has one more axis than
   an unbatched one, and both go through this code. *)

let rank (x : 'a t) = Array.length (Nx.shape x)
let last shape = shape.(Array.length shape - 1)

(* The axes before the matrix: none for one matrix, the lanes for a batch. *)
let batch_shape (x : 'a t) = Array.sub ~pos:0 ~len:(rank x - 2) (Nx.shape x)

(* [pick x idx] is [idx] on the last axis, all of it elsewhere. *)
let pick (x : 'a t) idx = Nx.slice (List.init (rank x - 1) ~f:(fun _ -> A) @ [ idx ]) x
let col (x : 'a t) j : 'a t = pick x (I j)
let cols (x : 'a t) lo hi : 'a t = pick x (R (lo, hi))

(* Row [i] of the last two axes: index the second-to-last, keep the last. *)
let row (x : 'a t) i : 'a t =
  let before = List.init (rank x - 2) ~f:(fun _ -> A) in
  Nx.slice ((before @ [ I i ]) @ [ A ]) x

(* A column's shape: the batch axes, then the row axis. *)
let column_shape (x : 'a t) = Nx.shape (col x 0)

(* [row] already drops the axis it indexes, so its result is the vector over
   the last axis that the recurrence wants. *)
let row_vec (x : 'a t) i = row x i

(* [Σ_k x.(..., i, k) r.(..., k)] for every [i], as a broadcast product reduced
   over the last axis: the form that batches, where {!Nx.matmul} does not. *)
let dot ~(x : 'a t) ~(r : 'a t) =
  Nx.sum ~axes:[ -1 ] (Nx.mul x (Nx.unsqueeze ~axes:[ -2 ] r))

(* The same for two vectors per batch, as the triangular solves need. *)
let dot_vec ~(x : 'a t) ~(r : 'a t) = Nx.sum ~axes:[ -1 ] (Nx.mul x r)

(* The unblocked recurrence: one column at a time.

   Right for a small matrix and for a batch of them, and wrong for a big one on
   a device: every column rebuilds the whole tensor with `concatenate`, so an
   `n x n` factorisation moves `O(n^3)` elements and does its arithmetic in
   `n`-long steps. `cholesky` below blocks it; this stays as the panel kernel
   and as the small-`n` path. *)
let cholesky_unblocked a =
  let shape = Nx.shape a in
  let n = last shape in
  if not (Int.equal n shape.(Array.length shape - 2))
  then
    invalid_arg
      (Printf.sprintf
         "Linalg.cholesky: %s is not square in its last two axes"
         (String.concat ~sep:"x" (List.map (Array.to_list shape) ~f:Int.to_string)));
  let at = Nx.placement a in
  let idx = Nx.arange_f_at at Nx.float64 0.0 (Float.of_int n) 1.0 in
  let l = ref (Nx.zeros_like a) in
  for j = 0 to n - 1 do
    (* L[i,j] = (A[i,j] - Σ_{k<j} L[i,k] L[j,k]) / L[j,j], where the diagonal's
       own numerator is the pivot, so its quotient is the square root and one
       division covers the diagonal and below. Entries above are zero. *)
    let pivot =
      if Int.equal j 0
      then col a j
      else Nx.sub (col a j) (dot ~x:(cols !l 0 j) ~r:(pick (row !l j) (R (0, j))))
    in
    let diagonal =
      Nx.maximum
        (Nx.sqrt (Nx.maximum (pick pivot (I j)) (Nx.scalar_at at Nx.float64 0.0)))
        (Nx.scalar_at at Nx.float64 1e-300)
    in
    (* [diagonal] is one value per batch: a scalar when unbatched. Broadcast it
       along the row axis so the division is elementwise. *)
    let divisor =
      Nx.broadcast_to
        (column_shape a)
        (Nx.reshape (Array.append (batch_shape a) [| 1 |]) diagonal)
    in
    (* One mask per row of the column, broadcast over the batch axes. *)
    let at_or_below =
      Nx.broadcast_to
        (column_shape a)
        (Nx.greater_equal idx (Nx.scalar_at at Nx.float64 (Float.of_int j)))
    in
    let column = Nx.where at_or_below (Nx.div pivot divisor) (Nx.zeros_like pivot) in
    l
    := Nx.concatenate
         ~axis:(-1)
         [ cols !l 0 j; Nx.unsqueeze ~axes:[ -1 ] column; cols !l (j + 1) n ]
  done;
  !l

(* The blocked factorisation: a right-looking Cholesky.

   Panel by panel, with the trailing matrix kept as the Schur complement:
   factor the diagonal block with the unblocked recurrence (small: `block x
   block`), solve the panel below it (`L21 = C21 L11^-T`), update the trailing
   block (`C22 - L21 L21^T`, one matmul), and move on. The column blocks are
   collected and joined once at the end, so the factorisation copies the matrix
   `O(1)` times and puts its arithmetic in matmuls over `n`-long rows instead of
   `n` sequential steps of length one. That difference is the whole point: on a
   device the unblocked form was latency-bound at about one percent of what the
   hardware can do. *)
let cholesky_blocked ~block a =
  let shape = Nx.shape a in
  let n = last shape in
  let lead = Array.length shape - 2 in
  let batch = Array.sub ~pos:0 ~len:lead shape in
  let at = Nx.placement a in
  (* Rank-relative: [lead] axes of [A], then a row range and a column range. *)
  let window r0 r1 c0 c1 =
    Nx.slice (List.init lead ~f:(fun _ -> A) @ [ R (r0, r1); R (c0, c1) ])
  in
  let zeros rows cols = Nx.zeros_at at Nx.float64 (Array.append batch [| rows; cols |]) in
  let blocks = ref []
  and trailing = ref a
  and k = ref 0 in
  while !k < n do
    let b = Int.min block (n - !k) in
    let m = n - !k in
    let c11 = window 0 b 0 b !trailing in
    let l11 = cholesky_unblocked c11 in
    let panel =
      if Int.equal b m
      then l11
      else (
        let c21 = window b m 0 b !trailing in
        (* [L21 = C21 L11^-T], as one triangular solve of the transposed panel. *)
        let l21 =
          Nx.matrix_transpose
            (Nx.solve_triangular ~triangle:Lower l11 (Nx.matrix_transpose c21))
        in
        let c22 = window b m b m !trailing in
        let c22 = Nx.sub c22 (Nx.matmul l21 (Nx.matrix_transpose l21)) in
        trailing := c22;
        Nx.concatenate ~axis:(-2) [ l11; l21 ])
    in
    (* The rows above this panel are zero: the block is the panel, padded. *)
    blocks
    := (if Int.equal !k 0 then panel else Nx.concatenate ~axis:(-2) [ zeros !k b; panel ])
       :: !blocks;
    k := !k + b
  done;
  Nx.concatenate ~axis:(-1) (List.rev !blocks)

(* [block] is in elements: a panel of `block x block` is where the unblocked
   recurrence is still cheap, and where a matmul becomes wide enough to be
   worth issuing. Sixteen keeps the panel work small and gives four panels at
   the corpus's row counts (48-202). *)
let cholesky ?(block = 16) a =
  let shape = Nx.shape a in
  let n = last shape in
  if not (Int.equal n shape.(Array.length shape - 2))
  then
    invalid_arg
      (Printf.sprintf
         "Linalg.cholesky: %s is not square in its last two axes, so it cannot be \
          factored"
         (String.concat ~sep:"x" (List.map (Array.to_list shape) ~f:Int.to_string)));
  if n <= block then cholesky_unblocked a else cholesky_blocked ~block a

(* [b] is a column per batch, [batch; n; 1]; the recurrence is easier on the
   [batch; n] vector inside it, so the shape is flattened on the way in and
   restored on the way out. *)
let solve l b =
  let n = last (Nx.shape l) in
  (* Work on at least two axes — a batch of one when there is no batch — so
     that a column is a vector to slice and concatenate rather than a scalar. *)
  let lead =
    match batch_shape l with
    | [||] -> [| 1 |]
    | s -> s
  in
  let flat = Nx.reshape (Array.append lead [| n |]) b in
  (* Forward: y[i] = (b[i] - Σ_{k<i} L[i,k] y[k]) / L[i,i]. *)
  let y = ref flat in
  for i = 0 to n - 1 do
    let acc =
      if Int.equal i 0
      then Nx.zeros_at (Nx.placement b) Nx.float64 lead
      else dot_vec ~x:(cols !y 0 i) ~r:(pick (row_vec l i) (R (0, i)))
    in
    let value = Nx.div (Nx.sub (col !y i) acc) (pick (row_vec l i) (I i)) in
    y
    := Nx.concatenate
         ~axis:(-1)
         [ cols !y 0 i; Nx.unsqueeze ~axes:[ -1 ] value; cols !y (i + 1) n ]
  done;
  (* Backward: x[i] = (y[i] - Σ_{k>i} L[k,i] x[k]) / L[i,i], from the end. *)
  let x = ref !y in
  for i = n - 1 downto 0 do
    let acc =
      if Int.equal i (n - 1)
      then Nx.zeros_at (Nx.placement b) Nx.float64 lead
      else dot_vec ~x:(cols !x (i + 1) n) ~r:(pick (col l i) (R (i + 1, n)))
    in
    let value = Nx.div (Nx.sub (col !x i) acc) (pick (row_vec l i) (I i)) in
    x
    := Nx.concatenate
         ~axis:(-1)
         [ cols !x 0 i; Nx.unsqueeze ~axes:[ -1 ] value; cols !x (i + 1) n ]
  done;
  Nx.reshape (Nx.shape b) !x

let solve_spd a b = solve (cholesky a) b
