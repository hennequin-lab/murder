(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Core iLQR tensor mathematics: the per-step linearization tape, the Riccati
   backward pass (regularized, Cholesky-based), gain rollouts, and the
   perturbed sweeps used by the implicit gradient.

   Conventions (column vectors throughout):
   - state [n], control [m]; dynamics [f] : x -> u -> x'
   - running loss l(x,u) with gradient [lx] ([n]), [lu] ([m]) and Hessians
     [lxx] ([n;n]), [luu] ([m;m]), [lux] ([m;n]) (d²l/du dx)
   - final loss phi(x) with [fx], [fxx]
   - a "tape" holds the linearization of one iLQR iteration along a nominal
     trajectory. Its per-step blocks are tensors with the horizon as their
     leading axis, so a [Rune.scan] over them is one step of the sweep: a
     fixed-iteration solve traces one step, not the whole horizon, and
     JIT-compiles as a loop.

   Every per-step tensor may carry leading batch axes between the horizon and
   the dimensions ([horizon; batch...; n]): reductions run over the horizon
   and dimension axes only, and matrix-vector products broadcast over the
   batch, so these functions are batch-correct as they stand.

   Every function here is a pure tensor computation with host-static control
   flow: safe to trace, safe to jit-compile. *)

open Base

(* The per-step blocks of one linearization, with the horizon as their leading
   axis: [a] is [[horizon; n; n]], [lx] [[horizon; n]], and so on. *)
module Rows = struct
  type 'a t =
    { a : 'a (** ∂f_k/∂x, shape [horizon; n; n] *)
    ; b : 'a (** ∂f_k/∂u, shape [horizon; n; m] *)
    ; lx : 'a (** ∂l_k/∂x, shape [horizon; n] *)
    ; lu : 'a (** ∂l_k/∂u, shape [horizon; m] *)
    ; lxx : 'a (** ∂²l_k/∂x², shape [horizon; n; n] *)
    ; luu : 'a (** ∂²l_k/∂u², shape [horizon; m; m] *)
    ; lux : 'a (** ∂²l_k/∂u∂x, shape [horizon; m; n] *)
    }

  let walk c { a; b; lx; lu; lxx; luu; lux } =
    let open Nx.Ptree.Walk in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    let lx = field c "lx" leaf lx in
    let lu = field c "lu" leaf lu in
    let lxx = field c "lxx" leaf lxx in
    let luu = field c "luu" leaf luu in
    let lux = field c "lux" leaf lux in
    { a; b; lx; lu; lxx; luu; lux }
end

type ('a, 'b) tape =
  { blocks : ('a, 'b) Nx.t Rows.t
  ; fx : ('a, 'b) Nx.t (** ∂φ/∂x at the final state, shape [n] *)
  ; fxx : ('a, 'b) Nx.t (** ∂²φ/∂x² at the final state, shape [n; n] *)
  }

type ('a, 'b) gains =
  { feedback : ('a, 'b) Nx.t (** K_k, shape [horizon; m; n] *)
  ; feedforward : ('a, 'b) Nx.t (** k_k, shape [horizon; m] *)
  }

(** Linear (first-order) cost terms of a Riccati pass. The primal pass uses the
    tape's own gradients; the implicit-gradient pass substitutes cotangent
    slices. *)
type ('a, 'b) lin =
  { rlx : ('a, 'b) Nx.t (** [horizon; n] *)
  ; rlu : ('a, 'b) Nx.t (** [horizon; m] *)
  ; rfx : ('a, 'b) Nx.t (** [n] *)
  }

let lin_of_tape (tape : ('a, 'b) tape) : ('a, 'b) lin =
  { rlx = tape.blocks.lx; rlu = tape.blocks.lu; rfx = tape.fx }

(* [chol_solve ~upper ~rhs] solves [u^T u @ x = rhs] for upper Cholesky factor
   [u]: two triangular substitutions, one per factor. A right-hand side one
   rank below [upper] is a vector: it is promoted to a [..., n, 1] matrix for
   the substitution and squeezed back (the traced triangular solve requires
   matrix right-hand sides). Batched over every leading axis. *)
let chol_solve ~(upper : ('a, 'b) Nx.t) ~(rhs : ('a, 'b) Nx.t) : ('a, 'b) Nx.t =
  let vector = Array.length (Nx.shape rhs) = Array.length (Nx.shape upper) - 1 in
  let rhs = if vector then Nx.unsqueeze ~axes:[ -1 ] rhs else rhs in
  let z = Nx.solve_triangular ~upper:true ~transpose:true ~unit_diag:false upper rhs in
  let x = Nx.solve_triangular ~upper:true ~transpose:false ~unit_diag:false upper z in
  if vector then Nx.squeeze ~axes:[ -1 ] x else x

(* [sym x] is the symmetrized part of [x]. *)
let sym x = Nx.div_s (Nx.add x (Nx.matrix_transpose x)) 2.0

(* [matvec m v] is [m @ v], batched over every leading axis. [Nx.matmul] would
   contract the batch axis of a batched vector, so the vector is promoted to a
   column and the result squeezed back. *)
let matvec m v = Nx.squeeze ~axes:[ -1 ] (Nx.matmul m (Nx.unsqueeze ~axes:[ -1 ] v))

(* [batch_zero x] is a zero with [x]'s shape without its last axis: the shape
   of the per-member scalars a batched solve accumulates ([B] here, [] when
   unbatched). *)
let batch_zero x =
  let s = Nx.shape x in
  Nx.zeros (Nx.dtype x) (Array.init (Array.length s - 1) ~f:(fun i -> s.(i)))

(* One Riccati step from the blocks of step [k]. [vx]/[vxx] is the value
   function at step [k+1]; returns the feedforward [k_k] ([m]), the feedback
   [K_k] ([m; n]) and the value function at step [k]. Regularization follows
   Tassa: [Quu + mu B Bᵀ] and [Qux + mu B Aᵀ].

   A regularized [Quu] that is not positive definite gives a NaN Cholesky
   factor, and every result the step computes from it is NaN: the adaptive
   solve retries with more regularization, the fixed solve's acceptance mask
   rejects the trial. *)
let riccati_step ~(mu : float) ~(row : ('a, 'b) Nx.t Rows.t) ~vx ~vxx
  : ('a, 'b) Nx.t
    * ('a, 'b) Nx.t
    * ('a, 'b) Nx.t
    * ('a, 'b) Nx.t
    * ('a, 'b) Nx.t
    * ('a, 'b) Nx.t
  =
  let a = row.a in
  let b = row.b in
  let at = Nx.matrix_transpose a in
  let bt = Nx.matrix_transpose b in
  let qx = Nx.add row.lx (matvec at vx) in
  let qu = Nx.add row.lu (matvec bt vx) in
  let qxx = Nx.add row.lxx (Nx.matmul at (Nx.matmul vxx a)) in
  let qux = Nx.add row.lux (Nx.matmul bt (Nx.matmul vxx a)) in
  let quu = Nx.add row.luu (Nx.matmul bt (Nx.matmul vxx b)) in
  (* Tassa regularization = adding [mu I] to the running value function:
     quu += mu BᵀB, qux += mu BᵀA, qxx += mu AᵀA. All three terms are needed
     for the recursion to stay stable as mu grows. *)
  let quu_r =
    if Float.equal mu 0.0 then quu else Nx.add quu (Nx.mul_s (Nx.matmul bt b) mu)
  in
  let qux_r =
    if Float.equal mu 0.0 then qux else Nx.add qux (Nx.mul_s (Nx.matmul bt a) mu)
  in
  let qxx =
    if Float.equal mu 0.0 then qxx else Nx.add qxx (Nx.mul_s (Nx.matmul at a) mu)
  in
  let u = Nx.cholesky ~upper:true quu_r in
  let k_k = Nx.neg (chol_solve ~upper:u ~rhs:qu) in
  let k_mat = Nx.neg (chol_solve ~upper:u ~rhs:qux_r) in
  let vxx' = sym (Nx.add qxx (Nx.matmul (Nx.matrix_transpose qux_r) k_mat)) in
  let vx' = Nx.add qx (matvec (Nx.matrix_transpose qux_r) k_k) in
  (* Expected decrease accumulators: d1 = Σ kᵀQ_u ≤ 0 against the true
     gradient model, d2 = Σ kᵀQ_uu^μ k ≥ 0 against the regularized curvature
     (matching the gains actually used), so J(α) ≈ J(0) + α·d1 + ½α²·d2. The
     sums run over the control axis only, so a batched solve gets one pair of
     coefficients per member. *)
  let d1 = Nx.sum ~axes:[ -1 ] (Nx.mul k_k qu) in
  let d2 = Nx.sum ~axes:[ -1 ] (Nx.mul k_k (matvec quu_r k_k)) in
  k_k, k_mat, vx', vxx', d1, d2

(* [backward ~mu ~tape ~lin] runs the Riccati sweep over the whole horizon as
   one scan, from the last step to the first. Returns the gains and the total
   expected-decrease coefficients [d1], [d2] (one scalar per member). *)
let backward ~(mu : float) ~(tape : ('a, 'b) tape) ~(lin : ('a, 'b) lin)
  : ('a, 'b) gains * ('a, 'b) Nx.t * ('a, 'b) Nx.t
  =
  let rows = Nx.Ptree.instantiate (module Rows) in
  let flip x = Nx.flip ~axes:[ 0 ] x in
  let zero = batch_zero lin.rfx in
  let (_, (_, (d1, d2))), (feedforward, feedback) =
    Rune.scan
      Nx.Ptree.(pair tensor (pair tensor (pair tensor tensor)))
      Nx.Ptree.(pair (pair tensor tensor) rows)
      Nx.Ptree.(pair tensor tensor)
      ~f:(fun (vx, (vxx, (d1, d2))) ((rlx, rlu), row) ->
        let row = { row with Rows.lx = rlx; Rows.lu = rlu } in
        let k_k, k_mat, vx', vxx', d1', d2' = riccati_step ~mu ~row ~vx ~vxx in
        (vx', (vxx', (Nx.add d1 d1', Nx.add d2 d2'))), (k_k, k_mat))
      ~init:(lin.rfx, (tape.fxx, (zero, zero)))
      ((flip lin.rlx, flip lin.rlu), Nx.Ptree.map rows (fun _ x -> flip x) tape.blocks)
  in
  { feedback = flip feedback; feedforward = flip feedforward }, d1, d2

(* The rows of one step of a gain rollout: the nominal state and control, the
   step index, and the gains. *)
module Roll_rows = struct
  type 'a t =
    { x : 'a
    ; u : 'a
    ; k : 'a
    ; ff : 'a
    ; fb : 'a
    }

  let walk c { x; u; k; ff; fb } =
    let open Nx.Ptree.Walk in
    let x = field c "x" leaf x in
    let u = field c "u" leaf u in
    let k = field c "k" leaf k in
    let ff = field c "ff" leaf ff in
    let fb = field c "fb" leaf fb in
    { x; u; k; ff; fb }
end

(* [rollout ~dyn ~x0 ~xs ~us ~gains ~alpha] integrates the true dynamics under
   the control law û_k = ū_k + α·k_k + K_k·(x̂_k − x̄_k), returning the new
   state path [[horizon+1; n]] and control path [[horizon; m]] as one scan.
   [xs] is [[horizon+1; n]] and [us] is [[horizon; m]], with any batch axes
   between the horizon and the dimensions. *)
let rollout
      ~(dyn : k:('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t)
      ~(x0 : ('a, 'b) Nx.t)
      ~(xs : ('a, 'b) Nx.t)
      ~(us : ('a, 'b) Nx.t)
      ~(gains : ('a, 'b) gains)
      ~(alpha : float)
  : ('a, 'b) Nx.t * ('a, 'b) Nx.t
  =
  let roll_rows = Nx.Ptree.instantiate (module Roll_rows) in
  let horizon = (Nx.shape us).(0) in
  let k = Nx.arange (Nx.dtype x0) 0 horizon 1 in
  let rows =
    { Roll_rows.x = Nx.slice [ R (0, horizon) ] xs
    ; u = us
    ; k
    ; ff = gains.feedforward
    ; fb = gains.feedback
    }
  in
  let _, (u, x_next) =
    Rune.scan
      Nx.Ptree.tensor
      roll_rows
      Nx.Ptree.(pair tensor tensor)
      ~f:(fun x row ->
        let dx = Nx.sub x row.Roll_rows.x in
        let u = Nx.add row.u (Nx.add (Nx.mul_s row.ff alpha) (matvec row.fb dx)) in
        let x' = dyn ~k:row.k x u in
        x', (u, x'))
      ~init:x0
      rows
  in
  Nx.concatenate ~axis:0 [ Nx.unsqueeze ~axes:[ 0 ] x0; x_next ], u

(* [rollout_linearized ~tape ~gains ~x0] integrates the linearized dynamics
   δx_{k+1} = A δx_k + B δu_k with δu_k = k_k + K_k δx_k from δx_0 = [x0], as
   one scan. Used by the implicit gradient's perturbation solve. *)
let rollout_linearized
      ~(tape : ('a, 'b) tape)
      ~(gains : ('a, 'b) gains)
      ~(x0 : ('a, 'b) Nx.t)
  : ('a, 'b) Nx.t * ('a, 'b) Nx.t
  =
  let _, (du, dx_next) =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair (pair tensor tensor) (pair tensor tensor))
      Nx.Ptree.(pair tensor tensor)
      ~f:(fun dx ((a, b), (k_k, k_mat)) ->
        let u = Nx.add k_k (matvec k_mat dx) in
        let dx' = Nx.add (matvec a dx) (matvec b u) in
        dx', (u, dx'))
      ~init:x0
      ((tape.blocks.a, tape.blocks.b), (gains.feedforward, gains.feedback))
  in
  Nx.concatenate ~axis:0 [ Nx.unsqueeze ~axes:[ 0 ] x0; dx_next ], du
