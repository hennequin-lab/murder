(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Core iLQR tensor mathematics: the per-step linearization tape, the Riccati
   backward pass (regularized, Cholesky-based) in the two halves a derivative
   reuses, the LQR solve and its JVP, gain rollouts, and the perturbed sweeps
   used by the implicit VJP.

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
   batch, so these functions are batch-correct as they stand. A tangent axis
   is just such a batch axis.

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

(* [flip x] reverses the horizon axis. *)
let flip x = Nx.flip ~axes:[ 0 ] x

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

(* The per-step factors that the primal and every tangent of one LQR solve
   share. *)
type ('a, 'b) common =
  { quu_chol : ('a, 'b) Nx.t
    (** the upper Cholesky factor of the regularized [Q_uu], [horizon; m; m] *)
  ; feedback : ('a, 'b) Nx.t (** K_k = -Q_uu⁻¹ Q_ux, [horizon; m; n] *)
  }

(* [regularize ~mu ...] adds Tassa's damping. All three terms are needed for the
   recursion to stay stable as mu grows. *)
let regularize ~mu ~at ~bt ~a ~b ~quu ~qux ~qxx =
  match mu with
  | None -> quu, qux, qxx
  | Some mu ->
    ( Nx.add quu (Nx.mul (Nx.matmul bt b) mu)
    , Nx.add qux (Nx.mul (Nx.matmul bt a) mu)
    , Nx.add qxx (Nx.mul (Nx.matmul at a) mu) )

(* One step of the Hessian recursion: the value Hessian at step [k] from its
   successor, the regularized [Q_uu] factor, and the feedback gain. A
   regularized [Q_uu] that is not positive definite gives a NaN Cholesky
   factor, and every quantity computed from it is NaN. *)
let common_step ~mu ~(row : ('a, 'b) Nx.t Rows.t) ~vxx =
  let a = row.a
  and b = row.b in
  let at = Nx.matrix_transpose a
  and bt = Nx.matrix_transpose b in
  let qxx = Nx.add row.lxx (Nx.matmul at (Nx.matmul vxx a)) in
  let qux = Nx.add row.lux (Nx.matmul bt (Nx.matmul vxx a)) in
  let quu = Nx.add row.luu (Nx.matmul bt (Nx.matmul vxx b)) in
  let quu, qux, qxx = regularize ~mu ~at ~bt ~a ~b ~quu ~qux ~qxx in
  let u = Nx.cholesky ~upper:true quu in
  let k_mat = Nx.neg (chol_solve ~upper:u ~rhs:qux) in
  let vxx' = sym (Nx.add qxx (Nx.matmul (Nx.matrix_transpose qux) k_mat)) in
  u, k_mat, vxx'

let rows_structure () = Nx.Ptree.instantiate (module Rows)

(* [backward_common ?mu tape] runs the Hessian recursion over the whole horizon
   as one scan, from the last step to the first. *)
let backward_common ?mu (tape : ('a, 'b) tape) : ('a, 'b) common =
  let rows = rows_structure () in
  let _, (quu_chol, feedback) =
    Rune.scan
      Nx.Ptree.tensor
      rows
      Nx.Ptree.(pair tensor tensor)
      ~f:(fun vxx row ->
        let u, k_mat, vxx' = common_step ~mu ~row ~vxx in
        vxx', (u, k_mat))
      ~init:tape.fxx
      (Nx.Ptree.map rows (fun _ x -> flip x) tape.blocks)
  in
  { quu_chol = flip quu_chol; feedback = flip feedback }

(* [costate tape] is the primal costate [λ_{k+1}] at every step, [horizon; n]:
   the value gradient of the local model with the tape's own linear terms. One
   scan, and the same tensor for every tangent of the solve. *)
let costate (tape : ('a, 'b) tape) : ('a, 'b) Nx.t =
  let _, lam =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      ~f:(fun lam (lx_k, a_k) ->
        (* Emit the incoming carry, so the stack is [λ_K; …; λ_1] and flips to
           [λ_1; …; λ_K]: row [k] is [λ_{k+1}], the value gradient at the state
           the step starts from. *)
        Nx.add lx_k (matvec (Nx.matrix_transpose a_k) lam), lam)
      ~init:tape.fx
      (flip tape.blocks.lx, flip tape.blocks.a)
  in
  flip lam

(* The per-step outputs of the linear half of the backward pass. *)
module Lin_rows = struct
  type 'a t =
    { k : 'a
    ; d1 : 'a
    ; d2 : 'a
    }

  let walk c { k; d1; d2 } =
    let open Nx.Ptree.Walk in
    let k = field c "k" leaf k in
    let d1 = field c "d1" leaf d1 in
    let d2 = field c "d2" leaf d2 in
    { k; d1; d2 }
end

(* [backward_lin ~common ~tape ~lin] is the linear half of the Riccati pass,
   with the factorization already in hand. *)
let backward_lin ~(common : ('a, 'b) common) ~(tape : ('a, 'b) tape) ~(lin : ('a, 'b) lin)
  : ('a, 'b) gains * ('a, 'b) Nx.t * ('a, 'b) Nx.t
  =
  let _, out =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(
        pair (pair tensor tensor) (pair (pair tensor tensor) (pair tensor tensor)))
      (Nx.Ptree.instantiate (module Lin_rows))
      ~f:(fun vx ((a, b), ((chol, k_mat), (rx, ru))) ->
        let qu = Nx.add ru (matvec (Nx.matrix_transpose b) vx) in
        let qx = Nx.add rx (matvec (Nx.matrix_transpose a) vx) in
        let k_k = Nx.neg (chol_solve ~upper:chol ~rhs:qu) in
        (* [Q_uu k_k = -Q_u], so [k_kᵀ Q_uu k_k = -k_kᵀ Q_u]: the curvature the
           gains actually used is minus the modelled first-order decrease. *)
        let d1 = Nx.sum ~axes:[ -1 ] (Nx.mul k_k qu) in
        let d2 = Nx.neg d1 in
        let vx' = Nx.add qx (matvec (Nx.matrix_transpose k_mat) qu) in
        vx', { Lin_rows.k = k_k; d1; d2 })
      ~init:lin.rfx
      ( (flip tape.blocks.a, flip tape.blocks.b)
      , ((flip common.quu_chol, flip common.feedback), (flip lin.rlx, flip lin.rlu)) )
  in
  ( { feedback = common.feedback; feedforward = flip out.Lin_rows.k }
  , Nx.sum ~axes:[ 0 ] out.Lin_rows.d1
  , Nx.sum ~axes:[ 0 ] out.Lin_rows.d2 )

(* [backward ?mu ~tape ~lin] runs the whole Riccati sweep over the horizon. *)
let backward ?mu (tape : ('a, 'b) tape) ~(lin : ('a, 'b) lin)
  : ('a, 'b) gains * ('a, 'b) Nx.t * ('a, 'b) Nx.t
  =
  let common = backward_common ?mu tape in
  backward_lin ~common ~tape ~lin

(* A solution trajectory, as one tensor per trajectory. *)
type ('a, 'b) solution =
  { xs : ('a, 'b) Nx.t (** [horizon + 1; n] *)
  ; us : ('a, 'b) Nx.t (** [horizon; m] *)
  }

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

(* [rollout_tensor ~dyn ~x0 ~xs ~us ~gains ~alpha] integrates the true
   dynamics under the control law û_k = ū_k + α·k_k + K_k·(x̂_k − x̄_k),
   returning the new state path [[horizon+1; n]] and control path [[horizon; m]]
   as one scan. [xs] is [[horizon+1; n]] and [us] is [[horizon; m]], with any
   batch axes between the horizon and the dimensions, and [alpha] is a scalar
   tensor so that a traced line search can pass the step size as an argument. *)
let rollout_tensor
      ~(dyn : k:('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t)
      ~(x0 : ('a, 'b) Nx.t)
      ~(xs : ('a, 'b) Nx.t)
      ~(us : ('a, 'b) Nx.t)
      ~(gains : ('a, 'b) gains)
      ~(alpha : ('a, 'b) Nx.t)
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
        let u = Nx.add row.u (Nx.add (Nx.mul row.ff alpha) (matvec row.fb dx)) in
        let x' = dyn ~k:row.k x u in
        x', (u, x'))
      ~init:x0
      rows
  in
  Nx.concatenate ~axis:0 [ Nx.unsqueeze ~axes:[ 0 ] x0; x_next ], u

(* [rollout] is [rollout_tensor] with the step size as a host float: the
   eager form, for a caller that does not trace the line search. *)
let rollout ~dyn ~x0 ~xs ~us ~gains ~alpha =
  rollout_tensor ~dyn ~x0 ~xs ~us ~gains ~alpha:(Nx.scalar_like x0 alpha)

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

(* [solve ?mu ~tape ~x0] is the solution of the quadratic problem: the Riccati
   backward pass, then the gain rollout from [x0] under the linearized
   dynamics. *)
let solve ?mu (tape : ('a, 'b) tape) ~(x0 : ('a, 'b) Nx.t) : ('a, 'b) solution =
  let common = backward_common ?mu tape in
  let gains, _, _ = backward_lin ~common ~tape ~lin:(lin_of_tape tape) in
  let xs, us = rollout_linearized ~tape ~gains ~x0 in
  { xs; us }

(** The tangent of a tape's first-order data: what {!solve_jvp} consumes. *)
type ('a, 'b) tangent =
  { da : ('a, 'b) Nx.t (** ∂A_k along the tangent, shape [horizon; n; n] *)
  ; db : ('a, 'b) Nx.t (** ∂B_k, shape [horizon; n; m] *)
  ; df : ('a, 'b) Nx.t
    (** ∂f_k, the tangent of the dynamics' value at the nominal step,
        shape [horizon; n] *)
  ; dlx : ('a, 'b) Nx.t (** ∂(∂l_k/∂x), shape [horizon; n] *)
  ; dlu : ('a, 'b) Nx.t (** ∂(∂l_k/∂u), shape [horizon; m] *)
  ; dfx : ('a, 'b) Nx.t (** ∂(∂φ/∂x) at the final state, shape [n] *)
  }

(* The per-step inputs of the tangent backward pass. *)
module Tangent_rows = struct
  type 'a t =
    { a : 'a
    ; b : 'a
    ; da : 'a
    ; db : 'a
    ; lxx : 'a
    ; lux : 'a
    ; lx : 'a
    ; dlx : 'a
    ; dlu : 'a
    ; g : 'a
    ; fb : 'a
    }

  let walk c { a; b; da; db; lxx; lux; lx; dlx; dlu; g; fb } =
    let open Nx.Ptree.Walk in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    let da = field c "da" leaf da in
    let db = field c "db" leaf db in
    let lxx = field c "lxx" leaf lxx in
    let lux = field c "lux" leaf lux in
    let lx = field c "lx" leaf lx in
    let dlx = field c "dlx" leaf dlx in
    let dlu = field c "dlu" leaf dlu in
    let g = field c "g" leaf g in
    let fb = field c "fb" leaf fb in
    { a; b; da; db; lxx; lux; lx; dlx; dlu; g; fb }
end

(* The per-step inputs of the tangent feedforward rollout. *)
module Tangent_roll_rows = struct
  type 'a t =
    { a : 'a
    ; b : 'a
    ; chol : 'a
    ; fb : 'a
    ; q : 'a
    }

  let walk c { a; b; chol; fb; q } =
    let open Nx.Ptree.Walk in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    let chol = field c "chol" leaf chol in
    let fb = field c "fb" leaf fb in
    let q = field c "q" leaf q in
    { a; b; chol; fb; q }
end

(* One step of the tangent backward pass: the tangent feedforward [q] from the
   per-step inputs [row] and the incoming [e], and the next [e]. *)
let tangent_step ~e (row : ('a, 'b) Nx.t Tangent_rows.t) : ('a, 'b) Nx.t * ('a, 'b) Nx.t =
  let x =
    Nx.add
      row.dlx
      (Nx.add (matvec (Nx.matrix_transpose row.da) row.lx) (matvec row.lxx row.g))
  in
  let u =
    Nx.add
      row.dlu
      (Nx.add (matvec (Nx.matrix_transpose row.db) row.lx) (matvec row.lux row.g))
  in
  let q = Nx.add u (matvec (Nx.matrix_transpose row.b) e) in
  let e' =
    Nx.add
      x
      (Nx.add
         (matvec (Nx.matrix_transpose row.a) e)
         (matvec (Nx.matrix_transpose row.fb) q))
  in
  q, e'

(* One step of the tangent gain rollout: the tangent control from the per-step
   inputs [row] and the incoming [v], and the next [v]. *)
let tangent_roll_step ~v (row : ('a, 'b) Nx.t Tangent_roll_rows.t)
  : ('a, 'b) Nx.t * ('a, 'b) Nx.t
  =
  let du = Nx.add (Nx.neg (chol_solve ~upper:row.chol ~rhs:row.q)) (matvec row.fb v) in
  du, Nx.add (matvec row.a v) (matvec row.b du)

(* [solve_jvp ~common ~tape ~costate ~tangent ~dx0] is the derivative of
   {!solve}'s solution. Derived as the transpose of the implicit VJP, and
   checked against it in the test suite: three cheap sweeps against the primal's
   factorization, no Cholesky and no triangular solve differentiated.

   1. [G], the trajectory the tangent of the dynamics drives: [G_0 = dx0],
   [G_{k+1} = A_k G_k + δf_k];
   2. a backward recursion for [E] and the tangent feedforward [q̂], which
   carries the tangents of the linear blocks and of the loss gradients, with
   the primal Hessians acting on [G];
   3. a forward rollout of the tangent control law against the shared factor. *)
let solve_jvp
      ~(common : ('a, 'b) common)
      ~(tape : ('a, 'b) tape)
      ~(costate : ('a, 'b) Nx.t)
      ~(tangent : ('a, 'b) tangent)
      ~(dx0 : ('a, 'b) Nx.t)
  : ('a, 'b) solution
  =
  let horizon = (Nx.shape tape.blocks.a).(0) in
  (* 1. The tangent trajectory. *)
  let _, g_next =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      ~f:(fun g (a, df) ->
        let g' = Nx.add (matvec a g) df in
        g', g')
      ~init:dx0
      (tape.blocks.a, tangent.df)
  in
  let g = Nx.concatenate ~axis:0 [ Nx.unsqueeze ~axes:[ 0 ] dx0; g_next ] in
  (* 2. The tangent backward pass. *)
  let rows =
    { Tangent_rows.a = flip tape.blocks.a
    ; b = flip tape.blocks.b
    ; da = flip tangent.da
    ; db = flip tangent.db
    ; lxx = flip tape.blocks.lxx
    ; lux = flip tape.blocks.lux
    ; lx = flip costate
    ; dlx = flip tangent.dlx
    ; dlu = flip tangent.dlu
    ; g = flip (Nx.slice [ R (0, horizon) ] g)
    ; fb = flip common.feedback
    }
  in
  let e_last = Nx.add tangent.dfx (matvec tape.fxx (Nx.slice [ I horizon ] g)) in
  let _, q_hat =
    Rune.scan
      Nx.Ptree.tensor
      (Nx.Ptree.instantiate (module Tangent_rows))
      Nx.Ptree.tensor
      ~f:(fun e row ->
        let q, e' = tangent_step ~e row in
        e', q)
      ~init:e_last
      rows
  in
  (* 3. The tangent gain rollout. *)
  let _, (du, v_next) =
    Rune.scan
      Nx.Ptree.tensor
      (Nx.Ptree.instantiate (module Tangent_roll_rows))
      Nx.Ptree.(pair tensor tensor)
      ~f:(fun v row ->
        let du, v' = tangent_roll_step ~v row in
        v', (du, v'))
      ~init:(Nx.zeros_like dx0)
      { Tangent_roll_rows.a = tape.blocks.a
      ; b = tape.blocks.b
      ; chol = common.quu_chol
      ; fb = common.feedback
      ; q = flip q_hat
      }
  in
  let v =
    Nx.concatenate ~axis:0 [ Nx.unsqueeze ~axes:[ 0 ] (Nx.zeros_like dx0); v_next ]
  in
  { xs = Nx.add g v; us = du }
