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
open Nx

(* The per-step blocks of one linearization, with the horizon as their leading
   axis: [a] is [[horizon; n; n]], [lx] [[horizon; n]], and so on. *)
module Rows = struct
  type ('a, _) t =
    { a : 'a (** ∂f_k/∂x, shape [horizon; n; n] *)
    ; b : 'a (** ∂f_k/∂u, shape [horizon; n; m] *)
    ; lx : 'a (** ∂l_k/∂x, shape [horizon; n] *)
    ; lu : 'a (** ∂l_k/∂u, shape [horizon; m] *)
    ; lxx : 'a (** ∂²l_k/∂x², shape [horizon; n; n] *)
    ; luu : 'a (** ∂²l_k/∂u², shape [horizon; m; m] *)
    ; lux : 'a (** ∂²l_k/∂u∂x, shape [horizon; m; n] *)
    }
  [@@deriving ptree]
end

type ('a, 'b, 'd) tape =
  { blocks : (('a, 'b, 'd) Nx.t, 'd) Rows.t
  ; fx : ('a, 'b, 'd) Nx.t (** ∂φ/∂x at the final state, shape [n] *)
  ; fxx : ('a, 'b, 'd) Nx.t (** ∂²φ/∂x² at the final state, shape [n; n] *)
  }

type ('a, 'b, 'd) gains =
  { feedback : ('a, 'b, 'd) Nx.t (** K_k, shape [horizon; m; n] *)
  ; feedforward : ('a, 'b, 'd) Nx.t (** k_k, shape [horizon; m] *)
  }

(** Linear (first-order) cost terms of a Riccati pass. The primal pass uses the
    tape's own gradients; the implicit-gradient pass substitutes cotangent
    slices. *)
type ('a, 'b, 'd) lin =
  { rlx : ('a, 'b, 'd) Nx.t (** [horizon; n] *)
  ; rlu : ('a, 'b, 'd) Nx.t (** [horizon; m] *)
  ; rfx : ('a, 'b, 'd) Nx.t (** [n] *)
  }

let lin_of_tape (tape : ('a, 'b, 'd) tape) : ('a, 'b, 'd) lin =
  { rlx = tape.blocks.lx; rlu = tape.blocks.lu; rfx = tape.fx }

(* [flip x] reverses the horizon axis. *)
let flip x = Nx.flip ~axes:[ 0 ] x

(* [chol_solve ~upper ~rhs] solves [u^T u @ x = rhs] for upper Cholesky factor
   [u]: two triangular substitutions, one per factor. A right-hand side one
   rank below [upper] is a vector: it is promoted to a [..., n, 1] matrix for
   the substitution and squeezed back (the traced triangular solve requires
   matrix right-hand sides). Batched over every leading axis. *)
let chol_solve ~(upper : ('a, 'b, 'd) Nx.t) ~(rhs : ('a, 'b, 'd) Nx.t) : ('a, 'b, 'd) Nx.t
  =
  let vector = Array.length (shape rhs) = Array.length (shape upper) - 1 in
  let rhs = if vector then unsqueeze ~axes:[ -1 ] rhs else rhs in
  let z =
    solve_triangular ~triangle:Upper ~transpose:Transpose ~unit_diag:false upper rhs
  in
  let x = solve_triangular ~triangle:Upper ~unit_diag:false upper z in
  if vector then squeeze ~axes:[ -1 ] x else x

(* [sym x] is the symmetrized part of [x]. *)
let sym x = Infix.((x + matrix_transpose x) /$ 2.0)

(* [matvec m v] is [m @ v], batched over every leading axis. [Nx.matmul] would
   contract the batch axis of a batched vector, so the vector is promoted to a
   column and the result squeezed back. *)
let matvec m v = squeeze ~axes:[ -1 ] Infix.(m *@ unsqueeze ~axes:[ -1 ] v)

(* The per-step factors that the primal and every tangent of one LQR solve
   share. *)
type ('a, 'b, 'd) common =
  { quu_chol : ('a, 'b, 'd) Nx.t
    (** the upper Cholesky factor of the regularized [Q_uu], [horizon; m; m] *)
  ; feedback : ('a, 'b, 'd) Nx.t (** K_k = -Q_uu⁻¹ Q_ux, [horizon; m; n] *)
  }

(* [regularize ~mu ...] adds Tassa's damping. All three terms are needed for the
   recursion to stay stable as mu grows. *)
let regularize ~mu ~at ~bt ~a ~b ~quu ~qux ~qxx =
  match mu with
  | None -> quu, qux, qxx
  | Some mu ->
    ( Infix.(quu + (bt *@ b * mu))
    , Infix.(qux + (bt *@ a * mu))
    , Infix.(qxx + (at *@ a * mu)) )

(* One step of the Hessian recursion: the value Hessian at step [k] from its
   successor, the regularized [Q_uu] factor, and the feedback gain. A
   regularized [Q_uu] that is not positive definite gives a NaN Cholesky
   factor, and every quantity computed from it is NaN. *)
let common_step ~mu ~(row : (('a, 'b, 'd) Nx.t, 'd) Rows.t) ~vxx =
  let a = row.a
  and b = row.b in
  let at = matrix_transpose a
  and bt = matrix_transpose b in
  let qxx = Infix.(row.lxx + (at *@ (vxx *@ a))) in
  let qux = Infix.(row.lux + (bt *@ (vxx *@ a))) in
  let quu = Infix.(row.luu + (bt *@ (vxx *@ b))) in
  let quu, qux, qxx = regularize ~mu ~at ~bt ~a ~b ~quu ~qux ~qxx in
  let u = cholesky ~triangle:Upper quu in
  let k_mat = Infix.(-chol_solve ~upper:u ~rhs:qux) in
  let vxx' = sym Infix.(qxx + (matrix_transpose qux *@ k_mat)) in
  u, k_mat, vxx'

let rows_structure () = Nx.Ptree.instantiate (module Rows)

(* [backward_common ?mu tape] runs the Hessian recursion over the whole horizon
   as one scan, from the last step to the first. *)
let backward_common ?mu (tape : ('a, 'b, 'd) tape) : ('a, 'b, 'd) common =
  let rows = rows_structure () in
  let _, (quu_chol, feedback) =
    Rune.scan
      Nx.Ptree.tensor
      rows
      Nx.Ptree.(pair tensor tensor)
      (fun vxx row ->
         let u, k_mat, vxx' = common_step ~mu ~row ~vxx in
         vxx', (u, k_mat))
      ~init:tape.fxx
      (Nx.Ptree.map rows (fun _ x -> flip x) tape.blocks)
  in
  { quu_chol = flip quu_chol; feedback = flip feedback }

(* [costate tape] is the primal costate [λ_{k+1}] at every step, [horizon; n]:
   the value gradient of the local model with the tape's own linear terms. One
   scan, and the same tensor for every tangent of the solve. *)
let costate (tape : ('a, 'b, 'd) tape) : ('a, 'b, 'd) Nx.t =
  let _, lam =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      (fun lam (lx_k, a_k) ->
         (* Emit the incoming carry, so the stack is [λ_K; …; λ_1] and flips to
            [λ_1; …; λ_K]: row [k] is [λ_{k+1}], the value gradient at the state
            the step starts from. *)
         Infix.(lx_k + matvec (matrix_transpose a_k) lam), lam)
      ~init:tape.fx
      (flip tape.blocks.lx, flip tape.blocks.a)
  in
  flip lam

(* The per-step outputs of the linear half of the backward pass. *)
module Lin_rows = struct
  type ('a, _) t =
    { k : 'a
    ; d1 : 'a
    ; d2 : 'a
    }
  [@@deriving ptree]
end

(* [backward_lin ~common ~tape ~lin] is the linear half of the Riccati pass,
   with the factorization already in hand. *)
let backward_lin
      ~(common : ('a, 'b, 'd) common)
      ~(tape : ('a, 'b, 'd) tape)
      ~(lin : ('a, 'b, 'd) lin)
  : ('a, 'b, 'd) gains * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t
  =
  let _, out =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(
        pair (pair tensor tensor) (pair (pair tensor tensor) (pair tensor tensor)))
      (Nx.Ptree.instantiate (module Lin_rows))
      (fun vx ((a, b), ((chol, k_mat), (rx, ru))) ->
         let qu = Infix.(ru + matvec (matrix_transpose b) vx) in
         let qx = Infix.(rx + matvec (matrix_transpose a) vx) in
         let k_k = Infix.(-chol_solve ~upper:chol ~rhs:qu) in
         (* [Q_uu k_k = -Q_u], so [k_kᵀ Q_uu k_k = -k_kᵀ Q_u]: the curvature the
            gains actually used is minus the modelled first-order decrease. *)
         let d1 = sum ~axes:[ -1 ] Infix.(k_k * qu) in
         let d2 = Infix.(-d1) in
         let vx' = Infix.(qx + matvec (matrix_transpose k_mat) qu) in
         vx', { Lin_rows.k = k_k; d1; d2 })
      ~init:lin.rfx
      ( (flip tape.blocks.a, flip tape.blocks.b)
      , ((flip common.quu_chol, flip common.feedback), (flip lin.rlx, flip lin.rlu)) )
  in
  ( { feedback = common.feedback; feedforward = flip out.Lin_rows.k }
  , sum ~axes:[ 0 ] out.Lin_rows.d1
  , sum ~axes:[ 0 ] out.Lin_rows.d2 )

(* [backward ?mu ~tape ~lin] runs the whole Riccati sweep over the horizon. *)
let backward ?mu (tape : ('a, 'b, 'd) tape) ~(lin : ('a, 'b, 'd) lin)
  : ('a, 'b, 'd) gains * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t
  =
  let common = backward_common ?mu tape in
  backward_lin ~common ~tape ~lin

(* A solution trajectory, as one tensor per trajectory. *)
type ('a, 'b, 'd) solution =
  { xs : ('a, 'b, 'd) Nx.t (** [horizon + 1; n] *)
  ; us : ('a, 'b, 'd) Nx.t (** [horizon; m] *)
  }

(* The rows of one step of a gain rollout: the nominal state and control, the
   step index, and the gains. *)
module Roll_rows = struct
  type ('a, _) t =
    { x : 'a
    ; u : 'a
    ; k : 'a
    ; ff : 'a
    ; fb : 'a
    }
  [@@deriving ptree]
end

(* [rollout_tensor ~dyn ~x0 ~xs ~us ~gains ~alpha] integrates the true
   dynamics under the control law û_k = ū_k + α·k_k + K_k·(x̂_k − x̄_k),
   returning the new state path [[horizon+1; n]] and control path [[horizon; m]]
   as one scan. [xs] is [[horizon+1; n]] and [us] is [[horizon; m]], with any
   batch axes between the horizon and the dimensions, and [alpha] is a scalar
   tensor so that a traced line search can pass the step size as an argument. *)
let rollout_tensor
      ~(dyn :
         k:('a, 'b, 'd) Nx.t
         -> ('a, 'b, 'd) Nx.t
         -> ('a, 'b, 'd) Nx.t
         -> ('a, 'b, 'd) Nx.t)
      ~(x0 : ('a, 'b, 'd) Nx.t)
      ~(xs : ('a, 'b, 'd) Nx.t)
      ~(us : ('a, 'b, 'd) Nx.t)
      ~(gains : ('a, 'b, 'd) gains)
      ~(alpha : ('a, 'b, 'd) Nx.t)
  : ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t
  =
  let roll_rows = Nx.Ptree.instantiate (module Roll_rows) in
  let horizon = (shape us).(0) in
  let k = arange_at (beside us) (Nx.dtype x0) 0 horizon 1 in
  let rows =
    { Roll_rows.x = slice [ R (0, horizon) ] xs
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
      (fun x row ->
         let dx = Infix.(x - row.Roll_rows.x) in
         let u = Infix.(row.u + ((row.ff * alpha) + matvec row.fb dx)) in
         let x' = dyn ~k:row.k x u in
         x', (u, x'))
      ~init:x0
      rows
  in
  concatenate ~axis:0 [ unsqueeze ~axes:[ 0 ] x0; x_next ], u

(* [rollout] is [rollout_tensor] with the step size as a host float: the
   eager form, for a caller that does not trace the line search. *)
let rollout ~dyn ~x0 ~xs ~us ~gains ~alpha =
  rollout_tensor ~dyn ~x0 ~xs ~us ~gains ~alpha:(scalar_like x0 alpha)

(* [rollout_linearized ~tape ~gains ~x0] integrates the linearized dynamics
   δx_{k+1} = A δx_k + B δu_k with δu_k = k_k + K_k δx_k from δx_0 = [x0], as
   one scan. Used by the implicit gradient's perturbation solve. *)
let rollout_linearized
      ~(tape : ('a, 'b, 'd) tape)
      ~(gains : ('a, 'b, 'd) gains)
      ~(x0 : ('a, 'b, 'd) Nx.t)
  : ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t
  =
  let _, (du, dx_next) =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair (pair tensor tensor) (pair tensor tensor))
      Nx.Ptree.(pair tensor tensor)
      (fun dx ((a, b), (k_k, k_mat)) ->
         let u = Infix.(k_k + matvec k_mat dx) in
         let dx' = Infix.(matvec a dx + matvec b u) in
         dx', (u, dx'))
      ~init:x0
      ((tape.blocks.a, tape.blocks.b), (gains.feedforward, gains.feedback))
  in
  concatenate ~axis:0 [ unsqueeze ~axes:[ 0 ] x0; dx_next ], du

(* [solve ?mu ~tape ~x0] is the solution of the quadratic problem: the Riccati
   backward pass, then the gain rollout from [x0] under the linearized
   dynamics. *)
let solve ?mu (tape : ('a, 'b, 'd) tape) ~(x0 : ('a, 'b, 'd) Nx.t) : ('a, 'b, 'd) solution
  =
  let common = backward_common ?mu tape in
  let gains, _, _ = backward_lin ~common ~tape ~lin:(lin_of_tape tape) in
  let xs, us = rollout_linearized ~tape ~gains ~x0 in
  { xs; us }

(** The tangent of a tape's first-order data: what {!solve_jvp} consumes. *)
type ('a, 'b, 'd) tangent =
  { da : ('a, 'b, 'd) Nx.t (** ∂A_k along the tangent, shape [horizon; n; n] *)
  ; db : ('a, 'b, 'd) Nx.t (** ∂B_k, shape [horizon; n; m] *)
  ; df : ('a, 'b, 'd) Nx.t
    (** ∂f_k, the tangent of the dynamics' value at the nominal step,
        shape [horizon; n] *)
  ; dlx : ('a, 'b, 'd) Nx.t (** ∂(∂l_k/∂x), shape [horizon; n] *)
  ; dlu : ('a, 'b, 'd) Nx.t (** ∂(∂l_k/∂u), shape [horizon; m] *)
  ; dfx : ('a, 'b, 'd) Nx.t (** ∂(∂φ/∂x) at the final state, shape [n] *)
  }

(* The per-step inputs of the tangent backward pass. *)
module Tangent_rows = struct
  type ('a, _) t =
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
  [@@deriving ptree]
end

(* The per-step inputs of the tangent feedforward rollout. *)
module Tangent_roll_rows = struct
  type ('a, _) t =
    { a : 'a
    ; b : 'a
    ; chol : 'a
    ; fb : 'a
    ; q : 'a
    }
  [@@deriving ptree]
end

(* One step of the tangent backward pass: the tangent feedforward [q] from the
   per-step inputs [row] and the incoming [e], and the next [e]. *)
let tangent_step ~e (row : (('a, 'b, 'd) Nx.t, 'd) Tangent_rows.t)
  : ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t
  =
  let open Infix in
  let x = row.dlx + (matvec (matrix_transpose row.da) row.lx + matvec row.lxx row.g) in
  let u = row.dlu + (matvec (matrix_transpose row.db) row.lx + matvec row.lux row.g) in
  let q = u + matvec (matrix_transpose row.b) e in
  let e' = x + (matvec (matrix_transpose row.a) e + matvec (matrix_transpose row.fb) q) in
  q, e'

(* One step of the tangent gain rollout: the tangent control from the per-step
   inputs [row] and the incoming [v], and the next [v]. *)
let tangent_roll_step ~v (row : (('a, 'b, 'd) Nx.t, 'd) Tangent_roll_rows.t)
  : ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t
  =
  let open Infix in
  let du = -chol_solve ~upper:row.chol ~rhs:row.q + matvec row.fb v in
  du, matvec row.a v + matvec row.b du

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
      ~(common : ('a, 'b, 'd) common)
      ~(tape : ('a, 'b, 'd) tape)
      ~(costate : ('a, 'b, 'd) Nx.t)
      ~(tangent : ('a, 'b, 'd) tangent)
      ~(dx0 : ('a, 'b, 'd) Nx.t)
  : ('a, 'b, 'd) solution
  =
  let horizon = (shape tape.blocks.a).(0) in
  (* 1. The tangent trajectory. *)
  let _, g_next =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      (fun g (a, df) ->
         let g' = Infix.(matvec a g + df) in
         g', g')
      ~init:dx0
      (tape.blocks.a, tangent.df)
  in
  let g = concatenate ~axis:0 [ unsqueeze ~axes:[ 0 ] dx0; g_next ] in
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
    ; g = flip (slice [ R (0, horizon) ] g)
    ; fb = flip common.feedback
    }
  in
  let e_last = Infix.(tangent.dfx + matvec tape.fxx (slice [ I horizon ] g)) in
  let _, q_hat =
    Rune.scan
      Nx.Ptree.tensor
      (Nx.Ptree.instantiate (module Tangent_rows))
      Nx.Ptree.tensor
      (fun e row ->
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
      (fun v row ->
         let du, v' = tangent_roll_step ~v row in
         v', (du, v'))
      ~init:(zeros_like dx0)
      { Tangent_roll_rows.a = tape.blocks.a
      ; b = tape.blocks.b
      ; chol = common.quu_chol
      ; fb = common.feedback
      ; q = flip q_hat
      }
  in
  let v = concatenate ~axis:0 [ unsqueeze ~axes:[ 0 ] (zeros_like dx0); v_next ] in
  { xs = Infix.(g + v); us = du }
