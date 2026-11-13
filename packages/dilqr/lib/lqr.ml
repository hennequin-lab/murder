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
  ; value_hessian : ('a, 'b, 'd) Nx.t
    (** V_0 = ∂²/∂x_0² of the cost-to-go from step 0, [n; n] with any batch
        axes: the last carry of the Hessian recursion, after step 0's own cost
        has entered it. The value function the recursion leaves is
        [V (δ) = ½ δᵀ V_0 δ + Vx_0ᵀ δ + cst] on a deviation [δ] from the
        linearization point, and {!backward_lin} returns [Vx_0]. *)
  }

(* [shift ~damping vxx] is the value Hessian with Levenberg-Marquardt damping
   shifted into it, [vxx + mu I]. Expanding the shift through the recursion
   gives exactly the three terms Tassa's damping adds, [mu AᵀA], [mu BᵀA] and
   [mu BᵀB], so damping here costs one diagonal addition per step and nothing
   else. [mu] is a scalar tensor, or one per member broadcast against [eye]. *)
let shift ~damping vxx =
  match damping with
  | None -> vxx
  | Some (mu, eye) -> Infix.(vxx + (mu * eye))

(* One step of the Hessian recursion: the value Hessian at step [k] from its
   successor, the regularized [Q_uu] factor, and the feedback gain. A
   regularized [Q_uu] that is not positive definite gives a NaN Cholesky
   factor, and every quantity computed from it is NaN. *)
let common_step ~damping ~(row : (('a, 'b, 'd) Nx.t, 'd) Rows.t) ~vxx =
  let a = row.a
  and b = row.b in
  let at = matrix_transpose a
  and bt = matrix_transpose b in
  let vxx = shift ~damping vxx in
  let qxx = Infix.(row.lxx + (at *@ (vxx *@ a))) in
  let qux = Infix.(row.lux + (bt *@ (vxx *@ a))) in
  let quu = Infix.(row.luu + (bt *@ (vxx *@ b))) in
  let u = cholesky ~triangle:Upper quu in
  let k_mat = Infix.(-chol_solve ~upper:u ~rhs:qux) in
  let vxx' = sym Infix.(qxx + (matrix_transpose qux *@ k_mat)) in
  u, k_mat, vxx'

let rows_structure () = Nx.Ptree.instantiate (module Rows)

(* [backward_common ?mu tape] runs the Hessian recursion over the whole horizon
   as one scan, from the last step to the first. *)
let backward_common ?mu (tape : ('a, 'b, 'd) tape) : ('a, 'b, 'd) common =
  let rows = rows_structure () in
  (* The identity the damping shift needs. Built once: it lies where the tape
     does, and its size is the state dimension, the last axis of [fxx]. *)
  let damping =
    match mu with
    | None -> None
    | Some mu ->
      let shape_fxx = shape tape.fxx in
      let n = shape_fxx.(Array.length shape_fxx - 1) in
      Some (mu, Nx.eye_at (Nx.placement tape.fxx) (Nx.dtype tape.fxx) n)
  in
  let value_hessian, (quu_chol, feedback) =
    Rune.scan
      Nx.Ptree.tensor
      rows
      Nx.Ptree.(pair tensor tensor)
      (fun vxx row ->
         let u, k_mat, vxx' = common_step ~damping ~row ~vxx in
         vxx', (u, k_mat))
      ~init:tape.fxx
      (Nx.Ptree.map rows (fun _ x -> flip x) tape.blocks)
  in
  { quu_chol = flip quu_chol; feedback = flip feedback; value_hessian }

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
  : ('a, 'b, 'd) gains * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t
  =
  let value_grad, out =
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
  , sum ~axes:[ 0 ] out.Lin_rows.d2
  , value_grad )

(* [backward ?mu ~tape ~lin] runs the whole Riccati sweep over the horizon. *)
let backward ?mu (tape : ('a, 'b, 'd) tape) ~(lin : ('a, 'b, 'd) lin)
  : ('a, 'b, 'd) gains * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t
  =
  let common = backward_common ?mu tape in
  backward_lin ~common ~tape ~lin

(* The Gaussian the local model puts on the initial state, once a prior has
   been fused with the value function the backward pass leaves at step 0. *)
type ('a, 'b, 'd) initial =
  { mean : ('a, 'b, 'd) Nx.t (** [n], with any batch axes *)
  ; precision : ('a, 'b, 'd) Nx.t (** [n; n], with the same axes *)
  }

(* [initial_condition ~common ~value_grad ~nominal ~prior] is the Gaussian the
   local model implies for the initial state.

   The recursion leaves [V (δ) = ½ δᵀ V δ + Vxᵀ δ] on a deviation [δ] from
   [nominal]. With a prior [N (μ, Λ⁻¹)] on the absolute state [nominal + δ],
   its cost in deviation coordinates has gradient [Λ (nominal − μ)] and
   Hessian [Λ] at [δ = 0]: the joint cost is quadratic with precision
   [P = V + Λ] and stationary point [δ* = −P⁻¹ (Vx + Λ (nominal − μ))], and
   the answer is [nominal + δ*]. Omit [prior] when the tape's own step-0 cost
   already holds it — the boundary condition of a solve with a free initial
   state — and the answer is the stationary point of the value function
   itself, [δ* = −V⁻¹ Vx], at [P = V].

   [P] is the answer's precision, so its inverse — one [chol_solve] against
   the factor computed here — is its covariance. A [P] that is not positive
   definite gives NaNs: the prior's precision, or without one the tape's
   step-0 cost, is the only thing that regularizes the block. *)
let initial_deviation ~(common : ('a, 'b, 'd) common) ~(value_grad : ('a, 'b, 'd) Nx.t)
  : ('a, 'b, 'd) Nx.t
  =
  let upper = cholesky ~triangle:Upper common.value_hessian in
  Infix.(-chol_solve ~upper ~rhs:value_grad)

let initial_condition
      ~(common : ('a, 'b, 'd) common)
      ~(value_grad : ('a, 'b, 'd) Nx.t)
      ~(nominal : ('a, 'b, 'd) Nx.t)
      ~(prior : (('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t) option)
  : ('a, 'b, 'd) initial
  =
  match prior with
  | None ->
    (* No prior outside the tape: the value function is the cost, and its
       stationary point is the deviation. *)
    { mean = Infix.(nominal + initial_deviation ~common ~value_grad)
    ; precision = common.value_hessian
    }
  | Some (prior_mean, prior_precision) ->
    let precision = Infix.(common.value_hessian + prior_precision) in
    let rhs = Infix.(value_grad + matvec prior_precision (nominal - prior_mean)) in
    let upper = cholesky ~triangle:Upper precision in
    let delta = chol_solve ~upper ~rhs in
    { mean = Infix.(nominal - delta); precision }

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

(* {2 The local model's Gaussian} *)

let log_2pi = Float.log (2.0 *. Float.pi)

(* [last_dim t] is the extent of [t]'s last axis: the dimension of the
   distribution a row of it lives in. *)
let last_dim t =
  let shape = Nx.shape t in
  shape.(Array.length shape - 1)

(* [precision_logp ~upper z] is [log N (z; 0, (UᵀU)⁻¹)] summed over the last
   axis, one value per row of [z], for [upper] the upper Cholesky factor of the
   precision: the quadratic form is one product with the factor and the
   determinant is the factor's diagonal. [upper] is one matrix — [d; d] — and
   [z]'s leading axes broadcast against it. *)
let precision_logp ~upper z =
  let open Infix in
  let whitened = z *@ matrix_transpose upper in
  let quadratic = sum ~axes:[ -1 ] (whitened * whitened) in
  let log_det = sum ~axes:[ -1 ] (log (diag upper) *$ -2.0) in
  let normalizer = scalar_like z (Float.of_int (last_dim z) *. log_2pi) in
  (quadratic + (log_det + normalizer)) *$ -0.5

(* [rollout_controls ~tape ~x0 us] is the state path the linearized dynamics
   reach from [x0] under the given controls: [x_0 = x0] and
   [x_{k+1} = A_k x_k + B_k u_k], as one scan. It is {!rollout_linearized}
   without the feedback law. [us] is [horizon; m] with any batch axes between
   the horizon and the dimension, and [x0] carries the same ones. *)
let rollout_controls ~(tape : ('a, 'b, 'd) tape) ~(x0 : ('a, 'b, 'd) Nx.t) us =
  let _, xs_next =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair (pair tensor tensor) tensor)
      Nx.Ptree.tensor
      (fun x ((a, b), u) ->
         let x' = Infix.(matvec a x + matvec b u) in
         x', x')
      ~init:x0
      ((tape.blocks.a, tape.blocks.b), us)
  in
  concatenate ~axis:0 [ unsqueeze ~axes:[ 0 ] x0; xs_next ]

(* The per-step inputs of the innovation scan of {!Laplace.log_prob}: the
   deviation of a sample's controls from the nominal, the deviation the
   linearized dynamics reach, and the factors. Construct the structure with
   [Nx.Ptree.instantiate (module Innovation_rows)]. *)
module Innovation_rows = struct
  type ('a, _) t =
    { du : 'a (** [horizon; batch…; m] *)
    ; dx : 'a (** [horizon; batch…; n] *)
    ; ff : 'a (** [horizon; m] *)
    ; fb : 'a (** [horizon; m; n] *)
    ; chol : 'a (** [horizon; m; m] *)
    }
  [@@deriving ptree]
end

(* [row0 xs] is the state a path starts from. *)
let row0 xs = slice [ I 0 ] xs

(** The Gaussian the local model puts on its whole latent trajectory.

    The local quadratic model of one linearization — the {!tape} — is
    quadratic in [z = (x₀, u₀ … u_{K−1})] once the dynamics has been
    eliminated, and its Hessian factorizes exactly along the horizon:

    {[
      J (z) − J (z⋆) = ½ Δx₀ᵀ P Δx₀ + Σ_k ½ ‖U_k Δw_k‖²
      Δw_k = (u_k − ū_k) − k_k − K_k (x_k − x̄_k)
    ]}

    with [z⋆] the model's minimizer, [P] the precision of the initial state
    ([V₀] with the prior the tape already carries, and [V₀ + Λ] for a prior
    passed in), [U_k] the Cholesky factor of [Q_uu,k = l_uu + BᵀV B], [K] the
    feedback gain and [k] the feedforward one, and [(x̄, ū)] the nominal
    trajectory the tape was linearized at. The innovations [(Δx₀, Δw)] are
    therefore independent Gaussians of covariances [(P⁻¹, Q_uu,k⁻¹)], and the
    map between them and [(z − z⋆)] is triangular with a unit diagonal: its
    Jacobian is one.

    Three things follow, and all three are cheap — no dense covariance, no
    inverse beyond one triangular solve per block:

    - {!mean} is [z⋆]: the policy's rollout from the initial state's mean;
    - {!sample} draws the innovations and rolls the linear response out;
    - {!log_prob} is the innovations' density, which needs no log determinant
      of the joint.

    This is the truncation-point message of the Riccati pass read the other
    way round: for a linear dynamics with Gaussian losses the model is exact
    and this is the posterior itself, and for a nonlinear one it is the
    Gauss-Newton (Laplace) approximation at whatever trajectory the tape was
    linearized at. *)
module Laplace = struct
  type ('a, 'b, 'd) t =
    { nominal : ('a, 'b, 'd) solution
      (** the trajectory the tape was linearized at: [(x̄, ū)]. The Gaussian's
          mean is {!mean}'s, which is this plus the policy's rollout. *)
    ; feedback : ('a, 'b, 'd) Nx.t (** [K_k], [horizon; m; n] *)
    ; feedforward : ('a, 'b, 'd) Nx.t (** [k_k], [horizon; m] *)
    ; quu_chol : ('a, 'b, 'd) Nx.t
      (** [U_k], the upper Cholesky factor of [Q_uu,k], [horizon; m; m]. The
          undamped one: a solve's regularization is a device that finds the
          trajectory, not part of the model whose posterior this is. *)
    ; initial : ('a, 'b, 'd) initial option
      (** the initial state's block, [None] when it is fixed to the nominal's:
          then there is no factor over [x₀] and every draw starts there. *)
    }

  (* [law t] is the control law the Gaussian is centred on. *)
  let law (t : ('a, 'b, 'd) t) : ('a, 'b, 'd) gains =
    { feedback = t.feedback; feedforward = t.feedforward }

  (** [mean ~tape t] is the Gaussian's mean: the local model's minimizer,
      which is the control law's rollout from the initial state's mean (or
      from the nominal's initial state when that one is fixed). One scan; the
      same [tape] the Gaussian was built from. *)
  let mean ~(tape : ('a, 'b, 'd) tape) (t : ('a, 'b, 'd) t) : ('a, 'b, 'd) solution =
    let nominal_x0 = row0 t.nominal.xs in
    let dx0 =
      match t.initial with
      | None -> zeros_like nominal_x0
      | Some init -> Infix.(init.mean - nominal_x0)
    in
    let dxs, dus = rollout_linearized ~tape ~gains:(law t) ~x0:dx0 in
    { xs = Infix.(t.nominal.xs + dxs); us = Infix.(t.nominal.us + dus) }

  (* [build ~nominal ~law ~common ~initial] is the Gaussian of the local model,
     linearized at [nominal], from the pass's own factors. *)
  let build ~nominal ~(law : ('a, 'b, 'd) gains) ~(common : ('a, 'b, 'd) common) ~initial =
    { nominal
    ; feedback = law.feedback
    ; feedforward = law.feedforward
    ; quu_chol = common.quu_chol
    ; initial
    }

  (** [make ?mu ~tape ~nominal ~prior] is the local model's Gaussian when the
      initial state is a latent. [prior] is [(mean, precision)] of a Gaussian
      on the absolute state when the caller holds one outside the tape, and
      [None] when the tape's own step-0 cost already holds it — which is how a
      solve with a free initial state prices it. [mu] damps the pass as a
      solve's regularization does: the Gaussian is then the damped model's,
      which is a different (stricter) model, not an approximation to the
      undamped one, so it is the escape hatch for a model whose [Q_uu] is not
      positive definite rather than a default. The undamped model is the
      default.

      The tape must be the linearization at [nominal], and neither carries
      batch axes: this is one trajectory's Gaussian. *)
  let make
        ?mu
        (tape : ('a, 'b, 'd) tape)
        ~(nominal : ('a, 'b, 'd) solution)
        ~(prior : (('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t) option)
    =
    let common = backward_common ?mu tape in
    let law, _, _, value_grad = backward_lin ~common ~tape ~lin:(lin_of_tape tape) in
    let initial =
      initial_condition ~common ~value_grad ~nominal:(row0 nominal.xs) ~prior
    in
    build ~nominal ~law ~common ~initial:(Some initial)

  (** [make_fixed ?mu ~tape ~nominal] is the same Gaussian when the initial
      state is not a latent: it is the nominal's, there is no factor over
      [x₀], and the Gaussian is over the controls alone. *)
  let make_fixed ?mu (tape : ('a, 'b, 'd) tape) ~(nominal : ('a, 'b, 'd) solution) =
    let common = backward_common ?mu tape in
    let law, _, _, _ = backward_lin ~common ~tape ~lin:(lin_of_tape tape) in
    build ~nominal ~law ~common ~initial:None

  (** [sample ~tape t ~log_scale ~key ~n_samples] draws from the Gaussian with
      every covariance scaled by [exp (2 · log_scale)]: the innovations
      [(Δx₀, Δw_k)] are drawn independently with covariances
      [(e^{2s} P⁻¹, e^{2s} Q_uu,k⁻¹)] and the linear response to them is
      rolled out, so a draw is

      {[
        x₀ = x₀⋆ + e^s U_P⁻¹ ξ₀,  u_k = u⋆_k + k_k + K_k δx_k + e^s U_k⁻¹ ξ_k
      ]}

      with [δx] the linearized dynamics' response. [s = 0] is the Laplace
      approximation; one scalar for every block keeps the family stationary in
      the horizon, and is the one parameter worth learning — a shape that the
      local model gets wrong can be widened or narrowed without giving [q] a
      covariance per step.

      The result is a {!solution} with a sample axis between the horizon and
      the dimensions: [xs] is [[horizon + 1; n_samples; n]] and [us]
      [[horizon; n_samples; m]], so row 0 of [xs] is the batch of initial
      states. *)
  let sample
        ~(tape : ('a, 'b, 'd) tape)
        (t : ('a, 'b, 'd) t)
        ~(log_scale : ('a, 'b, 'd) Nx.t)
        ~key
        ~n_samples
    : ('a, 'b, 'd) solution
    =
    if n_samples <= 0
    then invalid_arg "Dilqr.Lqr.Laplace.sample: n_samples must be positive";
    let dtype = Nx.dtype t.feedforward in
    let at = beside t.feedforward in
    let horizon = (Nx.shape t.feedforward).(0) in
    let m = last_dim t.feedforward in
    let n = last_dim t.nominal.xs in
    let scale = exp log_scale in
    let keys = Nx.Rng.split ~n:2 key in
    (* One innovation per step, drawn row by row: [U_k⁻¹ ξ] is a triangular
       solve with the draws as its columns, so the horizon is a batch axis of
       one call. *)
    let xi = Nx.Rng.normal_at at ~key:keys.(1) dtype [| horizon; n_samples; m |] in
    let noise =
      Infix.(
        transpose
          ~axes:[ 0; 2; 1 ]
          (solve_triangular ~triangle:Upper t.quu_chol (transpose ~axes:[ 0; 2; 1 ] xi))
        * scale)
    in
    let dx0 =
      match t.initial with
      | None -> zeros_at at dtype [| n_samples; n |]
      | Some init ->
        let xi0 = Nx.Rng.normal_at at ~key:keys.(0) dtype [| n_samples; n |] in
        let upper = cholesky ~triangle:Upper init.precision in
        let noise =
          matrix_transpose (solve_triangular ~triangle:Upper upper (matrix_transpose xi0))
        in
        Infix.(noise * scale)
    in
    let mean = mean ~tape t in
    let dxs, dus =
      rollout_linearized
        ~tape
        ~gains:{ feedback = t.feedback; feedforward = noise }
        ~x0:dx0
    in
    { xs = Infix.(unsqueeze ~axes:[ 1 ] mean.xs + dxs)
    ; us = Infix.(unsqueeze ~axes:[ 1 ] mean.us + dus)
    }

  (** [log_prob ~tape t ~log_scale ~x0 ~us] is the log density of that
      Gaussian at [(x0, us)], summed over the latents: one value per member of
      a batch. [x0] is [[batch…; n]] and [us] is [[horizon; batch…; m]], the
      shapes a recognition holds its draws in.

      The density is the innovations' — no determinant of the joint — because
      the map between the two is triangular with a unit diagonal. Each factor
      is scaled by [e^{-s}], which is the precision factor of
      [e^{2s} Σ]: the scale needs no case of its own. *)
  let log_prob
        ~(tape : ('a, 'b, 'd) tape)
        (t : ('a, 'b, 'd) t)
        ~(log_scale : ('a, 'b, 'd) Nx.t)
        ~(x0 : ('a, 'b, 'd) Nx.t)
        ~(us : ('a, 'b, 'd) Nx.t)
    =
    let horizon = (Nx.shape us).(0) in
    let m = last_dim us in
    (* The nominal's controls have no batch axis, so it takes one per sample
       axis the caller's [us] has. *)
    let batch_rank = Array.length (Nx.shape x0) - 1 in
    let ones = Array.create ~len:batch_rank 1 in
    let nominal_us =
      reshape (Array.concat [ [| horizon |]; ones; [| m |] ]) t.nominal.us
    in
    let open Infix in
    let du = us - nominal_us in
    let dx0 = x0 - row0 t.nominal.xs in
    let dxs = rollout_controls ~tape ~x0:dx0 du in
    let dx = slice [ R (0, horizon) ] dxs in
    let rows =
      { Innovation_rows.du; dx; ff = t.feedforward; fb = t.feedback; chol = t.quu_chol }
    in
    let _, per_step =
      Rune.scan
        Nx.Ptree.unit
        (Nx.Ptree.instantiate (module Innovation_rows))
        Nx.Ptree.tensor
        (fun () row ->
           (* The innovation is measured from the nominal, so the feedforward
              gain is part of it: it is what moves the mode off the nominal. *)
           let dw = row.Innovation_rows.du - (row.ff + matvec row.fb row.dx) in
           (), precision_logp ~upper:(row.chol * exp (log_scale *$ -1.0)) dw)
        ~init:()
        rows
    in
    let controls = sum ~axes:[ 0 ] per_step in
    match t.initial with
    | None -> controls
    | Some init ->
      let upper = cholesky ~triangle:Upper init.precision in
      let initial =
        precision_logp ~upper:(upper * exp (log_scale *$ -1.0)) (x0 - init.mean)
      in
      initial + controls
end

(* [initial_response ~common ~tape ~residual] is the trajectory perturbation a
   free initial state adds to a tangent solve: the local model's step in [x₀]
   that zeroes the leftover [residual] of the pinned system — the gradient the
   stationarity condition of [x₀] must leave at zero — and the closed loop's
   response to it. The step is Newton's (the model is quadratic in [x₀]) and
   the response is the feedback law's, with no feedforward of its own. *)
let initial_response
      ~(common : ('a, 'b, 'd) common)
      ~(tape : ('a, 'b, 'd) tape)
      ~(residual : ('a, 'b, 'd) Nx.t)
  : ('a, 'b, 'd) Nx.t * ('a, 'b, 'd) Nx.t
  =
  let d = initial_deviation ~common ~value_grad:residual in
  rollout_linearized
    ~tape
    ~gains:{ feedback = common.feedback; feedforward = zeros_like tape.blocks.lu }
    ~x0:d

(* [solve ?mu ~tape ~x0] is the solution of the quadratic problem: the Riccati
   backward pass, then the gain rollout from [x0] under the linearized
   dynamics. *)
let solve ?mu (tape : ('a, 'b, 'd) tape) ~(x0 : ('a, 'b, 'd) Nx.t) : ('a, 'b, 'd) solution
  =
  let common = backward_common ?mu tape in
  let gains, _, _, _ = backward_lin ~common ~tape ~lin:(lin_of_tape tape) in
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
      ~(free_initial : bool)
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
  (* The last carry is the tangent costate at step 0 — the tangent of the
     initial state's gradient, which a free initial state must leave at zero. *)
  let e0, q_hat =
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
  let xs = Infix.(g + v) in
  if not free_initial
  then { xs; us = du }
  else (
    (* The initial state is a variable, so the tangent must also satisfy its
       stationarity: the pinned tangent leaves [e0] there, and the correction is
       the local model's step in [x₀] with the closed loop's response to it.
       [V₀] is the block's marginal precision, so the step is Newton's. *)
    let dx1, du1 = initial_response ~common ~tape ~residual:e0 in
    { xs = Infix.(xs + dx1); us = Infix.(du + du1) })
