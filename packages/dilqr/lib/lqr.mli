(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The linear-quadratic core: one LQR problem as data, its solve, the Riccati
    factors the solve and every derivative of it share, the two implicit
    derivatives, and the rollouts.

    {b The problem.} A {!tape} is the quadratic problem the iLQR iteration
    solves at one linearization, written in deviation form: with
    [τ = (x, u)] the deviation of the state and control from the nominal
    trajectory, minimize

    {[
      Σ_{k<K} (½ τ_kᵀ C_k τ_k + l_τ,k ᵀ τ_k) + ½ x_Kᵀ φ_xx x_K + φ_x ᵀ x_K
      s.t.  x_{k+1} = A_k x_k + B_k u_k,  x_0 = x₀
    ]}

    [C_k] is the loss Hessian ([lxx], [luu], [lux]), [l_τ] the loss gradient
    ([lx], [lu]), and [φ_x], [φ_xx] the terminal gradient and Hessian. The
    constraint carries no constant term: the deviation dynamics of a rollout
    is linear, so the affine part of the local model is already absorbed by
    {!lin}.

    {b The derivative.} {!solve} solves that problem by a Riccati backward pass
    and a gain rollout. {!solve_jvp} is its derivative in forward mode,
    {!backward_common} and {!backward_lin} the two halves of the backward pass
    the derivative reuses.

    The derivative is {e implicit}: it is the derivative of the map
    [data ↦ argmin] of the quadratic problem above, not of any iteration that
    finds the argmin. Two ingredients make it cheap:

    - the factorization of the regularized [Q_uu] and the feedback gain [K]
      do not depend on the linear terms, so they are computed once
      ({!backward_common}) and reused by the primal solve and by every
      tangent ({!backward_lin}, {!solve_jvp});
    - no tangent of the quadratic blocks ([lxx], [luu], [lux], [φ_xx]) is
      ever needed: the tangent of [l_τ] and [φ_x] and the tangent of the
      dynamics' constant term carry the whole dependence on the problem's
      parameters.

    {b Batches.} Every per-step tensor may carry leading batch axes between
    the horizon and the dimensions ([horizon; batch...; n]), and every
    function reduces over the horizon and dimension axes only, so matrix
    products broadcast over the batch: these functions are batch-correct as
    they stand. A {e tangent} axis is just such a batch axis, which is what
    lets one call serve a whole batch of tangents ([horizon; k; n; n], say).
    Under [Rune.vmap] the map's lane axis is invisible, so the code reads as
    unbatched code either way. *)

(** The per-step blocks of one linearization, with the horizon as their
    leading axis. Construct one with [Nx.Ptree.instantiate (module Rows)] and
    let a [Rune.scan] over the horizon fill it in; that is how
    [Dilqr.Solver.blocks] builds a tape, and how a caller assembles a problem
    of their own. *)
module Rows : sig
  type 'a t =
    { a : 'a (** ∂f_k/∂x, shape [horizon; n; n] *)
    ; b : 'a (** ∂f_k/∂u, shape [horizon; n; m] *)
    ; lx : 'a (** ∂l_k/∂x, shape [horizon; n] *)
    ; lu : 'a (** ∂l_k/∂u, shape [horizon; m] *)
    ; lxx : 'a (** ∂²l_k/∂x², shape [horizon; n; n] *)
    ; luu : 'a (** ∂²l_k/∂u², shape [horizon; m; m] *)
    ; lux : 'a (** ∂²l_k/∂u∂x, shape [horizon; m; n] *)
    }

  (** [walk c rows] is the structure of [rows], for [Rune.scan] and
      [Nx.Ptree.map]. Exposed because [Nx.Ptree.instantiate (module Rows)] is
      how a tape is built and traversed. *)
  val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
end

(** The quadratic problem the iLQR iteration solves at one linearization. *)
type ('a, 'b) tape =
  { blocks : ('a, 'b) Nx.t Rows.t
  ; fx : ('a, 'b) Nx.t (** ∂φ/∂x at the final state, shape [n] *)
  ; fxx : ('a, 'b) Nx.t (** ∂²φ/∂x² at the final state, shape [n; n] *)
  }

(** A control law: [u_k = k_k + K_k x_k]. *)
type ('a, 'b) gains =
  { feedback : ('a, 'b) Nx.t (** K_k, shape [horizon; m; n] *)
  ; feedforward : ('a, 'b) Nx.t (** k_k, shape [horizon; m] *)
  }

(** The linear cost terms of a Riccati pass. The primal pass uses the tape's
    own gradients ({!lin_of_tape}); the adjoint pass substitutes cotangent
    slices. *)
type ('a, 'b) lin =
  { rlx : ('a, 'b) Nx.t (** [horizon; n] *)
  ; rlu : ('a, 'b) Nx.t (** [horizon; m] *)
  ; rfx : ('a, 'b) Nx.t (** [n] *)
  }

(** [lin_of_tape tape] is the tape's own linear terms: the loss gradients and
    the terminal gradient. *)
val lin_of_tape : (float, 'b) tape -> (float, 'b) lin

(** [sym x] is the symmetric part of [x]: [(x + xᵀ) / 2]. Riccati value
    Hessians are symmetric, so a computed one is symmetrized; a cotangent on
    a symmetric block is symmetrized for the same reason. *)
val matvec : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [matvec m v] is [m @ v] over every leading axis: a matrix broadcast against
    a batch of vectors. [Nx.matmul] would contract a batched vector's batch
    axis with the matrix, so the vector is promoted to a column and squeezed
    back. Every product with a value gradient here goes through it. *)

val sym : (float, 'b) Nx.t -> (float, 'b) Nx.t

(** The per-step factors that the primal and every tangent of one LQR solve
    share. Build one with {!backward_common}.

    It holds the upper Cholesky factor of the regularized [Q_uu] and the
    feedback gain [K = -Q_uu⁻¹ Q_ux], both with the horizon leading. That is
    all a tangent needs: the factor to solve for its feedforward gain, and [K]
    to roll a tangent state into a tangent control. Nothing here depends on the
    linear terms of the problem, so one copy serves a whole batch of
    directions, at [horizon · (m² + m·n)] — against the
    [O (horizon · n²)] tape it is built from. {!costate}, the only other
    quantity a tangent batch shares, is computed separately and just as
    cheaply. *)
type ('a, 'b) common =
  { quu_chol : ('a, 'b) Nx.t (** the upper Cholesky factor of [Q_uu^μ], [horizon; m; m] *)
  ; feedback : ('a, 'b) Nx.t (** K_k = -Q_uu⁻¹ Q_ux, [horizon; m; n] *)
  }

(** [backward_common ?mu tape] runs the Hessian recursion of the Riccati
    backward pass: per step, [Q_xx], [Q_ux], [Q_uu], the feedback gain
    [K = -Q_uu⁻¹ Q_ux], and the value Hessian [V = sym (Q_xx + Q_uxᵀ K)] the
    next step is built from, from the last step to the first. Only the factor
    and the gain are retained: [V] is internal to the recursion.

    [mu] is Levenberg-Marquardt damping, a scalar tensor: [Q_uu + μ BᵀB] and
    [Q_ux + μ BᵀA] are factorized instead, which is what makes the pass stable
    when the loss curvature alone is not positive definite. It is a tensor and
    not a [float] so that a schedule can be carried through a traced loop;
    omit it for exactly no damping (omitting it skips the terms, while passing
    a zero costs three products per step).

    The result does not depend on the linear terms of the problem, which is
    the whole point: the primal solve and every tangent of it share it. A
    regularized [Q_uu] that is not positive definite gives a NaN Cholesky
    factor, and every quantity computed from it is NaN: a caller that damps
    retries with a larger [mu] (see [Dilqr.Solver.solve]). *)
val backward_common : ?mu:(float, 'b) Nx.t -> (float, 'b) tape -> (float, 'b) common

(** [costate tape] is the primal costate [λ_{k+1}] at each step, shape
    [horizon; n]: row [k] is the value gradient the local model's own linear
    terms give at the state step [k] starts from ([λ_K = ∂φ/∂x], and
    [λ_k = A_kᵀ λ_{k+1} + ∂l_k/∂x]).

    A tangent of the local model needs it, and nothing about it depends on the
    tangent: compute it once and share it with every direction of a batch, as
    with {!backward_common}. One [Rune.scan] over the horizon. *)
val costate : (float, 'b) tape -> (float, 'b) Nx.t

(** [backward_lin ~common ~tape ~lin] is the linear half of the same pass,
    with the factorization already in hand: the feedforward gain
    [k_k = -Q_uu⁻¹ Q_u] and the value gradient it threads, one step at a
    time, and the expected-decrease coefficients [d1], [d2] (one scalar per
    member): [d1 = Σ k_kᵀ Q_u ≤ 0], the modelled first-order decrease, and
    [d2 = Σ k_kᵀ Q_uu^μ k_k ≥ 0], the curvature the gains actually used. Since
    [Q_uu^μ k_k = -Q_u] the two are negatives of each other, and [d2] is
    returned for the line search's model [α d1 + ½ α² d2] rather than
    recomputed.

    Cheap: two products with the value gradient and one triangular solve per
    step. Called once for the primal linear terms, and once per tangent batch
    by {!solve_jvp}. *)
val backward_lin
  :  common:(float, 'b) common
  -> tape:(float, 'b) tape
  -> lin:(float, 'b) lin
  -> (float, 'b) gains * (float, 'b) Nx.t * (float, 'b) Nx.t

(** [backward ?mu ~tape ~lin] is {!backward_common} followed by
    {!backward_lin}: the whole Riccati pass over the horizon as one scan. *)
val backward
  :  ?mu:(float, 'b) Nx.t
  -> (float, 'b) tape
  -> lin:(float, 'b) lin
  -> (float, 'b) gains * (float, 'b) Nx.t * (float, 'b) Nx.t

(** A solution trajectory, as one tensor per trajectory. *)
type ('a, 'b) solution =
  { xs : ('a, 'b) Nx.t (** [horizon + 1; n] *)
  ; us : ('a, 'b) Nx.t (** [horizon; m] *)
  }

(** [solve ?mu ~tape ~x0] is the solution of the quadratic problem: the Riccati
    backward pass, then the gain rollout from [x0] under the linearized
    dynamics. Pure tensor code with static control flow: traceable and
    jittable, and differentiable — but differentiating this code differentiates
    the Cholesky factorization too. {!solve_jvp} is the derivative the iLQR
    solver needs.

    [mu] damps the backward pass (see {!backward_common}); the rollout is the
    same either way. *)
val solve
  :  ?mu:(float, 'b) Nx.t
  -> (float, 'b) tape
  -> x0:(float, 'b) Nx.t
  -> (float, 'b) solution

(** The tangent of a tape's first-order data: what {!solve_jvp} consumes.

    The fields are the tangents of [A], [B], the dynamics value [f], the loss
    gradient, and the terminal gradient. The tangents of the quadratic blocks
    ([lxx], [luu], [lux], [φ_xx]) are not part of a tangent: the argmin's
    derivative is taken through the KKT system, whose matrix is the primal's,
    so they never enter. *)
type ('a, 'b) tangent =
  { da : ('a, 'b) Nx.t (** ∂A_k along the tangent, shape [horizon; n; n] *)
  ; db : ('a, 'b) Nx.t (** ∂B_k, shape [horizon; n; m] *)
  ; df : ('a, 'b) Nx.t
    (** ∂f_k, the tangent of the dynamics' value at the nominal step, shape
        [horizon; n]. It is what drives the tangent trajectory; a change of
        [A] or [B] at a fixed nominal does not. *)
  ; dlx : ('a, 'b) Nx.t (** ∂(∂l_k/∂x), shape [horizon; n] *)
  ; dlu : ('a, 'b) Nx.t (** ∂(∂l_k/∂u), shape [horizon; m] *)
  ; dfx : ('a, 'b) Nx.t (** ∂(∂φ/∂x) at the final state, shape [n] *)
  }

(** [solve_jvp ~common ~tape ~costate ~tangent ~dx0] is the tangent of
    {!solve}'s solution for the data tangent {!tangent} and the initial state
    tangent [dx0]: the derivative of the argmin along one direction in the
    space of problems.

    Linear in the tangent data and in [dx0], which is what lets one call serve
    a whole batch of directions: give the {!tangent} tensors and [dx0] a
    leading direction axis ([horizon; k; n; n] and so on) and the result is
    [k; horizon + 1; n] and [k; horizon; m]. The factorization in [common] and
    the [costate] are then paid for once, outside the directions.

    This takes the tangents of the first-order data as a materialized
    {!tangent}, which is the largest direction-multiplied tensor of the whole
    computation. A problem-level caller should prefer
    [Dilqr.Solver.tangent_solve], which runs the same {!tangent_step} and
    {!tangent_roll_step} but computes those tangents step by step and never
    stores them.

    Three sweeps, and no Cholesky factor or triangular solve differentiated:

    {v
      G_0 = dx0,   G_{k+1} = A_k G_k + δf_k
      X_k = δlx_k + δA_kᵀ λ_{k+1} + lxx_k G_k      (X_K = δfx + φ_xx G_K)
      U_k = δlu_k + δB_kᵀ λ_{k+1} + lux_k G_k
      E_K = X_K
      q̂_k = U_k + B_kᵀ E_{k+1}
      E_k = X_k + A_kᵀ E_{k+1} + K_kᵀ q̂_k
      v_0 = 0
      δu_k = -Q_uu^μ⁻¹ q̂_k + K_k v_k,   v_{k+1} = A_k v_k + B_k δu_k
      δx_k = G_k + v_k
    v}

    with [λ] the [costate] of the tape's own linear terms and [Q_uu^μ] the
    regularized factor in [common]. So a direction costs one sweep of
    [n]-vectors and two products with the tape's Hessian blocks per step: it is
    cheaper than the primal's Hessian recursion, which is the point of sharing
    [common].

    {b The adjoint law.} This map is the transpose of the implicit VJP that
    [Dilqr] runs ([Dilqr.bwd]) within the idealization they share: for every
    cotangent [ȳ] on the trajectory and every tangent,

    {v
      ⟨ȳ, δτ⟩ = ⟨cotangent on A, δA⟩ + ⟨cotangent on B, δB⟩
              + ⟨cotangent on f, δf⟩ + ⟨cotangent on l_τ, δl_τ⟩
              + ⟨cotangent on φ_x, δφ_x⟩ + ⟨gradient on x₀, dx0⟩
    v}

    so a scalar objective gets the same derivative in both modes. That law is
    the test: it holds pointwise, needs no step size, and fails loudly if
    either side is off by a term. The test suite checks it for random tapes at
    fixed points, and checks {!solve_jvp} against central differences of
    {!solve}. *)
val solve_jvp
  :  common:(float, 'b) common
  -> tape:(float, 'b) tape
  -> costate:(float, 'b) Nx.t
  -> tangent:(float, 'b) tangent
  -> dx0:(float, 'b) Nx.t
  -> (float, 'b) solution

(** [rollout ~dyn ~x0 ~xs ~us ~gains ~alpha] integrates the {e true}
    dynamics under the control law [û_k = ū_k + α k_k + K_k (x̂_k - x̄_k)],
    returning the new state path [[horizon + 1; n]] and control path
    [[horizon; m]] as one scan. [xs] is [[horizon + 1; n]] and [us] is
    [[horizon; m]], with any batch axes between the horizon and the
    dimensions. One line-search trial. *)
val rollout
  :  dyn:(k:(float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t)
  -> x0:(float, 'b) Nx.t
  -> xs:(float, 'b) Nx.t
  -> us:(float, 'b) Nx.t
  -> gains:(float, 'b) gains
  -> alpha:float
  -> (float, 'b) Nx.t * (float, 'b) Nx.t

(** [rollout_tensor ~dyn ~x0 ~xs ~us ~gains ~alpha] is {!rollout} with the
    step size as a scalar tensor. A traced line search passes [alpha] as an
    argument of the compiled trial, so one program serves every trial step;
    with a host [float] each step would be a separate program. *)
val rollout_tensor
  :  dyn:(k:(float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t)
  -> x0:(float, 'b) Nx.t
  -> xs:(float, 'b) Nx.t
  -> us:(float, 'b) Nx.t
  -> gains:(float, 'b) gains
  -> alpha:(float, 'b) Nx.t
  -> (float, 'b) Nx.t * (float, 'b) Nx.t

(** [rollout_linearized ~tape ~gains ~x0] integrates the linearized dynamics
    [x_{k+1} = A_k x_k + B_k u_k] under [u_k = k_k + K_k x_k] from [x0], as
    one scan. It is {!solve}'s forward pass, and — with the gains of a
    perturbed pass — the last step of the implicit VJP. *)
val rollout_linearized
  :  tape:(float, 'b) tape
  -> gains:(float, 'b) gains
  -> x0:(float, 'b) Nx.t
  -> (float, 'b) Nx.t * (float, 'b) Nx.t

(** The per-step inputs of the tangent backward pass, with the horizon as their
    leading axis. A composition that streams the linearization instead of
    materializing a {!tangent} (as [Dilqr.Solver.tangent_solve] does) fills one
    of these per step and calls {!tangent_step}. Construct the structure with
    [Nx.Ptree.instantiate (module Tangent_rows)]. *)
module Tangent_rows : sig
  type 'a t =
    { a : 'a (** ∂f_k/∂x, shape [horizon; n; n] *)
    ; b : 'a (** ∂f_k/∂u, shape [horizon; n; m] *)
    ; da : 'a (** its tangent, [horizon; n; n] *)
    ; db : 'a (** [horizon; n; m] *)
    ; lxx : 'a (** ∂²l_k/∂x², [horizon; n; n] *)
    ; lux : 'a (** ∂²l_k/∂u∂x, [horizon; m; n] *)
    ; lx : 'a (** the costate [λ_{k+1}], [horizon; n] *)
    ; dlx : 'a (** the tangent of the loss gradient, [horizon; n] *)
    ; dlu : 'a (** [horizon; m] *)
    ; g : 'a (** the tangent trajectory [G_k], [horizon; n] *)
    ; fb : 'a (** the feedback gain [K], [horizon; m; n] *)
    }

  val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
end

(** The per-step inputs of the tangent gain rollout. *)
module Tangent_roll_rows : sig
  type 'a t =
    { a : 'a (** ∂f_k/∂x, [horizon; n; n] *)
    ; b : 'a (** ∂f_k/∂u, [horizon; n; m] *)
    ; chol : 'a (** the upper Cholesky factor of [Q_uu^μ], [horizon; m; m] *)
    ; fb : 'a (** the feedback gain [K], [horizon; m; n] *)
    ; q : 'a (** the tangent feedforward [q_k], [horizon; m] *)
    }

  val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
end

(** [tangent_step ~e row] is [(q, e')]: one step of the tangent backward pass,
    the tangent feedforward [q_k] and the next [e]. Linear in its inputs, so it
    serves a batch of directions as it stands. *)
val tangent_step
  :  e:(float, 'b) Nx.t
  -> (float, 'b) Nx.t Tangent_rows.t
  -> (float, 'b) Nx.t * (float, 'b) Nx.t

(** [tangent_roll_step ~v row] is [(δu, v')]: one step of the tangent gain
    rollout, a triangular solve against the shared factor. *)
val tangent_roll_step
  :  v:(float, 'b) Nx.t
  -> (float, 'b) Nx.t Tangent_roll_rows.t
  -> (float, 'b) Nx.t * (float, 'b) Nx.t
