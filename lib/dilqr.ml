(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Differentiable iLQR for discrete-time optimal control on Nx tensors.

    Given a problem — dynamics [x_{k+1} = f(x_k, u_k; θ)], a running loss and
    a final loss — {!Make} provides:

    - solves: {!Make.val-solve} is the classic adaptive iLQR (regularization
      retry, Armijo backtracking, host-side control flow — eager only);
      {!Make.val-solve_fixed} runs a fixed number of iterations with
      tensor-only control flow and is traceable by [Rune.jit]. Its time loops
      are [Rune.scan]s, which the compiler stages as loops: a program's size
      grows with the iteration count, not with the horizon.

    - differentiable solves: {!Make.val-solve_ilqr} and
      {!Make.val-solve_ilqr_fixed} wrap a solve in a custom VJP whose backward
      pass computes the implicit gradient — the adjoint system of the local
      LQR QP, solved by the same Riccati machinery, chained through the
      derivative blocks back to θ. Composing
      [Rune.jit (fun inputs -> Rune.grad (fun inputs -> loss (solve_ilqr_fixed inputs)) inputs)] compiles the solve and its gradient
      together, on CPU or GPU.

    - trajectory helpers: {!Make.val-taus_of}, {!Make.val-split_taus} and
      {!Make.val-loss_of_taus} move between per-step arrays and the stacked
      [(K+1); n+m] tensor the differentiable solve returns. *)

module Regularisation = Regularisation
module Linesearch = Linesearch
module Lqr = Lqr
include Lqr

(** The problem description. See {!module-type:Lqr.Ctx}. *)
module type Ctx = Lqr.Ctx

module Make (C : Ctx) = struct
  include Implicit_grad.Make (C)

  type value = C.value

  type solution = Solver.t =
    { xs : value array (** states, length [horizon + 1] *)
    ; us : value array (** controls, length [horizon] *)
    ; cost : value (** scalar tensor: running + final loss *)
    }

  (** The adaptive iLQR solve, wrapped with its implicit gradient. Eager
      only — do not jit. [max_iter], [conv_threshold], [c1] (Armijo) and
      [ls_depth] (backtracking depth) tune the solve; [reg0] seeds the
      regularization state. *)
  let solve_ilqr ?max_iter ?conv_threshold ?c1 ?ls_depth ?reg0 (inputs : value Inputs.t)
    : value
    =
    let solve theta x0 us0 =
      let r0 =
        match reg0 with
        | Some r -> r
        | None -> Regularisation.init
      in
      Solver.solve ?max_iter ?conv_threshold ?c1 ?ls_depth ~reg:r0 theta x0 us0
    in
    Rune.custom_vjp
      inputs_ptree
      Nx.Ptree.tensor
      (fun { Inputs.theta; x0; us0 } ->
         let sol = solve theta x0 us0 in
         ( taus_of sol.xs sol.us
         , fun ybar -> bwd { theta; x0; us0; xs = sol.xs; us = sol.us } ybar ))
      inputs

  (** The traceable solve, wrapped with its implicit gradient: put this
      inside [Rune.jit] (with a [Rune.grad] inside it) to compile the solve
      and its gradient together. [iters] fixes the iteration count; [alphas]
      is the trial-step schedule for the in-graph line search; [mu] is the
      fixed regularization; [c1] the Armijo constant. *)
  let solve_ilqr_fixed ~(iters : int) ?alphas ?mu ?c1 (inputs : value Inputs.t) : value =
    Rune.custom_vjp
      inputs_ptree
      Nx.Ptree.tensor
      (fun { Inputs.theta; x0; us0 } ->
         let sol = Solver.solve_fixed ?alphas ?mu ?c1 ~iters theta x0 us0 in
         ( taus_of sol.xs sol.us
         , fun ybar -> bwd { theta; x0; us0; xs = sol.xs; us = sol.us } ybar ))
      inputs
end
