(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Differentiable iLQR for discrete-time optimal control on Nx tensors.

    Given a problem — dynamics [x_{k+1} = f(x_k, u_k; θ)], a running loss and
    a final loss — {!Problem.t} carries the description, and the solver
    provides:

    - solves: {!solve_adaptive} is the classic adaptive iLQR. With a JIT
      option, its tensor kernels — the dynamics and cost scans, the
      linearization and Riccati pass, and one line-search rollout — are
      compiled by [Rune.jit], while the host keeps the regularization retries,
      backtracking and convergence test. {!solve_fixed} runs at most a budget
      of iterations with tensor-only control flow — its backtracking and
      damping are traced loops and branches — so it is traceable by [Rune.jit],
      it stops when its batch has converged, and it solves a batch of problems
      at once under one step. Its time loops are [Rune.scan]s and its
      iteration budget a [Rune.iterate], which the compiler stages as loops: a
      program's size grows with neither the horizon nor the budget.

    - implicit derivatives: both modes are attached to the same solve. Forward
      mode runs a tangent map that reuses one factorization of the local model
      at the solution; reverse mode applies the adjoint system of the local LQR
      QP as that map's transpose, solved by the same Riccati machinery and
      chained through the derivative blocks back to θ. Composing

    {[
    Rune.jit
      Nx.Ptree.(inputs problem @-> returns Nx.Ptree.tensor)
      (fun i ->
         Rune.grad (inputs problem) (fun i -> loss (solve_fixed problem ~max_iters:5 i)) i)
    ]}

    compiles the fixed solve and its gradient together, on CPU or GPU.

    - trajectory helpers: {!taus_of}, {!split_taus} and {!loss_of_taus} move
      between per-step arrays and the stacked [(K+1); n+m] tensor the
      differentiable solve returns.

    Everything is a value: a problem is a {!Problem.t} record, its parameter
    structure is an [Nx.Ptree.t], and no module or functor is involved. *)

module Regularisation = Regularisation
module Damping = Damping
module Linesearch = Linesearch
module Lqr = Lqr
include Lqr

(** The problem description. See {!module:Problem}. *)
module Problem = Problem

(** The solve inputs as a structure of tensors. *)
module Inputs = Problem.Inputs

(** [inputs problem] is the structure of the solve inputs: the problem's
    parameters at [theta], the initial state, and every control of the
    initial guess. This is what [Rune.grad], [Rune.jvp], [Rune.custom_jvp]
    and [Rune.jit] take. *)
let inputs = Problem.inputs

let taus_of = Problem.taus_of
let split_taus = Problem.split_taus
let loss_of_taus = Problem.loss_of_taus

module Solver = Solver

(** The differentiable solves. Each attaches the implicit derivative in both
    modes: [Rune.grad] is the VJP, [Rune.jvp] the JVP. *)

(** A differentiable solve: the solve inputs in, the stacked trajectory out. *)
type ('w, 'p, 'd) solve = ('p, ('w, 'd) Problem.value) Inputs.t -> ('w, 'd) Problem.value

let solve_adaptive p ?beam ?parallel = Implicit_grad.solve_adaptive p ?beam ?parallel
let solve_fixed p = Implicit_grad.solve_fixed p

type ('p, 'a) residual = ('p, 'a) Implicit_grad.residual

let bwd = Implicit_grad.bwd
