(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Differentiable iLQR for discrete-time optimal control on Nx tensors.

    Given a problem — dynamics [x_{k+1} = f(x_k, u_k; θ)], a running loss and
    a final loss — {!Problem.t} carries the description, and the solver
    provides:

    - solves: {!solve_adaptive} is the classic adaptive iLQR with its
      implicit reverse-mode derivative. With a JIT option, its tensor kernels
      — the dynamics and cost scans, the linearization and Riccati pass, and
      one line-search rollout — are compiled by [Rune.jit], while the host
      keeps the regularization retries, backtracking and convergence test.
      {!solve_fixed} runs a fixed number of iterations with tensor-only
      control flow, is traceable by [Rune.jit], and solves a batch of problems
      at once. Its time loops are [Rune.scan]s, which the compiler stages as
      loops: a program's size grows with the iteration count, not with the
      horizon.

    - forward mode: {!Jvp.solve_adaptive} and {!Jvp.solve_fixed} are the same
      primal solves with a custom JVP instead of a custom VJP. Use them when
      the number of directions is small or when a batch of directions has to
      share one solve. They are explicit because a function cannot carry both
      custom rules in Rune.

    - implicit derivatives: the reverse rule is the adjoint system of the
      local LQR QP, solved by the same Riccati machinery and chained through
      the derivative blocks back to θ. Composing

    {[
    Rune.jit
      Nx.Ptree.(inputs problem @-> returns Nx.Ptree.tensor)
      (fun i ->
         Rune.grad (inputs problem) (fun i -> loss (solve_fixed problem ~iters:5 i)) i)
    ]}

    compiles the fixed solve and its gradient together, on CPU or GPU.

    - trajectory helpers: {!taus_of}, {!split_taus} and {!loss_of_taus} move
      between per-step arrays and the stacked [(K+1); n+m] tensor the
      differentiable solve returns.

    Everything is a value: a problem is a {!Problem.t} record, its parameter
    structure is an [Nx.Ptree.t], and no module or functor is involved. *)

module Regularisation = Regularisation
module Linesearch = Linesearch
module Lqr = Lqr
include Lqr

(** The problem description. See {!module:Problem}. *)
module Problem = Problem

(** The solve inputs as a structure of tensors. *)
module Inputs = Problem.Inputs

(** [inputs problem] is the structure of the solve inputs: the problem's
    parameters at [theta], the initial state, and every control of the
    initial guess. This is what [Rune.grad], [Rune.custom_vjp] and
    [Rune.jit] take. *)
let inputs = Problem.inputs

let taus_of = Problem.taus_of
let split_taus = Problem.split_taus
let loss_of_taus = Problem.loss_of_taus

module Solver = Solver

(** The differentiable solves. The default entry points attach the implicit
    reverse-mode rule; {!module:Jvp} holds the forward-mode forms. *)

(** A differentiable solve: the solve inputs in, the stacked trajectory out. *)
type ('w, 'p) solve = ('p, 'w Problem.value) Inputs.t -> 'w Problem.value

let solve_adaptive p ?beam ?parallel = Implicit_grad.solve_adaptive_vjp p ?beam ?parallel
let solve_fixed p = Implicit_grad.solve_fixed_vjp p

module Jvp = struct
  let solve_adaptive p ?beam ?parallel =
    Implicit_grad.solve_adaptive_jvp p ?beam ?parallel

  let solve_fixed p = Implicit_grad.solve_fixed_jvp p
end

type ('p, 'a) residual = ('p, 'a) Implicit_grad.residual

let bwd = Implicit_grad.bwd
