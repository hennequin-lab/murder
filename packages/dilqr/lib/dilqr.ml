(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Differentiable iLQR for discrete-time optimal control on Nx tensors.

    Given a problem — dynamics [x_{k+1} = f(x_k, u_k; θ)], a running loss and
    a final loss — {!Problem.t} carries the description, and the solver
    provides:

    - solves: {!Solver.solve} is the classic adaptive iLQR (regularization
      retry, Armijo backtracking, host-side control flow — eager only);
      {!Solver.solve_fixed} runs a fixed number of iterations with
      tensor-only control flow and is traceable by [Rune.jit]. Its time loops
      are [Rune.scan]s, which the compiler stages as loops: a program's size
      grows with the iteration count, not with the horizon.

    - differentiable solves: {!solve_ilqr} and {!solve_ilqr_fixed} wrap a
      solve in a custom VJP whose backward pass computes the implicit
      gradient — the adjoint system of the local LQR QP, solved by the same
      Riccati machinery, chained through the derivative blocks back to θ.
      Composing

    {[
      Rune.jit
        Nx.Ptree.(inputs problem @-> returns Nx.Ptree.tensor)
        (fun i ->
           Rune.grad
             (inputs problem)
             (fun i -> loss (solve_ilqr_fixed problem ~iters:5 i))
             i)
    ]}

    compiles the solve and its gradient together, on CPU or GPU.

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
include Implicit_grad
