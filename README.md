# Dilqr

Differentiable iterative LQR for discrete-time optimal control, built on the
[Raven](https://github.com/raven-ml/raven) libraries: Nx tensors, Rune autodiff
and JIT compilation.

Dilqr solves problems of the form

```
min_{u_0..u_{K-1}}  φ(x_K) + Σ_{k<K} l(x_k, u_k)    s.t.  x_{k+1} = f(x_k, u_k; θ)
```

with iterative LQR, and **differentiates through the solve**. The gradient is
*implicit* rather than obtained by unrolling the iLQR iterations: the backward
pass is the adjoint system of the local LQR QP, solved by the same Riccati
machinery as the forward pass, then chained back to the problem parameters
with a nested `Rune.grad`.

Everything below the solve level is a pure tensor computation with static
control flow, and every time loop is a `Rune.scan`. One traced step therefore
stands for the whole horizon, which `Rune.jit` compiles as a loop: a compiled
program's size grows with the iteration count, **not with the horizon**.

## Highlights

- **Adaptive solve** (`Solver.solve`) — classic iLQR: Levenberg–Marquardt
  regularization with Cholesky retries, Armijo backtracking line search,
  convergence checks. Host-side control flow, runs eagerly.
- **Traceable solve** (`Solver.solve_fixed`) — fixed iteration count, fixed
  regularization, and a line search realized as a masked argmin over a fixed
  trial schedule, so there is no host-side control flow: the solve traces and
  JIT-compiles.
- **Implicit gradients** (`solve_ilqr`, `solve_ilqr_fixed`) — differentiable
  solves wrapped in a `Rune.custom_vjp`. Put them inside `Rune.jit` with a
  `Rune.grad` inside, and the forward solve and the implicit VJP compile
  together into one program, on CPU or GPU.
- **Static shapes, dynamic time** — state and control dimensions and the
  horizon are static, which is what the compiled scans are built from. The
  step index `k` travels as a scalar tensor, so time-varying problems index
  their parameters with it.
- **One tensor core** (`Dilqr.Lqr`) — the linearization tape, Riccati
  backward pass, gain rollouts and perturbed sweeps, exposed for anyone who
  wants the pieces rather than the solver.

## Quick start

A problem is a `Dilqr.Ctx`: dimensions, dtype, dynamics, running and final
losses, plus a structure of parameters. Here is a double integrator, written
in a module so the functor application stays readable:

```ocaml
(* problem.ml *)
open Base
module Nx = Nx

module Integrator = struct
  module Ctx = struct
    type 'a params = { mass : 'a } [@@deriving ptree]

    module Params = struct
      type 'a t = 'a params

      let walk = walk_params
    end

    type w = Nx.float64_elt
    type value = (float, w) Nx.t

    let dtype = Nx.float64
    let n = 2
    let m = 1
    let horizon = 20
    let verbose = false
    let dt = 0.1

    let dynamics { mass } ~k:_ ~x ~u =
      let pos = Nx.slice [ R (0, 1) ] x
      and vel = Nx.slice [ R (1, 2) ] x in
      let force = Nx.div u mass in
      Nx.concatenate
        ~axis:0
        [ Nx.add pos (Nx.mul_s vel dt); Nx.add vel (Nx.mul_s force dt) ]

    let running_loss _ ~k:_ ~x ~u =
      Nx.sum
        (Nx.add (Nx.mul_s (Nx.sum (Nx.mul x x)) 0.5) (Nx.mul_s (Nx.mul u u) 0.05))

    let final_loss _ ~x = Nx.mul_s (Nx.sum (Nx.mul x x)) 5.0
  end

  module D = Dilqr.Make (Ctx)
end
```

`[@@deriving ptree]` (from the `ppx_ptree` package, shipped with Raven) turns
the parameter record into a `Nx.Ptree`, which is what lets `Rune.grad` and
`Rune.jit` see each parameter as its own leaf.

```ocaml
(* solve.ml *)
open Base
open Stdio
module Nx = Nx

let vec64 a = Nx.create Nx.float64 [| Array.length a |] a
let scalar64 v = Nx.scalar Nx.float64 v

let theta : Integrator.Ctx.value Integrator.Ctx.Params.t = { mass = scalar64 1.0 }
let x0 = vec64 [| 1.0; 0.0 |]

let us0 =
  Array.init Integrator.Ctx.horizon ~f:(fun _ -> Nx.create Nx.float64 [| 1 |] [| 0.0 |])

let inputs : Integrator.D.value Integrator.D.Inputs.t =
  { Integrator.D.Inputs.theta; x0; us0 }

(* 1. Adaptive solve, eager. *)
let sol = Integrator.D.Solver.solve theta x0 us0
let () = printf "cost: %g\n%!" (Nx.item [] sol.cost)

(* 2. Differentiable solve: any scalar loss of the trajectory gets its
   implicit gradient in (theta, x0). *)
let objective (inputs : Integrator.D.value Integrator.D.Inputs.t) =
  Integrator.D.loss_of_taus
    inputs.Integrator.D.Inputs.theta
    (Integrator.D.solve_ilqr_fixed ~iters:5 inputs)

let grads = Rune.grad Integrator.D.inputs_ptree objective inputs

(* 3. The same thing compiled: solve and implicit VJP in one program. *)
let grads_compiled =
  Rune.jit
    Nx.Ptree.(Integrator.D.inputs_ptree @-> returns Integrator.D.inputs_ptree)
    (fun i -> Rune.grad Integrator.D.inputs_ptree objective i)
    inputs
```

The user code needs `ppx_ptree` preprocessed and the usual libraries:

```dune
(executable
 (name solve)
 (libraries base stdio nx rune dilqr)
 (preprocess
  (pps ppx_ptree)))
```

A few things worth knowing:

- `solve_ilqr` (adaptive forward, eager backward) is for small problems and
  reference results; do not put it inside `Rune.jit`. Use
  `solve_ilqr_fixed ~iters` for anything compiled.
- The differentiable solves return the trajectory **stacked** as one
  `[K+1; n+m]` tensor: row `k` is `(x_k; u_k)` for `k < K`, row `K` is
  `(x_K; 0)`. `split_taus` recovers the per-step arrays.
- The gradient with respect to the initial control guess `us0` is exactly
  zero: the idealized solve is a fixed point and does not depend on it.

## API tour

Everything lives under `Dilqr`, with `Dilqr.Make (C)` as the entry point.

- **`Dilqr.Make (C)`** — the problem functor. `C : Dilqr.Ctx` fixes `n`, `m`,
  `horizon`, the dtype, `dynamics`, `running_loss`, `final_loss` and the
  parameter structure `Params`.

- **`Solver.solve`** — adaptive iLQR. Defaults: `max_iter = 20`,
  `conv_threshold = 1e-6`, `c1 = 1e-2`, `ls_depth = 8`, seeded
  regularization `Regularisation.init`. Returns `{ xs; us; cost }`.

- **`Solver.solve_fixed`** — fixed-iteration iLQR with tensor-only control
  flow: `~iters`, an optional trial schedule `alphas` (default
  `1; 1/2; …`, 8 trials), fixed `mu` (default none) and Armijo `c1`.
  Traceable, JIT-compilable, never reads a tensor on the host.

- **`Solver.rollout`** — integrate the dynamics under a control sequence and
  return the state path and total cost; useful for evaluating an existing
  trajectory.

- **`Solver.blocks`** — linearize dynamics and losses along a trajectory and
  return the tape.

- **`solve_ilqr`, `solve_ilqr_fixed`** — the differentiable solves: a solve
  wrapped in `Rune.custom_vjp` whose backward pass is the implicit VJP.
  `split_taus`, `taus_of` and `loss_of_taus` are the trajectory helpers they
  work with.

- **`Inputs` / `inputs_ptree`** — the solve inputs (`theta`, `x0`, `us0`) as a
  structure of tensors and its `Nx.Ptree.t`: the signature that `Rune.grad`,
  `Rune.custom_vjp` and `Rune.jit` take.

- **`Dilqr.Lqr`** — the tensor core: `Rows`, `tape`, `gains`, `lin`,
  `lin_of_tape`, `riccati_step`, `backward`, `rollout`, `rollout_linearized`.
  Pure tensor math with host-static control flow; safe to trace and jit.

- **`Dilqr.Regularisation`** — the Levenberg–Marquardt schedule
  (`{ delta; mu }`, `init`, `increase`, `decrease`), with `mu` floored at
  `1e-6`.

- **`Dilqr.Linesearch`** — trial-step schedules (`schedule ~n ()` is
  `1; 1/2; 1/4; …`) and the Armijo predicate (`accepts`).

## How the gradient works

At the end of a solve, iLQR has built a local quadratic model of the problem
along the solution trajectory: minimize `Σ ½ τᵀ C τ + cᵀ τ` subject to the
linearized dynamics `x_{k+1} = A x_k + B u_k + a_k`, with `τ = (x, u)` and `C`
(resp. `c`) the loss-Hessian blocks (resp. linear terms).

The adjoint of that QP's KKT system is *another* LQR problem, so the implicit
VJP is computed with the same machinery as the forward pass:

1. a perturbed Riccati backward pass whose linear terms are the incoming
   cotangent `ȳ`, followed by a homogeneous linearized rollout from `δx₀ = 0`,
   yields the trajectory perturbation `δτ` and the costate perturbation `δλ`;
2. block cotangents on the tape — `∂/∂a_k = −δλ_{k+1}`,
   `∂/∂C_k = −sym(δτ_k τ_kᵀ)`, `∂/∂A_k = −(δλ_{k+1} x_kᵀ + δx_k δλ_{k+1}ᵀ)`,
   and so on;
3. a nested `Rune.grad` over the scalar surrogate `Σ ⟨cotangent, block(θ)⟩`
   chains those cotangents to `θ`, recomputing the blocks from `θ` exactly as
   the forward pass does.

Under a converged solve this is the true derivative of the solution map. With
`solve_ilqr_fixed ~iters`, the gradient converges to the adaptive solve's as
iterations accumulate (see the pendulum example, which prints both).

## Examples

Two self-contained examples live in [`example/`](example/):

- [`example/pendulum_swingup.ml`](example/pendulum_swingup.ml) — pendulum
  swing-up from hanging to upright. Demonstrates the adaptive solve, the
  compiled fixed-iteration solve, and the implicit gradient both eager and
  compiled.

  ```console
  $ dune exec ./example/pendulum_swingup.exe
  ```

- [`example/integrator_mpc.ml`](example/integrator_mpc.ml) — receding-horizon
  MPC on a double integrator, warm-starting each solve from the previous one.
  The full solve step (and a gradient step) is compiled with `Rune.jit`.

  ```console
  $ dune exec ./example/integrator_mpc.exe
  ```

## Building and testing

From the `dilqr` directory:

```console
$ dune build
$ dune runtest                 # or: dune exec ./test/test_dilqr.exe
```

The library depends on OCaml `>= 5.5`, dune `>= 3.24`, `base`, `stdio` and the
Raven libraries `nx` and `rune`; the test suite additionally uses `windtrap`
and `ppx_ptree`. Both `nx` and `rune` come with Raven, so a suitable opam
switch (or an in-tree Raven checkout) is enough.

The test suite covers the regularization schedule, LQR correctness (double
integrator: cost decrease and control stationarity, checked with independent
autodiff), nonlinear convergence (pendulum), implicit gradients against
central finite differences, the fixed solve's competitiveness with the
adaptive one, and JIT round-trips of both the solve and its gradient.

## Status

Research code, developed alongside the Raven libraries; the interface may
still move.

## License

ISC — see `dune-project`.
