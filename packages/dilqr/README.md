# Dilqr

Differentiable iterative LQR for discrete-time optimal control, built on the
[Raven](https://github.com/raven-ml/raven) libraries: Nx tensors, Rune autodiff
and JIT compilation.

Dilqr solves problems of the form

```
min_{u_0..u_{K-1}}  φ(x_K) + Σ_{k<K} l(x_k, u_k)    s.t.  x_{k+1} = f(x_k, u_k; θ)
```

with iterative LQR, and **differentiates through the solve**. The derivative is
*implicit* rather than obtained by unrolling the iLQR iterations, and both
modes are available: the **reverse** mode solves the adjoint system of the
local LQR QP with the same Riccati machinery as the forward pass and chains the
result to the problem parameters with a nested `Rune.grad`; the **forward**
mode reuses one factorization of that QP for a whole batch of directions. The
two are transposes of each other.

This machinery is what makes iLQR usable as an *inference engine* and not just
a controller: in iLQR-VAE ([Schimel et al., 2022](#references)), iLQR infers
the latent inputs and state trajectory that best explain neural population
activity inside an input-driven sequential generative model, and the solve is
differentiated to fit the model end to end. Dilqr provides that solver core on
the modern Raven stack.

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
- **Implicit JVPs** (`solve_ilqr_jvp`, `solve_ilqr_fixed_jvp`) — the same
  solves wrapped in a `Rune.custom_jvp`. One Riccati backward pass at the
  solution yields the factors every tangent shares — the Cholesky factor of
  the regularized `Q_uu`, the feedback gain, and the costate — and each
  direction then costs one cheap sweep plus one rollout. Map `Rune.jvp` over
  the directions with `Rune.vmap` and the primal solve is computed once.
- **Static shapes, dynamic time** — state and control dimensions and the
  horizon are static, which is what the compiled scans are built from. The
  step index `k` travels as a scalar tensor, so time-varying problems index
  their parameters with it.
- **One tensor core** (`Dilqr.Lqr`) — the linearization tape, Riccati
  backward pass, gain rollouts and perturbed sweeps, exposed for anyone who
  wants the pieces rather than the solver.

## Quick start

A problem is a `Dilqr.Problem.t` record: dimensions, dtype, dynamics, running
and final losses, plus a structure of parameters. Here is a double
integrator:

```ocaml
(* problem.ml *)
open Base
module Nx = Nx

module Integrator = struct
  module Params = struct
    type 'a t = { mass : 'a } [@@deriving ptree]
  end

  let dt = 0.1
  let n = 2
  let m = 1
  let horizon = 20
  let verbose = false

  let dynamics { Params.mass } ~k:_ ~x ~u =
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

  let problem : (Nx.float64_elt, Nx.float64_t Params.t) Dilqr.Problem.t =
    { dtype = Nx.float64
    ; n
    ; m
    ; horizon
    ; verbose
    ; params = Nx.Ptree.instantiate (module Params)
    ; dynamics
    ; running_loss
    ; final_loss
    }
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

let theta : Nx.float64_t Integrator.Params.t = { mass = scalar64 1.0 }
let x0 = vec64 [| 1.0; 0.0 |]

let us0 =
  Array.init Integrator.horizon ~f:(fun _ -> Nx.create Nx.float64 [| 1 |] [| 0.0 |])

let inputs : (Nx.float64_t Integrator.Params.t, Nx.float64_t) Dilqr.Inputs.t =
  { Dilqr.Inputs.theta; x0; us0 }

(* 1. Adaptive solve, eager. *)
let sol = Dilqr.Solver.solve Integrator.problem theta x0 us0
let () = printf "cost: %g\n%!" (Nx.item [] sol.cost)

(* 2. Differentiable solve: any scalar loss of the trajectory gets its
   implicit gradient in (theta, x0). *)
let objective (inputs : (Nx.float64_t Integrator.Params.t, Nx.float64_t) Dilqr.Inputs.t) =
  Dilqr.loss_of_taus
    Integrator.problem
    inputs.Dilqr.Inputs.theta
    (Dilqr.solve_ilqr_fixed Integrator.problem ~iters:5 inputs)

let inputs_ptree = Dilqr.inputs Integrator.problem
let grads = Rune.grad inputs_ptree objective inputs

(* 3. The same thing compiled: solve and implicit VJP in one program. *)
let grads_compiled =
  Rune.jit
    Nx.Ptree.(inputs_ptree @-> returns inputs_ptree)
    (fun i -> Rune.grad inputs_ptree objective i)
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

Everything lives under `Dilqr`. A problem is a `Dilqr.Problem.t` value, and
every solver and helper takes it as its first argument.

- **`Dilqr.Problem.t`** — the problem as a value: `dtype`, `n`, `m`,
  `horizon`, `verbose`, the parameter structure `params : 'p Nx.Ptree.t`, and
  the functions `dynamics`, `running_loss` and `final_loss`. The structure
  is built with `Nx.Ptree.instantiate (module Params)`. Being a record,
  problems can be stored, passed and chosen at runtime.

- **`Dilqr.inputs`** — the structure of the solve inputs (`theta`, `x0`,
  `us0`), built by walking the problem's parameter structure: what
  `Rune.grad`, `Rune.custom_vjp` and `Rune.jit` take. `Dilqr.Inputs.t` is the
  corresponding record.

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
  work with, each taking the problem first.

- **`solve_ilqr_jvp`, `solve_ilqr_fixed_jvp`** — the differentiable solves in
  forward mode: the same solves wrapped in `Rune.custom_jvp`. `Rune.jvp`
  applied to them is the trajectory tangent, computed without differentiating a
  Cholesky factor. See [Forward mode](#forward-mode).

- **`Dilqr.Lqr`** — the tensor core: `Rows`, `tape`, `gains`, `lin`,
  `lin_of_tape`, `backward_common` (the factors a solve shares with its
  tangents), `backward_lin`, `backward`, `costate`, `solve`, `solve_jvp`,
  `rollout`, `rollout_linearized`, and the `tangent` record a forward-mode
  solve consumes. Pure tensor math with host-static control flow; safe to trace
  and jit.

- **`Solver.tangent`** — the tangent of the tape's first-order data along a
  direction in the problem's parameters: what `solve_jvp` needs from a problem,
  and nothing else.

- **`Dilqr.Regularisation`** — the Levenberg–Marquardt schedule
  (`{ delta; mu }`, `init`, `increase`, `decrease`), with `mu` floored at
  `1e-6`.

- **`Dilqr.Linesearch`** — trial-step schedules (`schedule ~n ()` is
  `1; 1/2; 1/4; …`) and the Armijo predicate (`accepts`).

## Batched solves

`Solver.solve_fixed` and `solve_ilqr_fixed` solve a **batch** of problems at
once. Give every leaf of the parameters, of the initial state and of every
initial control a leading batch axis. Say the quick-start problem's
parameters also carry a per-member `target`, used by its running loss:

```ocaml
let b = 8                                  (* members *)
let theta =
  { mass = Nx.broadcast_to [| b |] (scalar64 1.0)      (* shared dynamics *)
  ; target = Nx.create Nx.float64 [| b; 1 |] targets   (* per-member cost *)
  }
let x0 = Nx.create Nx.float64 [| b; 2 |] x0s
let us0 = Array.init horizon ~f:(fun _ -> Nx.zeros Nx.float64 [| b; 1 |])

let sol = Dilqr.Solver.solve_fixed Integrator.problem ~iters:8 theta x0 us0
(* sol.cost : [b];  sol.xs.(k) : [b; 2];  sol.us.(k) : [b; 1] *)
```

The dynamics and the losses are still written for **one member**: the solver
maps them over the batch, so each member gets its own line search, its own
trajectory and its own gradient. A shared dynamics parameter is simply
repeated along the batch axis (`Nx.broadcast_to`).

`taus_of` stacks to `[b; K+1; n+m]` (the batch axes lead the horizon), and
`split_taus` / `loss_of_taus` accept the batched trajectory. Differentiating
is unchanged:

```ocaml
let inputs = { Dilqr.Inputs.theta; x0; us0 }
let grads =
  Rune.grad
    (Dilqr.inputs Integrator.problem)
    (fun i ->
       Nx.sum
         (Dilqr.loss_of_taus
            Integrator.problem
            i.Dilqr.Inputs.theta
            (Dilqr.solve_ilqr_fixed Integrator.problem ~iters:8 i)))
    inputs
(* grads.theta.target : [b; 1];  grads.x0 : [b; 2];  ... *)
```

Only the fixed solves are batched. `Solver.solve` and `solve_ilqr` are
unbatched and say so: their regularization retries and Armijo backtracking
are host-side control flow, and a batch would need one of each per member.

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

Forward mode is the transpose of the same linear map, which is why the two give
the same answer for any scalar objective:

```
  <reverse gradient, v>  =  <cotangent, JVP v>       for every v, cotangent
```

and it is computed without a single derivative of a factorization. With `λ` the
costate of the local model, `K` its feedback gain and `G` the trajectory the
tangent of the dynamics drives,

```
  G_0 = dx0,  G_{k+1} = A_k G_k + δf_k
  X_k = δl_x,k + δA_kᵀ λ_{k+1} + l_xx,k G_k          (X_K = δφ_x + φ_xx G_K)
  U_k = δl_u,k + δB_kᵀ λ_{k+1} + l_ux,k G_k
  E_K = X_K,   q_k = U_k + B_kᵀ E_{k+1},   E_k = X_k + A_kᵀ E_{k+1} + K_kᵀ q_k
  v_0 = 0,     δu_k = -Q_uu⁻¹ q_k + K_k v_k,  v_{k+1} = A_k v_k + B_k δu_k
  δx_k = G_k + v_k,  δu_k
```

That is three cheap sweeps against the primal's factors. The tangents of the
quadratic blocks (`l_xx`, `l_uu`, `l_ux`, `φ_xx`) are not needed: the
derivative goes through the KKT system, whose matrix is the primal's.

## Forward mode

`Rune.jvp` of a `*_jvp` solve is its derivative along a tangent of the solve
inputs. A **batch of directions** is one `Rune.vmap` around it, with the primal
solve — and the factorization every direction reuses — computed once, outside
the lanes:

```ocaml
let taus = Dilqr.solve_ilqr_fixed_jvp problem ~iters:4 inputs in
let dtaus =
  Rune.vmap
    Nx.Ptree.(Dilqr.inputs problem @-> returns Nx.Ptree.tensor)
    (fun dinputs ->
       snd
         (Rune.jvp
            (Dilqr.inputs problem)
            Nx.Ptree.tensor
            (Dilqr.solve_ilqr_fixed_jvp problem ~iters:4)
            inputs
            dinputs))
    dirs
(* taus : [K+1; n+m];   dtaus : [k; K+1; n+m] for k directions in [dirs] *)
```

`dirs` is a structure of the solve inputs whose every leaf carries the `k`
directions on a leading axis, and `inputs` is captured — that is what makes the
primal a constant of the map. The tangents of the first-order data are computed
step by step as the tangent sweeps reach them (`Solver.tangent_solve`), never
materialized, so one direction costs `O (horizon · (n + m))` of memory on top of
the primal's tape — the tape itself, the Hessian recursion and the Cholesky are
lane-free and paid for once — plus one sweep of `n`-vectors and one rollout of
compute.

A **batch of problems** is a different axis and composes: the leading batch
axis of the solved inputs is mapped over inside the solve, and the direction
axis stays visible to `vmap` outside it.

Forward mode is what a *sketch* needs (see [`sofo`](../sofo)): a sketch wants
the trajectory tangent to compute its curvature blocks, and does not want a
reverse pass. When marking little losses for a sketch, mark quantities derived
from the **returned trajectory**, not terms inside `Problem.running_loss` or
`Problem.final_loss`: the solver evaluates those once per line-search trial and
once per iteration, and every evaluation would add its block again.

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

- [`example/batched_integrator.ml`](example/batched_integrator.ml) — a batch of
  double-integrator targets over one shared dynamics, solved and differentiated
  in one call.

  ```console
  $ dune exec ./example/batched_integrator.exe
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
adaptive one, JIT round-trips of both the solve and its gradient, batched
solves and gradients against their per-member counterparts, the forward mode
against finite differences and against the reverse mode (the transpose law,
pointwise), and a batch of directions against the directions one at a time.

## References

- Marine Schimel, Ta-Chu Kao, Kristopher T. Jensen and Guillaume Hennequin.
  *iLQR-VAE: control-based learning of input-driven dynamics with applications
  to neural data.* ICLR 2022 (Oral).
  <https://cbl.eng.cam.ac.uk/hennequin/publications/schimel-iclr-2022/>

## Status

Research code, developed alongside the Raven libraries; the interface may
still move.

## License

ISC — see `dune-project`.
