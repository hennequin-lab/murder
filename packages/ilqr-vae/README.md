# Ilqr_vae

iLQR-VAE (Schimel, Kao, Jensen & Hennequin, ICLR 2022) on the modern Raven
stack: `nx` tensors, `rune` transformations, and
[`dilqr`](https://github.com/hennequin-lab/dilqr-raven) as the inference
engine — with a swappable recognition model.

A trial is a tensor of observations; a mini-batch stacks trials on leading
axes. Latent inputs (and, for an LFADS-style
encoder, an initial state) drive a state through a dynamics; a likelihood
emits each observation from the state reached by the inputs before it. The
recognition is a value of type `Recognition.t` with its own parameters;
`Model.make` takes one, and everything else — the ELBO, sampling, training,
JIT compilation — is unchanged by the choice. The model is a description with
no parameters in it: `Model.make` builds it and draws nothing, and
`Model.init` draws the parameters a fresh one starts from, one subkey per
component.

The model's dimensions are not arguments: `n` and `m` come from the
dynamics, `n_output` from the likelihood and `n_steps` from the recognition
(the only component that knows the horizon), and `Model.make` checks that
the components agree with each other.

```ocaml
open Base
open Stdio
open Ilqr_vae

let model =
  Model.make
    Nx.float64
    ~dynamics:
      (Dynamics.Linear_nonlinear.make
         ~at:Nx.Host.on
         ~dt:0.01
         ~tau:0.04
         ~n:3
         ~nh:128
         ~m:3
         ~phi:Nx.tanh
         ())
    ~likelihood:(Likelihood.Gaussian.make ~at:Nx.Host.on ~n:3 ~n_output:3 ())
    ~prior:(Prior.Student.make ~at:Nx.Host.on ~m:3 ())
    ~encoder_prior:(Prior.Gaussian.make ~at:Nx.Host.on ~m:3 ())
    ~recognition:(Recognition.Iilqr.make ~at:Nx.Host.on ~n:3 ~m:3 ~n_steps:32 ~n_output:3 ())
    ()

let key = Nx.Rng.key 0
let params = Model.init ~at:Nx.Host.on ~key model
let data = (* one trial: [n_steps; n_output] *)
let batch = (* a tensor mini-batch: [batch; n_steps; n_output] *)

(* Eager: adaptive encoder and diagnostics. *)
let () =
  let posterior = Model.infer model params ~n_samples:100 ~key data in
  let e = Elbo.compute model params ~posterior data in
  printf "loss %f\n%!" (Nx.item [] (Nx.neg e.Elbo.elbo))

(* Compiled: fixed encoder, implicit gradient through the solve. The
   mini-batch is an argument, so the compiled step is not tied to one
   OCaml array of trials. *)
let step =
  Rune.jit
    Nx.Ptree.(
      Model.structure model @-> Nx.Rng.ptree @-> tensor
      @-> returns (pair tensor (Model.structure model)))
    (fun params key batch ->
       Rune.value_and_grad
         (Model.structure model)
         (fun p -> Elbo.loss_fixed model p ~key ~n_samples:16 ~iters:4 batch)
         params)
```

## Recognitions

- `Recognition.Iilqr` — the iLQR-VAE recognition. The posterior mean is the
  solution of a finite-horizon optimal control problem solved with `dilqr`, and
  `dilqr`'s implicit VJP differentiates the solve without unrolling it.

  `~covariance` picks the covariance. `Recognition.Iilqr.kron`, the default,
  is the paper's: a Kronecker product of a learned factor over time and a
  learned factor over input dimensions, shared by every trial.
  `Recognition.Iilqr.laplace ()` is the local quadratic model's own Gaussian,
  one per trial, which the Riccati pass at the solution already holds — the
  same factors `Recognition.Iilqr.posterior_initial` reports for the initial
  state. Its mean is that model's mode and its factors its Hessian in
  `(x_0, u_0 …)`, so the draws are joint over the initial state and the
  inputs, and the one learned parameter is a scalar scale for every block —
  no `n_steps × n_steps` factor, and a family that does not grow with the
  horizon. On a linear-Gaussian model it is the exact posterior and the bound
  has no sampling noise; on a nonlinear one it is the Gauss-Newton
  approximation, whose width the scale can correct.
  `Recognition.Iilqr.laplace ~damping` regularizes the pass, which is what
  gives a model whose `Q_uu` is not positive definite a covariance at all.

  `~infer_initial:true` makes the initial state a latent instead of fixing it
  at zero, and the model must then pass `~initial_prior`: that prior is the
  step-0 running loss of the encoder's control problem, so the solve optimizes
  `x_0` jointly with the inputs and one solve returns the joint mode. Its
  covariance is the local model's Hessian in `x_0` up to the scale —
  [Σ = exp (2 log_scale) · P⁻¹] with the Kronecker covariance and the same
  scalar as every other block with the Laplace one. The precision is a value
  of the local model, taken stopped in the draws and in the density; the mean
  is the solve's own, so the bound's gradient reaches the model through the
  solve's implicit VJP. `Recognition.Iilqr.posterior_initial` reports the
  pair.
- `Recognition.Lfads` — an LFADS-style recognition (Pandarinath et al.,
  2018). Bidirectional encoder RNNs infer a Gaussian posterior over the
  initial state and a second encoder–controller pair produces an
  autoregressive Gaussian posterior over the inputs; every RNN is the
  library's `Dynamics.Mgu`. It needs an `~initial_prior`, and
  `~with_inputs:false` gives the original LFADS model without inferred
  inputs.

Swapping them is changing one argument of `Model.make`. The ELBO estimates its
KL from `log q` and the model's `log p` at the same reparameterized draws, with
the density's parameters stopped so that the gradient is the path-derivative
(sticking-the-landing) estimator of Roeder, Wu & Duvenaud (2017).

A recognition whose mean solves an iterative problem takes the solver as a
value: `Recognition.adaptive ?max_iter ?conv_threshold ()` stops on the
solver's own test and runs eagerly, `Recognition.fixed ~iters ()` runs a known
number of iterations with tensor-only control flow and compiles. Pass it as
`~solve` to `Model.infer`, `Model.mean`, `Model.log_prob` or any of the
`Ilqr_vae` front-page functions; leaving it out means `Recognition.default`,
the adaptive solve.

## KL warm-up

The bound is `log p(o | z) − β · KL(q ‖ p)`, and `β` is a scalar tensor
argument of the ELBO, not a host float. A step compiled with `Rune.jit`
therefore takes `β` as an argument and follows a schedule without
recompiling — a `β` captured from the closure would be a constant of the
compiled program:

```ocaml
let schedule = Schedule.linear_warmup ~steps:2000 Nx.float64 in
let step =
  Rune.jit
    Nx.Ptree.(Model.structure model @-> tensor @-> Nx.Rng.ptree @-> returns tensor)
    (fun params beta key ->
       Elbo.loss_fixed model params ~beta ~key ~n_samples:16 ~iters:4 data)
in
let loss = step params (Schedule.at schedule i) key
```

Raising `β` from `0` to `1` over the first few thousand steps keeps an
LFADS-style posterior from collapsing onto the prior before the decoder has
learned to read the data; iLQR-VAE needs it less, because its control cost
already contains the encoder prior.

## Training over MPI

The `Ilqr_vae_mpi` library spreads one mini-batch's trials over MPI ranks: each
rank estimates and differentiates the bound of its own trials, the ranks'
gradients are averaged over the mini-batch, the root takes the optimizer step,
and the new parameters are broadcast. The step's key is split into one subkey
per trial, as a batched `Elbo.loss_fixed` splits it, so a run's numbers do not
depend on how many ranks ran it; the ranks change the wall-clock time, not the
estimate.

```ocaml
open Ilqr_vae
open Ilqr_vae_mpi

let () =
  let model = … in
  let params = Model.init ~at:Nx.Host.on ~key:(Nx.Rng.key 0) model in
  let driver = make model ~iters:4 ~n_samples:16 () in
  let sgd lr params grads =
    Rune.Ptree.map2
      (Model.structure model)
      (fun _ p g ->
         Nx.Infix.(p - (Nx.cast (Nx.dtype p) (Nx.scalar Nx.float64 lr) * g)))
      params
      grads
  in
  let _params, losses =
    run
      driver
      params
      ~steps:2000
      ~key:(Nx.Rng.key 1)
      ~data:(fun i -> batches.(i mod Array.length batches))
      ~report:(fun i loss -> Printf.printf "step %d  loss %.4f\n%!" i loss)
      (sgd 1e-3)
  in
  if is_root driver then Printf.printf "final loss %.4f\n%!" losses.(1999)
```

Launch it with `mpirun -np 4 ./train.exe` (or `srun`). A program run without a
launcher is a world of one rank, so the same executable trains on a laptop.
Every rank holds the whole mini-batch and slices its share; `partition` is that
split, and it is also what a data pipeline calls to read only the trials its
rank needs. The `update` function runs on the root only, so an optimizer that
keeps state keeps it there; `rank`, `size` and `is_root` say where a call is
running.

The sublibrary needs the `mpi` opam package and an MPI implementation (OpenMPI
or MPICH) on the machine.

## Status

The library is **implemented**: the `.mli` files in `lib/` are the design, the
`.ml` files beside them are the implementation, and `test/` is the suite. See
the [`dilqr` repository](https://github.com/hennequin-lab/dilqr-raven) for
`dilqr` itself.

## Checking

```sh
dune build @packages/ilqr-vae/check @packages/ilqr-vae/fmt   # interfaces and formatting
dune build @packages/ilqr-vae/runtest                        # the suite
```

The MPI driver's suite runs as one rank under `runtest`; the distributed case
needs a launcher:

```sh
dune build packages/ilqr-vae/test/mpi/test_mpi.exe
mpirun -np 4 ./_build/default/packages/ilqr-vae/test/mpi/test_mpi.exe
```
