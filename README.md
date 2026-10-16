# Ilqr_vae

iLQR-VAE (Schimel, Kao, Jensen & Hennequin, ICLR 2022) on the modern Raven
stack: `nx` tensors, `rune` transformations, and
[`dilqr`](https://github.com/hennequin-lab/dilqr-raven) as the inference
engine — with a swappable recognition model.

A trial is a tensor of observations. Latent inputs (and, for an LFADS-style
encoder, an initial state) drive a state through a dynamics; a likelihood
emits each observation from the state reached by the inputs before it. The
recognition is a value of type `Recognition.t` with its own parameters;
`Model.make` takes one, and everything else — the ELBO, sampling, training,
JIT compilation — is unchanged by the choice.

```ocaml
open Base
open Stdio
open Ilqr_vae

let model =
  Model.make
    Nx.float64
    ~n:3
    ~m:3
    ~n_steps:32
    ~n_output:3
    ~dynamics:
      (Dynamics.Linear_nonlinear.make
         ~dt:0.01
         ~tau:0.04
         ~n:3
         ~nh:128
         ~m:3
         ~phi:Nx.tanh
         Nx.float64)
    ~likelihood:(Likelihood.Gaussian.make ~n:3 ~n_output:3 Nx.float64)
    ~prior:(Prior.Student.make ~m:3 Nx.float64)
    ~encoder_prior:(Prior.Gaussian.make ~m:3 Nx.float64)
    ~recognition:(Recognition.Iilqr.make ~n:3 ~m:3 ~n_steps:32 ~n_output:3 Nx.float64)
    ()

let params = Model.init model
let key = Nx.Rng.key 0
let data = (* one trial: [n_steps; n_output] *)
let trials = (* an array of trials *)

(* Eager: adaptive encoder and diagnostics. *)
let () =
  let posterior = Model.infer model params ~n_samples:100 ~key data in
  let e = Elbo.compute model params ~posterior data in
  printf "loss %f\n%!" (Nx.item [] (Nx.neg e.Elbo.elbo))

(* Compiled: fixed encoder, implicit gradient through the solve. *)
let step =
  Rune.jit
    Nx.Ptree.(Model.structure model @-> Nx.Rng.ptree @-> returns (pair tensor (Model.structure model)))
    (fun params key ->
       Rune.value_and_grad
         (Model.structure model)
         (fun p -> Elbo.batch_loss_fixed model p ~key ~n_samples:16 ~iters:4 trials)
         params)
```

## Recognitions

- `Recognition.Iilqr` — the iLQR-VAE recognition. The posterior mean is the
  solution of a finite-horizon optimal control problem solved with `dilqr`;
  the covariance is a Kronecker product of a factor over time and a factor
  over input dimensions; the initial state is fixed at zero. `dilqr`'s
  implicit VJP differentiates the solve without unrolling it.
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

## Status

The library is **implemented**: the `.mli` files in `lib/` are the design, the
`.ml` files beside them are the implementation, and `test/` is the suite (22
tests). See the [`dilqr`
repository](https://github.com/hennequin-lab/dilqr-raven) for `dilqr` itself.

## Checking

```sh
dune build @check @fmt   # interfaces and formatting
dune runtest             # the suite
```
