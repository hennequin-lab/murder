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

let params = Model.params model
let key = Nx.Rng.key 0
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
         (fun p -> Elbo.batch_loss_fixed model p ~key ~n_samples:16 ~iters:4 batch)
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

## Status

The library is **implemented**: the `.mli` files in `lib/` are the design, the
`.ml` files beside them are the implementation, and `test/` is the suite (25
tests). See the [`dilqr`
repository](https://github.com/hennequin-lab/dilqr-raven) for `dilqr` itself.

## Checking

```sh
dune build @check @fmt   # interfaces and formatting
dune runtest             # the suite
```
