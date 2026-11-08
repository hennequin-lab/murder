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
  solution of a finite-horizon optimal control problem solved with `dilqr`;
  the input covariance is a Kronecker product of a factor over time and a
  factor over input dimensions. `dilqr`'s implicit VJP differentiates the
  solve without unrolling it.

  `~infer_initial:true` makes the initial state a latent instead of fixing it
  at zero, and the model must then pass `~initial_prior`: that prior is the
  boundary condition of the encoder's control problem, so the encoder solves
  for `x_0` jointly with the inputs. Its mean is the joint mode of that
  problem's local quadratic model, and its covariance is that model's Hessian
  in `x_0` up to one learned scale — [Σ = exp (2 log_scale) · P⁻¹] — so the
  width is per trial and the scale is the block's only free parameter. Both
  are taken stopped, in the draws and in the density, as the Kronecker factors
  are; `Recognition.Iilqr.posterior_initial` reports them for diagnostics.
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
