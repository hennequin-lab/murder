# Ilqr_vae

iLQR-VAE (Schimel, Kao, Jensen & Hennequin, ICLR 2022) on the modern Raven
stack: `nx` tensors, `rune` transformations, and
[`dilqr`](https://github.com/hennequin-lab/dilqr-raven) as the inference
engine.

A trial is a tensor of observations. Latent inputs drive a state through a
dynamics; a likelihood emits each observation from the state reached by the
inputs before it. The posterior mean of the Gaussian recognition distribution
is the solution of a finite-horizon optimal control problem solved with iLQR,
and `dilqr`'s implicit VJP differentiates the solve without unrolling it. The
ELBO is estimated with reparameterized samples.

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
    ()

let params = Model.init model
let key = Nx.Rng.key 0
let data = (* one trial: [n_steps; n_output] *)
let trials = (* an array of trials *)

(* Eager: adaptive encoder and diagnostics. *)
let () =
  let mu_u = Encoder.posterior_mean model params data in
  let e = Elbo.compute model params ~mu_u ~key ~n_samples:100 data in
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

## Status

The library is **implemented**: the `.mli` files in `lib/` are the design, the
`.ml` files beside them are the implementation, and `test/` is a twelve-test
suite. `plan.md` is the port plan, the mapping from the Owl implementation, and
the list of decisions open to the maintainer. See the
[`dilqr` repository](https://github.com/hennequin-lab/dilqr-raven) for `dilqr`
itself.

## Checking

```sh
dune build @check @fmt   # interfaces and formatting
dune runtest             # the suite
```
