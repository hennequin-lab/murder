# SOFO

Ocaml implementation of Second-order Forward-mode Optimization (SOFO) -- a
second-order sketching method -- based on
[Raven](https://github.com/raven-ml/raven)'s `nx` tensors and `rune` autodiff.

SOFO is a second-order optimizer that never backpropagates. At each iteration it
measures the loss and its curvature along a random **k-dimensional subspace** of
parameter space — a _sketch_ — with a single batched forward-mode pass, solves
the resulting **k×k** damped system for a step (Algorithm 1 of [Yu et al.,
NeurIPS 2024](#reference)), and moves along it. With k ≪ P the solve is cheap and
the pass is parallel, and because nothing is taped, memory is constant in the
time horizon — the property that makes the method interesting for RNNs trained
(or even meta-trained) over long sequences.

The library splits along what can be compiled:

- **Sketching** (`Sofo.sketch`) — runs the model under `Rune.jvp_k`, which
  carries k tangent lanes through one forward pass. Every little loss marks
  itself with `Sofo.observe` (or the packaged `Sofo.mse` / `Sofo.softmax_ce`),
  and a collector accumulates the blocks `YᵀHY` of the sketched generalized
  Gauss–Newton matrix `G̃ = ΘᵀJᵀHJΘ` as they go by. The result is the loss, the
  gradient sketch `C = Θᵀ∇c`, `G̃`, and the sampled directions `Θ`. This half
  differentiates, and it jits (`Sofo.Optim.Compiled`).
- **Updating** (`Sofo.Optim`) — Algorithm 1's damped subspace solve
  `θ ← θ − η·Θ·U(S + γI)⁻¹VᵀC`, and the shift. It needs an SVD, which neither
  differentiates nor compiles, so it runs eagerly on the host at O(k³) — the
  cheap part, against a model with P ≫ k parameters.

A loss is ordinary OCaml either way. `Sofo.observe` is an effect that only a
sketch collector intercepts; every other Rune handler lets it pass. The same
`objective` therefore trains under `Rune.value_and_grad` (with, say,
`Vega.adam_step`), under `Rune.jvp_k` alone (a first-order subspace method), or
under `Sofo.sketch` — swap the driver, not the model.

## A quick look: `example/linear_simple.ml`

Student–teacher linear regression: draw a teacher `W*`, a student `W`, and
minibatches `y = x·W*`, then train the student with the SOFO update. This is the
whole example (run it with `dune exec example/linear_simple.exe`):

```ocaml
open Base
open Nx

let device = "CPU"
let d_in, d_out = 100, 3
let batch_size = 512
let max_iter = 10_000
let lr = 0.1
let n_tangents = 128
let damping : Sofo.Optim.damping = `Absolute 0.

module Model = struct
  module P = struct
    type t = float32_t [@@deriving ptree]
  end

  let init ~d_in ~d_out =
    let open Infix in
    randn float32 [| d_in; d_out |] /$ Float.(sqrt (of_int d_in))

  let forward (w : P.t) x =
    let open Infix in
    x *@ w
end

let student = Model.init ~d_in ~d_out

let minibatch =
  let open Infix in
  let teacher = Model.init ~d_in ~d_out in
  let input_cov_sqrt =
    let u, _ = qr (randn float32 [| d_in; d_in |]) in
    let lambda =
      Array.init d_in ~f:(fun i -> Float.(1. / (1. + square (of_int Int.(i + 1)))))
      |> create float32 [| d_in; 1 |]
      |> fun x -> x / mean x
    in
    sqrt lambda * u
  in
  fun ~key bs ->
    let x = Rng.normal key float32 [| bs; d_in |] *@ input_cov_sqrt in
    let y = Model.forward teacher x in
    x, y

module Aux = struct
  type t = Rng.key [@@deriving ptree]
end

module O = Sofo.Optim.Compiled (Model.P) (Aux)

let objective params key =
  let open Infix in
  let x, y = minibatch ~key batch_size in
  let y' = Model.forward params x in
  Sofo.mse y' y

(* JIT compilation machinery for a sketched objective *)
let sketch_step =
  Rune.jit2 ~device (module O.In) (module O.Out) (O.sketch ~k:n_tangents objective)

let rec loop ~i (params : Model.P.t) (state : Sofo.Optim.state) =
  if i >= max_iter
  then params
  else (
    let key_data = Rng.fold_in state.key 0 in
    let out = sketch_step { params; key = state.key; aux = key_data } in
    let params, state = O.update ~lr ~damping state params out in
    let loss = item [] out.loss in
    Stdio.printf "[%05i] loss = %.6f\n%!" i loss;
    loop ~i:(i + 1) params state)

let state = Sofo.Optim.init ~key:(Rng.key 1985) ()
let _ = loop ~i:0 student state
```

How to read it:

1. **The model is a parameter tree.** `type t = float32_t [@@deriving ptree]`
   gives `Nx.Ptree.S` — the way sofo, Rune and Vega see every parameter leaf,
   whatever the model's shape. `Aux` is the same idea for the loss's _non_-parameter
   inputs.
2. **The loss is a plain function that marks its own curvature.**
   `objective` draws a minibatch and returns a scalar. `Sofo.mse` computes the
   mean squared error _and_ observes it with the exact MSE curvature; for custom
   little losses, use `Sofo.observe ~y ~curv l`. Outside a sketch the marking is
   inert, so the same `objective` can be handed to `Rune.value_and_grad`.
3. **`Aux` is for everything that is not a parameter.** Here it is the RNG key
   the minibatch is drawn from. It is _data_: the sketch never differentiates it
   and the update never moves it, but it rides the input tree of the compiled
   step, so a new batch is a new input rather than a new trace. The direction key
   is kept separate (`Rng.fold_in state.key 0`), so data and subspace draws do
   not correlate.
4. **The sketching half is compiled once.** `O = Sofo.Optim.Compiled (Model.P) (Aux)`
   ties the sketching computation to the parameter and aux structures, and
   `Rune.jit2 ... (O.sketch ~k:n_tangents objective)` traces it as a pure
   function of `{ params; key; aux }`. The directions Θ are drawn _inside_ it,
   from the key leaf, so replaying it each step samples a fresh 128-dimensional
   subspace without retracing.
5. **The loop replays, solves, steps.** One compiled call produces the loss, `C`,
   `G̃` and Θ; `O.update` runs the eager host-side solve and parameter shift, and
   advances the direction stream with the parameters, so the next iteration
   cannot reuse the subspace it just measured. `damping = \`Absolute 0.`and`lr = 0.1` make it a scaled Newton step inside the subspace.

For the full deployment — an eager `Sofo.check` cross-check, a compiled
cross-check, the exact second-order model residual, a first-order control
replaying the _same_ compiled sketch, and timings — see `example/linear.ml`.

## API at a glance

| What you want                                      | Entry point                                                         |
| -------------------------------------------------- | ------------------------------------------------------------------- |
| Measure loss, `C`, `G̃` at `params`, eagerly        | `Sofo.sketch (module P) ~k loss params` → `'p sketch`               |
| Compile the measurement for a training run         | `Sofo.Optim.Compiled (P) (Aux).sketch` with `Rune.jit2`             |
| Cross-check that the observed little losses add up | `Sofo.check sk`, `O.check out`, `Sofo.diagnostics`                  |
| Describe a little loss's curvature                 | `Curv.scale`, `Curv.diag`, `Curv.softmax_ce`, `Curv.hvp`            |
| Mark a little loss by hand                         | `Sofo.observe ~y ~curv l`                                           |
| Packaged little losses (value + observation)       | `Sofo.mse`, `Sofo.softmax_ce`                                       |
| Sample, apply, whiten the subspace Θ               | `Sofo.Optim.directions`, `apply`, `gram`                            |
| Algorithm 1's damped solve                         | `Sofo.Optim.coordinates`                                            |
| One step / one complete eager iteration            | `Sofo.Optim.update`, `Sofo.Optim.step`                              |
| Damping and preconditioning policy                 | `` `Relative_from_top`` (default), `` `Absolute``, `` `Inverse``, … |

The API reference lives in [`lib/sofo.mli`](lib/sofo.mli), whose doc comments
are the intended documentation; `PLAN.md` records the design and the
implementation notes, including what each milestone settled.

## Examples

| Example                    | What it shows                                                                                                                                             |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `example/linear_simple.ml` | The minimal compiled loop shown above.                                                                                                                    |
| `example/linear.ml`        | The full linear deployment: eager and compiled checks, exact model residual, first-order control, timings; CLI-tunable k, damping, preconditioner.        |
| `example/lorenz_simple.ml` | A small RNN (`Rune.scan`) on the paper's Lorenz endpoint task, with the SOFO loop and an Adam loop (`value_and_grad` + `Vega`) over the same `objective`. |
| `example/lorenz.ml`        | The composition study: one loss driven by `value_and_grad`, `jvp_k` and `sketch`, with memory and jit-staging numbers.                                    |

```bash
dune exec example/linear_simple.exe
dune exec example/linear.exe -- --steps 40 --lanes 32
dune exec example/lorenz_simple.exe
dune exec example/lorenz.exe
```

## Building and testing

Requires OCaml ≥ 5.5 (algebraic effects), `nx`, `rune` and `ppx_ptree`; the
examples additionally use `vega`, `laguz`, `nx.io`, `bos`, `cmdliner`,
[`cmdargs`](https://github.com/hennequin-lab/cmdargs) and `stdio`.

```bash
dune build            # library, examples and tests
dune test             # the windtrap suites in test/
dune exec example/linear_simple.exe
```

`test/test_sketch.ml` checks the frontend (including the batched-AD path),
`test/test_curv.ml` checks every curvature description against
`Rune.hessian'`, `test/test_compose.ml` checks composition with scans, maps and
reverse mode, and `test/test_optim.ml` checks the solve, the step and the
compiled half — mostly reference-free, since a loss that is exactly quadratic
makes the sketch's second-order model exact.

## Status

A research prototype: the API is unstable and nothing is released on opam yet.
The sketching frontend (`Sofo.Curv`, `observe`, `mse`, `softmax_ce`, `sketch`,
`check`), Algorithm 1's update (`Sofo.Optim`, including the compiled half), the
examples and the test suites are all implemented.

## Reference

Youjing Yu, Rui Xia, Brooke Ma, Máté Lengyel and Guillaume Hennequin,
_Second-order forward-mode optimization of recurrent neural networks for
neuroscience_, NeurIPS 2024 —
[OpenReview](https://openreview.net/forum?id=Pox8jNQOo5) ·
[proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a791a086d7643ecf53608e57cd5889f0-Abstract-Conference.html).

The batched forward mode this library builds on is `Rune.jvp_k`, which has its
own documentation in the [Raven](https://github.com/raven-ml/raven) repository.

## License

ISC — see the headers of the source files.
