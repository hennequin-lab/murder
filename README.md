# SOFO

OCaml implementation of Second-order Forward-mode Optimization (SOFO) -- a
second-order sketching method -- based on
[Raven](https://github.com/raven-ml/raven)'s `nx` tensors and `rune`
transformations (`jvp`, `vmap`, `jit`, `custom_jvp`).

SOFO is a second-order optimizer that never backpropagates. At each iteration it
measures the loss and its curvature along a random **k-dimensional subspace** of
parameter space — a _sketch_ — with a single batched forward-mode pass, solves
the resulting **k×k** damped system for a step (Algorithm 1 of [Yu et al.,
NeurIPS 2024](#reference)), and moves along it. With k ≪ P the solve is cheap and
the pass is parallel, and because nothing is taped, memory is constant in the
time horizon — the property that makes the method interesting for RNNs trained
(or even meta-trained) over long sequences.

The library splits along what can be compiled:

- **Sketching** (`Sofo.sketch`) — runs the loss under a named `Rune.vmap`
  around `Rune.jvp`: the map carries k tangent lanes through one forward pass,
  so every tensor's tangent is its whole `k`-lane batch. Every little loss marks
  itself with `Sofo.observe` (or the packaged `Sofo.mse` / `Sofo.sse` /
  `Sofo.softmax_ce`); a mark is a unit-result `Rune.custom_jvp` whose rule
  gathers the prediction's lanes with `Rune.lanes` and adds the block `YᵀHY` to
  a `Rune.Total` the sketch driver collects. The result is the loss, the
  gradient sketch `C = Θᵀ∇c`, `G̃`, and the sampled directions `Θ`. This half
  differentiates, and it jits (`Sofo.Optim.sketch_jit`).
- **Updating** (`Sofo.Optim`) — Algorithm 1's damped subspace solve
  `θ ← θ − η·Θ·U(S + γI)⁻¹UᵀC`, and the shift. The solve needs the
  eigendecomposition of the symmetric `G̃`, which does not compile, so it runs
  eagerly on the host at O(k³) — the cheap part, against a model with P ≫ k
  parameters. The shift compiles, and runs where the parameters live.

A loss is ordinary OCaml either way. `Sofo.observe` is a `custom_jvp` that only
a sketch's total intercepts; every other Rune transformation runs its no-op
function and drops the additions. The same `objective` therefore trains under
`Rune.value_and_grad` (with, say, `Vega.adam_step`), under a bare `Rune.vmap`
around `Rune.jvp` (a first-order subspace method), or under `Sofo.sketch` —
swap the driver, not the model.

## A quick look: `example/linear.ml`

Student–teacher linear regression with ill-conditioned inputs: draw a teacher
`W*`, a student `W`, and minibatches `y = x·W*`, then train the student with
the SOFO update. This is the whole example, with `--device` read by
`example/devices.ml` (run it with `dune exec example/linear.exe`, and
add `-- --device cuda` to run it on a CUDA GPU):

```ocaml
open Base
open Nx

let d_in, d_out = 100, 3
let batch_size = 512
let max_iter = 10_000
let lr = 0.1
let n_tangents = 128
let damping : Sofo.Optim.damping = `Absolute 0.

(* [--device] lists the devices to try, in order, as [Devices.first] reads
   them: "cpu" (the default), "cuda", "cuda:1", "cuda,cpu". The sketch
   accumulates in float64, which Metal cannot compute. *)
let device = Devices.first Cmdargs.(get_string "--device" |> default "cpu")

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

let student =
  Model.init ~d_in ~d_out |> Nx.Ptree.place Model.P.ptree (Nx.Placement.on device)

let minibatch =
  let open Infix in
  let teacher = Model.init ~d_in ~d_out in
  (* draw inputs from an ill-conditioned Gaussian;
     this covariance is the GGN (and also the Hessian in this case)! *)
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

let objective params key =
  let open Infix in
  let x, y = minibatch ~key batch_size in
  let y' = Model.forward params x in
  Sofo.mse ~target:y y'

(* The sketch compiles for the device the parameters are placed on; the
   keys are host values that join them on each call. *)
let sketch_step = Sofo.Optim.sketch_jit ~k:n_tangents Model.P.ptree Rng.ptree objective

let rec loop ~i (params : Model.P.t) (state : Sofo.Optim.state) =
  if i >= max_iter
  then params
  else (
    let aux = Rng.fold_in state.key 0 in
    let out = sketch_step ~key:state.key ~aux params in
    let params, state = Sofo.Optim.update ~lr ~damping Model.P.ptree state params out in
    let loss = item [] out.loss in
    Stdio.printf "[%05i] loss = %.6f\n%!" i loss;
    loop ~i:(i + 1) params state)

let state = Sofo.Optim.init ~key:(Rng.key 1985) ()
let () = Stdio.printf "device: %s\n%!" (Nx.Device.name device)
let _ = loop ~i:0 student state
```

How to read it:

1. **The model is a parameter tree.** `type t = float32_t [@@deriving ptree]`
   gives `Model.P.ptree` — the way sofo, Rune and Vega see every parameter leaf,
   whatever the model's shape. Everything else the loss reads gets a structure
   the same way; here it is `Rng.ptree`, the key a minibatch is drawn from.
2. **The loss is a plain function that marks its own curvature.**
   `objective` draws a minibatch and returns a scalar. `Sofo.mse` computes the
   mean squared error _and_ observes it with the exact MSE curvature; for custom
   little losses, use `Sofo.observe ~y ~curv l`. Outside a sketch the marking is
   inert, so the same `objective` can be handed to `Rune.value_and_grad`.
3. **The aux tree is for everything that is not a parameter.** Here it is the
   RNG key the minibatch is drawn from. It is _data_: the sketch never
   differentiates it and the update never moves it, but it rides the input tree
   of the compiled step, so a new batch is a new input rather than a new trace.
   The direction key is kept separate (`Rng.fold_in state.key 0`), so data and
   subspace draws do not correlate.
4. **The sketching half is compiled once.** `Sofo.Optim.sketch_jit
   ~k Model.P.ptree Rng.ptree objective` builds the whole `(key, aux, params)`
   computation under `Rune.jit` and returns the function to call: the first call
   traces and compiles, every later one replays. The directions Θ are drawn
   _inside_ it, from the key input, so replaying it each step samples a fresh
   128-dimensional subspace without retracing.
5. **The loop replays, solves, steps.** One compiled call produces the loss, `C`,
   `G̃`, the Gram ΘΘᵀ and Θ; `Sofo.Optim.update` solves on the host from the
   k-sized numbers, shifts the parameters in a compiled step where they live,
   and advances the direction stream with the parameters, so the next iteration
   cannot reuse the subspace it just measured. The parameters are placed on the
   `--device` (`Nx.Ptree.place`), and everything that touches them is compiled
   for it. `damping = \`Absolute 0.` and `lr = 0.1` make it a scaled Newton step
   inside the subspace.

## API at a glance

| What you want                                    | Entry point                                                                                          |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Draw a subspace Θ                                 | `Sofo.Optim.directions structure ?key ~k params`                                                     |
| Measure loss, `C`, `G̃` at `params`, eagerly       | `Sofo.sketch structure loss params dirs` → `'p Sofo.sketch`                                          |
| Compile the measurement for a training run        | `Sofo.Optim.sketch_jit ~k structure aux_structure loss` → `~key ~aux params`                         |
| One step on a compiled measurement                | `Sofo.Optim.update structure ~lr ?damping ?preconditioner state params out`                          |
| One step on an eager sketch                       | `Sofo.Optim.update_sketch structure ~lr ?damping ?preconditioner sk params`                          |
| One complete eager iteration                      | `Sofo.Optim.step structure ~k ~lr state ~loss ~params`                                               |
| The damped subspace solve alone                   | `Sofo.Optim.coordinates ?damping ?preconditioner ?gram ggn c`                                        |
| Whiten the basis, apply coordinates               | `Sofo.Optim.gram`, `Sofo.Optim.apply`, `sk.apply`                                                    |
| Describe a little loss's curvature                | `Curv.scale`, `Curv.diag`, `Curv.softmax_ce`, `Curv.hvp`                                             |
| Mark a little loss by hand                        | `Sofo.observe ~y ~curv l`                                                                            |
| Packaged little losses (value + observation)      | `Sofo.mse`, `Sofo.sse`, `Sofo.softmax_ce`                                                            |
| Direction stream and iteration counter            | `Sofo.Optim.init`, `Sofo.Optim.next`                                                                 |
| Damping and preconditioning policy                | `` `Relative_from_top`` (default), `` `Absolute``, `` `Relative_from_bottom``, `` `Inverse``, `` `Inverse_sqrt`` |

The API reference lives in [`lib/sofo.mli`](lib/sofo.mli), whose doc comments
are the intended documentation; `PLAN.md` records the design and the
implementation notes, including what each milestone settled.

## Examples

| Example                    | What it shows                                                                                                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `example/linear.ml`        | The minimal compiled loop shown above: ill-conditioned linear regression, one `Rune.jit`ed sketch, the host solve and the compiled shift.                                                                                |
| `example/lorenz.ml`        | The paper's Lorenz endpoint task: a small RNN (`Rune.scan`) trained by the compiled SOFO loop, with an Adam loop (`Rune.value_and_grad` + `Vega.adam_step`) over the same `objective` kept (commented out) for comparison.                                                  |

```bash
dune exec example/linear.exe -- --device cuda
dune exec example/lorenz.exe -- -d /tmp --device cuda
```

## Building and testing

Requires OCaml ≥ 5.5 (algebraic effects), `nx`, `rune` and `ppx_ptree`; the
examples additionally use `vega`, `laguz`, `nx.io`, `bos`, `cmdliner`,
[`cmdargs`](https://github.com/hennequin-lab/cmdargs) and `stdio`.

```bash
dune build            # library, examples and tests
dune test             # the windtrap suites in test/
dune exec example/linear.exe
```

`test/test_sketch.ml` checks the frontend (including the batched-AD path),
`test/test_curv.ml` checks every curvature description against
`Rune.jacfwd' (Rune.grad' f)`, `test/test_compose.ml` checks composition with
scans, maps and reverse mode, and `test/test_optim.ml` checks the solve, the
step and the compiled half — mostly reference-free, since a loss that is exactly
quadratic makes the sketch's second-order model exact.

## Status

A research prototype: the API is unstable and nothing is released on opam yet.
The sketching frontend (`Sofo.Curv`, `observe`, `mse`, `sse`, `softmax_ce`,
`sketch`), Algorithm 1's update (`Sofo.Optim`, including the compiled half), the
examples and the test suites are all implemented.

## Reference

Youjing Yu, Rui Xia, Brooke Ma, Máté Lengyel and Guillaume Hennequin,
_Second-order forward-mode optimization of recurrent neural networks for
neuroscience_, NeurIPS 2024 —
[OpenReview](https://openreview.net/forum?id=Pox8jNQOo5) ·
[proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a791a086d7643ecf53608e57cd5889f0-Abstract-Conference.html).

The batched forward mode this library builds on is a named `Rune.vmap` around
`Rune.jvp`, with `Rune.lanes` and `Rune.Total`; the sketch driver is the only
place all four are composed. See the [Raven](https://github.com/raven-ml/raven)
repository.

## License

ISC — see the headers of the source files.
