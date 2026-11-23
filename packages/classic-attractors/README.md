# classic-attractors

Six chaotic, low-dimensional attractors, and two trajectory generators. Every
attractor is a vector field integrated with Jera's classical fourth-order
Runge–Kutta:

| Module | Field | Parameters | `dim` | Initial conditions |
| --- | --- | --- | --- | --- |
| `Lorenz` | `x' = σ(y − x)`, `y' = x(ρ − z) − y`, `z' = xy − βz` | σ = 10, ρ = 28, β = 8/3 | 3 | (1, 1, 1) |
| `Rossler` | `x' = −y − z`, `y' = x + ay`, `z' = b + z(x − c)` | a = 0.2, b = 0.2, c = 5.7 | 3 | (1, 1, 0) |
| `Thomas` | `x' = sin y − bx`, … (cyclic) | b = 0.19 | 3 | (1, 1, 1) |
| `Halvorsen` | `x' = −ax − 4y − 4z − y²`, … (cyclic) | a = 1.89 | 3 | (−1.48, −1.51, 2.04) |
| `Aizawa` | sphere with a tube through it | a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1 | 3 | (0.1, 0, 0) |
| `Duffing` | `x' = v`, `v' = −δv − αx − βx³ + γcos(ωt)` | δ = 0.02, α = −1, β = 1, γ = 0.5, ω = 1 | 2 | (0, 0) |

Each module is a `Classic_attractors.S`; the equations, the box initial
conditions are drawn in and the recommended `dt` are documented in its own
interface.

## Generators

Both return a host tensor of shape `[n_steps; n; dim]` of the dtype asked for
(`Nx.float32` or `Nx.float64`), with consecutive states one RK4 step of `dt`
apart, and both drop the first `discard_first` states of the run so that the
answer still spans `n_steps` states:

```ocaml
open Classic_attractors

(* 4 unrelated trajectories of 100 states each, 0.01 apart, after a transient *)
let runs = Lorenz.simulate Nx.float32 ~dt:0.01 ~n_steps:100 ~discard_first:1000 4
(* [| 100; 4; 3 |] *)

(* 1024 consecutive chunks of 17 states, cut from one long run and shuffled *)
let chunks = Lorenz.snippets Nx.float32 ~dt:0.01 ~n_steps:17 ~discard_first:1000 1024
(* [| 17; 1024; 3 |] *)
```

`simulate` draws `n` initial conditions uniformly in the attractor's box
(`~key`, `Nx.Rng.key 0` by default) and integrates them together; `snippets`
starts from the attractor's classic initial condition, cuts the long run into
`n` non-overlapping chunks, and returns them in a random order (`~key` again).
A chunk tensor of `n_steps = horizon + 1` is ready for a forecasting task:
`slice [I 0]` is each chunk's initial state and `slice [R (1, horizon + 1)]` the
`horizon` states that follow.

The vector field is public as `field t y` (a state is `[n; dim]`), so a model or
a reference rollout can integrate it with `Jera.Ode` directly.

## Tests

```sh
dune build @packages/classic-attractors/runtest
```
