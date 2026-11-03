---
name: raven-rune
description: "Differentiate, vectorize, loop and compile Nx code with Rune in the Hennequin lab monorepo (murder). Use when a task needs Rune.grad/value_and_grad/jvp/vjp, Rune.scan/iterate, Rune.vmap, Rune.switch/cond, Rune.jit, custom_jvp/custom_vjp/custom_transpose, Rune.root, Rune.remat, Rune.Total, Rune.check_grads, or when debugging tracing errors such as 'the value is traced by Rune.jit', 'the function captures a value another transformation traces', or a jitted function that retraces every call. Triggers: Rune., jit, grad, vmap, scan, custom VJP, implicit differentiation, compiled training step."
---

# Rune

Rune transforms ordinary OCaml functions over ordinary values. There is no
special tensor type and no encoded parameter tree: a record is a record, and
its `walk` says which fields are arrays.

Read `references/entry-points.md` for signatures and contracts,
`references/tracing-and-jit.md` for what traces and what does not, and
`references/custom-rules.md` for custom derivatives.

## House rules

- Bind a compiled function once. `Rune.jit s f` makes a compiled function and
  `Rune.jit s f x` applies it; written at each call it makes a new function
  and traces every call. Put it at a module's top level or in a `let` that
  runs once.
- Every transformation takes a structure (`Nx.Ptree.t`) for each value it
  reads and a signature (`Nx.Ptree.fn`) for each function. Use the lab's
  spelling: `Nx.Ptree.(inputs_ptree @-> returns Nx.Ptree.tensor)`.
- Prefer `Rune.value_and_grad` to separate `f` and `Rune.grad` when you need
  both the loss and the gradient; it is one forward and one backward pass.
- Keep traced functions pure and their shapes static. Host-static control
  flow is fine; data-dependent control flow is `Rune.switch`, `Rune.cond` or
  `Nx.where`.
- A time loop over a horizon is a `Rune.scan`, not an OCaml loop. A fixed
  number of solver iterations may be an OCaml loop around a `scan`; a
  program's size then grows with the iterations, not with iterations times
  horizon.
- Test a derivative against a finite difference with `Rune.check_grads`, and
  a custom transpose with `Rune.check_transpose`. Use `float64` parameters for
  reliable differences.
- A GPU set has no eager kernels. Initialise models, optimizer state, data and
  keys on the host, and place them at the boundary of the jitted function that
  consumes them; see `raven-rig`.

## The two families

**Transformations** run their function once per call, over values that carry
more than their elements:

| Entry point | What it does |
| --- | --- |
| `jvp p q f params tangents` | forward-mode JVP, one pass |
| `vjp p q f params` | reverse-mode VJP; returns `(f params, pullback)` |
| `grad p f params` | gradient of a real scalar; `vjp` of `1` |
| `value_and_grad p f params` | `(loss, grads)` in one pass |
| `value_and_grad_aux p x f params` | `(loss, grads, aux)` |
| `linearize p q f params` | `(f params, jvp_at)` |
| `linear_transpose p q f params` | transpose of a linear `f` |
| `vmap ?axis s f` | map `f` over axis 0 of every argument |

**Constructs** trace their functions over stand-ins and run a program:

| Entry point | What it does |
| --- | --- |
| `jit ?beam ?parallel s f` | compile through tolk, keyed by argument shapes |
| `scan c x y f ~init xs` | fold a step over rows, stacking outputs |
| `iterate c ~max ~until f init` | run a step until a test holds |
| `switch b i branches` | data-dependent branch; only the taken branch computes |
| `cond b p t f` | `switch` on a boolean |
| `root ?linear_solve x ~residual solve` | implicit function theorem |
| `remat s f` | recompute in the backward pass to save memory |
| `callback s ~like f` | run OCaml code inside a traced function |

The one-array forms (`grad'`, `value_and_grad'`, `jvp'`, `vjp'`, `jacfwd'`,
`jacrev'`, `jit'`, `scan'`, `iterate'`, `vmap'`, `cond'`) are the same with
`Nx.Ptree.tensor` filled in. Use them for a function of one array.

## Choosing an entry point

| You need | Use |
| --- | --- |
| gradient of a scalar loss w.r.t. one array | `Rune.grad'` |
| gradient w.r.t. a record / nested structure | `Rune.grad` with its `Nx.Ptree.t` |
| loss and gradient together | `Rune.value_and_grad` |
| a Jacobian | `Rune.jacfwd'` (tall) or `Rune.jacrev'` (wide) |
| per-example gradients | `Rune.vmap` of `Rune.grad'`, or `Rune.vmap` around `Rune.grad` |
| Hessian-vector product | `Rune.jvp'` of `Rune.grad'` |
| derivative of a fixed-point or optimization solve | `Rune.custom_jvp` with a `Rune.custom_transpose`, or `Rune.root` |
| a loop over time | `Rune.scan` |
| a loop until convergence | `Rune.iterate` with `~max` and `~until` |
| a branch on a traced value | `Rune.switch` / `Rune.cond` |
| compiled code | `Rune.jit`, bound once |
| lower memory in the backward pass | `Rune.remat` |

`Rune.grad` requires a real scalar result (one float element, any shape);
use `vjp` for anything else. Integer and boolean arrays have no derivative:
their gradient is zero. Cast to a float dtype before differentiating.

## Traced code: what is allowed

Inside a construct, arrays are stand-ins with a dtype, shape and placement but
no elements. That means:

- `Nx.item`, `Nx.to_array`, `Nx.print`, `Nx.map_item` of a traced value
  raise. So does an OCaml `if` on one. Use `Rune.switch`, `Rune.cond`,
  `Nx.where`, or `Rune.iterate`.
- A draw without `~key` raises. Draw from a key the step receives, captures
  or computes. In a loop, use the step index:

```ocaml
(* [k] is a key the step captures; step [i] draws from its own key. *)
Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.tensor
  (fun x i ->
     Nx.Rng.with_key (Nx.Rng.fold_in_tensor k i) (fun () ->
         let x = Nx.add x (Nx.Rng.normal Nx.float32 [| 3 |]) in
         (x, x)))
  ~init:x0
  (Nx.arange Nx.int32 0 n 1)
```

- A step's carry must keep the visits, dtypes and shapes it received. A
  carry whose shape depends on the step raises at the trace.
- A traced function reads a captured `ref`, mutable value or domain-local
  state at the trace. Under `jit`, that is at the first call of each key. Do
  not use mutable state to communicate with a traced function; pass it as an
  argument.
- An operation that reads no stand-in computes once, at the call, except in
  `switch`, `remat`, a `root`'s `solve` and `jit`, which stage every
  operation their functions apply, captured values included.

## `jit`

```ocaml
let step =
  Rune.jit
    Nx.Ptree.(params @-> returns (pair tensor params))
    (fun ((w, b) as p) ->
       let loss, (gw, gb) = Rune.value_and_grad params mse p in
       (loss, (Infix.(w - (gw *$ 0.1)), Infix.(b - (gb *$ 0.1)))))
```

- A call traces `f` into a program keyed by the arguments' dtype, shape,
  placement, visits, the lane counts it asked about, total placements and
  probe collects. Elements are not part of the key: two calls with the same
  shapes share a program.
- The first call of a key traces; the first call that runs a program compiles
  it for the call's layout and donation variant. Later calls run it.
- Results are fresh, at the placement eager computation gives them, except a
  result that takes a donated argument's memory.
- A call consumes donated handles among its arguments. The donor dies after
  the call checks its arguments. Do not use a donated value afterwards.
- Checks (`Nx.check`, a sampler's parameter domain, an `iterate` past `max`,
  a `switch` index out of range) are computed with the program. On the host
  they raise at the call; on another set they raise at the next wait on that
  set.
- `~beam:n` searches each kernel for the fastest variant, `~parallel:n`
  compiles on up to `n` domains. Both make the first call much longer. Leave
  them off unless you are tuning a reused program.
- A jitted call inside another `jit` compiles as the outer program does.
- `RUNE_JIT_DEBUG=1` prints why a call retraced.

### Where values live

A compiled call's results lie where eager computation places them. A GPU set
has no eager kernels, so initialise on the host and cross to the device at the
jitted boundary: place model parameters and optimizer state just before the
first call of a recursive training loop, and the loop's jitted step returns
them on the device and keeps them there. Place a key with
`Nx.Rng.place D.on k` at the same boundary, not when it is created. Reading a
device value with `Nx.item` waits; do it at the reporting interval.

```ocaml
(* Host: build and initialise. *)
let init_key = Nx.Rng.key 0 in
let train_key = Nx.Rng.key 1 in
let params = Model.init ~key:init_key ~d_in ~d_out in
let state = Optim.init params in

(* Device: cross once, then let the jitted loop keep them. *)
let params = Nx.Ptree.place (module Model.P) D.on params in
let state = Nx.Ptree.place (module Optim.State) D.on state in
let train_key = Nx.Rng.place D.on train_key in
let rec train ~i (params, state) =
  if i = 0 then params
  else
    let params, state, _ = step train_key (params, state) in
    train ~i:(i - 1) (params, state)
```

The classic mistakes:

- `Rune.jit s f x` inside a loop: a new compiled function per iteration.
- A shape that changes between calls: a new key, a new trace and compile.
- Reading a traced value with `Nx.item` to decide something.
- Drawing without a key, or reusing one key for two independent draws.
  Split or `fold_in` instead; a key is spent once used.
- Capturing a value another transformation tracks. Pass it as an argument:
  `["Rune.jit: the function captures a value another transformation traces;
  pass it as an argument"]`.

## Custom derivatives

Use a custom rule when the derivative is cheaper than transposing the
forward code, or when the function is a solve whose derivative follows from
the implicit function theorem.

- `custom_jvp p q rule` where `rule args = (value, tangent_map)`. The tangent
  map must be **linear in its tangents** if reverse mode will transpose it.
  An `Nx.exp` of a tangent raises when transposed.
- `custom_transpose p q f ~transpose` gives a linear `f` a transpose with a
  better form than its operations' transposes. Put it inside a tangent map to
  carry both rules with one `custom_jvp`:

```ocaml
Rune.custom_jvp
  (Problem.inputs p)
  Nx.Ptree.tensor
  (fun inputs ->
     let sol = solve inputs in
     (taus_of sol, tangent_rule ~sol))
  inputs
```

  where `tangent_rule` is a `custom_transpose` whose `~transpose` is the
  implicit VJP. `packages/dilqr/lib/implicit_grad.ml` is the reference.
- `custom_vjp p q rule` gives a reverse-mode rule only. Forward mode around
  it raises unless a reverse mode applies the pullback. Use it when there is
  no forward rule.
- The rule must not use a value its own differentiation tracks. Pass such a
  value as an argument; reading it with `Nx.item` is allowed and reads the
  primal.
- A pullback runs after the differentiation returns. It must not use or read
  a value that differentiation tracks. It may read values captured at the
  call.
- `Rune.root` states that the result is a zero of a residual and
  differentiates that statement. It is the right tool for a Newton solve or a
  fixed point with a square residual.

## Totals and probes

`Rune.Total` is a write-only sum: code anywhere inside a function adds to it,
and the caller reads it when the function returns. `Rune.Probe` collects
values emitted inside a function, one row per execution. Use them for
diagnostics and for accumulating a quantity without threading it through a
carry.

```ocaml
let t : (float, Nx.float64_elt, 'd) Rune.Total.t = Rune.Total.make ()
let value, total =
  Rune.Total.collect t ~init:(Nx.scalar Nx.float64 0.) (fun () -> f params)
```

A total's `init` fixes its set and shape. An addition of another shape or set
raises where it is made.

## Pitfalls

- `Rune.vmap` maps over **axis 0**. Lanes are rows; a 0-d argument raises.
- `Nx.item` of a lane raises. A branch per lane is `Rune.switch`, a select
  per lane is `Nx.where`.
- `Rune.lanes` and `Rune.lane_index` need the name of a map that is around.
  Create the name with `Rune.axis ()`, pass it to `vmap ~axis`, and pass the
  same value to `lanes`/`lane_index`. `Rune.lane_count` is safe anywhere and
  returns `None` outside the map.
- `Rune.scan`'s step is traced once. Its side effects run once per trace, not
  once per row.
- `Rune.iterate` tests `until` before each step. It raises if the test is
  still false after `max` steps.
- `Rune.switch` traces every branch; only the taken one computes. All
  branches must return the same structure, dtypes and shapes.
- A `custom_jvp` tangent map that calls `Rune.lanes` must ask
  `Rune.lane_count` first, because it is traced whether or not a map is
  around.
- `Rune.detach x` has a zero tangent under every differentiation around the
  call. Use it to stop a gradient through a constant or a target.
- A second derivative of a `custom_vjp` differentiates the rule's code and
  the pullback, not a hidden forward rule.

## References

- `references/entry-points.md` — signatures, semantics, error cases.
- `references/tracing-and-jit.md` — keys, donation, checks, staging,
  retracing, debugging.
- `references/custom-rules.md` — custom JVP/VJP/transpose, `root`, `remat`,
  totals, probes, callbacks.
