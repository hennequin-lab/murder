# Custom rules, roots, rematerialization and side channels

Read the Custom differentiation rules, Roots, Rematerialization, Totals,
Probes and Callbacks sections of `rune.mli`.

## When to write a custom rule

- The derivative is cheaper than transposing the forward code: an implicit
  solve, a factorization reused across tangents, a linear solve whose
  transpose has a better form.
- The function is a fixed point or an optimization solve, and its derivative
  follows from the implicit function theorem.
- The forward code is not differentiable in the sense Rune needs (a
  non-smooth clamp, a custom numerical routine), and you can state the
  derivative you want.
- A library boundary: the caller should get a correct derivative without
  seeing the implementation.

Do not write one to hide a bug or to avoid a NaN at a kink. Use
`Nx.where`, `detach` or `cond` for the kink, and keep the real derivative.

## `custom_jvp`

```ocaml
val custom_jvp :
  ('p,'d) Ptree.t -> ('q,'d) Ptree.t -> ('p -> 'q * ('p -> 'q)) -> 'p -> 'q
```

`rule args` returns the value and a tangent map. Forward mode applies the
map; reverse mode transposes it. The map must be **linear in its tangents**:

- `Nx.mul`, `Nx.matmul`, `Nx.add`, `Nx.solve`, `Nx.reshape` and friends are
  fine.
- `Nx.exp`, `Nx.log`, `Nx.pow` of a tangent raise when reverse mode
  transposes the map: `["Rune.grad: a custom_jvp tangent map applies Exp to a
  tangent; a tangent map must be linear in its tangents"]`.
- An `iterate` in the map whose `until` reads a tangent also raises, at the
  call of `grad`, `vjp` or `linearize`.

The rule runs on values in two cases: no argument belongs to any code and no
construct stages the call, or one differentiation tracks an argument and no
transformation or traced function lies between. There the rule may read its
arguments with `Nx.item`, as a convergence test does. Elsewhere it is traced,
and a read raises.

The rule receives its arguments without their derivatives. It must not use
another value the same differentiation tracks; pass that value as an
argument. A read with `Nx.item` is allowed and reads the primal.

The derivative's result must match the value in visits, dtypes, shapes and
placements. The check names the first path where they differ.

### The paired-rule idiom

One `custom_jvp` can carry both directions when its tangent map is a
`custom_transpose` whose transpose is the reverse rule. This is how
`packages/dilqr/lib/implicit_grad.ml` attaches the implicit VJP and JVP to
one solve:

```ocaml
let tangent_rule p ~theta ~x0 ~us0 ~sol dinputs =
  Rune.custom_transpose
    (Problem.inputs p)
    Nx.Ptree.tensor
    (fun dinputs -> tangent_of_solve p ~sol ~dx0:dinputs.x0 ~dtheta:dinputs.theta)
    ~transpose:(fun ybar -> bwd p { theta; x0; us0; xs = sol.xs; us = sol.us } ybar)
    dinputs

let solve_fixed p ~iters inputs =
  Rune.custom_jvp
    (Problem.inputs p)
    Nx.Ptree.tensor
    (fun { theta; x0; us0 } ->
       let sol = solve p ~iters theta x0 us0 in
       (taus_of_stacked p sol, tangent_rule p ~theta ~x0 ~us0 ~sol))
    inputs
```

`Rune.grad` applies `bwd`; `Rune.jvp` applies the streaming tangent solve.
Neither transposes the sweeps operation by operation.

## `custom_vjp`

```ocaml
val custom_vjp :
  ('p,'d) Ptree.t -> ('q,'d) Ptree.t -> ('p -> 'q * ('q -> 'p)) -> 'p -> 'q
```

`rule args` returns the value and a pullback. Every reverse mode that tracks
an argument applies the pullback. There is no forward rule: forward mode
around it raises unless a reverse mode applies the pullback (as in a
Hessian-vector product).

The pullback runs in the backward pass, after the differentiation returns.
It must not use or read a value that differentiation tracks. It may read
values captured at the call, and it may use what it read there.

Use `custom_vjp` when only reverse mode matters, or when the pullback is
naturally written as a function of cotangents. Use `custom_jvp` plus
`custom_transpose` when both modes matter.

## `custom_transpose`

```ocaml
val custom_transpose :
  ('p,'d) Ptree.t -> ('q,'d) Ptree.t -> ('p -> 'q)
  -> transpose:('q -> 'p) -> 'p -> 'q
```

`f` must be linear in its argument; `transpose` must satisfy
`<w, f v> = <transpose w, v>` for every `v` and `w`. That is the caller's
promise, tested by `Rune.check_transpose`.

`f` and `transpose` may read residuals, in which they need not be linear. A
draw without `~key` in either raises. Neither may add to a total: each also
runs as the other's transpose, away from the call.

The classic use is a solve:

```ocaml
let solve_linear a b =
  let x = Nx.solve a b in
  let along v =
    Rune.custom_transpose Nx.Ptree.tensor Nx.Ptree.tensor
      (fun v -> Nx.solve a v)
      ~transpose:(fun w -> Nx.solve (Nx.matrix_transpose a) w)
      v
  in
  (x, along)
```

`Rune.check_transpose p q f ~transpose args` tests the promise on three fixed
random pairs. It reads results, so it must not be called on a traced
function. Use it in tests for every custom transpose.

## `root`

```ocaml
val root :
  ?linear_solve:(('x -> 'x) -> 'x -> 'x) ->
  ('x,'d) Ptree.t -> residual:('x -> 'x) -> (unit -> 'x) -> 'x
```

`solve ()` returns a value stated to be a zero of `residual`. Under
differentiation, the result's tangent `u` solves `J u + r = 0`, where `J` is
the derivative of `residual` at the result and `r` is the tangent of
`residual` there with the result held. The derivative is taken at the
returned point, whether or not `residual` vanishes there.

- The system must be square: `residual`'s result has the solution's
  structure, dtypes and shapes.
- `solve` has no derivative. A tracked value it reads is its value alone.
- `solve`, `residual` and `linear_solve` are traced. `Nx.item` of a value
  `residual` computes from its argument raises.
- `solve` stages every operation it applies, so `Nx.item` inside it of a
  value a transformation tracks raises.
- `linear_solve op b` returns `v` with `op v = b` for a linear `op`. It is
  traced once against an abstract `op`; a derivative fills in `J`, reverse
  mode `J`'s transpose. Without it, a derivative builds `J`'s matrix with one
  map of unit vectors and solves it with `Nx.solve`, which suits small
  systems. A complex `J` need only be real-linear.
- Under `vmap`, each lane solves its own system.

```ocaml
(* sqrt a by Newton, with the derivative 1 / (2 sqrt a) from the residual *)
let sqrt_newton a =
  let residual x = Nx.Infix.((x * x) - a) in
  Rune.root Nx.Ptree.tensor ~residual (fun () ->
      Rune.iterate' ~max:50
        ~until:(fun x -> Nx.all (Nx.less_s (Nx.abs (residual x)) 1e-12))
        (fun x -> Nx.Infix.((x + (a / x)) /$ 2.))
        a)
```

`root` is the right tool when the solve has a square residual. A solve with
constraints or a non-square system is usually a `custom_jvp` with an explicit
VJP.

## `remat`

```ocaml
val remat : ('a -> 'b, 'd) Ptree.fn -> ('a -> 'b) -> 'a -> 'b
```

`remat s f` is `f`, traced and evaluated. Its reverse mode keeps `f`'s
arguments and captures and recomputes intermediates, trading computation for
memory. Every transformation sees `remat s f` as `f`. Inside a `jit`, the
backward pass recomputes each intermediate that reads a value the program
computes; a value computed from the program's arguments and constants alone
is kept.

Use it on a long chain whose activations would otherwise fill memory. It does
not change values, only memory and time.

## Totals and probes

### `Rune.Total`

```ocaml
module Total : sig
  type ('v,'s,+'d) t
  val make : unit -> ('v,'s,'d) t
  val add : ('v,'s,'d) t -> ('v,'s,'d) Nx.t -> unit
  val collect : ('v,'s,'d) t -> init:('v,'s,'d) Nx.t -> (unit -> 'r) -> 'r * ('v,'s,'d) Nx.t
end
```

A total is a write-only sum. Code anywhere inside `collect`'s function adds
to it; the caller reads `init` plus the additions when the function returns.
Nothing reads a total before `collect` returns, so an addition never changes
a computed value. With no `collect` open, `add` does nothing.

- `init` fixes the total's set and shape. An addition of another shape or set
  raises where it is made, inside a traced function as it is traced.
- A traced function's additions reach the `collect` when the construct
  returns, so a construct that raises adds none.
- A `scan` or `iterate` step adds once per step; `until` adds once per test.
- Under `vmap`, each lane adds its own; a `collect` outside the map receives
  them in lane order.
- Under a differentiation, an addition adds its value's primal.

Use it for a diagnostic (a norm per step, a divergence term) without
threading it through a carry. `'d` is covariant, so a total made at a
library's top level serves every device set.

### `Rune.Probe`

```ocaml
module Probe : sig
  type 'a t
  val make : ('a,'d) Ptree.t -> 'a t
  val emit : 'a t -> 'a -> unit
  val collect : 'a t -> (unit -> 'r) -> ('r, exn * Printexc.raw_backtrace) result * 'a list
end
```

A probe receives values emitted inside a function. Each `emit` outside a
traced function offers the value with a leading axis of one; inside one, a
site offers once per execution, with enclosing rows, steps and lanes merged
into the leading axis. Emissions offer primals, once per execution; a
backward pass or a transpose offers nothing.

`collect` returns the function's result or its exception with a backtrace,
and the offers that reached it. Use it to log per-step values from inside a
`jit` or a `scan` without changing the computation.

## `callback`

```ocaml
val callback :
  ?lanes:('a -> 'b) -> ('a -> 'b, 'd) Ptree.fn -> like:'b -> ('a -> 'b) -> 'a -> 'b
```

`callback s ~like f` runs the OCaml function `f` on values, once per
execution of the call, so it may have effects such as a run of a solver
outside nx. `f` must return arrays of `like`'s dtypes, shapes and placements;
`like` may be an argument or an `eval_shape` result. A callback has no
derivative: a differentiation that tracks an argument raises at the call. A
custom rule around the call can supply one.

`~lanes:g` makes a `vmap` run `g` in `f`'s place, once per execution of the
map, on arrays that lead with an axis of the lanes. Use it for a batched
external solver.

Callbacks are also what makes a compiled program split: each stretch without
a callback is compiled, and the callback runs between them. Keep callbacks
out of a hot loop unless the external work dominates.

## Testing a custom rule

```ocaml
let () =
  match Rune.check_grads ~eps:1e-6 Nx.Ptree.tensor f x with
  | Ok () -> ()
  | Error msg -> failwith msg

let () =
  match Rune.check_transpose Nx.Ptree.tensor Nx.Ptree.tensor f ~transpose x with
  | Ok () -> ()
  | Error msg -> failwith msg
```

Use `float64` parameters: central differences at `float32` are noisy. Test
both modes where both are promised: `Rune.jvp` against the finite difference,
and `Rune.grad` against the transpose pairing. A rule that is correct in
forward mode and wrong in reverse mode is a common bug when the tangent map
is not linear.
