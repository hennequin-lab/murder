# Rune entry points

`'p`, `'q`, `'c`, `'x`, `'y` are structures (`Nx.Ptree.t`). `'d` is the
device set. In the one-array forms, `t` is `('v,'s,'d) Nx.t`. Read
`rune.mli` for the full contract; this file is the map.

## Derivatives

```ocaml
val jvp : ('p,'d) Ptree.t -> ('q,'d) Ptree.t -> ('p -> 'q) -> 'p -> 'p -> 'q * 'q
```
`jvp p q f params tangents` is `(f params, dy)`: the Jacobian-vector product
in one forward pass. `tangents` has `params`' structure, dtypes and shapes.
The tangent of an integer or boolean parameter is ignored, and `dy` holds
zeros there. Raises if `params` holds no float or complex array, or if
`tangents` differ from `params` in visits, dtype, shape or placement.

```ocaml
val grad : ('p,'d) Ptree.t -> ('p -> ('v,'s,'d) Nx.t) -> 'p -> 'p
```
Gradient of a real scalar (one float element, any shape). The gradient of an
integer or boolean array, or of one `f` does not depend on, is zero. Raises
`["Rune.grad: the objective must return a real scalar, got complex128 []"]`
otherwise. Use `vjp` for a non-scalar result.

```ocaml
val value_and_grad : ('p,'d) Ptree.t -> ('p -> ('v,'s,'d) Nx.t) -> 'p -> ('v,'s,'d) Nx.t * 'p
val value_and_grad_aux :
  ('p,'d) Ptree.t -> ('x,'d) Ptree.t -> ('p -> ('v,'s,'d) Nx.t * 'x) -> 'p
  -> ('v,'s,'d) Nx.t * 'p * 'x
```
One forward and one backward pass. `value_and_grad_aux` returns an auxiliary
value that contributes nothing to the gradient.

```ocaml
val vjp : ('p,'d) Ptree.t -> ('q,'d) Ptree.t -> ('p -> 'q) -> 'p -> 'q * ('q -> 'p)
```
`(f params, pullback)`. The pullback maps the result's cotangents to
`params`'. It runs no part of `f` again, except the pullbacks of custom rules
`f` calls. Outside every transformation it may be applied any number of
times, from any domain. Inside a transformation, apply it where the
transformation expects.

```ocaml
val linearize : ('p,'d) Ptree.t -> ('q,'d) Ptree.t -> ('p -> 'q) -> 'p -> 'q * ('p -> 'q)
val linear_transpose : ('p,'d) Ptree.t -> ('q,'d) Ptree.t -> ('p -> 'q) -> 'p -> 'q -> 'p
```
`linearize` returns the JVP at `params` without running `f` again.
`linear_transpose` transposes a function linear in its argument: it runs `f`
once over stand-ins, drops any term that does not depend on the argument, and
then applies the transpose. An `iterate` in `f` is transposed step by step
where its stop reads no value computed from the argument.

```ocaml
val check_grads : ?eps:float -> ?tol:float -> ('p,'d) Ptree.t
  -> ('p -> ('v,'s,'d) Nx.t) -> 'p -> (unit, string) result
val check_transpose : ?tol:float -> ('p,'d) Ptree.t -> ('q,'d) Ptree.t
  -> ('p -> 'q) -> transpose:('q -> 'p) -> 'p -> (unit, string) result
```
`check_grads` compares the gradient with central differences along two fixed
directions. `check_transpose` checks `<w, f v> = <transpose w, v>` on three
fixed random pairs. Both read results, so neither may be called on a traced
function. Use `float64` for reliable differences.

```ocaml
val detach : ('v,'s,'d) Nx.t -> ('v,'s,'d) Nx.t
```
Zero tangent under every differentiation around the call; the value itself
outside one.

## Maps

```ocaml
type axis
val axis : unit -> axis
val vmap : ?axis:axis -> ('a -> 'b, 'd) Ptree.fn -> ('a -> 'b) -> 'a -> 'b
val lanes : axis -> ('v,'s,'d) Nx.t -> ('v,'s,'d) Nx.t
val lane_index : axis -> ('v,'s,'d) Nx.t -> 'd Nx.int32_t
val lane_count : axis -> int option
```

`vmap s f` maps `f` over axis 0 of every argument's arrays. Lane `k` receives
row `k`. Each result array is the lanes' arrays stacked on a new axis 0; an
array that depends on no lane is broadcast along it. Lanes lie where their
rows lie, so a batch cut along axis 0 across devices computes each lane on
the devices that hold its row.

`lanes a x` gathers every lane's `x` on a new leading axis, inside the map
named `a`. It is linear, and its transpose sums the lanes' cotangents.
`lane_index a x` is the calling lane's index, a 0-d int32. `lane_count a` is
`Some n` inside the map, `None` outside; it reads no value and is safe in a
custom rule's tangent map.

A map applied without `~axis` has a fresh name, so no code inside can read
its lanes. Give a name when a custom rule or a nested map must.

Raises at a call if the arguments hold no array, a 0-d array, or arrays of
different leading lengths.

## Loops

```ocaml
val scan : ('c,'d) Ptree.t -> ('x,'d) Ptree.t -> ('y,'d) Ptree.t
  -> ('c -> 'x -> 'c * 'y) -> init:'c -> 'x -> 'c * 'y
val iterate : ('c,'d) Ptree.t -> max:int -> until:('c -> 'd Nx.bool_t)
  -> ('c -> 'c) -> 'c -> 'c
```

`scan` folds the step over the rows of `xs`, a structure whose arrays share a
leading length. Step `i` gets `f carry row_i`; the outputs are stacked on a
new axis 0. With `n = 0` the result is `init` and outputs of leading length
0. Under reverse mode a row's cotangent is stacked as the outputs are.

`iterate` applies the step until `until carry` holds, testing before each
step. Raises if the test is still false after `max` steps, naming the first
lane still running under a map. A derivative of an `iterate` is the
derivative of the steps taken; the test has none.

Both trace their step once per trace of the loop. Side effects in a step run
once per trace. A step that produces a row but has no `scan` output writes it
into a carry array with `Nx.scatter ~axis:0 p row a`, where `p` is the step's
count in the carry.

## Branches

```ocaml
val switch : ('b,'d) Ptree.t -> 'd Nx.int32_t -> (unit -> 'b) list -> 'b
val cond : ('b,'d) Ptree.t -> 'd Nx.bool_t -> (unit -> 'b) -> (unit -> 'b) -> 'b
```

`switch b i branches` takes branch `i` of `branches`. Every branch is traced;
only the taken one computes, so a branch not taken does no work, raises no
check, adds nothing to a total, and produces no NaN. All branches must return
the same visits, dtypes, shapes and device sets. `cond` is `switch` on a
boolean; a NaN compares false.

Under `vmap`, if `i` is the same in every lane the branch runs over the lanes.
If it differs, each branch some lane takes runs over every lane, and a lane
that did not take it is held on the first lane that did.

## Roots and rematerialization

```ocaml
val root : ?linear_solve:(('x -> 'x) -> 'x -> 'x) -> ('x,'d) Ptree.t
  -> residual:('x -> 'x) -> (unit -> 'x) -> 'x
val remat : ('a -> 'b, 'd) Ptree.fn -> ('a -> 'b) -> 'a -> 'b
```

`root x ~residual solve` is `solve ()`, stated to be a zero of `residual`.
Under differentiation the result's tangent solves `J u + r = 0`, where `J` is
the derivative of `residual` at the result and `r` is its tangent with the
result held. The system must be square. `linear_solve` may replace the
default `Nx.solve`; it is traced against an abstract `op` that a derivative
fills in with `J`.

`remat s f` is `f`, traced and evaluated, whose reverse mode keeps `f`'s
arguments and captures and recomputes intermediates. Every transformation
sees `remat s f` as `f`.

## Compilation

```ocaml
val jit : ?beam:int -> ?parallel:int -> ('a -> 'b, 'd) Ptree.fn -> ('a -> 'b) -> 'a -> 'b
val jit' : ?beam:int -> ?parallel:int -> (t -> t) -> t -> t
```

`jit s f` is `f` compiled. See `tracing-and-jit.md`.

## One-array forms

```ocaml
val jvp'   : (t -> t) -> t -> t -> t * t
val jacfwd' : (t -> t) -> t -> t
val grad'  : (t -> t) -> t -> t
val value_and_grad' : (t -> t) -> t -> t * t
val vjp'   : (t -> t) -> t -> t * (t -> t)
val jacrev' : (t -> t) -> t -> t
val scan'  : (t -> t -> t * t) -> init:t -> t -> t * t
val cond'  : 'd Nx.bool_t -> (unit -> t) -> (unit -> t) -> t
val iterate' : max:int -> until:(t -> 'd Nx.bool_t) -> (t -> t) -> t -> t
val vmap'  : ?axis:axis -> (t -> t) -> t -> t
val jit'   : ?beam:int -> ?parallel:int -> (t -> t) -> t -> t
```

`jacfwd' f x` has shape `shape (f x) @ shape x`; column `k` is the tangent
along element `k` of `x` in C order, computed in one forward pass.
`jacrev'` is the same from one backward pass; it raises if `x` is real and
`f x` complex, since `x`'s dtype cannot hold that Jacobian.

## Debug counters

```ocaml
Rune.derivations ()   (* programs derived from compiled calls, this domain *)
Rune.evaluations ()   (* compiled calls the compiler declined, this domain *)
Rune.variants ()      (* variants lowered and linked, this domain *)
```

These are for the suites, not for application code.
