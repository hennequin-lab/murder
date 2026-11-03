# Tracing, keys and `jit`

Read the Tracing, Compilation, Donation, Checks and Numerics sections of
`rune.mli` with this file.

## What runs when

| Kind | Example | Behaviour |
| --- | --- | --- |
| transformation | `jvp`, `grad`, `vjp`, `vmap` | runs its function once per call, over values carrying tangents or lanes |
| construct | `scan`, `iterate`, `switch`, `root`, `remat`, `jit` | traces its function over stand-ins and runs a program |

A construct other than `jit` traces its functions each time it is called: at
each call outside a `jit`, and inside one at each trace of the `jit`. A
`jit` traces once per key. A loop's step is traced once per trace of the
loop, whatever the number of steps, so the step's code and side effects run
once per trace.

`Rune.program` and `Rune.eval_shape` trace without compiling or computing:

```ocaml
val program    : ('a,'d) Ptree.t -> ('b,'d) Ptree.t -> ('a -> 'b) -> 'a -> program
val pp_program : Format.formatter -> program -> unit
val eval_shape : ('a,'d) Ptree.t -> ('b,'d) Ptree.t -> ('a -> 'b) -> 'a -> 'b
```

`program a b f x` runs `f` once over stand-ins of `x`'s arrays and returns
what it staged. `pp_program` prints constants, parameters, equations,
constructs and outputs; the numbering is stable across traces, so two
functions can be diffed. `eval_shape` returns `f x`'s structure with each
array a stand-in; `Nx.shape`, `Nx.dtype` and `Nx.placement` work, anything
else raises. Use `eval_shape` to check shapes before a run, and `program` to
see why a trace fails.

## Stand-ins and what a traced function may do

A stand-in has a dtype, a shape and a placement, and no elements. Therefore:

- `Nx.item`, `Nx.to_array`, `Nx.to_bigarray`, `Nx.print` and `Nx.map_item`
  of a stand-in raise, naming the construct: `["Nx.item: the value is traced
  by Rune.scan"]`.
- An OCaml `if` on a stand-in cannot work. Use `switch`, `cond`, `where`, or
  `iterate`.
- A draw without `~key` raises, naming the construct. Draw from a key the
  step receives, captures or computes, with `~key` or inside
  `Nx.Rng.with_key`.
- An operation that reads no stand-in computes once, at the call, except in
  `switch`, `remat`, a `root`'s `solve` and `jit`, which stage every
  operation their functions apply, captured values included.

A traced function reads what it captures when it is traced. That includes a
`ref`'s contents, a mutable value, a handler of the caller's own effect and
domain-local state. Under `jit`, that is the first call of each key. Pass
state as an argument instead.

`Rune.lanes` and `Rune.lane_index` raise outside the map that names them.
`Rune.lane_count` is safe everywhere and returns `None` outside.

## Keys

`jit s f` keeps one program per key. A key is the abstract value of a call's
arguments:

- their `Nx.Ptree.val-visits`, including integers and variant cases the walk
  reports;
- each array's dtype, shape and placement;
- for each name `f` passes to `lanes`, `lane_index` or `lane_count` as it is
  traced, the number of lanes of that map, or that none is around;
- the placement of the running value of each `Total` `f` adds to, where it
  differs from the first trace;
- whether a `Probe.collect` of each probe `f` emitted to is open.

Elements are not part of the key. Neither is whether an argument is donated.

Consequences:

- Two calls with the same shapes and dtypes share a program even when their
  values differ. That is the point.
- A shape that changes between calls is a new key: a new trace and compile.
  Pad to a fixed shape, or bucket the shapes.
- An integer in a structure that a `Walk.int` reports is part of the key. A
  window or block size written as a plain OCaml `int` outside the structure
  is captured at the trace and silently frozen.

A call whose key is being traced waits for that trace, except on the tracing
domain, where it traces again. Domains share programs.

## Life cycle of a compiled function

1. `let g = Rune.jit s f` makes a compiled function with no program. `g` must
   be bound once.
2. The first call of `g` with a new key traces `f` into a program.
3. The first call that runs the program compiles it for the call's variant.
4. Later calls with that key and variant run the compiled program.
5. A transformation around a call derives programs from its key's, once per
   derivation, and compiles them as it does the key's.

A variant is the layout of the arguments in memory (strides, offset phase
from the last 16-byte boundary, donation) together with tolk's settings and
its `DEBUG`. A view reads in place from a program compiled for its layout.

`?beam:n` searches each kernel for the fastest variant and times candidates
on the device, which makes the first call much longer. `?parallel:n`
compiles on up to `n` domains. Both override the `BEAM`, `JITBEAM` and
`PARALLEL` settings. A call inlined in an enclosing `jit` compiles as that
one's program does.

## Donation under `jit`

A call consumes the donated handles among its arguments, as one operation
consumes its operands. The values they give up die once the call has checked
its arguments; a later use raises naming the entry point. A result may take a
donated argument's memory where dtype, element count and placement match, no
check reads the argument, and the result either reads it at each element's
own index or is the argument itself. A donated argument whose memory another
value shares, or that is a view off 16 bytes or out of C order, lends nothing.

The result's values are the same either way, so donation is a memory and
bandwidth decision, not a numerical one. `Nx.Ptree.donate` donates every
array of a structure in walk order.

## Placement under `jit`

A compiled call's results are fresh, at the placement eager computation gives
them, except a result that takes a donated argument's memory. A jitted
training step therefore returns its parameters and state on the device, and a
recursive loop that threads them keeps them there.

Initialise on the host. A GPU set has no eager kernels, so a model or
optimizer state placed at initialisation cannot be touched by host code.
Place model parameters, optimizer state, inputs, targets and keys at the
boundary of the first jitted call:

```ocaml
let params = Nx.Ptree.place (module Model.P) D.on params in
let state = Nx.Ptree.place (module Optim.State) D.on state in
let train_key = Nx.Rng.place D.on train_key in
```

A value `f` captures on another device is copied there once per program, and
the program holds the copy. That makes a captured host constant work, but it
also hides the copy; explicit placement at the boundary keeps the cost visible
and leaves the host value usable.

Host reads (`Nx.item`, `Nx.to_array`) wait on the set. Read the loss at the
reporting interval, not every step.

## Checks under `jit`

`Nx.check`, a sampler's parameter domain, an `iterate` past `max` and a
`switch` index out of range are computed with the program. They are one check
of the call, made as it returns:

- on the host it raises at the call and the results are lost;
- on another set the call returns its results and the failure raises at the
  next wait on that set.

That difference matters for a GPU loop: an `iterate` that fails eagerly
raises at the call, but compiled it raises at the next `Nx.item` or `Nx.sync`.
Add a wait at the end of a program whose checks you need to see.

## Evaluated programs

The compiler declines some programs and runs them as nx operations, giving
eager's values. A program is evaluated when it holds:

- a map's `lanes` or `lane_index`;
- an emission that must be offered before a failing check on the host;
- the tangents of a `custom_vjp`'s results;
- an operation tolk does not lower;
- a check whose flags lie on another set than the program's device;
- no value, or an argument or result on another device than the program's, or
  a value over several devices;
- a device that computes nothing (`Rig.computes`), or something tolk cannot
  compile.

A declined program that runs a `callback` is split around it: each maximal
stretch with no callback and no total addition becomes a compiled call of its
own, and the callbacks and additions between them are evaluated. A loop, a
switch, a `remat` or a custom rule whose function runs a callback is
evaluated.

On a set without kernels, a stretch that holds values on several devices
raises, because nx cannot compute it.

## Debugging a retrace

Set `RUNE_JIT_DEBUG` to a nonzero integer before rune initialises. Each
retrace prints on standard error the first difference between its key and the
latest call's key, and the first call to find its program or a variant
declined prints why.

Other tools:

```ocaml
Rune.derivations ()   (* programs derived from compiled calls *)
Rune.evaluations ()   (* compiled calls the compiler declined *)
Rune.variants ()      (* variants lowered and linked *)
```

The counters are per domain and intended for suites. They are still the
quickest way to confirm that a program is not retracing.

Typical causes of a retrace:

| Symptom | Cause |
| --- | --- |
| retrace every call | `Rune.jit s f x` written at the call site |
| retrace when a batch size changes | shape is part of the key; bucket or pad |
| retrace after adding a field | visits changed; the walk reports a new leaf or report |
| retrace inside `vmap` | `lane_count` or a named map's lane count changed |
| trace error naming a capture | a value another transformation tracks is captured; pass it as an argument |
| trace error naming `Nx.item` | a read of a stand-in; use `switch`/`cond`/`where` |
| trace error naming a draw | a sampler without `~key`; scope a key |

## Numerics

A compiled call computes each nx function to the accuracy `rune.mli` states.
A float result may differ from eager's within that bound. A float reduction,
scan, contraction or scatter by `Add` combines terms in an order the compiler
chooses, so its bits are not eager's. A move of elements, a cast, a
comparison, an extreme and integer arithmetic are exact.

For a numerical regression, compare compiled and eager results with a stated
tolerance, not bit for bit. For a NaN, check shapes and dtypes first; a
broadcast that eager accepted and a compiled kernel declines raises at the
call, not inside the program.
