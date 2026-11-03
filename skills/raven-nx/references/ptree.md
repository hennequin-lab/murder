# Structures: `Nx.Ptree`

A structure is an ordinary OCaml value with arrays at known positions. `walk`
is the one function that enumerates them. Everything else — gradients,
optimizer steps, placement, casting, saving, compiled signatures — is built on
`walk`.

Read `Nx.Ptree` in `nx.mli` and `nx/doc/06-structures.md` for the full
contract.

## Writing a walk

Prefer `ppx_ptree` (`[@@deriving ptree]`), as `AGENTS.md` asks. The deriver
generates `walk_<type>`, and the module only needs to name the type:

```ocaml
type ('a, _) params =
  { w : 'a
  ; b : 'a
  }
[@@deriving ptree]

module Params = struct
  type ('a, 'd) t = ('a, 'd) params
  let walk = walk_params
end

let params = Nx.Ptree.instantiate (module Params)
```

For types the deriver does not cover, write `walk` with `Nx.Ptree.Walk`:

```ocaml
module Linear = struct
  type ('a, _) t = { w : 'a; b : 'a option }

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" (option leaf) b in
    { w; b }
end
```

Rules:

- Walk each part once, with a sequence of `let`s. A record literal evaluates
  fields in an unspecified order; if order matters, bind first.
- `leaf c x` walks a position of the parameter. Its type changes from the
  cursor's input to its output, so a forgotten field is a type error.
- `tensor c x` walks a fixed array (a constant that is not a parameter).
- `int c n` reports an integer that changes what a program computes (a
  window, a block shape, a layer kind). `case c tag` reports a variant case
  before the case's parts. `option`, `list` report presence and length.
- `field c "name" walk x`, `index c i walk x`, `under outer inner c x`,
  `structure s c x`.

`under U.walk V.walk` walks a structure whose positions are themselves
structures. `structure s c x` walks a value of a description at the cursor's
path. `Rng.walk` walks a key, and `Rng.ptree` describes one alone.

A `walk` that returns every part unchanged rebuilds an equal value. A test of
`Nx.Ptree.visits` catches a forgotten fixed array, integer or case, which the
type checker cannot.

## Using a description

```ocaml
val instantiate : (module U : S) -> ((('v,'s,'d) tensor, 'd) U.t, 'd) t
val nest : (module U : S) -> ('s,'d) t -> (('s,'d) U.t, 'd) t
val v : ('a 'b. cursor -> 's -> 's) -> ('s,'d) t
val tensor : (('v,'s,'d) tensor, 'd) t
val unit : (unit, 'd) t
val pair : ('a,'d) t -> ('b,'d) t -> ('a * 'b, 'd) t
val option : ('a,'d) t -> ('a option, 'd) t
val list : ('a,'d) t -> ('a list, 'd) t
```

- `instantiate (module P)` is the common case: arrays at the parameter's
  positions.
- `nest (module P) s` nests a module inside another structure. An optimizer's
  state over a model is `nest (module State) params`.
- `v walk` is a structure without a module, for records assembled from other
  structures. `dilqr/lib/problem.ml` builds its solve inputs this way:

```ocaml
Nx.Ptree.v (fun c { Inputs.theta; x0; us0 } ->
  let open Nx.Ptree.Walk in
  let theta = field c "theta" (structure problem.params) theta in
  let x0 = field c "x0" tensor x0 in
  let us0 = field c "us0" (under Controls.walk tensor) us0 in
  { Inputs.theta; x0; us0 })
```

- `pair`, `option`, `list` and `unit` compose descriptions without a module.

A description is at one dtype and one set, fixed by its first use.

## Derived maps and folds

```ocaml
val map    : ('s,'d) t -> (Path.t -> ('v,'r,'d) tensor -> ('v,'r,'d) tensor) -> 's -> 's
val map2   : ('s,'d) t
           -> ('v 'r. Path.t -> ('v,'r,'d) tensor -> ('v,'r,'d) tensor
                       -> ('v,'r,'d) tensor)
           -> 's -> 's -> 's
val fold   : ('s,'d) t -> (Path.t -> ('v,'r,'d) tensor -> 'acc -> 'acc) -> 's -> 'acc -> 'acc
val donate : ('s,'d) t -> 's -> 's
```

The callback is polymorphic in dtype, because one value may hold arrays of
several dtypes. `map` is how the lab flips a horizon axis in a whole tape:

```ocaml
Nx.Ptree.map rows (fun _ x -> flip x) tape.blocks
```

`fold` takes the accumulator last, as `Map.fold` does. `sofo` uses it to find
the first tensor leaf:

```ocaml
Nx.Ptree.fold structure
  (fun _ leaf acc -> Option.fold ~none:(Some (Nx.P leaf)) ~some:Option.some acc)
  x None
```

## Placement, casting, payloads

```ocaml
val place : (module U : S) -> 'e Placement.t -> (('v,'s,'d) tensor, 'd) U.t
          -> (('v,'s,'e) tensor, 'e) U.t
val cast  : (module U : S) -> ('w,'r) dtype -> (('v,'s,'d) tensor, 'd) U.t
          -> (('w,'r,'d) tensor, 'd) U.t
```

`place` moves every array, fixed ones included. It raises what a walk raises;
`Rng.walk` refuses a key at a placement that cuts the words' axis, so place
keys at `Placement.uncut p` with `Nx.Rng.place`.

`cast` casts only the parameter's positions; fixed arrays are kept.

`Nx.Ptree.place` is how a whole model or optimizer state crosses to a device.
Place it at the boundary of the jitted function that consumes it, not at
initialisation: a GPU set has no eager kernels, so host code cannot use a
placed structure. Once a jitted step returns the structure, a recursive loop
keeps it on the device. See `raven-rig`.

`Nx.Ptree.Payload.map/map2/fold` are for structures whose positions hold a
payload (a mask, a learning rate per array, a plan), not arrays.

## Arithmetic over a structure

These treat the float and complex arrays of a structure as one real vector.
An integer or boolean array is not part of the vector: `dot` and `norm` skip
it, `scale` and `axpy` keep it.

```ocaml
val dot   : ('s,'d) t -> (float,'c) dtype -> 's -> 's -> (float,'c,'d) tensor
val norm  : ('s,'d) t -> (float,'c) dtype -> 's -> (float,'c,'d) tensor
val scale : ('s,'d) t -> (float,'c,'d) tensor -> 's -> 's
val axpy  : ('s,'d) t -> (float,'c,'d) tensor -> 's -> 's -> 's
```

These are the primitives of optimizer steps, integrators and samplers over a
parameter tree. `dot` starts from `[0]` whole on the placement of the first
array in walk order; `norm` computes without overflowing a square.

## Paths and visits

```ocaml
type seg = Field of string | Index of int
Nx.Ptree.Path.root
Nx.Ptree.Path.v [ Field "layers"; Index 0; Field "w" ]
Nx.Ptree.Path.to_string p       (* "layers.0.w" *)
Nx.Ptree.Path.equal p q
Nx.Ptree.visits s x             (* Leaf p | Report (p, report) list, in walk order *)
Nx.Ptree.pp_visit fmt v
```

A path names an array in a saved file. Two paths that print alike may differ:
the single field `"a.b"` and the field `a` then `b` print the same and compare
unequal.

`visits` is the test hook for a walk: compare it with the arrays, integers and
cases the walk must visit. `Rune.jit` keys programs on visits.

## Flattening and signatures

```ocaml
val flatten : ('s,'d) t -> 's -> 'd packed list * Skeleton.t
val rebuild : ('s,'d) t -> like:'s -> 'd packed list -> 's

type (_, 'd) fn =
  | Returns : ('a,'d) t -> ('a,'d) fn
  | Arg : ('a,'d) t * ('b,'d) fn -> ('a -> 'b,'d) fn
val ( @-> ) : ('a,'d) t -> ('b,'d) fn -> ('a -> 'b,'d) fn
val returns : ('a,'d) t -> ('a,'d) fn
```

A signature is what `Rune.jit`, `Rune.vmap`, `Rune.remat` and `Rune.callback`
take. Each argument has its own description, so a two-array function can take
two dtypes:

```ocaml
Nx.Ptree.(tensor @-> tensor @-> returns tensor)
Nx.Ptree.(inputs_ptree @-> returns Nx.Ptree.tensor)
Nx.Ptree.(pair p.params tensor @-> returns tensor)
```

`Skeleton.equal/hash/diff` compare two values' visits, which is what a
compiled function keys on. `Nx.Ptree.rebuild s ~like (fst (flatten s like))` is
`like`.
