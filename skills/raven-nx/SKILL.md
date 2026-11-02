---
name: raven-nx
description: "Write, review, and debug Nx code in the Hennequin lab monorepo (murder). Use when a task touches Nx arrays in OCaml: creating tensors, shapes and dtypes, slicing, broadcasting, reductions, linear algebra, Nx.Ptree parameter records, Nx.Rng keys, device sets and placement, donation, or checks. Triggers: Nx., 'd float32_t, Nx.Ptree, [@@deriving ptree], Nx.Rng, Nx.devices, Nx.Placement, matmul, einsum, broadcast errors, 'different device set', eager tensor code."
---

# Nx

Nx is the array layer under the Raven stack. Read this file first; the
`references/` files carry the API detail.

## House rules

These are the lab's rules, from `AGENTS.md` and the code in `packages/`.

- `open Nx` at the top of a numerical module. Do not write `Nx.sum`,
  `Nx.matmul`, `Nx.shape` when `open Nx` is already there.
- Use infix operators for maths, scoped tightly:
  `let open Infix in ...`. Do not build long prefix chains
  (`mul (add a b) (sub c d)`).
- Shapes and dtypes are explicit. Name them in a comment when they are not
  obvious from the code: `(* [horizon; n; n] *)`.
- Constants that belong to a specification are named `let`s, not inline
  literals. A one-off literal that reads clearly may stay inline.
- Do not compare against other numerical libraries in comments or docs.
  Explain the behaviour from first principles.

## The type

```ocaml
type ('v, 's, 'd) Nx.t
```

- `'v` is the OCaml value type (`float`, `int`, `Complex.t`, `bool`).
- `'s` is the storage dtype (`Nx.float32_elt`, ...).
- `'d` is the **device set** brand, a phantom type.

Named aliases hide the first two:

```ocaml
Nx.float32_t   (* = (float, Nx.float32_elt, 'd) Nx.t *)
Nx.float64_t
Nx.int32_t
Nx.bool_t
```

A value carries a shape (`int array`, at most `Nx_array.Layout.max_rank`
axes) and a placement over its set. `Nx.shape`, `Nx.dtype`, `Nx.ndim`,
`Nx.dim i`, `Nx.numel` are cheap and never read elements.

Dtype values are ordinary values: `Nx.float32`, `Nx.int64`, `Nx.bool`.
`Nx.Dtype.name (dtype x)` gives the name; `Nx.Dtype.is Nx.Dtype.Float dt`
tests the kind. A scalar literal in `+$`, `*$` and friends is `'v`, so
`x *$ 2.0` works on floats and `i +$ 1` on ints.

## Two kinds of value: host and placed

Creation without a placement suffix makes a **host** value:

```ocaml
Nx.zeros Nx.float32 [| 2; 3 |]        : Nx.host Nx.float32_t
Nx.create Nx.float64 [| 2 |] [| 1.; 2. |]
Nx.arange Nx.int32 0 n 1
Nx.scalar Nx.float32 0.3
```

The `_at` forms make a value at a placement, and the `_like` forms make one
where another value lies:

```ocaml
Nx.zeros_at placement Nx.float32 [| 2; 3 |]
Nx.zeros_like x
Nx.scalar_like x 1.0
```

Inside a function that must work on every device set, write constants beside
the operand and annotate the set, or OCaml infers `host`:

```ocaml
let shift (type d) (x : (float, 's, d) Nx.t) =
  Nx.add x (Nx.arange_f_at (Nx.beside x) (Nx.dtype x) 0. 3. 1.)
```

A host constant meeting a placed value is a **type error**, not a runtime
broadcast. `Nx.beside x` is `Placement.uncut (placement x)`: whole on every
device `x` lies on, so a value made there combines with `x`.

## Device sets and placement

```ocaml
let device = ...                      (* Rig.host or a GPU, see raven-rig *)
module D = (val Nx.devices [ device ])
let placement = D.on                  (* whole value on every device *)
```

`Nx.devices` mints a fresh brand. Two sets never meet in typed code, even
over the same devices: `Nx.devices (Nx.rigs Nx.Host.v)` is a second, distinct
host set, which the tests use to prove code is placement-polymorphic.

Placements (`Nx.Placement`):

| Function | Meaning |
| --- | --- |
| `on s` | whole value on every device of `s` |
| `device s d` | whole value on `d` alone |
| `split ~axis s` | cut `axis` into one window per device, evenly |
| `mesh m cuts` | named mesh axes, major first |
| `uncut p` | whole on each device `p` cuts |
| `equal p q`, `same p q` | same windows / same brand |
| `devices p` | the set |

`Nx.place p x` moves a value. Across sets it changes the brand; within a set
it rearranges. `Nx.placement x` reads where a value lies. Operations compute
where their operands lie and move nothing: a mismatch raises
`Invalid_argument` and the message names the `Nx.place` call to write.

On a set without kernels, only host-only sets compute eagerly (through
`nx.cpu`). A GPU set is computed by `Rune.jit` alone; eager operations on it
raise and name the placement or compiler that would compute them. This is why
the lab's GPU programs are compiled.

Read `references/devices.md` for routing rules, multi-device placement, and
the GPU boundary.

## Shape, movement, indexing

```ocaml
reshape [| 4; 3 |] x          (* view when the layout allows *)
unsqueeze ~axes:[ -1 ] v
squeeze ~axes:[ -1 ] x
transpose x                   (* all axes reversed *)
transpose ~axes:[ 0; 2; 1 ] x
broadcast_to [| 4; 3 |] row
concatenate ~axis:0 [ a; b ]
stack ~axis:0 [ a; b ]
split ~axis:0 3 x
pad [| (0, 1); (0, 0) |] 0. x
tile [| 2; 1 |] x
repeat ~axis:1 3 x
flip ~axes:[ 0 ] x
```

Indexing uses one constructor per axis, in `Nx.slice`:

```ocaml
slice [ I 0; R (0, 2); A ] x
slice [ Rs (-1, -6, -1) ] x        (* reverse a 5-long axis *)
slice [ N; A ] row                 (* add an axis *)
get [ 1; 2 ] x                     (* view of one element position *)
take ~axis:0 positions x           (* gather; positions are int64 data *)
scatter ~axis:0 positions updates x
```

| Constructor | Selects |
| --- | --- |
| `I i` | one position; the axis goes |
| `R (start, stop)` | `start .. stop-1`, negative counts from the end |
| `Rs (start, stop, step)` | strided range |
| `A` | the whole axis |
| `N` | a new axis of extent 1 |
| `L [i; j]` | written positions; a gather |
| `T t` | positions in data; a gather |
| `D (start, n)` | a window of `n` from a data start; a gather |
| `M mask` | positions where a boolean mask holds; a gather |

Written positions are checked and may be negative. Positions held in data
(`T`, `take`, `scatter`) never raise and never count from the end: out of
range reads zero and its write is dropped. That is what makes gather and
add-scatter adjoint; state a loud failure with `Nx.check` if you need one.

`reshape` and `transpose` return views when the layout allows. A view is
fine; a `donate`d operand still cannot be written in place through a view.

## Maths

Open `Nx.Infix` locally for operators:

| Operator | Function |
| --- | --- |
| `+ - * / **` | `add sub mul div pow` |
| `+$ -$ *$ /$ **$` | scalar right operand |
| `*@` | `matmul` |
| `=% <>$ <$ <=$ >$ >=$` | comparisons to a scalar |
| `= <> < <= > >=` | tensor comparisons |
| `&& \|\|` | `logical_and`, `logical_or` (both sides computed) |
| `.%{p}` / `.${idx}` | `get` / `slice` |

`Nx.max` and `Nx.min` are reductions; `Nx.maximum` and `Nx.minimum` are
elementwise binary; `maximum_s`/`minimum_s` take a scalar. The lab's ReLU is
`maximum x (zeros_like x)`.

Reductions take `~axes:int list` (not `axis`), `~keepdims`, and preserve the
dtype: `sum`, `prod`, `mean`, `max`, `min`, `var ~ddof`, `std ~ddof`,
`logsumexp`, `argmax ~axis`, `cumsum ~axis`, `softmax ~axes ~scale`,
`log_softmax`. `associative_scan` is the tree scan for an associative
elementwise `f`.

Broadcasting aligns from the right: `[2;3]` with `[3]` works, `[2;3]` with
`[2]` does not. Print shapes before an operation that surprises you.

Linear algebra: `matmul`, `dot`, `contract`, `einsum`, `rearrange`,
`matrix_transpose`, `matrix_power`, `tril`, `triu`, `diagonal`, `diag`,
`eye`, `cholesky ~triangle`, `qr ~factors`, `lu`, `svd`, `svdvals`, `eigh`,
`eigvalsh`, `det`, `solve`, `solve_triangular`, `inv`, `lstsq`.

`contract` and `einsum` take an `Nx.Pattern.t`. Write sums explicitly after
`|`; a name silently dropped from the result raises rather than returning a
plausible wrong shape:

```ocaml
let contraction = Pattern.v "b i, b j, k i j -> b k | i j"
let by_column = Pattern.v "b i j, b j -> b i | j"
```

`rearrange` is the one-operand form (split, permute, merge axes).

## Structures: `Nx.Ptree`

A model, optimizer state or cache is an ordinary OCaml record of arrays with
a `walk`. Prefer `ppx_ptree` over a hand-written `walk`:

```ocaml
type ('a, _) params = { w : 'a; b : 'a } [@@deriving ptree]

module Params = struct
  type ('a, 'd) t = ('a, 'd) params
  let walk = walk_params
end

let params = Nx.Ptree.instantiate (module Params)
```

Then `Rune.grad params f`, `Nx.Ptree.place (module Params) placement p`,
`Nx.Ptree.cast (module Params) Nx.float64 p`, `Nx.Ptree.map`,
`Nx.Ptree.map2`, `Nx.Ptree.fold`, `Nx.Ptree.dot`, `Nx.Ptree.norm`,
`Nx.Ptree.scale`, `Nx.Ptree.axpy` all work on it.

For a structure built from other structures, use `Nx.Ptree.v` with
`Nx.Ptree.Walk.structure` and `Walk.under`; `Nx.Ptree.nest` nests a module
inside another structure. `dilqr/lib/problem.ml` builds its solve inputs this
way.

Read `references/ptree.md` for walks, paths, visits and flattening.

## Random numbers

A key is the whole generator state: a draw is a pure function of its key,
dtype and shape.

```ocaml
let draw () =
  Nx.Rng.with_key (Nx.Rng.key 0) @@ fun () ->
  Nx.Rng.normal Nx.float32 [| 64; 3 |]
```

- `Nx.Rng.key seed` is a host key; equal seeds give equal keys.
- `with_key k f` runs `f` in a scope rooted at `k`; draws without `~key`
  take the scope's next key.
- `split ?n k`, `split_batch ~n k`, `fold_in k i`, `fold_in_tensor k i`
  derive independent keys. A key is spent once used: do not reuse one two
  ways.
- `Nx.Rng.place p k` moves a key; a placement that cuts the words' axis
  holds no key.
- Samplers: `normal`, `uniform`, `exponential`, `gumbel`, `randint`,
  `bernoulli`, `gamma`, `beta`, `poisson`, `binomial`, `von_mises`, `bits`,
  each with `_at`.

Inside `Rune.jit`, a draw without `~key` raises. In a scan or iterate step,
draw from `fold_in_tensor k i` with the step index; see `raven-rune`.

## Donation and checks

`Nx.donate x` gives `x` up as a handle. A consumer may write its result over
the donor's memory, and the donor dies after the consumer checks its
operands. In practice:

```ocaml
let x = Nx.set [ I i ] row (Nx.donate x)   (* writes in place *)
let x = Nx.scatter ~axis:0 p u (Nx.donate x)
```

Without `donate`, `set` and `scatter` copy. A view, a `place` borrow, or any
other value over the memory makes it shared and forces fresh memory.

`Nx.check s ok data fail` raises a data-dependent failure. On the host it
raises at the call; on another set it is deferred to the next wait on that
set, so a check never makes the host wait. `fail` receives the first failing
index and the structure at that index.

## Reading values

`Nx.to_array`, `Nx.item` and `Nx.to_bigarray` copy to the host and wait for
the set. Use them at the edge of a computation, not inside a loop or a
traced function. Under `Rune.jit`, `Nx.item` of a traced value raises.

`Nx.to_string`, `Nx.print` and `Nx.pp` are for diagnostics. `Nx.map_item f x`
applies an OCaml function per element on the host; it is not traceable and
should not appear in hot paths.

## Pitfalls

- A host literal in a set-polymorphic function makes the set `host`.
  Annotate `(type d) (x : (_, _, d) Nx.t)` and build constants with `_at
  (Nx.beside x)` or `_like`.
- Two `Nx.devices` mints are two sets. Do not expect values to mix.
- `sum` takes `~axes`, `argmax` takes `~axis`, `transpose` takes `~axes`.
- `Nx.max` is a reduction. `Nx.maximum` is elementwise.
- A negative `R` bound counts from the end and then clips; a negative `T`
  or `take` position reads zero.
- Integer and boolean arrays have no derivative. Cast to a float dtype
  before differentiating.
- Eager operations on a GPU set raise. Compile with `Rune.jit` or use the
  host.
- `Nx.donate` is not a free win: a view or a second value over the memory
  makes it shared, so the consumer allocates fresh memory instead of
  writing in place, and a dead value raises naming its consumer.

## References

- `references/devices.md` — device sets, placement, routing, multi-device,
  GPU boundary.
- `references/ptree.md` — walks, paths, `instantiate`/`nest`/`v`, derived
  maps, signatures.
- `references/api.md` — function catalogue with signatures.
