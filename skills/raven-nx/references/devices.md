# Device sets, placement and routing

Read the Device sets section of `nx.mli` alongside this file. The monorepo
depends on this machinery more than on any other part of Nx.

## Why the set is a type parameter

A value is `('v, 's, 'd) Nx.t`. `'d` is a brand minted by `Nx.devices`. Values
of two brands never meet in typed code, so a function that is polymorphic in
`'d` is placement-polymorphic by construction, and a function that hard-codes
`Nx.host` fails to typecheck when applied to a GPU value.

`Nx.devices` is generative:

```ocaml
let module A = (val Nx.devices [ Rig.host ]) in
let module B = (val Nx.devices [ Rig.host ]) in
(* A.d and B.d are distinct. A value of A.d does not combine with one of B.d. *)
```

`Nx.Placement.same` and `Nx.Placement.equal` compare placements across
brands; `same` returns a type equality when the brands name one set.

## Minting a set

```ocaml
val devices : ?kernels:(module Nx_kernel.S) -> Rig.t list -> (module Devices)
```

- The list must be non-empty and repeat no device (`Rig.equal`).
- Without `~kernels`, a set whose every device runs on the host is computed
  by `nx.cpu`. Any other set is computed by no library: creations are made on
  the host and placed, and every other operation raises where it would make
  elements, naming the `place` onto a set with kernels or the compiler that
  would compute it. `Rune.jit` supplies that compiler.
- `~kernels` must compute on every device of the set. Raven at this pin ships
  `nx.cpu` (library `nx.cpu`, module `Nx_cpu`) for host memory; there is no
  GPU kernel library, which is why the lab's GPU code is always compiled.

`module type Devices` is:

```ocaml
type d
val v : d devices
val on : d Placement.t
val split : axis:int -> d Placement.t
```

`Nx.Host : Devices with type d = host` is the process host.

```ocaml
module D = (val Nx.devices [ device ])
module P = struct
  type (_, 'd) t = 'd float32_t [@@deriving ptree]
end
let placement = D.on
let params = Nx.Ptree.place (module P) placement params
```

This is the idiom in `packages/sofo/example/*.ml` and `packages/sofo/README.md`.

## Placements

```ocaml
val on : 'd devices -> 'd t
val device : 'd devices -> Rig.t -> 'd t
val split : axis:int -> 'd devices -> 'd t
val mesh : 'd Mesh.t -> (int * string list) list -> 'd t
val uncut : 'd t -> 'd t
val devices : 'd t -> 'd devices
val equal : 'd t -> 'e t -> bool
val same : 'd t -> 'e t -> ('d, 'e) Type.eq option
val pp : Format.formatter -> 'd t -> unit
```

- `on s` holds the whole value on every device: every device computes the
  same bytes. This is the default for small parameters and replicated state.
- `device s d` holds the whole value on one device.
- `split ~axis s` cuts one axis into windows, one per device in order,
  evenly: the first `n mod k` windows get `n / k + 1`.
- `mesh m cuts` lays the devices over named axes, major first. An axis cut
  over several mesh axes is cut over the product of their extents, numbered
  major first.
- `uncut p` is `p` with every axis whole, each mesh axis holding copies.
  `Nx.beside x` is `uncut (placement x)`.
- `with_leading_axis` / `without_leading_axis` are for interpreters that map
  over a new leading axis (what `Rune.vmap` does). Not for application code.

`Nx.Placement.pp` prints a placement as `CPU`, `on [CUDA:0; CUDA:1]`,
`split ~axis:0 [CUDA:0; CUDA:1]`, or a mesh's extents and cuts.

## Creating at a placement

Every host creation has an `_at` form, and every creation has a `_like` form:

```ocaml
zeros_at   : 'd Placement.t -> ('v,'s) dtype -> int array -> ('v,'s,'d) t
full_at    : 'd Placement.t -> ('v,'s) dtype -> int array -> 'v -> ...
arange_at  : 'd Placement.t -> ('v,'s) dtype -> int -> int -> int -> ...
eye_at     : ?m:int -> ?k:int -> 'd Placement.t -> ('v,'s) dtype -> int -> ...
normal_at  : 'd Placement.t -> ?key:'d key -> (float,'s) dtype -> int array -> ...
zeros_like : ('v,'s,'d) t -> ('v,'s,'d) t
scalar_like: ('v,'s,'d) t -> 'v -> ('v,'s,'d) t
```

`scalar_at` places at `uncut p`: a 0-d value has no axis to cut.

An `_at` form raises `Invalid_argument` where the placement cuts an axis the
shape lacks. A `_like` form makes the value where its argument lies, or whole
on each of its devices where the shape has no axis the placement cuts.

A function meant for every set writes its constants beside its operands:

```ocaml
let shift (type d) (x : (float, 's, d) Nx.t) =
  Nx.add x (Nx.arange_f_at (Nx.beside x) (Nx.dtype x) 0. 3. 1.)
```

Without the annotation OCaml infers `x`'s set as `host`, and the error appears
at the application, not at the constant.

## Moving values

```ocaml
val place : 'e Placement.t -> ('v,'s,'d) t -> ('v,'s,'e) t
val placement : ('v,'s,'d) t -> 'd Placement.t
val beside : ('v,'s,'d) t -> 'd Placement.t
```

`place p x` gives `x`'s elements at `p`. Each device of `p` holds its window
over `x`'s memory where `x` already lies there, or where the host addresses
both; otherwise it copies the window. Across sets it changes the brand.
Placing onto `Nx.Host` from another set waits on `x`'s set; elsewhere it
returns as `Rig.Buffer.submit_copy` returns.

Raises:

- `Invalid_argument` if `p` cuts an axis `x` does not have;
- `Rig.Lost` for a lost device;
- a failed check, as a wait does.

`beside x` is where a constant of another shape combines with `x`. Note the
warning in `nx.mli`: a value at `placement x` (not `beside x`) is cut along its
own axes, which need not line up with `x`'s. For `x` of shape `[4; 3]` split
along axis 0, `arange_at (placement x) ...` is split along its only axis,
which `add x` reads along axis 1, and raises. Use `beside`.

## Routing: what an operation does with placements

An operation computes where its operands lie and moves none of them.

- The result lies at the union of the operands' cuts, joined in operand
  order. Where that order cannot read an operand where it lies, a gather's or
  scatter's positions go last.
- A cut of an axis that becomes a result axis cuts that axis over the same
  devices.
- A cut of an axis read whole cuts nothing: an operand whole on every device
  is read by a view of each device's window.
- A reduction or contraction along a cut axis computes by partials where
  every operand holding the axis is cut alike, and cuts nothing in the result.
- A scan keeps the cut along its axis. A scatter writes the updates that land
  in each device's window.

A refusal raises `Invalid_argument` naming the function, the operand (numbered
from 0 in `Prim.t` order), where it lies, where it is read, and the `place`
call that fixes it:

```
Nx.matmul: operand 1 lies at split ~axis:1 [CUDA:0; CUDA:1] and is read at
split ~axis:0 [CUDA:0; CUDA:1]; Nx.place (Placement.split ~axis:0 s) y
```

Read the message literally: it names the operand, the two placements and the
call to write. Most "different device set" or "cannot read" errors are fixed
by one `Nx.place` at the boundary, not by moving every operand.

A gather whose positions cannot be read where they lie moves only the
positions, 8 bytes each, to the devices that read them. No other operand
moves.

## Multi-device patterns

Data-parallel batch:

```ocaml
module D = (val Nx.devices [ gpu0; gpu1 ])
let p = D.split ~axis:0                    (* cut the batch axis *)
let x = Nx.place p x
let y = Nx.sum ~axes:[ 1 ] x               (* each device sums its window *)
let total = Nx.sum y                       (* partials close in the result *)
```

Model parallel by mesh:

```ocaml
let m = Nx.Mesh.v D.v [ "row", 2; "col", 2 ] in
let p = Nx.Placement.mesh m [ 0, [ "row" ]; 1, [ "col" ] ] in
```

A reduction over a cut axis produces a result cut nothing: the devices
compute partials and the operation closes them. A broadcast's cotangent is
summed over repeated axes first, so the gradient of a bias added to a cut
batch moves no activation (see `rune.mli`, "Placement" under tangents).

## Waiting

A set's devices compute while the caller goes on. The host waits only at a
wait: `Nx.sync`, a read (`item`, `to_array`, `to_bigarray`), or `Nx.place`
onto `Nx.Host`. A loop that never waits holds the values of the steps its
devices' queues hold; a submit waits for room once a queue is full.

A check on a non-host set is computed where its data lies and raises at the
next wait on that set, on the domain that issued it. A check that no wait
follows never raises.

## The GPU boundary

At this pin:

- `Nx.devices [ gpu ]` has no kernels. Creations are made on the host and
  placed; every other eager operation raises.
- `Rune.jit` compiles through tolk and runs on the device. A compiled call's
  results lie where eager computation would place them.
- `Rune.jit` traces the operations, so a program on a GPU set can create
  constants with `_at` and operate on them.
- A value `f` captures on another device, which `f` places where the program
  computes, is copied there once per program; the program holds the copy.

So: eager CPU for host logic, compiled for GPU. The lab's examples select the
device at startup and compile the numerical kernel; see `raven-rig`.
