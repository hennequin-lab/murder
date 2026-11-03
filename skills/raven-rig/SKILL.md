---
name: raven-rig
description: "Run Raven work on CPUs, GPUs, files and remote machines with Rig in the Hennequin lab monorepo (murder). Use when a task selects or opens a device, moves an Nx value to a GPU, adds a -dev flag, writes or reads Rig buffers, submits kernels through Rig.Submission/Rig.Image, handles a lost device, or explains why an eager Nx operation on a GPU set raises. Triggers: Rig.host, Rig.open_, Rig_cuda, Rig_metal, Rig_amd_amdgpu, Rig_nv_nvidia, Nx.devices, placement, GPU, device selection, buffer, submission, Point.wait, Lost."
---

# Rig

Rig runs work on devices: the host, GPUs, files, and other machines. It is the
layer under Nx and Rune. Most lab code touches only three things: choose a
device, mint an Nx set over it, and print its name. Library code that submits
compiled work touches buffers, submissions and images.

Read `references/devices.md` for the device model and openers, and
`references/buffers-and-submissions.md` for memory, ordering and work.

## The two audiences

**Application code** selects a device and hands it to Nx and Rune. The lab's
idiom, from `packages/sofo/example/*.ml` and `packages/sofo/README.md`:

```ocaml
let device =
  let gpu =
    match Cmdargs.get_string "-dev" with
    | Some "amd" | Some "AMD" ->
      Rig.open_ (Rig_amd_amdgpu.gpu 0) |> Stdlib.Result.get_ok |> Option.some
    | Some "nv" | Some "NV" ->
      Rig.open_ (Rig_nv_nvidia.gpu 0) |> Stdlib.Result.get_ok |> Option.some
    | _ -> None
  in
  match gpu with
  | Some g -> g
  | None -> Rig.host

module D = (val Nx.devices [ device ])
let placement = D.on

let () = Stdio.printf "device: %s\n%!" (Rig.name device)
```

### Where values live

A GPU set (AMD or NVIDIA) has no eager Nx kernels at this pin: creations are
made on the host and placed, every other eager operation raises, and
`Rune.jit` supplies the compiler. Host-only sets (`Rig.host`, or a set of host
devices) compute eagerly through `nx.cpu`.

So initialise models, optimizer state and data on the host, and move a value
to the device only at the boundary of the jitted function that consumes it:

```ocaml
let placement = D.on

(* Host: build the model, the optimizer state, the data and the keys. *)
let init_key = Nx.Rng.key 0 in
let train_key = Nx.Rng.key 1 in
let params = Model.init ~key:init_key ~d_in ~d_out in
let state = Optim.init params in
let x, y = load_data () in

(* Cross to the device once, just before the first jitted call. *)
let params = Nx.Ptree.place (module Model.P) placement params in
let state = Nx.Ptree.place (module Optim.State) placement state in
let x = Nx.place placement x in
let y = Nx.place placement y in
let train_key = Nx.Rng.place placement train_key in

(* [step] is jitted. Its results lie on the device, so the recursive loop
   keeps them there and never places them again. *)
let rec train ~i (params, state) =
  if i = 0 then params
  else
    let params, state, _loss = step train_key (params, state) x y in
    train ~i:(i - 1) (params, state)
```

`Model.P` and `Optim.State` are the structures' modules (`Nx.Ptree.S`).
`Nx.place` moves one array; `Nx.Ptree.place` moves every array of a structure.

The rule has three parts:

- Keep a value on the host while host code builds or reads it. Do not place
  at initialisation.
- Place model parameters, optimizer state, inputs, targets and any auxiliary
  structure with `Nx.place` or `Nx.Ptree.place` immediately before the first
  jitted call that needs them.
- Place a key with `Nx.Rng.place D.on k` at that same boundary, not when the
  key is created. A host key draws on the host; a placed key draws where the
  jitted program computes.

Once the first jitted step returns the parameters and state, they are device
values. The recursive training loop threads them and keeps them there; moving
them back per iteration would force a copy and a wait. A host read of the loss
(`Nx.item`) does force a wait, so read it at the reporting interval, not every
step.

Only values on the jitted path need the device. Logging, schedules, host
control flow and anything read with `Nx.item` stay on the host.

**Library code** opens a device, allocates buffers, prepares a submission and
submits it. That is the rest of this skill.

## Devices

```ocaml
type t
val host : t
val open_    : ?machine:t -> 'a Rig_edge.opener -> (t, string) result
val open_io  : ?machine:t -> 'a Rig_edge.io_opener -> (t, string) result
val memory_device : string -> (t, string) result
val close : t -> unit
val name : t -> string
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
val arch : t -> string
val bus : t -> string option
val computes : t -> bool
val runs_on_host : t -> bool
val alignment : t -> int
val capability : t -> 'a Type.Id.t -> 'a option
val host_of : t -> t
val reaches : t -> t -> bool
```

An opener is the only place a driver and a device index are named. `open_`
returns `Error why` when the device cannot be opened and raises
`Invalid_argument` when misused. A device name is open at most once per
machine.

| Device | Open with |
| --- | --- |
| host cores | `Rig.host` |
| memory device (host memory, work runs before the call returns) | `Rig.memory_device "name"` |
| Apple GPU (Metal) | `Rig.open_ (Rig_metal.gpu 0)` |
| NVIDIA GPU (CUDA) | `Rig.open_ (Rig_cuda.gpu 0)` |
| NVIDIA GPU (kernel driver) | `Rig.open_ (Rig_nv_nvidia.gpu 0)` |
| NVIDIA GPU (no kernel driver, Linux) | `Rig.open_ (Rig_nv_pci.gpu 0)` |
| AMD GPU (`amdgpu`) | `Rig.open_ (Rig_amd_amdgpu.gpu 0)` |
| AMD GPU (no kernel driver, Linux) | `Rig.open_ (Rig_amd_pci.gpu 0)` |
| files (io device) | `Rig_disk.device`, then `Rig_disk.of_file path` for a buffer |
| another machine | `Rig_remote.connect ~key agents` opens a job and each machine's hosts |
| ConnectX NIC | `Rig_mlx5.open_`, its own opener |
| Broadcom NetXtreme NIC | `Rig_bnxt.open_`, its own opener |

Each driver is a separate library (`rig.metal`, `rig.cuda`, `rig.nv.nvidia`,
`rig.nv.pci`, `rig.amd.amdgpu`, `rig.amd.pci`, `rig.remote`, `rig.disk`, ...).
Add it beside `rig` in the executable's `libraries`.

`Rig.name` is for logs. `Rig.equal` is the device comparison; do not use `=`.
`Rig.runs_on_host d` is what `Nx.devices` uses to decide whether a set has
eager kernels. `Rig.reaches a b` says whether `a` can map `b`'s memory, which
is what `Nx.place` and `Buffer.borrow` depend on.

`Rig.close d` ends a device after waiting for the work submitted before the
call. It raises `Invalid_argument` for `Rig.host`; a library must leave open a
device it shares.

## Ordering: no events, no fences

Work on a GPU is asynchronous: the submit returns before the work runs. Rig
numbers the work submitted to a device with consecutive integers. A **point**
is a device and a value; the point is reached once all work up to it has
completed.

Every allocation records, per device, the point of its last write and its
last use. Rig orders work and host access by those stamps. That is why no call
takes an event or a fence:

- whatever reads memory follows every device's last write of it;
- whatever writes memory follows every use of it;
- `submit ~waits` adds points for work that shares no memory with the earlier
  work it must follow.

```ocaml
val submitted : t -> int
val signaled  : t -> int
module Point : sig
  type t
  val device : t -> t option
  val value  : t -> int
  val wait   : t -> unit
  val reached : t -> bool
  val pp : Format.formatter -> t -> unit
end
val submit : ?waits:Point.t array -> Submission.Run.t -> buffers:Buffer.t array -> Point.t
```

A `Point.wait` returns once the work up to the point has completed. A read
(`Buffer.read`, `Buffer.blit_to_bytes`) waits for the last write of its
memory. A `Buffer.copy` returns once the bytes are in the destination.

The host waits only at these points. A loop that never waits holds the values
of the steps the queues hold; a submit waits for room once a queue is full.

## Buffers

```ocaml
module Buffer : sig
  type t
  type device := t
  type kind = Device | Pinned | Mapped
  val create : ?memory:kind -> device -> int -> t
  val of_string : string -> t
  val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> t
  val of_io : device -> 'r Type.Id.t -> 'r -> access:access -> int -> t
  val view : t -> first:int -> length:int -> t
  val borrow : device -> t -> t option
  val free : ?why:string -> t -> bool
  val device : t -> device
  val length : t -> int
  val access : t -> access            (* Read | Read_write *)
  val dead : t -> string option
  val spans : t -> bool
  val overlaps : t -> t -> bool
  val read :
    t -> ('a, 'b) Bigarray.kind
    -> (('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> 'c) -> 'c
  val write :
    t -> ('a, 'b) Bigarray.kind
    -> (('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> 'c) -> 'c
  val export :
    t -> ('a, 'b) Bigarray.kind
    -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  val copy : src:t -> dst:t -> unit
  val submit_copy : ?waits:Point.t array -> src:t -> dst:t -> Point.t
  val blit_from_string : string -> int -> t -> int -> int -> unit
  val blit_to_bytes : t -> int -> bytes -> int -> int -> unit
  val fill : t -> char -> unit
  val submit_fill : ?waits:Point.t array -> t -> char -> Point.t
  val aligned : t -> at:int -> int -> bool
  val offset : t -> int
end
```

- `create d n` allocates `n` bytes of `d`'s memory. `Pinned` is host memory
  the device also addresses; `Mapped` is device memory the host also
  addresses, slowly.
- `view` is a range of another buffer's bytes in the same allocation.
  `borrow d b` is a buffer on `d` over `b`'s memory without a copy, or `None`
  if `d` cannot map it.
- `read` and `write` hand the host the memory as a bigarray, after waiting
  for the last use or write. The callback must not keep the array.
  `blit_to_bytes` copies out of memory the host does not address.
- Compare buffers with `==`. Polymorphic comparison does not look at bytes;
  `=` raises for two buffers of one device.
- A dead buffer raises `Invalid_argument` on use. `Buffer.dead` gives the
  reason.
- `Buffer.free` returns the memory early when nothing else claims it.

## Submissions and images

```ocaml
module Submission : sig
  type device := t
  type t
  type ref
  val read      : int -> at:int -> ref
  val write     : int -> at:int -> ref
  val unordered : int -> at:int -> ref
  val entry     : Image.func -> at:int -> ref

  type part
  type side = Slot of int | Buffer of Buffer.t
  val copy   : src:side -> dst:side -> part
  val launch : ?refs:ref array -> Image.func -> params:int -> part
  val fill   : ?refs:ref array -> nativeint -> params:int -> part
  val words  : Buffer.t -> part

  val make : ?keep:'a -> ?fixed:(Buffer.t * access) list -> device -> part array -> t
  val block : t -> int -> block
  val address : t -> Buffer.t -> int

  module Run : sig
    type submission := t
    type t
    val make    : submission -> t
    val take    : t -> bool
    val release : t -> unit
    val groups  : t -> block -> int -> int -> int -> unit
    val threads : t -> block -> int -> int -> int -> unit
    val shared  : t -> block -> int -> unit
    val int32   : t -> block -> int -> int -> unit
    val int64   : t -> block -> int -> int64 -> unit
    val float32 : t -> block -> int -> float -> unit
    val float64 : t -> block -> int -> float -> unit
  end
end
val submit : ?waits:Point.t array -> Submission.Run.t -> buffers:Buffer.t array -> Point.t
```

A **submission** is prepared work: parts, refs and fixed memory. A **run** is
the per-submit storage a launch reads its grid and parameters from. `submit`
runs a run once with one buffer per slot and returns its point.

- A ref makes 8 bytes of the parameters an offset into one of the submit's
  buffers (its slot), or the value by which the device refers to a function
  (`entry`). `read`, `write` and `unordered` state how the work uses the slot.
- A copy's ends are a slot or a fixed buffer. `Slot i` is the buffer the
  submit passes at index `i`.
- A launch's grid comes from `Run.groups`, `Run.threads` and `Run.shared`;
  its parameters from `Run.int32`, `int64`, `float32`, `float64` or the refs.
- A run is taken by the calling thread (`Run.take`), stored into, and
  released or submitted. A store into a run the thread has not taken raises.
- A submission is made once and submitted many times. `Run.make` is cheap;
  `Submission.make` is the expensive preparation.

```ocaml
module Image : sig
  type t
  type device := t
  type func
  val load : device -> string -> (t, string) result
  val device : t -> device
  val resolve : t -> string -> func option
  val name : func -> string
  val image : func -> t
  val params : func -> int option
  val threads : func -> int option
  val entry : func -> int
end
```

An image is a binary compiled for the device: a metallib on Apple, PTX or
cubin on NVIDIA, an ELF on AMD. `Image.resolve` finds a function by name.
Application code rarely calls `Image` directly; `Rune.jit` compiles and runs
its own programs.

## Donation and claims

A function may write its result over an argument it no longer needs. A claim
records that something reads an allocation, so a donation writes only memory
nothing else reads.

```ocaml
module Claim : sig
  type t
  type hold
  val share : Buffer.t -> unit
  val with_ : read:Buffer.t list -> donate:Buffer.t list -> (t -> 'a) -> 'a
  val exclusive : t -> Buffer.t -> bool
  val consume : t -> why:string -> Buffer.t -> Buffer.t
  val consumable : t -> Buffer.t -> bool
  val hold : Buffer.t list -> hold
  val release : hold -> unit
end
```

A donor writes over memory it holds exclusive, by the only claim on it. It
first consumes that memory, so every buffer made over it until then is dead.
`Claim.with_ ~read ~donate f` reads some buffers and gives others up for the
call. This is what `Nx.donate` and a jitted call's donation do underneath.

## Loss

```ocaml
exception Lost of t * string
val lost : t -> string option
val fail : string -> unit
val failure : unit -> string option
```

A device is lost when it can run no more work, for good: a driver fault, a
hang, a close, a fork. A use that the loss makes invalid raises `Lost (d,
why)`. Every other error of an operation raises in its call. `Rig.lost d` is
`Some why` without waiting. `Rig.fail why` fails the process and loses every
device but the host.

Handle `Lost` at the top of a program, not around every call. The lost
device itself cannot be used again; opening its name makes a new, unequal
device, and `close` runs the lost one's owed stop.

## Per-device values and memory budget

```ocaml
module Local : sig
  type device := t
  type 'a key
  val new_key : (device -> 'a) -> 'a key
  val get : 'a key -> device -> 'a
end
val budget : t -> int
val allocated : t -> int
val set_budget : t -> int -> unit
val free_cache : t -> unit
```

`Local` keeps a value per device, such as an image or a workspace. Make the
key once, at a module's top level; `get` initialises on first use and takes no
lock afterwards. `Local` is for libraries, not for application state.

`Rig.budget` is the device's memory budget; `Rig.allocated` what rig holds on
it. `set_budget` lowers or raises it; `free_cache` returns cached allocations.

## Testing

Use `Rig.memory_device "name"` for a device that runs anywhere: its memory is
the host's and its work runs before `submit` returns. The Nx device-set tests
use `Nx.devices (Nx.rigs Nx.Host.v)` to prove a computation is placement-
polymorphic without a GPU. Do not fake a GPU; skip cleanly when no device
opens.

```ocaml
let test_device_set () =
  let module D = (val Nx.devices (Nx.rigs Nx.Host.v)) in
  let on x = Nx.place D.on x in
  (* [f] is the computation under test. *)
  equal ~msg:"the result" (array float_exact) (to_array (f x)) (to_array (f (on x)))
```

## Pitfalls

- `Rig.open_` returns a result. `Result.get_ok` is fine in an example; in a
  library, report `Error why` to the caller.
- An eager Nx operation on a GPU set raises. Compile it with `Rune.jit`, or
  do that part on the host.
- Do not name a driver in a library. Take a `Rig.t` and let the caller open
  it.
- Do not use `=` on devices or buffers.
- Do not wait after every submit. Let the stamps order the work; wait at the
  edge where the host needs the result.
- Do not keep the bigarray handed to `Buffer.read` or `Buffer.write`.
- A lost device is not recoverable. Let `Lost` reach the top.
- `Local.new_key` in a function makes a new key each call and leaks the
  values. Put it at a module's top level.

## References

- `references/devices.md` — device model, openers, timelines, ordering, loss,
  budget, remote machines.
- `references/buffers-and-submissions.md` — buffers, copies, claims,
  submissions, launches, images, and the Nx/Rune boundary.
