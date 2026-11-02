# Rig devices

Read the preamble and the Devices, Timelines, Order, Domains, Loss and
Reclamation sections of `rig.mli`. This file is the practical summary.

## What a device is

A device holds memory and, unless it is an io device, runs work on it.

- `Rig.host` is the process: its memory is the process's memory and its work
  is the process's own code.
- A driver's device, such as a GPU, runs work its driver places on the
  device's queues. A program opens it with an opener.
- An io device, such as a file, holds memory the host reads and writes through
  its library's calls.
- Another machine's devices open through its host, with
  `Rig_remote.connect`.

A machine has at most one open device per name. Opening a name again while it
is open returns the open device. Opening a lost device's name runs its owed
stop first, then makes a new device, unequal to the lost one, with a timeline
of its own.

A device holds one of the process's 65,535 slots while it is open, while
memory of it or a value its submissions keep remains to return, and while it
is lost and has not stopped. A device that never stops, the process's first
failure, and a device whose timeline passed `2^46` hold their slot for the
rest of the process.

## Openers

`Rig_edge.opener` is the driver's description of a device. The opener is the
only place a driver and an index are named.

| Device | Opener | Library |
| --- | --- | --- |
| host cores | `Rig.host` (no opener) | `rig` |
| memory device | `Rig.memory_device "name"` | `rig` |
| Apple GPU | `Rig_metal.gpu 0` | `rig.metal` |
| NVIDIA, CUDA | `Rig_cuda.gpu 0` | `rig.cuda` |
| NVIDIA, kernel driver | `Rig_nv_nvidia.gpu 0` | `rig.nv.nvidia` |
| NVIDIA, no kernel driver | `Rig_nv_pci.gpu 0` | `rig.nv.pci` |
| AMD, `amdgpu` | `Rig_amd_amdgpu.gpu 0` | `rig.amd.amdgpu` |
| AMD, no kernel driver | `Rig_amd_pci.gpu 0` | `rig.amd.pci` |

An opener is passed to `Rig.open_`. Two device families do not use one:

| Device | How it opens | Library |
| --- | --- | --- |
| files (io device) | `Rig_disk.device`, then `Rig_disk.of_file path` for a buffer | `rig.disk` |
| another machine | `Rig_remote.connect ~key agents` opens a job and each machine's hosts | `rig.remote` |
| ConnectX NIC | `Rig_mlx5.open_`, its own opener | `rig.mlx5` |
| Broadcom NetXtreme NIC | `Rig_bnxt.open_`, its own opener | `rig.bnxt` |

```ocaml
val open_   : ?machine:t -> 'a Rig_edge.opener -> (t, string) result
val open_io : ?machine:t -> 'a Rig_edge.io_opener -> (t, string) result
```

`Error why` means the device cannot be opened; `why` names the device and the
driver's reason. `Invalid_argument` means the call is misused. `open_` holds
the name's lock while the opener runs, so opens of one name serialise and
opens of other names do not.

`Rig.memory_device name` is the test device: its memory is the host's and its
work runs before `submit` returns. Use it for examples and tests that must run
on any machine.

## Properties

```ocaml
val name        : t -> string
val equal       : t -> t -> bool
val pp          : Format.formatter -> t -> unit
val arch        : t -> string
val bus         : t -> string option
val computes    : t -> bool
val runs_on_host: t -> bool
val alignment   : t -> int
val capability  : t -> 'a Type.Id.t -> 'a option
val host_of     : t -> t
val reaches     : t -> t -> bool
val close       : t -> unit
```

- `name` is for logs. Do not parse it.
- `equal` is the comparison. Do not use `=`: a device is an abstract value.
- `computes d` is false for an io device.
- `runs_on_host d` is what `Nx.devices` uses to decide whether a set has
  eager kernels. A host-backed set computes through `nx.cpu`; a GPU set needs
  `Rune.jit`.
- `alignment d` is the byte alignment of an allocation on `d`.
- `capability d k` is a driver-specific capability under a type key.
- `host_of d` is the host of `d`'s machine.
- `reaches a b` says whether `a` can map `b`'s memory. `Nx.place` and
  `Buffer.borrow` depend on it.
- `close d` waits for the work submitted before the call, loses the device
  with the reason `"closed"`, and returns once the driver's stop has
  returned. It raises `Invalid_argument` for `Rig.host`. A library must
  leave open a device it shares; the code that owns the process's use of a
  device closes it.

## Timelines and points

Work on a GPU is asynchronous. Rig numbers the work submitted to a device with
consecutive integers, in the order the device accepts it. These are the
device's **timeline** values.

```ocaml
val submitted : t -> int      (* last value assigned *)
val signaled  : t -> int      (* last value the device reports reached *)

module Point : sig
  type t [@@immediate]
  val device : t -> device option
  val value  : t -> int
  val wait   : t -> unit
  val reached: t -> bool
  val pp     : Format.formatter -> t -> unit
end
```

A value is **reached** once every value up to it has completed. A point is a
device and a value. `Point.wait` commits the device's work up to that value,
then blocks until it is reached. It raises `Lost` if the device was lost
before reaching it. `Point.reached` answers without blocking and commits as
`wait` does, so a spin loop on it ends once the work runs.

A point is an immediate value: storing one allocates nothing. `Point.device`
returns `None` once the device has stopped and a later device took its slot.

A driver may hold submitted work until it is committed. Every wait commits
the work it waits for; a driver also commits on its own at least once every
`k` values. A program never commits by hand. `signaled` commits nothing, so
the last values submitted may stay unreported until a wait.

## Ordering

Rig orders work by the memory it uses, not by events or fences. Each
allocation records, per device, the point of its last write and the point of
its last use: its **stamps**. The rules:

1. **Device order.** On one device, the work of value `v` follows the work of
   every value below `v`, on every queue.
2. **Parts.** Within a submission, a part follows each earlier part that may
   share memory with it, where one of the two writes. Parts that need no
   order may run concurrently.
3. **Memory.** Work that reads an allocation follows every device's last
   write of it. Work that writes it follows every use of it, by any device.
4. **Waits.** `submit ~waits` makes the work follow the points given.
5. **Host access.** Host code after `Point.wait p`, or after `reached p`
   answers true, follows `p`'s work. `Buffer.read` follows every last write;
   `Buffer.write` follows every use. Copies and fills order the same way.
6. **Host writes.** Work submitted after a host write to memory the device
   addresses sees the write.

Work through an `Submission.unordered` ref is the exception to rule 3: it
follows only device order, and nothing later follows it. Order it yourself
with atomics on flags.

Consequences for application code:

- You do not need to wait after a submit before another submit that reads its
  result: the stamps order them.
- You do need `~waits` when two pieces of work on different devices share no
  memory but must be ordered (for example, a signal that one stage finished
  in a separate flag buffer would share memory, so it would be ordered; a
  pure ordering dependency does not).
- An access follows the stamps that exist when it reads them. If two domains
  submit work on two devices over one allocation and either writes it, order
  the submits yourself.

## Domains

Calls may be made from any domain, concurrently. While a function waits it
holds no lock of rig's, except:

- `open_` holds its name's lock while the opener runs;
- while a driver's `room`, `submit` or `commit` blocks, where the driver
  declares `may_block`, other submits and commits on that device wait.

Host access to one buffer from several domains needs your own
synchronisation, as for any shared bigarray.

## Loss

A device is lost when:

- its driver reports a fault, or fails to take or commit work;
- another exception escapes its driver while it changes memory or code;
- its progress stands still past its hang bound;
- its queue waits for a value a lost device never reached;
- `close` ends it, `fail` fails the process, or a child of `fork` inherits a
  driver's device.

A loss is final. It spreads only along queue waits for values the lost device
never reached.

```ocaml
exception Lost of t * string
val lost    : t -> string option
val fail    : string -> unit
val failure : unit -> string option
```

`Lost (d, why)` prints as `"NAME lost: why"`. `why` is the driver's message,
an escaped exception's constructor and arguments, `"no progress for n ms"`,
`"NAME lost"`, `"closed"`, `"forked"`, `"out of memory"`, or what `fail` was
given.

After a loss:

- every use of the lost device, of its memory through any buffer, and of a
  borrow on it raises `Lost`;
- every use of other memory that depends on a point the lost device never
  reached raises `Lost`, because that work may never have run;
- memory the lost device only read raises it only for uses that depend on
  every stamp, and only until the device has stopped.

An array from `Buffer.export` made before the loss reads and writes with no
loss check, and its bytes may be the old ones, the lost work's, or a mix.

`Rig.fail why` fails the process: every device but the host that is not lost
is lost with `why`, and every later open returns `Error why`.

Handle `Lost` at the top of a program. A lost device is not recoverable;
`close` runs its owed stop.

## Reclamation and budget

Memory returns on its own once collected:

1. **Collection.** Once every buffer and exported array over the allocation
   is unreachable, the OCaml collector collects it.
2. **Waiting.** It waits until every stamp point is reached or dropped.
3. **Reuse.** It enters the device's cache, which `Buffer.create` reuses with
   empty stamps.
4. **Release.** Cached memory returns to the driver when the cache is freed,
   the budget is lowered, or an allocation needs the room.

```ocaml
val budget     : t -> int
val allocated  : t -> int
val set_budget : t -> int -> unit
val free_cache : t -> unit
```

`budget` starts at `max_int` for the host and at the driver's budget for a
GPU. `allocated` is what the device holds now. `set_budget` lowers or raises
it; `free_cache` returns cached memory.

A buffer kind changes what counts where:

- `Device` is the device's own memory, charged to its budget.
- `Pinned` is host memory both address, charged to the host's budget, except
  on a device whose memory the host addresses, where it is the device's.
- `Mapped` is device memory the host also addresses through a write-combined
  window, charged to the device's budget; it falls back to `Pinned` where
  there is no window or no room.

An allocation larger than the budget raises `Out_of_memory` immediately. A
program that keeps every buffer reachable and frees none runs out of memory
however large the budget.

## Other machines

`Rig_remote.connect ~key agents` starts a job with an agent at each host and
port, and opens each machine's host. Its devices take the same calls as this
machine's. A submit's work on a remote device follows the same ordering
rules; a point of a remote device may be waited on locally. `Rig_remote` is a
library of its own; see `rig/doc/07-machines.md` in the Raven tree for the
protocol.
