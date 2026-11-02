# Buffers, submissions and the Nx boundary

Read the Buffers, Submitting, Claims and Reclamation sections of `rig.mli`.

## Buffers

A buffer is a range of bytes in an allocation on a device. An allocation is
the memory one call made; a view is a buffer over part of another buffer's
bytes in the same allocation; a borrow is a buffer on another device over the
same allocation, without a copy.

```ocaml
module Buffer : sig
  type t
  type device := t
  type kind = Device | Pinned | Mapped

  val create : ?memory:kind -> device -> int -> t
  val of_string : string -> t
  val of_bigarray : ('a,'b,Bigarray.c_layout) Bigarray.Array1.t -> t
  val of_io : device -> 'r Type.Id.t -> 'r -> access:access -> int -> t
  val view : t -> first:int -> length:int -> t
  val borrow : device -> t -> t option
  val free : ?why:string -> t -> bool

  val device : t -> device
  val length : t -> int
  val access : t -> access
  val dead : t -> string option
  val spans : t -> bool
  val overlaps : t -> t -> bool
  val is_borrowed : t -> bool
  val aligned : t -> at:int -> int -> bool
  val offset : t -> int
end
```

- `create d n` allocates `n` bytes on `d`. `n < 0` raises. An empty buffer
  allocates nothing. A non-empty one starts on the device's alignment; a host
  buffer of at least 64 KiB and four pages starts on a page, so devices can
  borrow it.
- `of_string` and `of_bigarray` make a host buffer. `of_bigarray` does not
  copy: it keeps the array reachable, and its memory is outside the claims.
  Wrap an array once and share the buffer through `view`/`borrow`; two calls
  over the same bytes are two allocations and rig does not order work between
  them.
- `of_io` wraps a region of an io device (a file's pages, say). The region's
  memory is outside the claims.
- `view b ~first ~length` is a range of `b`'s bytes. It shares the
  allocation's stamps and claims, so work through it is work on the
  allocation.
- `borrow d b` maps `b`'s memory on `d`, or is `None` if `d` cannot map it
  (`Rig.reaches`). It shares the stamps and claims.
- `free b` tries to return the memory early. It is `false`, changing
  nothing, if `b` does not span its memory, another claim holds it, or it is
  outside the claims. Otherwise it consumes the memory with the reason, and
  every buffer over it is dead.

Compare buffers with `==`. Polymorphic comparison does not look at bytes:
`compare` terminates with an order that says nothing about contents, and `=`
raises for two buffers of one device and finds buffers of different devices
unequal. `Hashtbl.hash` depends on addresses and state, which can change
while a buffer lives.

`Buffer.dead b` is `Some why` for a dead buffer. A call that takes a dead
buffer raises `Invalid_argument`.

## Host access

```ocaml
val wait : t -> access -> unit
val reached : t -> access -> bool
val read :
  t -> ('a,'b) Bigarray.kind
  -> (('a,'b,Bigarray.c_layout) Bigarray.Array1.t -> 'c) -> 'c
val write :
  t -> ('a,'b) Bigarray.kind
  -> (('a,'b,Bigarray.c_layout) Bigarray.Array1.t -> 'c) -> 'c
val export :
  t -> ('a,'b) Bigarray.kind
  -> ('a,'b,Bigarray.c_layout) Bigarray.Array1.t
```

- `Buffer.wait b Read` waits for every device's last write of the memory.
  `wait b Read_write` waits for every use.
- `read b k f` calls `f` on the memory as a bigarray, once every last write
  is done. `f` must not write through the array and must not keep it after it
  returns.
- `write b k f` calls `f` on the memory once every use is done. Work
  submitted after `write` returns sees the writes. `b` must admit writes.
- `export b k` hands out the array for good. From then on the memory is
  outside the claims, the caller orders access, and the array reads without
  a loss check.

`Buffer.read` and `write` need memory the host addresses. For memory it does
not address, use `blit_to_bytes`/`blit_from_string` or `copy`.

## Copies and fills

```ocaml
val copy : src:t -> dst:t -> unit
val submit_copy : ?waits:Point.t array -> src:t -> dst:t -> Point.t
val blit_from_string : string -> int -> t -> int -> int -> unit
val blit_to_bytes : t -> int -> bytes -> int -> int -> unit
val fill : t -> char -> unit
val submit_fill : ?waits:Point.t array -> t -> char -> Point.t
```

`copy` copies and returns once the bytes are in the destination; it is
`submit_copy` plus a wait. The two buffers must have one length and share no
byte, and `dst` must admit writes. A copy that the host makes returns a point
that is already reached; a copy on a device's copy engine is asynchronous.

`blit_to_bytes` copies out of memory the host does not address.

## Claims and donation

A donation writes a result over memory an argument no longer needs. A claim
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
  type hold
  val hold : Buffer.t list -> hold
  val release : hold -> unit
end
```

`Claim.with_ ~read ~donate f` takes a read claim on every buffer in both
lists. A buffer in `donate` whose memory is not otherwise read becomes
exclusive; `f` may then donate it. `consume c ~why b` marks the memory
consumed, so every buffer over it is dead, and returns `b`. `share b` puts
the memory outside the claims for good, which a library does before handing
out a value its own claims do not track.

The rules matter to library authors, not to application code: `Nx.donate`
and a jitted call's donation already do the right thing. If you write a
function that writes over an input, take a claim first, and consume before
you write.

## Submissions

A submission is prepared work on one device: parts, refs and fixed memory. A
run is the per-submit storage a launch or fill reads its grid and parameters
from.

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

### Refs

A ref places a value in 8 parameter bytes at `at` when the work runs. `at`
must be a multiple of 8 and its 8 bytes must lie among the part's parameters;
two refs of a part must not share an `at`.

- `read slot ~at` — the work reads the buffer at that slot.
- `write slot ~at` — the work reads and writes it; the buffer must admit
  writes.
- `unordered slot ~at` — the work reads and writes it at any time, ordered
  only by device order.
- `entry fn ~at` — the 8 bytes hold the value by which the device refers to
  `fn`. The submission keeps `fn` reachable.

The 8 bytes of a buffer ref hold a byte offset into the slot's buffer; when
the work runs they hold that offset plus the buffer's address as the device
addresses it. Through a buffer ref the work may address only that buffer's
bytes, and write them only through a `write` or an `unordered` ref. Rig
checks neither rule.

### Parts

- `copy ~src ~dst` copies one buffer into another. An end is `Slot i` (the
  buffer the submit passes at index `i`) or `Buffer b` (a fixed buffer).
- `launch ?refs fn ~params` launches an image function with `params` bytes of
  parameters.
- `fill ?refs value ~params` fills memory with a 32-bit value.
- `words b` is a part whose effect rig cannot see. It follows every earlier
  part, and later parts on its queue follow it.

`Submission.make ~keep ~fixed d parts` prepares the parts once. `~keep` is a
value the work uses beyond memory (a driver object); the submission keeps it
reachable. `~fixed` is the submission's fixed memory. `make` reads the parts
once and no buffer bytes, so a caller may write `address`es into them
afterwards. Its slots are `0` to `n - 1`, where `n` is one more than the
highest slot named.

### Runs and blocks

`Run.make s` makes a run of `s`, taken by the calling thread. `Run.take r`
reserves it; `Run.release r` ends the reservation keeping the bytes. A store
into a run the thread has not taken raises.

`Run.groups r b x y z` sets the grid, `Run.threads r b x y z` the threads per
group, `Run.shared r b n` the dynamic shared memory. `Run.int32`, `int64`,
`float32`, `float64` store parameters. Bytes a run never stores read 0 and
keep their last store across submits.

A library that submits from several domains keeps a few runs beside each
cached submission: take the first run whose `take` is true, otherwise make
one and add it, and release it if the call raises before its submit.

### Submitting

`submit ~waits r ~buffers` runs the work of `r`'s submission once and returns
its point. `buffers` holds one buffer per slot. The work follows the points of
`~waits` besides those its memory orders.

`submit` reads each element of `buffers` once; a store into `r` after it
returns changes nothing of the work. It uses `r` until it returns or raises,
and frees it then. It blocks while the device's queues have no room, while it
waits for a point on the host, and where the driver blocks.

Raises `Invalid_argument` for an invalid submission or submit (dead buffer,
wrong slot count, a write through a `Read` buffer, a copy whose ends differ
in length, a ref into handle-only memory, ...), `Out_of_memory` if the queues
must grow and the device has no memory, and `Lost`.

### Example: a copy on a memory device

```ocaml
open Rig

let () =
  let device = Result.get_ok (memory_device "example") in
  let src = Buffer.create device 4 in
  let dst = Buffer.create device 4 in
  Buffer.write src Bigarray.int32 (fun a -> a.{0} <- 42l);
  let work =
    Submission.(make device [| copy ~src:(Buffer src) ~dst:(Buffer dst) |])
  in
  let run = Submission.Run.make work in
  let done_ = submit run ~buffers:[||] in
  Point.wait done_;
  Buffer.read dst Bigarray.int32 (fun a -> Printf.printf "copied %ld\n" a.{0})
```

## Images

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

`load d binary` loads a binary in the device's format: a metallib on Apple,
PTX or cubin on NVIDIA, a code object on AMD, a relocatable ELF object for
the host. Each call loads the binary again, so each image's module variables
are its own. `resolve i "name"` finds a function and makes it ready to
launch. `params fn` is the number of parameter bytes it reads; `threads fn`
is the most threads a group may have.

A launch's `params` must be at least `Image.params fn`. `Run.threads`
refuses more than `Image.threads`.

## The Nx and Rune boundary

An Nx value is an `Nx_array` over a `Rig.Buffer`:

```ocaml
Nx_array.buffer : ('v,'s) Nx_array.t -> Rig.Buffer.t
Nx_array.layout : ('v,'s) Nx_array.t -> Nx_array.Layout.t
Nx_array.dtype  : ('v,'s) Nx_array.t -> ('v,'s) Nx.Dtype.t
Nx.Repr         (* crossing between Nx values and Nx_array, borrow/copy *)
```

Application code should not need these. What matters:

- `Nx.devices [d]` mints a set. A set whose devices all run on the host
  computes eagerly through `nx.cpu`. Any other set has no eager kernels:
  creations are made on the host and placed, and every other eager operation
  raises, naming the placement or the compiler that would compute it.
- `Rune.jit` compiles through tolk and runs on the device, so a GPU program
  is a jitted program. Its results lie where eager computation would place
  them.
- `Nx.place` moves a value. Across sets it changes the brand; onto the host
  it waits on the source set; elsewhere it returns as `Buffer.submit_copy`
  returns. Each device holds its window over the source's memory where the
  host addresses both, otherwise a copy.
- `Nx.donate` and a jitted call's donation use rig's claims underneath. The
  result may take the donated argument's memory when dtype, element count and
  placement match, no check reads the argument, and the result reads it at
  each element's own index. A donated argument whose memory another value
  shares, or that is a view off 16 bytes or out of C order, lends nothing.
- `Nx.check` on a non-host set is computed with the program and raises at the
  next wait on that set. Add a `Nx.sync` or a read at the end of a program
  whose checks you need to see.
- A device lost while it runs a value's work raises `Rig.Lost` at the next
  use of the memory that work wrote: an operation or submit that reads it, a
  wait, or a read.

## Host code and kernels

`Rig.host` runs work the process links, such as a function in a relocatable
ELF object (`Image.load Rig.host`). This is the path for a host kernel
library: `nx.cpu` is one. `Rig_edge` is the contract a driver implements;
read it only if you are writing a driver, not if you are using one.
