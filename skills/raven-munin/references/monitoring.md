# Monitoring the machine and its devices

A run records its machine beside its metrics: CPU busy, memory, disk rates,
GPU memory and busy time. munin samples readings on a schedule into the run's
log, at the session's latest step, so a reader aligns the machine with the
training.

## Readings, levels and rates

A **reading** is a function `unit -> (string * float) list`; its keys follow
the metric key rules, and in one run a key is a reading's or a metric's,
never both.

- A **level** is a value at the call: memory in use, temperature.
- A **rate** is a counter's change per second over the window since the
  reading's previous call, such as bytes read per second. The window's length
  is logged under a key of the reading's own.

The first call of a reading that gives rates starts the window and returns
nothing, so its first samples arrive one period after `sample`. A reading
keeps its own counters: give it to one sampler only, and never call it from
another thread.

`munin.sys` makes readings of the machine and of this process. munin itself
reads no device: the program builds device readings from its own library,
which in this lab is rig.

## The machine

```ocaml
let host = Munin_sys.host ()
let pairs = host ()   (* [] on the first call: rates have no window yet *)
```

| Key | Kind | Meaning |
| --- | --- | --- |
| `sys/cpu_busy_pct` | rate | percent of all CPUs' time not idle and not waiting for I/O |
| `sys/load_per_core` | level | one-minute load average over logical CPUs; absent on Windows |
| `sys/mem_used_bytes` | level | total memory less available memory |
| `sys/mem_total_bytes` | level | the machine's memory |
| `sys/swap_used_bytes` | level | swap in use |
| `sys/disk_read_bytes_per_s` | rate | summed over physical disks |
| `sys/disk_write_bytes_per_s` | rate | summed over physical disks |
| `sys/disk_busy_pct` | rate | mean over disks of percent of the window with requests in flight; Linux only |
| `sys/interval_s` | rate window | seconds since the previous call |

A key the system cannot read is left out of the pairs: no load average on
Windows, no disk busy percent elsewhere. `host ()` raises `Sys_error` only
when it can read nothing.

## This process

```ocaml
let process = Munin_sys.process ()
```

| Key | Kind | Meaning |
| --- | --- | --- |
| `sys/proc_cpu_cores` | rate | CPU seconds per second; `2.` is two cores busy |
| `sys/proc_rss_bytes` | level | resident memory |
| `sys/proc_interval_s` | rate window | seconds since the previous call |

## Devices

munin fixes the keys, and the program fills what its device library reports:

```ocaml
type device = {
  allocated : int option;     (* dev/<name>/allocated_bytes *)
  budget : int option;        (* dev/<name>/budget_bytes *)
  memory : int option;        (* dev/<name>/memory_bytes *)
  used : int option;          (* dev/<name>/used_bytes *)
  busy : float option;        (* dev/<name>/busy_pct *)
  activity : float option;    (* dev/<name>/activity_pct *)
  temperature : float option; (* dev/<name>/temperature_c *)
  power : float option;       (* dev/<name>/power_w *)
}

val unknown : device
val device : string -> (unit -> device) -> unit -> (string * float) list
```

- `allocated` is what this process holds on the device, its cache included.
- `budget` is what this process may hold before the runtime reclaims memory.
  A device without a bound reports `max_int`, which a reading records as no
  budget.
- `memory` is the device's memory from which its driver allocates, for every
  process; `used` is `memory` less what the driver would still give. `used`
  counts this process's memory, and errs where the driver's free figure errs.
- `busy` is the percent of the window the compute engine ran work of any
  process; a small kernel through the whole window reads 100.
- `activity` is the firmware's own measure across compute units, a different
  quantity from `busy`.
- `temperature` is the edge sensor in degrees Celsius; `power` is watts.
  rig reads neither, so they stay `None`.

A field that is `None` gives no pair. Every vendor's device gives its
quantities under the same keys. The name is one segment: non-empty UTF-8,
with no `/`, no control character and no leading `.`.

## Glue over rig

A program that computes on a set of rig devices samples each device's
allocation and budget, and its driver's memory where the driver reports it:

```ocaml
let every = 15.

let held d () : Munin_sys.device =
  match (Rig.allocated d, Rig.budget d) with
  | exception Rig.Lost (_, why) -> raise (Sys_error why)
  | allocated, budget ->
      {
        Munin_sys.unknown with
        allocated = Some allocated;
        budget = (if budget = max_int then None else Some budget);
      }

let memory d () : Munin_sys.device =
  match Rig.capacity d with
  | exception Rig.Lost (_, why) -> raise (Sys_error why)
  | None -> held d ()
  | Some c ->
      {
        (held d ()) with
        memory = Some c.total;
        used = Some (c.total - c.free);
      }

let reading d = Munin_sys.device (Rig.name d) (memory d)

let monitor s devices =
  Session.sample s ~every
    (Munin_sys.host () :: Munin_sys.process ()
    :: List.map reading (Nx.Devices.to_list devices))
```

- `Rig.capacity d` is `None` where the driver reports no free memory (host,
  software devices, Metal), and the glue falls back to `held`.
- On NVIDIA's kernel driver, add the GPU's busy time by its index:

```ocaml
let nvidia i d () : Munin_sys.device =
  match Rig_nv_nvidia.usage ~window:every i with
  | Error why -> raise (Sys_error why)
  | Ok u -> { (memory d ()) with busy = u.busy }
```

`Rig_nv_nvidia.usage` keeps the process's client of the driver, and the GPU's
device file open until the process exits. Read it only on an NVIDIA machine.

## Sampling a run

```ocaml
Session.within
  (Session.start ~store ~experiment:"lorenz" ~steps:total_steps ())
@@ fun s ->
Session.sample s ~every:15.
  (Munin_sys.host () :: Munin_sys.process ()
  :: List.map reading (Nx.Devices.to_list devices));
(* training *)
```

- `Session.sample` calls each reading now, in the caller's thread, then on a
  thread of its own every `every` seconds until the session finishes.
- Each call's pairs become one `reading` record, at the call's time and the
  session's latest step (the last `metrics` or `resume` record; 0 before
  any). That step is how a reader aligns the machine with the training.
- `finish` stops the samplers and waits for a reading in progress.
- A reading that raises `Sys_error` gives nothing that time and the sampler
  goes on; any other exception stops the sampler, and `finish` raises it
  after writing the run's end.
- A reading that gives an invalid key, a key twice, or a metric's key stops
  the sampler. `sys/` and `dev/` are reserved for readings.
- `every` must be positive and finite. Fifteen seconds is a good default for
  a long run; one second makes a log that is mostly readings.

The history keeps readings as it keeps metrics:

```ocaml
let h = History.read run in
History.readings h;                      (* the reading keys, sorted *)
History.stats h "sys/proc_rss_bytes";    (* peak RSS as sampled *)
History.series h "sys/cpu_busy_pct";     (* against the run's steps *)
```

Samples of readings are never orphans: the machine's state at a time happened
whatever a later resume did to the steps. A view over the run's steps drops
the readings abandoned by a resume; `History.surviving h k` gives that view,
and the dashboard uses it.

## Tests and CI

- Do not fake a device, and do not require one. A test that reads the machine
  skips cleanly when the key or the device is absent.
- A test of logging uses a store at a temporary root and its own reading, a
  plain function returning fixed pairs, not `Munin_sys`.
- A test does not depend on rates or timings: a first call gives nothing, and
  a window's length comes from the clock.
- In a benchmark or a CI job, sample at a long period or not at all; the
  sampler allocates a thread and a record per period.
