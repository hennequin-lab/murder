# Munin API

Signatures are from the pinned Raven sources. `Munin` re-exports the modules
below, so `open Munin` then `Session.start`, `Store.runs`, `History.read`,
`Run.id`, `Metric.best`, `Artifact.version`, `Provenance.detect`,
`Value.pp`, `Store_format`.

## Modules

| Module | Holds |
| --- | --- |
| `Munin.Store` | the store's root, its runs and artifacts, tags, notes, deletion, `gc`, `check` |
| `Munin.Session` | the one writer of a run: start, resume, metrics, samples, files, artifacts, finish |
| `Munin.Run` | a run as a listing reads it: start, params, provenance, tags, notes, status |
| `Munin.History` | a run's log folded: series, statistics, summary, files, uses, resumes |
| `Munin.Metric` | samples, series, statistics, goals, `best` |
| `Munin.Artifact` | one version: name, number, digest, path, size, metadata, producer, step |
| `Munin.Provenance` | command, cwd, git, toolchain, machine, seeds, captured env |
| `Munin.Value` | params, summaries and metadata values, and their printing |
| `Munin.Store_format` | the files of a store, in full |
| `Munin_sys` | readings of the machine, the process and devices (`munin.sys`) |

`munin` (the library) depends on `unix`, `threads.posix` and a float
printer. It depends on no Raven library. `munin.sys` reads the system
through C stubs and depends on no part of munin.

## Names, steps, values

- A **text** is a non-empty UTF-8 string with no control character
  (U+0000..U+001F, U+007F). Experiments, run names, and param, summary and
  metadata keys are texts.
- A **tag** is a text holding no comma.
- A **key** is one or more segments joined by `/`, where a segment is a text
  with no `/` that does not start with `.`: `train/loss`, `val/acc`,
  `sys/proc_rss_bytes`, `dev/CUDA:0/used_bytes`.
- An **artifact name** and an **alias** are one segment each.
- A **step** is an integer from 0 to 2^53. Steps are the program's counter
  and need not rise from one sample to the next.
- A **value** is `` `Bool b ``, `` `Int n ``, `` `Float f `` or
  `` `String s ``. A string must be UTF-8. A float reads back exactly; JSON
  has no NaN or infinity, and the store writes them as objects, so a NaN
  survives a round trip.
- A **time** is seconds since the Unix epoch. `Session.log` takes an
  optional `?time`, which defaults to the wall clock now.

## Munin.Store

```ocaml
type t

val open_ : ?root:string -> unit -> t
val root : t -> string

val runs : ?prev:Run.t list -> t -> Run.t list
val run : t -> string -> Run.t option

val add_tags : t -> string -> string list -> unit
val remove_tags : t -> string -> string list -> unit
val set_notes : t -> string -> string option -> unit
val delete_run : t -> string -> unit

val artifacts : ?name:string -> t -> Artifact.t list
val artifact :
  t -> string -> [ `Latest | `Version of int | `Alias of string ] ->
  Artifact.t option
val aliases : t -> string -> (string * int) list
val set_alias : t -> Artifact.t -> string -> unit
val remove_alias : t -> name:string -> string -> unit
val delete_artifact : t -> Artifact.t -> unit

val gc : t -> int
val check : t -> (string * string) list
```

`open_ ()` reads and creates nothing. The root is, in order: `?root`, made
absolute against the current directory; `$RAVEN_TRACKING_DIR` when set and
not empty; `$XDG_DATA_HOME/raven/munin`; `$HOME/.local/share/raven/munin`;
on Windows `%LOCALAPPDATA%\raven\munin`. A missing root is an empty store,
and the first write creates it.

`runs` is newest first by start time, then id. Each run costs its first and
last records, its tags and notes, and a lock probe when it has no end.
`prev` is a listing made earlier; with it, `runs` reads only the new runs and
the runs `prev` saw without an end, and re-reads tags and notes whose file
changed. Poll with it.

`run s id` is `None` when the store has no such run, and raises `Failure`
when the log does not start as a run of this format.

`delete_run` takes the run's lock and removes the directory. It refuses a
run whose writer holds the lock and a `` `Silent `` run. The versions the run
produced stay, still naming it as producer. `gc` removes blobs no version
names, writer work that crashed, and what a deletion cut short; it returns
how many entries it removed. `check` reads every log, tag file and blob and
gives `(path, problem)` for each break of the format.

## Munin.Session

```ocaml
type t
type metric
type reading = unit -> (string * float) list

val start :
  ?store:Store.t -> ?name:string -> ?parent:string ->
  ?tags:string list -> ?params:(string * Value.t) list -> ?steps:int ->
  ?provenance:Provenance.t -> experiment:string -> unit -> t

val resume : ?provenance:Provenance.t -> Run.t -> step:int -> t
val within : t -> (t -> 'a) -> 'a
val finish : ?status:[ `Finished | `Failed of string | `Killed ] -> t -> unit
val sync : t -> unit
val id : t -> string
val store : t -> Store.t

val metric :
  t -> ?goal:Metric.goal -> ?step_metric:metric -> string -> metric
val key : metric -> string
val log : metric -> step:int -> ?time:float -> float -> unit
val log_metrics :
  t -> step:int -> ?time:float -> (metric * float) list -> unit
val set_summary : t -> (string * Value.t) list -> unit

val sample : t -> every:float -> reading list -> unit

val log_file : t -> step:int -> key:string -> string -> unit

val log_artifact :
  t -> ?step:int -> ?metadata:(string * Value.t) list -> name:string ->
  string -> Artifact.t
val use_artifact : t -> Artifact.t -> unit
```

### Start

`start ~experiment ()` creates the run and writes its `start` record:
experiment, name, parent, tags, params, steps and provenance. It syncs the
log. It raises `Invalid_argument` if a name or key is not a text, a param key
repeats, a string value is not UTF-8, or `steps` is not a step; `Failure` if
`parent` names no run of the store or `git diff` fails; `Sys_error` if the
store cannot be written or cannot lock.

`parent` is the id of the run this one belongs to, such as a sweep. `steps`
is the largest step the run will log. `provenance` defaults to
`Provenance.detect ()`; when the tree is dirty, `start` also keeps
`git diff --binary HEAD` beside the log.

### Lifecycle

`within s f` calls `f s` and then finishes `s`: `` `Finished `` when `f`
returns, `` `Killed `` on `Sys.Break`, `` `Failed (Printexc.to_string e) ``
on another exception. It re-raises the exception with its backtrace. If
finishing raises after `f` raised, `f`'s exception wins and the run has no
end.

`finish s` stops the samplers, writes the end, syncs, writes `stats.json`
and releases the lock. Finishing twice does nothing. `sync s` makes every
record written so far survive a power loss.

`resume run ~step` is a new writer of `run`, continuing from `step`. The
program has restored a state that includes every sample at or below `step`.
Samples and files above `step` become orphans. It reads the log once, taking
the run's end, the kind of every key, and every key's statistics, then cuts
what the last writer left unwhole, writes a `resume` record with this
machine's identity and provenance, and syncs. It raises `Failure` if the run
ended or another writer holds the lock.

While a session is open, its process holds an exclusive lock on the log.
A reader on the same machine and boot sees `Running`; a reader that cannot
probe sees `Silent`; once no process holds the lock and the log has no end,
the run reads `Crashed`.

### Metrics

`metric s key` declares `key` and is its handle. `~goal` is `Minimize` or
`Maximize`, which `Metric.best` needs. `~step_metric` names another metric
used as this one's x axis, paired by step.

The first declaration of a key in a run appends a `metric` record; an equal
one later, as after a resume, writes nothing and is another handle. It raises
`Invalid_argument` if the key is not a key, the run declared it with another
goal or step metric, a reading of the run gave it, or `step_metric` is
another session's.

`log m ~step ?time v` appends one sample. `log_metrics s ~step ?time pairs`
appends one record holding a sample of each metric, all at one step and time;
an empty list writes nothing. A metric from another session, or twice in
`pairs`, raises. `log_metrics` is the call for training: one record per
report step.

`set_summary s values` appends values that replace the run's earlier value of
each key. A summary is what the program states about the run, such as a test
score; metric statistics are computed by readers.

A session's values may be called from any domain or thread, and take effect
one at a time. A write on a finished session raises `Invalid_argument`. A
write the system refuses raises `Sys_error` and leaves the log as before the
call; after a failed sync, every later write raises `Failure`.

### Sampling

```ocaml
val sample : t -> every:float -> reading list -> unit
```

`sample s ~every readings` calls each reading now in the caller's thread,
then every `every` seconds on a thread of its own until `s` finishes. Each
call's pairs become one `reading` record at the call's time and the session's
latest step (the last `metrics` or `resume` record, 0 before any). A call
that gives no pair logs nothing. A reading of a rate gives nothing at its
first call, so its first sample arrives one period later. Samples of
readings are never orphans.

A reading raising `Sys_error` skips that reading that time; another exception
stops the sampler, and `finish` raises it after writing the run's end. A key
that is not a key, appears twice, or belongs to a metric of the run also
stops the sampler. Each reading keeps its own counters and must go to one
sampler only.

`every` must be positive and finite.

### Files and artifacts

`log_file s ~step ~key path` copies a file into the run and appends a `file`
record; readers find it with `History.files`. The copy is complete before the
record is written.

`log_artifact s ~name path` publishes the file or directory `path` as a new
version of `name`, produced by this run at `?step`, with `?metadata`. It
copies `path` once and digests the bytes it copies, so a later change to
`path` does not change the version. Content already stored is shared as one
blob and still makes a new version. It syncs the blob, the version file and
the run's log before returning. It raises `Invalid_argument` for a bad name,
step, metadata key or string, or a directory holding a symbolic link to a
directory, and `Sys_error` when `path` cannot be read.

`use_artifact s a` appends a `use` record: this run read version `a`.
`History.uses` lists them.

## Munin.Run

```ocaml
type status =
  [ `Running | `Silent | `Crashed | `Finished | `Failed of string | `Killed ]

type t

val id : t -> string
val dir : t -> string
val experiment : t -> string
val name : t -> string option
val parent : t -> string option
val params : t -> (string * Value.t) list
val steps : t -> int option
val provenance : t -> Provenance.t
val diff : t -> string option
val started_at : t -> float
val tags : t -> string list
val notes : t -> string option
val status : t -> status
val ended_at : t -> float option
val updated_at : t -> float
```

A `Run.t` is a run as it was when read; its accessors do no I/O. It costs the
first and last records of the log, the tag files, the notes, and a lock probe
when it has no end.
`Run.status` is:

| Status | Meaning |
| --- | --- |
| `` `Running `` | a process holds the run's lock |
| `` `Silent `` | the reader could not probe the lock, so live and dead look alike |
| `` `Crashed `` | no end in the log, and no process holds the lock |
| `` `Finished `` | the writer finished it |
| `` `Failed e `` | the writer finished it with error text `e` |
| `` `Killed `` | the writer finished it as killed |

`updated_at` is the log's modification time, from the file system's clock.
`diff` is the path of `git.diff` when the run had one.

## Munin.History

```ocaml
type t
type file = { key : string; step : int; time : float; path : string }
type resume = { step : int; time : float; provenance : Provenance.t }
type key_stats = {
  kind : [ `Metric | `Reading ];
  goal : Metric.goal option;
  stats : Metric.stats;
}

val read : ?limit:int -> ?series:string list -> Run.t -> t
val follow : ?limit:int -> t -> t
val restarted : t -> bool

val metrics : t -> string list
val readings : t -> string list
val goal : t -> string -> Metric.goal option
val step_metric : t -> string -> string option
val series : t -> string -> Metric.series
val surviving : t -> string -> Metric.series
val orphans : t -> string -> Metric.series
val stats : t -> string -> Metric.stats option

val summary : t -> (string * Value.t) list
val files : t -> file list
val uses : t -> (string * int) list
val resumes : t -> resume list
val damaged : t -> int

val stats_of : Run.t -> (string * key_stats) list
val run : t -> Run.t
```

`read run` folds the whole log up to its last whole line. `~limit` stops at
the first whole line that ends past that many bytes from the log's start, for
a bounded read. `~series` names the keys whose samples the history keeps and
defaults to every key; the statistics of every key are kept regardless. It
probes the lock and re-reads tags and notes.

`follow h` folds the whole lines the log gained since `h`, keeping the keys
`h` keeps, and costs only the new bytes. It re-reads the log from its start
when the log shrank or was replaced, or when a resume orphaned samples of a
key whose samples `h` does not keep; `restarted h` says so. `series (follow
h) k` extends `series h k` unless the result restarted or a resume orphaned
`k`'s samples.

`metrics h` is the metric keys with at least one non-orphan sample, sorted;
`readings h` the reading keys, sorted. `series h k` is the samples that are
not orphans, in log order, and raises `Invalid_argument` if `h` does not keep
them. `surviving h k` drops the readings a resume abandoned even though they
are not orphans; it is what a view over the run's steps shows. `orphans h k`
is the abandoned samples.

Damaged lines are counted by `damaged`; a damaged line is skipped, not
fatal.

`stats_of run` reads the `stats.json` hint beside an ended run when its
format, rules and length match, else folds the log, at the cost of the whole
log. It is how a listing shows best values for many long runs.

`History.run h` is the run as `h` last read it, including tags, notes and
status.

## Munin.Metric

```ocaml
type goal = [ `Minimize | `Maximize ]
type sample = { step : int; time : float; value : float }
type series
type finite = { least : sample; greatest : sample; mean : float }
type stats = {
  count : int;
  last : sample;
  finite : finite option;
  non_finite : (int * sample) option;
}

val length : series -> int
val ordered : series -> bool
val step : series -> int -> int
val time : series -> int -> float
val value : series -> int -> float
val sample : series -> int -> sample
val best : goal -> stats -> sample option
```

A sample is **finite** when its value is neither NaN nor infinite.
Statistics that compare or add range over finite samples alone. `best goal
st` is the least finite sample under `` `Minimize ``, the greatest under
`` `Maximize ``, earliest among equals; `None` if none is finite.

A series is a snapshot. Its samples are indexed from 0 in log order, and the
accessors raise `Invalid_argument` outside `[0, length s - 1]`. `ordered s`
is true when no step is below the one before it.

## Munin.Artifact

```ocaml
type t

val name : t -> string
val version : t -> int
val digest : t -> string
val path : t -> string
val size_bytes : t -> int
val metadata : t -> (string * Value.t) list
val run : t -> string
val step : t -> int option
val created_at : t -> float
```

Versions are numbered from 1 and never change. `digest` is
`blake2b-256:` followed by 64 hexadecimal digits. `path` is the blob, a file
or a directory, shared by every version with the same content: read it,
never write it. `run` stays after the producing run is deleted. A deleted
version's number names nothing again; its blob stays until `gc`.

## Munin.Provenance

```ocaml
type git = { commit : string; dirty : bool }

type t = {
  command : string list;
  cwd : string;
  hostname : string option;
  pid : int;
  ocaml : string;
  raven : string;
  os : string;
  arch : string;
  cpu : string option;
  git : git option;
  seeds : (string * int) list;
  env : (string * string) list;
}

val detect :
  ?seeds:(string * int) list -> ?capture_env:string list -> ?cwd:string ->
  unit -> t
val pp : Format.formatter -> t -> unit
```

`detect ()` runs
`git status --porcelain=v2 --branch --untracked-files=no` once in `cwd`,
without a shell; `git` is `None` when that fails. The seeds are the one part
only the program knows: name every generator the program seeds, such as
`[ ("init", 42); ("data", 7) ]`. `capture_env` records named environment
variables, skipping unset ones. A resume records its own provenance, since
it may run other code.

## Munin.Value

```ocaml
type t = [ `Bool of bool | `Int of int | `Float of float | `String of string ]

val pp : Format.formatter -> t -> unit
val to_float : t -> float option
val to_int : t -> int option
val to_string : t -> string option
val to_bool : t -> bool option
```

`pp` prints a float in the fewest digits that read back, a NaN as `nan`, and
a string with tabs and line feeds escaped, so a value never splits a
tab-separated line. The CLI prints values this way.

## Munin_sys

```ocaml
val host : unit -> unit -> (string * float) list
val process : unit -> unit -> (string * float) list

type device = {
  allocated : int option;
  budget : int option;
  memory : int option;
  used : int option;
  busy : float option;
  activity : float option;
  temperature : float option;
  power : float option;
}

val unknown : device
val device : string -> (unit -> device) -> unit -> (string * float) list
```

`host ()` and `process ()` are readings; `device name read` turns a device
record into one. The fields and their keys are in `references/monitoring.md`.

## Store_format

The module specifies every file; `references/cli-and-store.md` summarises the
layout and the common `jq` reads. Use `Store_format` when a program reads a
store without munin, or when a test asserts on the files: the format is part
of the contract, not an implementation detail.

The format version is the value of `Munin.Store_format.version`.
