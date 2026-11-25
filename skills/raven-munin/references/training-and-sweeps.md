# Training runs and sweeps

A run is one execution of a program. The session is its one writer; readers
list runs, fold their logs, and compare them. This file covers the training
loop, the layout of a sweep, checkpoints and resume, and the lab conventions
that make a store readable later.

Commands are written `munin`. In a murder checkout, prefix them with
`dune exec --`; `dune exec` parses its own options after the program name, so
the `--` is what keeps `--experiment` and `--parent` for munin. A command
with no flags also runs bare, as `dune exec munin runs`.

## One experiment per question, one run per trial

- An **experiment** names what the runs are attempts at: a data set and a
  model, such as `lorenz` or `mnist`. Runs of one experiment are compared
  with each other, and the dashboard picks a key metric per experiment.
- A **run** is one configuration: one learning rate, one seed, one device.
- A **sweep** is a parent run plus the trials that name it as `~parent`.
- Do not put two configurations in one run to save a process. The store then
  cannot compare them, and a run has one end.
- Do not name an experiment after a sweep (`sweep-3`). Name it after the
  question, give the sweep run a `~name`, and give each trial `~parent`.

Seeds are trials, not metadata only: a second seed is a second run with
`("seed", `Int s)` as a param. Replicates let `munin compare` show their
best values side by side, and let the dashboard draw both curves.

## The loop

```ocaml
open Munin

let experiment = "lorenz"
let total_steps = 100_000
let report_every = 100
let eval_every = 1_000

let train_run s ~lr ~seed ~params =
  let train_loss = Session.metric s ~goal:`Minimize "train/loss" in
  let val_loss = Session.metric s ~goal:`Minimize "val/loss" in
  let step_ms = Session.metric s "perf/step_ms" in
  let key = Nx.Rng.fold_in (Nx.Rng.key seed) 1 in
  let params = ref params in
  let best = ref infinity in
  for step = 1 to total_steps do
    let t0 = Unix.gettimeofday () in
    let next, loss = train_step ~lr (Nx.Rng.fold_in key step) !params in
    params := next;
    if step mod report_every = 0 then begin
      let loss = Nx.item [] loss in                 (* the device wait *)
      let dt = Unix.gettimeofday () -. t0 in
      Session.log_metrics s ~step
        [ (train_loss, loss); (step_ms, dt *. 1e3) ]
    end;
    if step mod eval_every = 0 then begin
      let v = evaluate !params in
      Session.log val_loss ~step v;
      best := Float.min !best v
    end;
    (* Checkpoints are published at a slower cadence; see below. *)
  done;
  Session.set_summary s [ ("best/val_loss", `Float !best) ]
```

The session is started by the caller, which owns the params and the run's
end:

```ocaml
let () =
  let store = Store.open_ () in
  Session.within
    (Session.start ~store ~experiment ~name:"adam-lr3e-4"
       ~tags:[ "adam" ]
       ~params:
         [
           ("optimizer", `String "adam");
           ("lr", `Float 3e-4);
           ("batch", `Int 512);
           ("seed", `Int seed);
           ("device", `String device_name);
         ]
       ~steps:total_steps
       ~provenance:
         (Provenance.detect ~seeds:[ ("init", seed); ("data", 42) ] ())
       ())
  @@ fun s ->
  Printf.printf "run %s\n%!" (Session.id s);
  Session.sample s ~every:15. [ Munin_sys.host (); Munin_sys.process () ];
  train_run s ~lr:3e-4 ~seed
    ~params:(init (Nx.Rng.fold_in (Nx.Rng.key seed) 0))
```

The `train_run` above lives in the executable. A library function in
`packages/` stays free of munin: give it a small `log` hook, or keep the loop
in the executable. A stable library does not depend on munin.

## What to log

Log at a report cadence, not every micro-step. Each sample is one appended
record, and reading a jitted loss with `Nx.item` synchronises with the
device. Report every 50 to 1,000 steps for long runs, every epoch for short
ones; the dashboard reduces long series to its columns.

| Key | Goal | What |
| --- | --- | --- |
| `train/loss` | `Minimize` | the loss the step optimizes, per report step |
| `train/loss_avg` | none | an epoch's mean, at the epoch's last step |
| `val/loss` | `Minimize` | validation loss, less often than training |
| `val/acc` | `Maximize` | validation accuracy or another score |
| `epoch` | none | the epoch, as an x axis for once-per-epoch keys |
| `perf/step_ms` | none | wall time of one step or one report window |
| `perf/images_per_sec` | none | throughput, if the run reports it |

- `train/*` is smoothed by default in the dashboard charts. A validation
  metric named `val/loss` with a goal becomes the experiment's key metric:
  the dashboard prefers a goal metric outside `train/` whose key ends in
  `loss`, then any goal metric outside `train/`, then any goal metric.
- A metric without a goal still has a series and statistics; it has no best,
  and `munin compare` shows its last value.
- `log_metrics s ~step [ ... ]` puts one record for a step. Use it when
  several metrics share a step; separate `log` calls are separate records at
  the same step and are fine too.
- Metrics logged at a coarser cadence can still use the step as their x
  axis. If the x axis is the epoch instead, log `epoch` at the same step and
  declare `~step_metric:epoch` on the metric.
- A summary holds final values that are not a series: `best/val_loss`,
  `test/acc`, `n_params`, `wall_time_s`. It is what a reader shows on the run,
  and what survives when the series is long.

A file a run logs belongs to a step, such as a sample grid or a diagnostic
figure:

```ocaml
Session.log_file s ~step ~key:"viz/samples" path
```

The store copies the file before the record is written, so the program may
overwrite `path` afterwards. `History.files h` gives `{ key; step; time;
path }` for each; `key` is a key, and `path` is the copy. Log one image per
few epochs, not per step: a store grows by its files.

Do not log a metric under `sys/` or `dev/`. Those prefixes belong to readings,
and a key is a metric's or a reading's in one run, never both.

## Params

Params are the run's settings, each a `Munin.Value.t`. A reader compares
them and the CLI prints only those that differ, so include everything that
could explain a difference in result:

- optimizer and its settings (`lr`, `wd`, `momentum`, schedule);
- model shape (`width`, `depth`, `n_params`), if it varies;
- data (`dataset`, `split`, `n_train`, preprocessing switches);
- `seed` for every generator the run seeds, and `batch`;
- `device`, backend and compiler options (`jit`, `beam`, `parallel`);
- code-selecting switches (`loss`, `reparam`, `solver`) and their versions.

Do not put a run's identity in a param (`run_name`, `timestamp`); the run has
an id and a start time. Do not put long text in a param; use a summary or the
notes.

`Provenance.detect ~seeds` records the seeds as provenance too. A run's
provenance includes the command, cwd, git commit and dirty flag, the
toolchain, the machine and the captured environment; a dirty tree also keeps
`git diff --binary HEAD` beside the log, which is how a run's code is
recovered.

## Sweeps in one process

Sequential trials in one driver are the simplest layout: one store, one
parent run, one session per trial, each closed before the next starts.

```ocaml
let configs =
  [ ("adam-lr1e-4", 1e-4); ("adam-lr3e-4", 3e-4); ("adam-lr1e-3", 1e-3) ]

let trial store ~parent (name, lr) =
  Session.within
    (Session.start ~store ~experiment:"lorenz" ~name ~parent
       ~params:
         [
           ("optimizer", `String "adam");
           ("lr", `Float lr);
           ("seed", `Int 42);
         ]
       ~steps:total_steps ())
  @@ fun s ->
  train_run s ~lr ~seed:42
    ~params:(init (Nx.Rng.fold_in (Nx.Rng.key 42) 0))

let () =
  let store = Store.open_ () in
  Session.within
    (Session.start ~store ~experiment:"lorenz" ~name:"sweep-lr" ())
  @@ fun sweep ->
  List.iter
    (fun config ->
      match trial store ~parent:(Session.id sweep) config with
      | () -> ()
      | exception e ->
          Printf.eprintf "trial failed: %s\n%!" (Printexc.to_string e))
    configs
```

`within` ends a failed trial as `Failed` with the exception's text before it
re-raises. The driver catches and goes on; the store keeps the failure, and
`munin runs --status failed --parent ID` finds it.

## Parallel sweeps across processes

One process per trial, each starting its own session, with the parent's id
passed on the command line:

```sh
# The driver prints the sweep run's id and exits.
sweep=$(./sweep.exe)
for lr in 1e-4 3e-4 1e-3; do
  ./train.exe -parent "$sweep" -lr "$lr" -seed 42 &
done
wait
munin runs --parent "$sweep"
```

- Spawn with `Unix.create_process`, a shell or `exec`, not with `fork`
  alone. A child forked without `exec` inherits the writer's log descriptor
  and lock, so the parent run reads `running` until the child exits. munin
  opens its log with `O_CLOEXEC`, so an executed child inherits nothing.
- Each process must create its own run; never start a session and then fork
  the training loop.
- `~parent` needs an existing run id. The parent session may finish before
  the children start; the parent's end does not end the children.
- Keep the parent id: write it to a file, print it, or set it in the
  environment (`MUNIN_PARENT`), so a background trial that loses its terminal
  is still findable by `munin runs --parent`.
- A trial that exits nonzero should have ended `Failed` through `within`.
  If it was killed or crashed, the run reads `Crashed`; `munin end RUN` ends
  it as `Killed` once its writer is gone.
- Parallel trials write to one store. Publications and `gc` serialize, and
  deletions of runs wait for each other; do not run `gc` or delete runs
  while a sweep is writing.

## Comparing a sweep

```sh
parent=$(munin runs --experiment lorenz | awk -F'\t' '$5 == "sweep-lr" { print $1 }')
ids=$(munin runs --parent "$parent" | cut -f1)
munin compare $ids
```

`compare` prints:

- `param.*` rows for params the runs do not all share;
- `metric.*` rows, one column per run, with the best value under a shared
  goal (or the last value without a goal);
- `*` on a row's best when every run that logged the key shares one
  goal.

The columns follow the order of the arguments, which is newest first from
`munin runs`. Sort the ids or pass them explicitly when the order matters.
A NaN or an infinity never takes the star.

Programmatically, read an ended run without folding its log:

```ocaml
let best_of run key =
  match List.assoc_opt key (History.stats_of run) with
  | Some { goal = Some goal; stats; _ } -> Metric.best goal stats
  | _ -> None
```

## Checkpoints

Publish a checkpoint on a cadence slower than reporting, and give it the step
that produced it:

```ocaml
let ckpt = Filename.concat dir "checkpoint" in
save_params ckpt !params;
let v = Session.log_artifact s ~step ~name:"model" ckpt in
if val_loss <= !best then Store.set_alias (Session.store s) v "best"
```

- `log_artifact` copies the file or directory once and digests the bytes it
  copied. Overwriting the source afterwards does not change the version.
- `~metadata` records what a reader needs to interpret it, such as
  `[ ("format", `String "safetensors"); ("optimizer", `String "adam") ]`.
- The alias moves and versions stay immutable, so `best` is always the
  current pick and old picks remain addressable by number.
- `Store.artifacts ~name:"model"` is ordered by name then number; the run's
  own versions are
  `List.filter (fun a -> Artifact.run a = Run.id run) (Store.artifacts ~name:"model" store)`.
  Do not use `` `Latest `` for a resume: it may name another run's version.
- A version's `Artifact.path` is a shared blob. Read it, never write it.
- Directories holding a symbolic link to a directory are refused. A file or a
  plain tree is fine.

A version published with `~step` and synced before the call returns is the
one place to resume from: the run's log has every sample up to that step,
even after a power loss.

## Resuming a run

```ocaml
let resume_from store run ~lr ~seed =
  let mine =
    List.filter
      (fun a -> Artifact.run a = Run.id run)
      (Store.artifacts ~name:"model" store)
  in
  match List.rev mine with
  | [] -> failwith "no checkpoint of this run"
  | last :: _ ->
      let step = Option.get (Artifact.step last) in
      let params = load_params (Artifact.path last) in
      Session.within (Session.resume run ~step) @@ fun s ->
      Session.use_artifact s last;
      train_run s ~lr ~seed ~params
```

- `Session.resume` is a new writer of the same run. It fails if the run ended
  or another writer holds the lock, so a resume driver treats those as "not
  resumable".
- The restored state must include every sample at or below `step`. Samples
  and files the run logged above `step` become orphans, which readers leave
  out of series and statistics.
- The resumed writer re-declares its metrics. Goal and step metric must match
  the run's earlier declarations, or `Session.metric` raises.
- The checkpoint must hold everything the resumed loop needs: parameters,
  optimizer state, the step, and the RNG state and data position if an exact
  resume matters. munin records the step, not the generator state.
- `use_artifact` records what was read, so `History.uses` links the resume to
  the checkpoint version.
- A run that you want to continue with a *different* configuration is a new
  run that uses the checkpoint, not a resume. `Session.start` with
  `use_artifact` is the evaluation pattern.

## Evaluation runs

An evaluation is its own run of its own experiment:

```ocaml
let () =
  let store = Store.open_ () in
  let best = Option.get (Store.artifact store "model" (`Alias "best")) in
  Session.within
    (Session.start ~store ~experiment:"lorenz-eval" ~name:"test"
       ~params:[ ("checkpoint", `Int (Artifact.version best)) ]
       ~steps:0 ())
  @@ fun s ->
  Session.use_artifact s best;
  let params = load_params (Artifact.path best) in
  Session.set_summary s [ ("test/mse", `Float (test_mse params)) ]
```

`steps:0` is right for a run that logs no series: the dashboard shows no
progress bar rather than a fraction of zero. The eval run's history lists
`("model", n)` in `History.uses`, and `Store.artifacts` still names the
producing run.

## Tags and notes

Tags and notes may change during and after the run, from any process:

```sh
munin tag RUN baseline to-review
munin note RUN "Loss flat after step 60; lr may be too high."
munin runs --tag baseline
```

Use tags for cross-cutting selection (`baseline`, `gpu`, `paper-3`), not for
the configuration (params do that). Use notes for the sentence a future
reader needs: why the run was stopped, what looked wrong, which checkpoint to
use. `Store.set_notes` replaces the notes whole; `Store.add_tags` and
`Store.remove_tags` are idempotent.

## Replacing the lab's text logs

| Before | With munin |
| --- | --- |
| `append_row info [| step; loss; drift; w |]` | `Session.log_metrics s ~step [ ... ]`, one handle per column |
| a `_results_<optimizer>/` directory per configuration | `~params` and `~name` on one experiment; the store compares them |
| `checkpoint.bin` overwritten in place | `Session.log_artifact ~step ~name:"model"`; alias the best |
| printing the best value at the end | `~goal` on the metric; `munin compare`, `History.stats_of` |
| re-running to recover a number | read the store; tag or note it instead |
| `git log` and shell history to find the command | the run's provenance: command, commit, diff, seeds |

Keep the last-epoch console print if it helps while training. The store is
what a later reader and the next agent use.

## Costs and limits

- A store lists runs from the first and last record of each log, so a listing
  of hundreds of runs is still cheap. `History.read` of a long run costs its
  whole log; use `History.stats_of` for ended runs and `History.follow` for
  live ones.
- Do not poll a store in a tight loop. The dashboard polls once a second and
  uses `Store.runs ~prev`.
- A reader keeps 24 bytes per sample per kept key; `History.read ~series`
  limits which samples it keeps while still computing every key's statistics.
- A store is a directory; do not put it in the repository. Set
  `RAVEN_TRACKING_DIR` for a project that wants its own store, and exclude
  that directory from backups if it grows.
- `Store.check` is the integrity check; run it after a crash or a copy. It
  reads every log and blob.

## Pitfalls

- `Session.start` without `~store` writes the default store, which is shared
  by every project on the machine. Pass `~store` in code that may run under
  another project's environment.
- `steps` is a claim, not a limit: logging above it is allowed, and the
  dashboard's progress bar stays full. Declare the largest step the run
  really logs.
- A metric handle belongs to one session. After a resume the handles are new;
  logging through an old handle raises.
- `Session.log` accepts steps out of order. A run that goes back without a
  resume keeps both branches in its series; `Metric.ordered` reports it, and
  the dashboard draws the log order. Use `resume` when the program really
  restarted.
- Reading a device scalar inside a jitted function raises. Return it, read it
  with `Nx.item` at the report boundary.
- Do not call `Session.log` inside a `Rune.jit` or `Rune.scan` body. The trace
  runs the call at trace time, and the tensor is not a float.
- A provenance `cwd` is the process's cwd at `start`. Run from the work tree,
  or build it with `Provenance.detect ~cwd` and pass `~provenance`; otherwise
  the recorded commit is not the code that ran.
- A failed write (`Sys_error`) leaves the log unchanged, but once a sync
  fails every later write raises `Failure`. Stop the run; do not retry the
  loop.
- `finish` raises what stopped a sampler after writing the run's end, so a
  broken reading does not lose the run.
