---
name: raven-munin
description: "Record, resume and compare training runs with Munin in the Hennequin lab monorepo (murder) and the projects that pin it. Use when a task starts, logs, sweeps, resumes or compares training runs or experiments in OCaml: sessions, params, metrics, steps and goals, summaries, artifact versions and checkpoints, tags and notes, provenance and seeds, readings of the machine and its GPUs, the munin command, or the terminal dashboard. Triggers: Munin., Munin_sys, Session.start, Session.within, Session.metric, log_metrics, log_artifact, use_artifact, Session.resume, Store.open_, RAVEN_TRACKING_DIR, munin runs, munin compare, munin watch, experiment tracking, training sweep, checkpoint registry, finding an old run."
---

# Munin

Munin records what a program does as it runs: params, metrics step by step,
readings of the machine and its devices, files, and versioned artifacts such
as checkpoints. It writes a store, a directory of JSON and text files, and
reads it back. No server, no account, and `jq`, `tail -f` or another program
reads a store as well as munin does.

The point for this lab is training sweeps: one run per trial, each with its
params, its metrics, its checkpoint versions and its provenance, and a
command that compares them. Read this file first; the references carry the
detail.

- `references/api.md` — every module and signature.
- `references/training-and-sweeps.md` — training loops, sweep layouts,
  checkpoints, resume, evaluation runs, provenance.
- `references/monitoring.md` — readings of the machine, the process and
  rig's devices, and the sampler's rules.
- `references/cli-and-store.md` — the `munin` command, the dashboard, the
  store's files, and `jq` recipes.

For the tensors see `raven-nx`; for compiled steps and `scan` see
`raven-rune`; for device selection and placement see `raven-rig`.

## Workflow

1. In a murder checkout, `munin` is already in the lock through
   `murder-dev`: a private executable adds `(libraries munin)` (and
   `munin.sys` for machine readings) and builds. A project repo, or a package
   that installs the executable, adds `munin` to its `depends` and runs
   `./dev lock`. A stable library keeps a logging hook instead.
2. Start one session per trial with `Session.start`, every setting in
   `~params`, and `~steps`; bracket it with `Session.within`.
3. Log `train/*`, `val/*` and `perf/*` at a report cadence, and sample
   readings every 15 seconds.
4. Publish checkpoints with `Session.log_artifact`, alias the best, and
   resume a crashed run with `Session.resume`.
5. Compare with `munin compare` or the dashboard, and keep the store out of
   the repository.

## House rules

- A stable library does not link munin. The executable that owns the loop
  does, as `packages/munin/examples/x-kaun-mnist` does for Kaun: Kaun knows
  nothing of munin, and the program logs.
- `murder-dev` depends on `munin` and `quill`, so a murder checkout runs
  `dune exec munin` and `dune exec quill` with no change. A package that
  links `munin` adds it to its own `depends` (a dependency change: ask, then
  `./dev lock`, and include the lock diff); a project repo does the same for
  itself. The package brings `jsont`, `bytesrw`, `mosaic` and `matrix` into a
  lock, because the CLI and dashboard are in the same package.
- One session writes one run. Start it once, bracket it with
  `Session.within`, and let that write the run's end.
- Name every setting that changes the result as a param, including the seed,
  the device, the data set and the switches that select code. A run's name is
  for humans; it is not a param.
- Log at a report cadence. Each sample is one appended record, and reading a
  jitted loss back with `Nx.item` waits for the device, so report every N
  steps rather than every step.
- Follow the key conventions the dashboard reads: `train/*` for training
  metrics (smoothed in charts), `val/*` for validation, `perf/*` for timings.
  `sys/*` and `dev/*` belong to readings and are never metric keys.
- A test opens a store at a temporary root, never the default store, and does
  not assert on ids or times: ids hold the start time and random bits, and
  times come from the clock.

## The shape of a tracked run

```ocaml
open Munin

let experiment = "lorenz"
let total_steps = 100_000
let report_every = 100

(* [seed], [device_name], [params0], [train_step], [evaluate] and
   [best_val_loss] come from the program. *)
let () =
  let store = Store.open_ () in
  let provenance =
    Provenance.detect ~seeds:[ ("init", seed); ("data", 42) ]
      ~capture_env:[ "OMP_NUM_THREADS" ] ()
  in
  Session.within
    (Session.start ~store ~experiment ~name:"adam-lr3e-4"
       ~params:
         [
           ("optimizer", `String "adam");
           ("lr", `Float 3e-4);
           ("batch", `Int 512);
           ("seed", `Int seed);
           ("device", `String device_name);
         ]
       ~steps:total_steps ~provenance ())
  @@ fun s ->
  Printf.printf "run %s\n%!" (Session.id s);
  Session.sample s ~every:15. [ Munin_sys.host (); Munin_sys.process () ];
  let train_loss = Session.metric s ~goal:`Minimize "train/loss" in
  let val_loss = Session.metric s ~goal:`Minimize "val/loss" in
  let step_ms = Session.metric s "perf/step_ms" in
  let params = ref params0 in
  for step = 1 to total_steps do
    let t0 = Unix.gettimeofday () in
    let next, loss = train_step !params in
    params := next;
    if step mod report_every = 0 then begin
      (* The device wait belongs here, at the report boundary. *)
      let loss = Nx.item [] loss in
      let dt = Unix.gettimeofday () -. t0 in
      Session.log_metrics s ~step
        [ (train_loss, loss); (step_ms, dt *. 1e3) ]
    end;
    if step mod eval_every = 0 then
      Session.log val_loss ~step (evaluate !params)
  done;
  Session.set_summary s [ ("best/val_loss", `Float (best_val_loss ())) ]
```

`within` finishes the run as `Finished`, or as `Killed` on `Sys.Break`, or as
`Failed` with the exception's text, and re-raises. `Session.start` writes the
start and syncs it; each later call appends one record that a reader in
another process sees once the call returns.

`~steps` is the largest step the run will log. Readers show progress as the
latest step over it, and the dashboard sizes the axis to it while the run is
live. `Session.metric` with `~goal` is what lets a reader pick the best
sample: `train/loss` and `val/loss` minimize, `val/acc` maximizes.
`~step_metric:epoch` pairs a metric with another metric used as its x axis.

Keys are one or more segments joined by `/`; a segment holds no `/` and does
not start with `.`. Experiment names, run names and param keys are non-empty
UTF-8 texts with no control character. A tag holds no comma. A step is an
integer from 0 to 2^53.

## Sweeps

A sweep is a run that is the parent of the trial runs. The parent is a
marker: it holds the experiment, the shared settings and the list of trials
that name it.

```ocaml
let trial store ~parent (name, lr, seed) =
  Session.within
    (Session.start ~store ~experiment:"lorenz" ~name ~parent
       ~params:
         [
           ("lr", `Float lr);
           ("seed", `Int seed);
           ("batch", `Int 512);
         ]
       ~steps:total_steps ())
  @@ fun s ->
  train_run s ~lr ~seed
    ~params:(init (Nx.Rng.fold_in (Nx.Rng.key seed) 0))

let () =
  let store = Store.open_ () in
  Session.within
    (Session.start ~store ~experiment:"lorenz" ~name:"sweep-lr" ())
  @@ fun sweep ->
  List.iter (trial store ~parent:(Session.id sweep)) configs
```

- Use one experiment per question (data set and model), and let the params
  say what differs. `munin compare` prints only the params the runs do not
  all share, so a constant setting need not repeat as a column.
- One run per configuration, never several configurations in one run.
- A trial that fails ends as `Failed` with the exception; `within` re-raises,
  so a driver that wants to continue catches and goes on. Do not finish a
  failed trial as finished.
- For parallel trials, execute the training program per configuration with
  `-parent ID` and spawn it with `Unix.create_process`, a shell or `exec`,
  not with `fork` alone. A forked child that never execs inherits the
  writer's log descriptor and its lock, so the parent run reads `running`
  until that child exits. munin opens its log with `O_CLOEXEC`, so an
  executed child inherits nothing.
- Child runs need only the parent's id. The parent session may finish once
  the trials are started; the children do not write it.
- Tag the parent, such as `~tags:[ "sweep" ]`. It logs no metrics; compare
  `munin runs --parent ID` so it is not a column.

Compare and read the sweep from the store (prefix each command with
`dune exec --` in a murder checkout):

```sh
munin runs --experiment lorenz
parent=$(munin runs --experiment lorenz | awk -F'\t' '$5 == "sweep-lr" { print $1 }')
munin compare $(munin runs --parent "$parent" | cut -f1)
munin metrics "$(munin runs --parent "$parent" | head -1 | cut -f1)"
```

`munin compare` prints a row per differing param and per metric, one column
per run, with `*` on the best of a row when every run that logged the key
shares one goal. `references/training-and-sweeps.md` has the in-process and
subprocess layouts.

## Checkpoints, artifacts and resume

An artifact is a name with immutable versions numbered from 1. A run
publishes a file or a directory; the store keeps the bytes once, named by
their digest.

```ocaml
(* In the training run: publish, and move the alias when it improves. *)
let v =
  Session.log_artifact s ~step ~name:"model"
    ~metadata:[ ("optimizer", `String "adam"); ("lr", `Float lr) ]
    checkpoint_dir
in
if val_loss < !best then Store.set_alias (Session.store s) v "best"
```

A checkpoint directory is copied once, so overwriting the directory
afterwards does not change the version. `log_artifact` syncs the blob, the
version and the run's log before it returns, so a run resumed from the
version finds every sample up to its step in the log.

Resume a run whose writer is gone from its latest checkpoint:

```ocaml
let restart store run ~lr ~seed =
  let mine =
    List.filter
      (fun a -> Artifact.run a = Run.id run)
      (Store.artifacts ~name:"model" store)
  in
  match List.rev mine with
  | [] -> failwith "this run published no checkpoint"
  | last :: _ ->
      let step = Option.get (Artifact.step last) in
      let params = load_params (Artifact.path last) in
      Session.within (Session.resume run ~step) @@ fun s ->
      Session.use_artifact s last;
      (* The run's metrics are re-declared, then training logs steps above
         [step]. *)
      train_run s ~lr ~seed ~params
```

`Session.resume` fails if the run has ended or another writer holds its lock.
The samples the run logged above `step` become orphans: `History.series` and
`History.stats` leave them out, and `History.orphans` gives them. A resumed
writer re-declares its metrics, and the goal and step metric must match what
the run declared, or `Session.metric` raises. A run named in a new run's
`use_artifact` is how a test or an evaluation says which version it read.

The CLI ends a crashed run as killed, and maintenance removes what nothing
names:

```sh
dune exec -- munin end RUN          # crashed or silent run, as killed
dune exec -- munin artifacts model  # versions, digests, producers, aliases
dune exec -- munin gc               # blobs no version names, crash leftovers
dune exec -- munin check            # every break of the store's format
```

## Reading a store

`Store.runs` lists runs newest first from the first and last records of each
log, so it costs the same for a long run as for a short one.
`History.read` folds one log into series and statistics; `History.follow`
folds only what the log gained since, which is how a reader follows a live
run. `History.stats_of run` reads an ended run's `stats.json` hint and folds
the log only when the hint does not match.

```ocaml
let store = Store.open_ () in
let run = Option.get (Store.run store id) in
let h = History.read run in
match (History.goal h "val/loss", History.stats h "val/loss") with
| Some goal, Some stats ->
    Option.iter
      (fun (b : Metric.sample) ->
        Printf.printf "best val/loss %.6f at step %d\n" b.value b.step)
      (Metric.best goal stats)
| _ -> ()
```

`Metric.best` ranges over finite samples alone. A NaN or an infinity never
becomes a best, a minimum, a maximum or a mean; `stats.non_finite` counts
them and dates the earliest, and the dashboard marks the run with `×`.

`dune exec munin` runs the executable from the locked Raven package:

```sh
dune exec munin runs
dune exec -- munin runs --experiment lorenz
dune exec -- munin watch RUN
```

`dune exec` parses its own options after the program name too, so put `--`
before munin's flags. Outside a murder checkout, build or install munin from
a Raven checkout and run that binary. Each command takes `--root DIR` after
its name, as `munin runs --root DIR`; without it the store is
`$RAVEN_TRACKING_DIR`, else `$XDG_DATA_HOME/raven/munin`, else
`~/.local/share/raven/munin`. The default store is shared by every program
on the machine, so set `RAVEN_TRACKING_DIR` when a project wants its own.
`munin` with no command, on a terminal, opens the dashboard; `munin watch
[RUN]` shows one run as it trains.

## Pitfalls

- A key is a metric's or a reading's in one run, never both. `Session.metric`
  raises if a reading already gave the key, and a reading raises if it gives
  a metric's key. Keep the `sys/` and `dev/` prefixes for munin.
- A reading of a rate gives nothing on its first call: its counter window
  starts there. Give a reading to one sampler only, and call it from no other
  thread.
- A metric handle belongs to the session that declared it. Declare the
  handles in the session that logs them, and declare them again after a
  resume; `Session.log` through a handle of a finished session raises.
- Logging steps out of order is allowed. Readers keep the order the records
  were written, and `Metric.ordered` says whether steps rise. A resume is the
  only thing that makes orphans.
- A failure to write raises `Sys_error` and leaves the log before the call.
  After a failed `sync`, every later write raises `Failure`, since records
  already written may be lost. Let the run end as failed and start again.
- `Nx.item` of a traced value inside `Rune.jit` raises. Read device scalars
  between steps, at the report boundary, not inside the compiled program.
- Never call `Session.log` or `Session.log_metrics` inside a `Rune.jit` or
  `Rune.scan` body: a trace runs the call once at trace time, and a traced
  tensor is not a float. Return the value and log it outside.
- A run's provenance is detected at `start`, in the process's `cwd`. Run the
  training executable from the work tree, or build the provenance with
  `Provenance.detect ~cwd` and pass it as `~provenance`; otherwise the
  recorded commit is the wrong one. A dirty tree
  also keeps `git diff --binary HEAD` beside the log.
- `Store.delete_run` refuses a running or silent run. End a crashed or silent
  run first, and remember that a silent run's writer cannot be probed: end it
  only when that writer is known to be gone.
- Deleting a run keeps the artifact versions it produced, still naming it as
  producer. A version's number is never reused; the blob goes when `gc` finds
  no version naming it.

## References

- `references/api.md` — modules, signatures, types.
- `references/training-and-sweeps.md` — the loop, sweeps, checkpoints,
  resume, provenance.
- `references/monitoring.md` — `Munin_sys`, device glue over rig, sampler
  rules.
- `references/cli-and-store.md` — commands, dashboard, store layout, `jq`.

The API follows the Raven commit pinned in `dune-project`. munin depends on
no Raven library, and `munin.sys` depends on no part of munin; a program
links `munin` and, for machine readings, `munin.sys`.
