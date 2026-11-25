# The command, the dashboard and the files

`munin` reads a store and changes what may change after it was written: tags,
notes, aliases, deletions, `gc`. Everything else in a store is append-only.

In a murder checkout, `dune exec munin` runs the executable from the locked
Raven package, and `dune exec` builds it on first use:

```sh
dune exec munin runs
dune exec -- munin runs --experiment lorenz
dune exec -- munin watch RUN
```

`dune exec` parses its own options after the program name too, so put `--`
before the program's flags. Elsewhere, build or install munin from a Raven
checkout (`dune exec packages/munin/bin/main.exe -- runs`) or put its
`target/bin` on `PATH`. The command examples below are written `munin`; in a
murder checkout, prefix them with `dune exec --`.

Every command takes `--root DIR` after its name, as `munin runs --root DIR`,
so a Raven checkout can read a project's store. Without it the store is `$RAVEN_TRACKING_DIR`, else
`$XDG_DATA_HOME/raven/munin`, else `~/.local/share/raven/munin`. Use the
same environment variable or `--root` as the program that wrote the run, or
the command reads a different store.

## Output

Each command prints tab-separated lines without a header, one record a line,
so `cut`, `awk`, `sort` and `grep` take the output as it is. A value prints
as `Value.pp` prints it, so a string with a tab or a line feed never splits a
line. Times are RFC 3339 in UTC. An error goes to stderr and exits 1. `--root`
that does not exist is an error, so a typo fails instead of reading an empty
store.

## Commands

| Command | Prints or does |
| --- | --- |
| `munin runs` | id, status, start, experiment, name, tags, newest first |
| `munin runs --experiment E --status S --tag T --parent ID` | the runs that match; repeat `--tag` for every tag |
| `munin show RUN` | the run as `key value` lines: params, provenance, tags, notes, summary, each metric's count, last, best, non-finite count |
| `munin metrics RUN` | each key with a sample: key, kind, count, last, best, non-finite count |
| `munin metrics RUN KEY` | the key's series: step, time, value, orphans left out |
| `munin compare RUN...` | one column per run: differing params, then each metric's best, or last without a goal; `*` on the best of a row when every run that logged it shares a goal |
| `munin artifacts [NAME]` | name, version, digest, size, producing run, step, aliases |
| `munin alias NAME VERSION ALIAS` | make `ALIAS` name that version |
| `munin alias --remove NAME ALIAS` | remove the alias |
| `munin rm-artifact NAME VERSION` | delete a version and its aliases |
| `munin tag RUN TAG...` | add tags; `--remove` removes them |
| `munin note RUN TEXT` | replace the notes; without `TEXT` prints them; `--clear` removes them |
| `munin end RUN` | end a crashed or silent run as killed |
| `munin rm RUN...` | delete runs and every file they hold |
| `munin gc` | remove blobs no version names and crashed writers' work; prints `removed N` |
| `munin check` | a `path problem` line per break of the format; exit 1 if any |
| `munin root` | the store's directory |
| `munin` | the dashboard of the store's runs |
| `munin watch [RUN]` | one run as it trains; without `RUN`, the newest run still going, else the newest run |

`runs` status names are `running`, `silent`, `crashed`, `finished`, `failed`,
`killed`. `show` prints no line for an absent name, end or note, so
`grep ^ended` tells whether a run ended.

## Queries

```sh
munin runs --experiment lorenz --status failed
munin runs --tag baseline --tag gpu
munin show "$(munin runs | head -1 | cut -f1)"

# the least value of a minimize metric and the step it happened at
run=$(munin runs --experiment lorenz | head -1 | cut -f1)
munin metrics "$run" val/loss | sort -t "$(printf '\t')" -k3,3g | head -1

# a sweep, its trials, and their comparison
parent=$(munin runs --experiment lorenz | awk -F'\t' '$5 == "sweep-lr" { print $1 }')
ids=$(munin runs --parent "$parent" | cut -f1)
munin compare $ids

# a live run's newest sample
munin metrics "$run" train/loss | tail -1
```

`compare` makes one column per argument, so pass the ids in the order you
want to read. A run that did not log a metric leaves its cell `-`. The best
of a row is starred only when every run that has the key shares one goal.

## The dashboard

`munin` with no command, on a terminal, shows the store: runs pane, centre,
run pane. `munin watch [RUN]` shows one run. Both read the store again once a
second and never write it. Without a terminal, `munin` prints its help,
`munin watch` fails, and a script that called them by mistake stops.

Keys, one meaning at every depth:

| Key | Does |
| --- | --- |
| `up` `down`, `k` `j` | move the run cursor; the strip at the strips depth |
| `left` `right`, `h` `l` | the metric; the step cursor at zoom, shift for ten |
| `[` `]` | previous, next run |
| `g` `G` | first, last run |
| `Enter` | one depth deeper: charts, strips, zoom; open a folded experiment |
| `Esc` | one depth back; at charts, clear overlays, then filter, then grey runs |
| `Space` | overlay the cursor's run, or take it off; four at most |
| `r` | the experiment's other runs in grey |
| `c` | clear the overlays |
| `s` | smoothing on or off |
| `y` | the other y scale for this metric |
| `+` `-` | narrow, widen the steps shown about zoom's cursor |
| `/` | filter the runs |
| `o` | order the runs: started, best, name |
| `e` | fold or open the cursor's experiment |
| `i` | everything about the run |
| `1` `3` | show or hide the runs pane, the run pane |
| `?` | list the keys |
| `q`, `Ctrl+C` | quit, except in the filter, where `q` is typed |

A run's state word comes from its status and from when its samples arrived,
on the dashboard's clock: `starting`, `running`, `live`, `stalled`,
`writing`, `silent`, `crashed`, `finished`, `failed`, `killed`. A glyph says
the same: `●` live, `○` running, `◌` stalled or silent, `✓` finished, `■`
killed, `✗` failed or crashed, `×` a run with a non-finite sample.

The filter after `/` matches the name, id, experiment, tag, state word, or a
param written `key=value`; `lr<0.002` and `lr>2e-3` compare a numeric param.
`i` shows params, notes, tags, source, metrics, used artifacts, files,
resumes and damaged lines. The key metric of an experiment is the first goal
metric outside `train/` whose key ends in `loss`, else the first goal metric
outside `train/`, else the first goal metric, else the first metric.

## The store's files

```text
<root>/
  runs/<id>/events.jsonl            the run's log, one JSON record a line
  runs/<id>/tags/<digest>           one tag file per tag, named by its digest
  runs/<id>/notes                   UTF-8 notes, no trailing newline required
  runs/<id>/git.diff                uncommitted changes at its start
  runs/<id>/stats.json              the keys' statistics, once ended
  runs/<id>/writer                  the latest writer's identities
  runs/<id>/files/<key>/<file>      a file the run logged
  artifacts/<name>/<n>.json         version <n> of the artifact
  artifacts/<name>/aliases/<alias>  the version the alias names
  blobs/blake2b-256/<hex>           the content of versions
  lock                              orders collection after publication
  deletions                         orders deletions of runs
```

`<id>` is `YYYYMMDDTHHMMSSZ-xxxxxxxx`: the start time and 32 random bits.
Nothing in munin parses an id; compare them, never order by their contents.

### A run's start and end

```sh
cd "$(munin root)"
ID=$(munin runs | head -1 | cut -f1)

head -n 1 runs/$ID/events.jsonl | jq -c .params
tail -n 1 runs/$ID/events.jsonl | jq -r 'select(.type == "end") | .status'
```

A log without an end belongs to a run that is running, silent or crashed. On
Linux the writer holds an exclusive `flock` on the log for its life, so on
the machine and boot that wrote it:

```sh
flock -s -n runs/$ID/events.jsonl true && echo crashed || echo running
```

A child forked without `exec` shares the lock, so `running` proves a process
holds the lock, not that the program that started the run is alive.

### A metric's series

```sh
jq -r --arg k val/loss 'select(.type == "metrics" and (.values | has($k)))
  | [.step, .values[$k]] | @tsv' runs/$ID/events.jsonl
```

A key holds `/`, so it goes through `--arg`; `jq` cannot read `val/loss` as a
path.

A resumed run holds orphans: samples above the step of a later `resume`
record. This drops them as it meets each resume:

```sh
jq -rn --arg k val/loss '
  reduce inputs as $r ([];
    if $r.type == "resume" then map(select(.[0] <= $r.step))
    elif $r.type == "metrics" and ($r.values | has($k))
    then . + [[$r.step, $r.values[$k]]]
    else . end)
  | .[] | @tsv' runs/$ID/events.jsonl
```

A NaN or an infinity is an object, `{"float":"NaN"}`, since JSON has no
number for it. `@tsv` prints that object as JSON text.

### Following a live run

A writer appends whole lines and a reader stops at the first line that is not
whole, so `tail -f` and `jq` follow a run as it writes:

```sh
tail -n +1 -f runs/$ID/events.jsonl \
  | jq -c 'select(.type == "metrics") | [.step, .values]'
```

For a metric key, use `--arg k` and `[$r.step, $r.values[$k]]` as above.

### Statistics of an ended run

`stats.json` is a hint. It is valid when its `format` and `rules` match the
reader's and its `length` is the log's length in bytes; otherwise fold the
log.

```sh
[ "$(jq .length runs/$ID/stats.json)" -eq "$(wc -c < runs/$ID/events.jsonl)" ] \
  && jq -c '.keys["val/loss"].finite.least' runs/$ID/stats.json
```

### Tags and notes

```sh
for f in runs/$ID/tags/*; do cat "$f"; echo; done
cat runs/$ID/notes 2>/dev/null
grep -lx baseline runs/*/tags/*
```

A tag is the file's bytes with no line feed; the file's name is the BLAKE2b-256
digest of those bytes. Any process may add a tag by writing a `.`-named file
and renaming it, which is what `munin tag` does.

### Artifacts and blobs

```sh
cat artifacts/model/aliases/best
jq -r .digest artifacts/model/2.json
digest=$(jq -r .digest artifacts/model/2.json)
b2sum -l 256 "blobs/blake2b-256/${digest#blake2b-256:}"
```

A version's content is the blob named by its digest. A blob present under its
name is complete, so `b2sum` of a file blob gives its name back. A directory
blob's digest is that of a listing of its entries; a program reads
`Artifact.path` rather than reconstructing it. A deleted version's file holds
`"deleted":true`; the latest version is the highest number not deleted.

### Recovering the code

```sh
git checkout "$(head -n 1 runs/$ID/events.jsonl | jq -r .provenance.git.commit)"
git apply runs/$ID/git.diff
```

`git.diff` exists only when the work tree was dirty at `start`; it is
`git diff --binary HEAD` run in the run's cwd, so it restores the tracked
files the run started from. Untracked files are not kept.

## Maintenance

- `munin rm RUN...` deletes runs and their files. It refuses a run whose
  writer holds the lock and a silent run, reports why, and goes on with the
  rest. Versions a deleted run produced stay, still naming it as producer.
- `munin end RUN` resumes a crashed or silent run at a step that orphans
  nothing and finishes it as killed. For a silent run, trust that its writer
  is gone: its lock cannot be probed here.
- `munin gc` removes blobs no version names, `.`-named writer work, and what
  a cut-short deletion left. It waits for publications in progress, and a
  publication that starts waits for it. Do not run it during a sweep.
- `munin check` reads every log, tag file and blob. A `stats.json` that
  matches its log's length but whose statistics differ is reported; a damaged
  line is reported; a blob whose bytes do not match its digest is reported.
- A store is a directory. Do not commit it. Set `RAVEN_TRACKING_DIR` per
  project or accept the shared default store, and never point a test at
  either.
