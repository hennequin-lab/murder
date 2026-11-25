# Dependencies

Dune package management is the only package manager. The root `dune.lock/` is
committed; `dune build` uses it without solving.

## Pins

External sources are pinned in `dune-project`:

- **Raven** — the whole Raven repository at a full commit SHA. The pin lists
  every Raven package this repo and its downstream projects use (`lz`,
  `nx`, `rune`, `tolk`, `vega`, `jera`, `hugin`, `rig`, `munin`, `quill`,
  `ppx_ptree`, `kaun`). Raven is not a package of this monorepo.
- **windtrap** and **ppx_windtrap** — windtrap's `v0.3.0` release, the
  version just ahead of the commit Raven's CI uses. The release is on opam,
  but Raven and litany pin windtrap themselves, and a pin inherited through a
  pinned project wins over the package repositories, so this repo pins the
  release too and names it. Drop this pin, and Raven's and litany's, to take
  the opam release.
- **litany** — a full SHA; it is not on opam.

Pins belong in `dune-project`, not `dune-workspace`, because Dune resolves the
pins of a pinned project recursively. A project that pins murder therefore
builds against the same Raven commit, windtrap and litany without declaring
its own pins. `dune-workspace` only holds workspace settings: the cache, the
lock directory, and the environment.

A downstream project that needs a Raven package missing from the list above
must add it to the Raven pin in `dune-project`; packages not named there come
from the package repositories. The same applies to murder packages a project
wants to use.

## Tools in the checkout

`murder-dev` depends on the two Raven applications the lab runs from a
checkout: `munin`, the experiment tracker and dashboard, and `quill`, the
notebook. Dune package management builds a locked package's executables into
`_build/_private/<context>/.pkg/<pkg>/target/bin` and puts those directories
on `PATH` when it runs a program, so no extra setup is needed:

```sh
dune exec munin runs          # the store's runs, newest first
dune exec quill notebook.md   # open a notebook
```

`dune exec` parses its own options after the program name too, so put `--`
before the program's flags:

```sh
dune exec -- munin runs --root DIR
```

Both are applications, not libraries. A package that wants munin's library in
its own code adds `munin` to its `depends` and relocks, and an ongoing project
does the same in its own repository. Because they are dependencies of
`murder-dev`, `@check` and `@all @install` build them and their dependencies
(mosaic, matrix, toffee, cmarkit, tree-sitter, ...), in CI as well as locally;
the shared Dune cache makes a later checkout restore them.

## Updating the lock

Use the wrapper, not `dune pkg lock` directly:

```sh
./dev lock
```

`scripts/relock.sh` runs `dune pkg lock`. The temporary Raven-main lock patch
is gone: Raven builds with `@install` and this lock resolves without it.
Review the lock diff, keep it in a dedicated `chore(deps):` commit, and run
`./dev check` before merging.

## Updating Raven

1. Change the Raven SHA in `dune-project`.
2. Run `./dev lock`.
3. Build and test: `dune build @all @install && dune runtest`.
4. Fix API changes in the packages.
5. Commit the pin, the lock diff, and the code changes together.

Do not pin a branch or a moving tag. Do not vendor Raven.
