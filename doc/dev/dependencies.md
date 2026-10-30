# Dependencies

Dune package management is the only package manager. The root `dune.lock/` is
committed; `dune build` uses it without solving.

## Pins

External sources are pinned in `dune-project`:

- **Raven** — the whole Raven repository at a full commit SHA. The pin lists
  every Raven package this repo and its downstream projects use (`compress`,
  `nx`, `rune`, `tolk`, `vega`, `jera`, `hugin`, `rig`, `ppx_ptree`, `kaun`).
  Raven is not a package of this monorepo.
- **windtrap** and **ppx_windtrap** — the commit Raven's own CI uses, so our
  tests and Raven's agree.
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

## Updating the lock

Use the wrapper, not `dune pkg lock` directly:

```sh
./dev lock
```

`scripts/relock.sh` runs `dune pkg lock` and then `scripts/patch-lock.py`.
The patch is temporary and exists for two upstream issues:

1. Raven main's private test libraries under `packages/hugin/test` are built by
   `@install` even when Dune builds another package with `-p`. One of them
   copies `packages/hugin/lib/marks.{ml,mli}` from outside its package and
   fails. Building `<pkg>.install` avoids the private tests.
2. GitHub's archive endpoint returns 504 for some Jane Street tarballs. The
   patch points those sources at codeload; the lock's checksum still verifies.

Remove the patch when Raven main builds with `@install` and GitHub serves the
archives. Until then, review the lock diff, keep it in a dedicated
`chore(deps):` commit, and run `./dev check` before merging.

## Updating Raven

1. Change the Raven SHA in `dune-project`.
2. Run `./dev lock`.
3. Build and test: `dune build @all @install && dune runtest`.
4. Fix API changes in the packages.
5. Commit the pin, the lock diff, and the code changes together.

Do not pin a branch or a moving tag. Do not vendor Raven.
