# Contributing

## Set up

- Install Dune 3.24.2 or newer.
- Clone the repo and run `./dev setup`. This installs the git hooks, downloads
  the locked dependencies, and builds the `.cmt` files.
- First command: `dune exec getting-started/01-nx/ex01_arrays.exe`.

## Daily loop

- `./dev fmt` before committing: formats OCaml and dune files.
- `./dev check` before pushing: formatting, lint, and tests.
- `dune runtest` while iterating; `dune build @packages/<pkg>/runtest` for one
  package.

Direct equivalents:

```sh
dune build @fmt            # check formatting
dune fmt                   # apply formatting
dune build @lint           # in-build lint
dune exec -- litany check  # full lint
dune build @ci             # fmt + lint + tests
```

## Parallel work

Use a worktree when another person or agent is using the checkout:

```sh
./dev wt new <name>        # ../murder-wt/<name> on branch wt/<name>
./dev wt list
./dev wt rm <name>         # after the branch is merged
```

Worktrees live outside the repository, each with its own `_build`; the shared
Dune cache makes the first build in a new worktree cheap. Never create a
worktree inside the tree. See `doc/dev/worktrees.md`.

## Packages and applications

This repo is for stable shared libraries. Applications and ongoing projects
live in their own repositories and depend on a pinned murder commit or a
released opam version. See `doc/dev/dependencies.md` for the pin and lock
workflow.

## Formatting

- One root `.ocamlformat`; `dune build @fmt` is the check, `dune fmt` applies.
- Do not reformat unrelated code in a feature PR.
- A mass reformat gets a `chore(<package>): Format ...` commit and its hash in
  `.git-blame-ignore-revs`.

## Linting

- `dune build @lint` for editor diagnostics.
- `dune exec -- litany check` for the full roster.
- The gate is aggressive on purpose: `(select all)` stable rules. Add
  restriction rules by exact name when the team adopts a house policy.
- Suppress one finding only with a reason: `[@litany.allow "..."]`.
  `unused-allow` is itself a finding, so suppressions cannot rot.

## Tests

- Every package has a windtrap suite under `test/`.
- State contracts, not implementation; use `prop`/`stateful` where they fit.
- Bug fixes include a test that failed before the fix.
- Keep a package's `runtest` within about two minutes warm. No network in tests.

## Dependencies

- Dune package management is the only package manager.
- Add the dependency to the package's `dune-project` stanza, run `./dev lock`,
  and commit the lock diff with the change.
- Raven is pinned in `dune-workspace` to its GitHub repository. Move the SHA
  deliberately and relock.
- Pin git dependencies to a full SHA; never a moving branch. Do not vendor
  Raven.

## Commits

- Subject: `type(scope): Imperative summary`; scope is the package or area.
- The `commit-msg` hook checks it locally; CI checks the PR title.
- Body: blank line, wrapped at 72, says why. No process narration.
- One logical change per commit and PR.

## Pull requests

- PR title follows the same convention; squash merge uses it.
- Fill the PR checklist; keep changes small and reviewable.
- Public API changes need `.mli` docs and a `CHANGES.md` entry.

## Adding a package

- `./dev new <name>`, add the `(package ...)` stanza, fill `lib/`, `test/`,
  `doc/`.
- Internal package names are bare. Add a `murder-` public prefix only at opam
  publication.

## Getting unstuck

- Read the first error, not the last.
- `rm -rf _build` for cache weirdness; regenerate the lock for lock weirdness.
- Ask in the lab channel with the command, the full error, and the commit.
