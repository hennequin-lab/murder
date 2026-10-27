# Worktrees

A git worktree is a second checkout of the same repository, on its own branch,
with its own index and `_build`. Use one per person or agent working at the
same time.

```text
~/code/
|-- murder/                  main checkout, usually on main
`-- murder-wt/
    |-- geom-motor/          worktree, branch wt/geom-motor
    `-- symo-curvature/      worktree, branch wt/symo-curvature
```

## Create and remove

```sh
./dev wt new geom-motor                     # wt/geom-motor from main
./dev wt new geom-motor fix/motor-scale     # explicit branch name
./dev wt new geom-motor fix/motor-scale main  # name, branch, base
./dev wt list
./dev wt rm geom-motor                      # refuses a dirty checkout
```

`new` creates the worktree under `../murder-wt` and runs `./dev setup` in it:
git hooks and the `.cmt` files. Set
`MURDER_WT_ROOT` to put worktrees elsewhere. Plain git works too:

```sh
git worktree add ../murder-wt/geom-motor -b wt/geom-motor main
```

## Keep worktrees outside the repository

Never create a worktree inside the checkout, including under `.agents/`. Dune
reads every `dune-project` below the workspace root, so a nested worktree
breaks both checkouts:

- Visible directory: the outer build loads both projects and stops with
  `The package "cmdargs" is defined more than once`.
- Hidden directory (`.agents/nested`): the outer scan skips it, but `dune` run
  inside it walks up to the outer root and builds the outer checkout:
  `Don't know about directory .agents/nested specified on the command line`.

Either way the checkout is unusable: the visible case fails loudly, the
hidden case builds the outer tree. `./dev wt new` and `MURDER_WT_ROOT` keep
worktrees out of the tree.

## Build directory and cache

Each worktree has its own `_build`. That is what allows parallel builds. Do not
share, symlink, or point `--build-dir` at another checkout's `_build`: Dune
locks a build directory, so the two builds serialize, and `dune promote` could
write into the wrong tree.

Rebuilds are cheap because the Dune cache is machine-wide. `dune-workspace`
sets `(cache enabled)`; entries live in `~/.cache/dune` and are keyed by rule
inputs, not by path, so a rule built in one checkout is reused in another.
When the cache and `_build` are on the same filesystem, Dune restores files by
hardlink (its default storage mode), so a restored `_build` shares blocks with
the cache instead of duplicating them.

Measured here on 12 cores with a warm cache:

| Scenario | Wall time |
| --- | --- |
| `dune build @ci`, cold cache | about 2.5 min |
| `dune build @ci`, fresh `_build`, warm cache | about 1 s |
| no-op `dune build` | about 0.1 s |

The cold number is dominated by the locked OCaml compiler
(`ocaml-base-compiler` from `dune.lock`), which every `_build` needs. The
default cache mode, `enabled-except-user-rules`, skips the rules that build it,
so a new worktree would pay for it again; `(cache enabled)` caches those rules
as well, together with this repository's formatting, lint, and test rules.
A cache hit restores the recorded result, so a cached `@ci` does not re-run
tests. Dune re-runs a rule when any input changes, so tests still run on the
code you edit. Use `DUNE_CACHE=disabled <command>` to force a real run.

Package sources are cache entries too, including the git pins. A new worktree
on the same lock builds with no network access once the cache is warm.

`dune cache size` reports cache bytes that no build directory hardlinks; a
small number means the cache and the worktrees share their files. The cache
grows with the repository; trim it with `dune cache trim --size=10GB` when it
is large.

Consequences of hardlinked outputs:

- Never edit files under `_build`. They are hardlinked into the cache, so an
  in-place edit corrupts the entry for every checkout. Use `dune promote` or
  `dune fmt`.
- Deleting a worktree deletes its links; the cache keeps the content.
- Do not copy `_build` between checkouts. Build, and let the cache restore it.

`dune clean` or `rm -rf _build` is allowed in a worktree you own. It is still
forbidden in a shared checkout.

## Parallel builds

- One writer per worktree: one agent, one branch, no checkout switching in a
  worktree someone else owns.
- Dune uses every core by default. With several worktrees building, cap each
  one: `DUNE_JOBS=4 dune build @ci`, or `dune build -j 4`.
- Iterate on the package alias, for example
  `dune build @packages/geom/runtest`, and leave `./dev check` for the gate.
- Create worktrees one at a time. The first build on a machine warms the
  cache; concurrent first builds each compile the same dependencies.

## Agent harnesses

A harness that allocates worktrees for agents must keep them outside the
repository: pi-subagents defaults to `{parent of the checkout}/worktrees` and
rejects a base directory inside the checkout. Harness worktrees do not run
`./dev setup`, so run it there before the first `./dev check`, or wire it into
the harness setup hook where one exists.

## Hooks

`core.hooksPath` lives in the shared git configuration, so one `./dev setup`
installs the hooks for every worktree; each worktree has its own `.githooks`
checkout.

## Dependencies and the lock

The lock belongs to the branch. A worktree on a branch that changes
`dune.lock` solves and builds its own package set, and downloads what the cache
does not have, while the other worktrees are unaffected. Run `./dev lock`
inside that worktree and commit the lock diff there.

## Git

Worktrees share one object database and one set of refs; indexes and working
trees are separate. A commit made in a worktree lands in the same repository,
and the human pushes it from either checkout.

- One branch per worktree. `git worktree add` refuses a branch that another
  worktree has checked out; do not work around that with a detached HEAD.
- After the work is merged, `./dev wt rm <name>` removes the worktree and
  leaves the branch. Delete it with `git branch -d`.
- If a worktree directory was deleted by hand, `./dev wt prune` drops the
  record.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `The package "x" is defined more than once` | worktree inside the repo | move it outside, `git worktree remove --force` |
| `Don't know about directory ...` | hidden worktree inside the repo | same |
| `is already checked out at ...` | branch used by another worktree | use another branch, or remove that worktree |
| build waits, then runs alone | two Dune builds share one `_build` | give each worktree its own `_build` |
| build looks stuck | a Dune server is busy | `dune rpc ping`, then read the first error |
