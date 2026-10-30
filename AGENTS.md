# AGENTS.md

Murder is the Hennequin lab's monorepo for stable Raven-based OCaml libraries,
plus a getting-started course. Applications and ongoing projects live in their
own repositories. Most work here happens in `packages/<name>/` or
`getting-started/`.

Read `CONTRIBUTING.md` for the human workflow and `doc/dev/` for the design,
testing, CI, dependency, and release notes. This file is the short version for
coding assistants.

## Hard rules

These protect other people's work and the reproducibility of the repo. Breaking
one causes real damage.

- NEVER push, force-push, amend published commits, or rewrite history. The
  human pushes. Do not commit unless explicitly asked.
- NEVER hand-edit `dune.lock/` or anything under `_build/`. `dune.lock` is
  written by `./dev lock` (which also runs the Raven-main workaround); `_build`
  is Dune's output.
- NEVER relock or delete `dune.lock` as a side effect. Relock only when the
  change adds or updates a dependency or pin, and include the lock diff.
- NEVER disable, bypass, or weaken `@fmt`, `@lint`, tests, or CI to make a
  change pass. Fix the cause, or suppress one finding with a reason.
- NEVER add a dependency without asking. When approved, use Dune package
  management and pin git dependencies to a full commit SHA.
- NEVER kill a dune or other process you did not start. Kill by PID, never by
  pattern (`pkill`, `killall`). Do not start a second watch server in a
  checkout that already has one.
- NEVER run `dune clean` in a shared checkout. If `_build` is wedged, confirm no
  dune process is running and remove only your own `_build`.
- NEVER exhaust a system resource (threads, processes, descriptors, memory,
  disk) in a test or command. Put limits on a child process.
- NEVER silence a warning or prefix a variable with `_` to hide it. A warning
  is a bug in the change.
- No secrets, credentials, private data, or large datasets in the repo.
- This repo is for stable libraries. Do not add application code or an ongoing
  project to `packages/`; that work belongs in its own repository.
- Do not run `git reset --hard`, `git checkout -- .`, `git clean`, or
  `git stash` in a shared checkout. Use a branch or worktree, and ask before
  destructive git.
- If a repository-wide decision changes, update `CONTRIBUTING.md` or the
  relevant `doc/dev/` note in the same change. Do not let the repo and its
  documentation drift.
- Never create a worktree inside the repository, including under `.agents/`.
  Dune reads every `dune-project` below the root, so a nested worktree breaks
  both checkouts. Use `./dev wt new` and keep worktrees in `../murder-wt/`.
- One agent per worktree. Never switch branches in, or remove, a worktree you
  do not own. Never share or symlink `_build` between checkouts, and never edit
  files under `_build`: they are hardlinked to the shared Dune cache.

## Layout

| Path               | Meaning                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------- |
| `packages/<name>/` | Shared libraries. A package's main library is `lib/`; tests are `test/`; docs are `doc/`. |
| `getting-started/` | Tested Raven/Nx/Rune/Vega course and capstone.                                            |
| `doc/dev/`         | Design, testing, CI, dependency, and release notes.                                       |
| `scripts/`         | Small repo scripts.                                                                       |
| `dune.lock/`       | Committed, generated dependency solution.                                                 |
| `litany`           | Linter configuration.                                                                     |
| `.githooks/`       | `commit-msg`, `pre-commit`, `pre-push`.                                                   |

- Internal package names are bare (`foo`, not `murder-foo`).
- Do not create an umbrella `murder` package while the repo is used as a cloned
  workspace.
- This repo is libraries only. It does not publish to opam. A separate project
  repo pins a murder commit and owns its lock, tests, and CI. Do not vendor
  these libraries into a project repo.

## Commands

- `./dev setup` — install hooks and dev tools, build `.cmt`.
- `./dev fmt` — apply formatting (`dune fmt`).
- `./dev check` — `dune build @ci`, then the full litany roster.
- `./dev test` — `dune runtest`.
- `./dev lint` — `dune exec -- litany check`.
- `./dev lock` — `dune pkg lock` plus the Raven-main lock patch.
- `./dev new <name>` — scaffold `packages/<name>/`.
- `./dev wt` — create, list, and remove worktrees (see Parallel work).
- One package's tests: `dune build @packages/<pkg>/runtest`.
- One tutorial: `dune exec getting-started/01-nx/ex01_arrays.exe`.

Use `timeout` for commands that can hang (tests, package solving). If a build
looks stuck, check `dune rpc ping` before killing anything.

## Parallel work

Several agents can work at once, one per worktree.

- `./dev wt new <name>` creates `../murder-wt/<name>` on branch `wt/<name>`
  from `main` and runs `./dev setup` there. `./dev wt list` and
  `./dev wt rm <name>` manage the rest.
- Each worktree has its own `_build` and index. The Dune cache in `~/.cache/dune`
  is shared: the first build on a machine pays for the compiler and
  dependencies, later worktrees restore them in seconds.
- Cap concurrent builds with `DUNE_JOBS=4`. Iterate on
  `dune build @packages/<pkg>/runtest`; leave `./dev check` for the gate.
- Full guide: `doc/dev/worktrees.md`.

## Code and docs

- Modules and variants are `Capitalized_snake_case`; values are `snake_case`.
- Public API lives in `.mli` files. Doc comments live in `.mli` and start
  `(** [f x] ...`. State invariants and preconditions; use `{[ ... ]}` for code.
- Keep code shallow: early returns, no arrow-shaped nesting, blank lines
  between logical blocks. Short function names; a variant or labelled argument
  when an argument selects behaviour.
- Name meaningful or recurring values as constants; keep a self-explanatory
  one-off inline. A value from a spec is always named.
- Hot paths allocate outside loops, prefer loops to closures, and use
  `unsafe_` accessors only after the bounds are already checked.
- Comments explain what a non-obvious block does and why, with an example when
  it helps. Do not narrate the code's arrangement. ASCII diagrams for whole
  systems.
- Error messages name the operation, the inputs, and the specific problem, with
  a hint when the fix is not obvious.
- User-facing docs explain behaviour from first principles. Do not compare to
  other numerical or machine-learning libraries.
- Prose in docs, comments, commits, and replies: short, factual, no
  superlatives, no rhetorical triples, no punchy closing lines. State bad news
  plainly.
- Do not rename during a tidy unless the new name is clearly better.
- A public API change needs tests, `.mli` docs, and a `CHANGES.md` entry.

## Tests

Tests use windtrap. A suite is an ordinary executable that `dune runtest` builds
and runs.

- State what the `.mli` promises, through the public interface. Do not test
  internals or read dune files and source text.
- Use the strongest tool the claim allows: `prop` for a law, `stateful` against
  a model for state across calls, `cases` for values the spec states, `expect`
  for long text.
- Expected values come from the spec. A value learned from the code is a
  baseline, written as `expect`, and says so.
- Draw inputs that break the code: empty and one-element shapes, bounds,
  every dtype, strided and broadcast views, NaN, infinities, int extremes.
  Cover the cases a law needs, often enough that every seed reaches them.
- Assert with verbs that print data (`equal`, `less ~than`, `raises`) and floats
  with a stated tolerance. Never `is_true`.
- Write a test where a user would get a wrong answer or a crash. For a bug,
  write the test first, watch it fail, fix, watch it pass.
- `test/` mirrors `lib/`. Shared helpers live in `test/support/`; goldens and
  generators live at the level that needs them.
- A package's `runtest` should finish within about 2 minutes warm. Fix a slow
  test at its cause; do not hide it behind a tag or a scheduled job.
- No network, wall clock, or hardware in tests unless the test skips cleanly
  without it. Do not fake a device. Keep data synthetic or tiny.
- Set `WINDTRAP_TIMEOUT` so a hung test fails. CI uses a higher property count;
  a nightly run may use more.
- Do not edit goldens by hand. Regenerate them with their generator.
- Tutorials are tests: mdx snippets and example executables must keep working.

## Dependencies and lock

- Dune package management is the only package manager. Do not use opam.
- Add a dependency to the package's `dune-project`, run `./dev lock`, and
  commit the lock diff with the change.
- Pin a git dependency to a full SHA and name the packages it provides. Never
  pin a moving branch. Remove version constraints on a pinned package while the
  pin is active. Declare pins in `dune-project`, not `dune-workspace`: Dune
  resolves them recursively, so projects that pin murder inherit them.
- Never hand-edit the lock. If a package is missing, relock; if the lock
  conflicts, take one side, relock, run `./dev check`, and commit the result.
- Keep dependencies few. A dependency is a contract with the lab, not a
  convenience.

## Commits and PRs

Subject:

```text
type(scope): Imperative summary
```

- Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `bench`, `chore`.
- Scope: a package name, or `repo`, `ci`, `docs`, `deps`, `tutorials`,
  `getting-started`, `dev`. One scope per subject.
- Capitalised summary, no trailing period, at most 72 characters; about 50 is
  better. It completes "If applied, this commit will ...".
- Blank line, then a body wrapped at 72 that says why: what was wrong, why it
  matters, why this approach, what it costs. Describe the change, never the
  process (no reviews, agents, phases, attempts).
- One logical change per commit and PR. Do not reformat unrelated code.
- A user-visible change gets a `CHANGES.md` entry under `Unreleased`, in the
  package's section, 1–3 lines, with identifiers in backticks.
- The `commit-msg` hook checks the subject locally. CI checks the PR title;
  squash merge uses it as the commit on `main`. Do not bypass either.
- If asked to commit: run `./dev fmt` and `./dev check` first, stage only the
  intended files, and include a lock diff only when dependencies changed.

## Reviewing

- Read `CONTRIBUTING.md`, the package's `README.md`, and the files the change
  touches before reviewing.
- A change to a public `.mli`, library behaviour, or a contract needs a review
  by someone who did not write it.
- Check tests, docs, error messages, dependency/lock changes, and unrelated
  formatting. Generated files and `_build` do not belong in a diff.
- Address every finding or answer it with evidence from the code, its callers,
  and its history. Do not dismiss a finding because it is inconvenient.

## When stuck

- Read the first error, not the last. Reproduce with the smallest command that
  shows it.
- For a numerical bug, print shapes, dtypes, and a small input. Check broadcasting
  and axis conventions before changing the algorithm.
- For a CI failure, run `./dev check` and see `doc/dev/ci.md`.
- Ask with the command, the full error, and the commit. Do not leave a
  background job running when you end a turn.

## Raven for numerical code

- Prefer to `open Nx` globally.
- Prefer to use infix operators for Nx maths, as they are more legible for humans; for this, use local `let open Infix in (...)`.
- Prefer to use `ppx_ptree` rather than write you own `walk` functions.
