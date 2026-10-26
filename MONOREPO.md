# Murder — Monorepo Plan

> **Status:** draft for discussion. This is the living plan for the Hennequin lab's
> monorepo of stable Raven-based libraries. Applications and ongoing projects
> live in their own repositories. Decisions are marked **recommended** where I
> have a clear opinion; open questions are collected at the end.
>
> **Name:** `murder` — a collective noun for crows, and a collection of
> Raven-based libraries.

---

## 0. TL;DR

| Question | Recommendation |
| --- | --- |
| Scope | **Stable libraries only** plus `getting-started/`. Applications and ongoing projects live in separate repositories. |
| One Dune project or several? | **One root `dune-project`** for all stable lab libraries. Ongoing projects live in separate repositories, not in this workspace. |
| Package manager | **Dune package management for everything.** No opam switch, no `opam install` in the normal workflow. |
| Commit `dune.lock/`? | **Yes.** Dune 3.24 solves one lock for Linux/macOS on x86_64/arm64. Commit it and treat it as generated. |
| Dune/OCaml versions | **Dune 3.24.2+**, **OCaml 5.5.x** (the lock pins 5.5.2; litany requires 5.5). |
| Package naming | **Bare library names inside the workspace.** Add a `murder-` public name only when a package is published to opam; no umbrella package until there are external consumers. |
| Raven dependency | Start with the **opam release `1.0.0~alpha3`**, frozen by `dune.lock`. Pin a Raven commit only when a specific unreleased fix is needed. |
| Tests | **Windtrap** for every package; `:with-test` dependencies; property/stateful/expect tests as appropriate. |
| Lint | **Litany** pinned to a commit, run as `@lint` locally and as a full `litany check` in CI. Start **aggressive**: `(select all)` stable rules; adopt restriction rules one at a time. |
| Formatting | **`.ocamlformat` pinned to 0.28.1**, `dune build @fmt` in CI, `./dev fmt` locally, opt-in hooks. |
| CI | **Format + lint + Linux/macOS build/test** on every PR; branch protection on `main`. |
| Onboarding | **`getting-started/`** with four tested chapters: Nx, Rune, Vega, capstone. |
| Coverage/mutation | Report first; gate only after the suite stabilizes. Mutation is a periodic reading list, not a PR gate. |

The rest of this document explains the choices, gives a concrete skeleton, and
lays out a phased roadmap.

### 0.1 Options considered

| Decision | Options | Why the recommendation |
| --- | --- | --- |
| Repository shape | One root Dune project; one project per package; Bazel/Nix | One root project gives one lock, one build graph, and one command to learn. Per-package projects multiply solving and CI. Bazel/Nix add a second build system the lab does not need for pure OCaml. |
| Package naming | Bare names; `murder-` public prefix at publication; umbrella package | Bare names keep internal code and package metadata short; the workspace is the namespace. A public prefix avoids opam collisions when a library leaves the lab. An umbrella package only helps external `opam install` users; it is pure maintenance while the monorepo is cloned as a workspace. |
| Lock directory | Commit; ignore; commit one per platform | Commit: Dune 3.24 solves Linux and macOS on x86_64/arm64 in one lock. Ignoring it brings back solver drift for Raven/windtrap/litany. Per-platform locks are only needed if Windows or an unusual distro enters scope. |
| Raven source | Released `1.0.0~alpha3`; pin `main`; vendor | Release + lock gives a stable API and a reproducible build. Pin a commit only for a specific unreleased fix; vendoring creates a maintenance fork. |
| CI platforms | Linux only; Linux + macOS; self-hosted | Linux gives fast PR feedback; macOS catches platform-specific breakage. Self-hosted is not worth the maintenance for this team size. Demote macOS to `main`/weekly if Actions minutes become a problem. |
| Lint policy | Litany default; `all`; all + restriction; no linter | `all` catches correctness/suspicious/perf/style/pedantic on new code without the churn of nursery rules. Restriction rules encode house policy and are adopted by exact name. Default is too timid for a new repo with no legacy. |
| Getting started | Link to Raven docs; written guide; tested tutorial | A tested tutorial is the only one that stays correct. Links alone do not teach the lab's idioms; untested prose rots. |
| Hooks | Mandatory; opt-in; none | Opt-in via `./dev setup`, with CI as the real gate. Mandatory hooks are bypassed and create false confidence; none lets the tree drift. |

---

## 1. Goals and non-goals

### Goals

1. **Low-friction onboarding.** A new student should go from a clean laptop to a
   running Raven example in under 30 minutes: install Dune, clone, build.
2. **Reproducibility.** The same commit builds the same dependency graph on every
   lab machine and in CI. Papers and internal results can point at a commit.
3. **A clean tree by default.** Formatting and linting are automated and
   enforced in CI, not remembered.
4. **One obvious way to add a library.** A package template, a naming
   convention, and a short checklist.
5. **Tests that teach.** Windtrap suites state contracts; the getting-started
   docs are executable and tested, so they cannot silently rot.
6. **A CI that catches regressions without becoming a second job.** Fast,
   deterministic, cached, and easy to reproduce locally.

### Non-goals (for now)

- Windows support. Linux and macOS only; `.gitattributes` keeps line endings
  sane if someone opens the repo on Windows.
- Bazel, Nix, Buck, or a custom build system. Dune is the interface.
- Independent release trains for every package. All packages move together at
  one version until there is a reason not to.
- Coverage or mutation gates on day one. They are useful, but they add noise
  before the test suite has settled.
- A website, documentation deployment, or opam publication on day one. Plan for
  them, but do not block the skeleton on them.

---

## 2. Guiding principles

- **Dune is the interface.** Build, test, format, lint, docs, and dependencies
  are all reached through `dune`. A thin `./dev` script may wrap common
  sequences, but it should not hide anything important.
- **Pin and lock, then upgrade deliberately.** A dependency change is a
  reviewable event, not something that happens implicitly on every build.
- **Generated files are generated.** Never hand-edit `dune.lock/` or
  `opam/*.opam`; change the source of truth and regenerate.
- **The local loop is fast; CI enforces what humans forget.** Formatting and
  linting should take seconds warm. Full tests can take minutes.
- **Tests state contracts, not implementation.** Prefer windtrap's `prop`,
  `stateful`, and `equal`/`raises` over assertions on internals.
- **Small packages with explicit interfaces.** Public modules have `.mli`
  files. Avoid a `utils` package; if shared code emerges, give it a domain
  name.
- **Onboarding is a product.** The getting-started directory, README, and
  CONTRIBUTING are reviewed and tested like code.

---

## 3. Repository layout

```text
murder/
├── dune-project              # the only project file; all packages declared here
├── dune-workspace            # pkg enabled, warnings, test env
├── dune.lock/                # committed, generated, cross-platform
├── dune                      # @lint, @lint-fix, @ci aliases
├── litany                    # linter configuration (s-expressions)
├── .ocamlformat              # pinned formatter version
├── .editorconfig             # whitespace and final newlines for non-OCaml files
├── .gitattributes            # LF line endings; binary files
├── .git-blame-ignore-revs    # mass-format commits
├── .gitignore
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── commit-lint.yml
│   ├── pull_request_template.md
│   └── ISSUE_TEMPLATE/
├── .githooks/
│   ├── commit-msg            # scripts/check-commit-msg.sh
│   ├── pre-commit            # dune build @fmt
│   └── pre-push              # dune build @fmt @lint
├── packages/
│   ├── README.md             # index of shared libraries (replaces an umbrella)
│   ├── murder-dev/           # dev-only package: litany, tooling; never published
│   ├── _template/            # ignored by Dune; copied by scripts/new-package.sh
│   └── <library>/
│       ├── README.md
│       ├── lib/
│       │   ├── dune
│       │   ├── <library>.ml
│       │   └── <library>.mli
│       ├── test/
│       │   ├── dune
│       │   └── test_<library>.ml
│       └── doc/
│           ├── dune
│           └── index.md
├── getting-started/
│   ├── README.md
│   ├── 00-setup.md
│   ├── 01-nx/
│   ├── 02-rune/
│   ├── 03-vega/
│   └── 04-capstone/
├── doc/
│   └── dev/
│       ├── design.md
│       ├── testing.md
│       ├── ci.md
│       ├── dependencies.md
│       └── release.md
├── scripts/
│   ├── check-commit-msg.sh
│   └── new-package.sh
├── CHANGES.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── AGENTS.md                # rules for coding assistants
├── LICENSE
├── MONOREPO.md               # this file
└── README.md
```

**Why `packages/`?** It keeps the root small and makes the package boundary
obvious. Raven does the same. `getting-started/` sits beside `packages/` because
it is not a published library.

**Stable libraries only.** Applications and ongoing projects live in their own
repositories. This repo accepts code when it has an owner, a README, tests, a
documented API, and a reason to be shared. A separate project repo depends on
the libraries here through a pin (until release) or an opam version (after
release); see §4.3.

---

## 4. One Dune project, many packages

### 4.1 The root project

The root `dune-project` declares every package, shared metadata, and pins. There
is no per-package `dune-project`. This gives one dependency solve, one lock
directory, one build graph, and one `dune build`.

A skeleton:

```lisp
(lang dune 3.24)

(name murder)
(version 0.1.0)

(source (github hennequin-lab/murder))
(homepage "https://github.com/hennequin-lab/murder")
(bug_reports "https://github.com/hennequin-lab/murder/issues")
(authors "Hennequin Lab")
(maintainers "Hennequin Lab")
(license ISC)

(generate_opam_files true)
(opam_file_location inside_opam_directory)
(implicit_transitive_deps false)
(using mdx 0.4)

; Pins are for packages opam does not have, or for a specific unreleased fix.
; Pin a commit SHA, never a moving branch.
(pin
 (url "git+https://github.com/invariant-hq/litany.git#<litany-sha>")
 (package
  (name litany)))

(package
 (name murder-dev)
 (allow_empty)
 (synopsis "Development tools for the murder monorepo")
 (description
  "Lint and development tooling. This package is not published.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  litany))

(package
 (name example)
 (synopsis "An example murder package")
 (description "Replace this with the first real library.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (nx
   (>= 1.0.0~alpha3))
  (rune
   (>= 1.0.0~alpha3))
  (vega
   (>= 1.0.0~alpha3))
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (ppx_windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test)))
```

Notes:

- Package names are **bare** inside the workspace (`example`, not
  `murder-example`). The workspace is the namespace. If a package is published
  to opam, keep the library name and set `(public_name murder-example)`; the
  package stanza becomes `(name murder-example)`. The module stays `Example`.
- Do **not** create an umbrella `murder` package while the repo is used as a
  cloned workspace. It only helps `opam install murder`; add it when there are
  external consumers.
- Dune requires a project with exactly one package to give that package the
  project's name. Keep the `example` placeholder until the first real package
  exists (or add a second real package); do not delete it while `murder-dev` is
  the only package.
- `(= :version)` on internal packages keeps the monorepo in lockstep.
- For Raven packages, use a lower bound, not an exact version. The lock file is
  the exact pin; the source constraint says what API you need.
- If you pin Raven to a commit, remove the version constraints on the pinned
  packages. Dune treats pinned packages as `dev`, and a `>= 1.0.0~alpha3`
  constraint can reject `dev`.
- `murder-dev` exists only to make `litany` a dependency of the workspace, so
  `dune exec -- litany` and `%{bin:litany}` work. It is not published. Its name
  keeps the `murder-` prefix because it is a repo-level tool, not a library.
- `(using mdx 0.4)` is for executable documentation. If you do not want mdx on
  day one, omit it and add it when the first doc page lands.

### 4.2 Package conventions

- **Package name:** bare `<name>` while internal; `murder-<name>` only for a
  published package (`public_name`). The library name stays `<name>` and the
  main module `<Name>` either way.
- **Directory:** `packages/<name>/`; public API in `lib/<name>.mli`.
- **Tests:** `test/` mirrors `lib/`; one `test_<module>.ml` per suite; shared
  helpers in `test/support/`.
- **Docs:** `doc/index.md` with mdx; package `README.md` for humans.
- **Dependencies:** declare every direct dependency; `implicit_transitive_deps
  false` means a missing direct dependency is a compile error, which is what we
  want.
- **No `open` in public interfaces** unless it is part of the contract. Keep
  module names qualified in examples.
- **`.mli` doc comments** follow Raven's style: start with `[f x] ...`, state
  invariants, use `{[ ... ]}` for code.
- **Error messages** name the operation, the inputs, and the specific problem;
  include a hint when the fix is not obvious.

### 4.3 Using murder from another repo

Ongoing work lives in its own repository. A Dune workspace is rooted at the
topmost directory and cannot include arbitrary external paths, so a project does
not attach itself to this workspace. It depends on the libraries here in one of
two ways:

1. **Git pin, until there is a release.** In the project's `dune-project`:

   ```lisp
   (pin
    (url "git+https://github.com/hennequin-lab/murder.git#<full-sha>")
    (package (name foo))
    (package (name bar)))
   ```

   List every murder package in the project's dependency closure, because
   monorepo packages depend on each other at the same version. Remove version
   constraints on the pinned packages while the pin is active. Run
   `dune pkg lock` in the project and commit its lock. The project's CI then
   builds against exactly that murder commit.
2. **Opam version, after release.** Depend on the published package names. If
   the public name is `murder-foo`, that is the name in `dune-project`; the
   library and module stay `foo`/`Foo`.

For active co-development, a project may use a **local path pin** to a
neighbouring checkout of this repo:

```lisp
(pin
 (url "../murder")
 (package (name foo)))
```

Keep that local pin out of the committed lock; switch to the git SHA before
merging. Do not copy or vendor the libraries into the project; pin the repo.
The project repo owns its tests and CI. This repo's CI only gates the libraries
and tutorials it contains.

When a library in a project repo is ready to be shared, move that library into
`packages/` here, with its tests, docs, and `CHANGES.md` entry. The project repo
can then switch its pin to the new commit (or to the released opam package).

### 4.4 Adding a package

1. Run `./dev new <name>` (copies `packages/_template/`).
2. Add the `(package ...)` stanza to the root `dune-project`.
3. Fill in `lib/<name>.mli`, `lib/<name>.ml`, `test/test_<name>.ml`, and
   `doc/index.md`.
4. Add a `CHANGES.md` entry.
5. Run `./dev fmt`, `./dev check`, then open a PR.

A template with placeholders is easier to keep correct than a long checklist.
The script should substitute `@NAME@` and print the stanza to paste; it should
not rewrite `dune-project` with fragile `sed`.

---

## 5. Dune package management and the lock directory

This is the most consequential decision in the plan, so it gets its own
section.

### 5.1 What Dune package management gives us

- Dune reads package metadata from `dune-project`, solves dependencies, and
  builds them from source. No opam switch is required for the project.
- Dune 3.24 can *clone the compiler*: `dune build` downloads and builds OCaml
  5.5.2 if the lock says so. A new member needs Dune and network access, not a
  system OCaml.
- A lock directory records exact package versions, sources, checksums, and
  build actions, so the same commit resolves identically later.
- `dune tools` installs developer tools (ocamlformat, ocamllsp, odoc) in
  separate lock directories under `_build`, versioned from project config.

### 5.2 Should we commit `dune.lock/`?

**Yes.** Recommended policy:

- Commit **one** `dune.lock/` at the repository root.
- Treat it as generated: it is changed only by `dune pkg lock`, reviewed as a
  dependency diff, and never hand-edited.
- Commit it in the same PR as the `dune-project` change that caused it.
- Never commit `_build/` or `_build/.dev-tools.locks/`; those are local.

Reasons:

1. **Onboarding.** `git clone && dune build` is the whole setup. Without a lock,
   every new member runs the solver and may get a different Raven alpha, a
   different windtrap, or a compiler that litany refuses.
2. **Reproducibility.** A paper, a figure, or a bug report can name a commit;
   the lock names the exact dependency sources behind it.
3. **CI determinism.** CI does not silently pick up a new upstream release in
   the middle of a deadline. Dependency updates are explicit PRs.
4. **Pinned git dependencies.** Litany is not on opam. Without a lock, its
   commit pin is re-resolved on every build; with a lock, the exact SHA and
   checksum are recorded.
5. **Cross-platform is now practical.** Dune 3.24 writes a lock solved for
   Linux and macOS on both x86_64 and arm64. It is not the old "one machine's
   lock" situation. (Windows remains out of scope.)

Caveats to state in CONTRIBUTING:

- `dune.lock/` is platform-solved, not human-merged. On a conflict, take one
  side, run `dune pkg lock`, run the tests, and commit the regenerated result.
- Lock updates can be large. Keep them in dedicated `chore(deps):` PRs when
  possible; do not bury a lock refresh in a feature PR.
- A lock is only as good as its source URLs. Pin git dependencies to full
  commit SHAs, not branches or tags that can move.
- Dune package management is still evolving. Pin the Dune version in CI and
  document the minimum (`3.24.2`). Re-test a Dune upgrade in its own PR.

If the team ever decides *not* to commit the lock, the fallback is:

- Commit exact pins for Raven, windtrap, and litany in `dune-project`.
- Run `dune pkg lock` in CI before building.
- Accept that non-pinned transitive dependencies can drift.

That fallback is strictly worse for a lab that cares about reproducible
figures. I would only use it if the team cannot agree on a single platform
matrix.

### 5.3 Lock workflow

```sh
# First setup, or after changing dependencies/pins:
dune pkg lock

# Build using the committed lock:
dune build

# See what a refresh would change, without touching the lock:
dune pkg outdated
```

Rules:

- `dune pkg lock` is the only writer of `dune.lock/`.
- After changing `dune-project` dependencies or pins, relock and commit.
- If `dune build` says a package is missing from the lock, relock; do not edit
  the lock by hand.
- Review lock diffs for unexpected new packages or source URLs.
- Refresh the lock on a schedule (monthly or quarterly) in a dedicated PR.
- `dune pkg validate-lockdir` exists, but in Dune 3.24 it can complain about
  `dune` itself; the build is the real check. Do not add it as a CI gate.

### 5.4 Pins: Raven, windtrap, litany

**Raven.** Use the released opam packages (`nx`, `rune`, `vega`, and any other
Raven packages a library needs) at `1.0.0~alpha3`. The lock freezes the exact
version. If a lab library needs an unreleased Raven fix:

```lisp
(pin
 (url "git+https://github.com/raven-ml/raven.git#<full-sha>")
 (package (name nx))
 (package (name rune))
 (package (name vega)))
```

Pin all Raven packages used, not just the one with the fix; Raven packages
depend on each other with `(= :version)`. Remove the version constraints on
those packages while the pin is active. Remove the pin once a release includes
the fix.

**Windtrap.** Use the opam release (`0.2.x`) unless a needed feature or fix is
only on `main`; then pin `windtrap` and `ppx_windtrap` to one SHA. Add both as
`:with-test` dependencies:

```lisp
(windtrap (and :with-test (>= 0.2.0)))
(ppx_windtrap (and :with-test (>= 0.2.0)))
```

**Litany.** Not on opam. Pin it to a full SHA and put it only in `murder-dev`:

```lisp
(pin
 (url "git+https://github.com/invariant-hq/litany.git#<full-sha>")
 (package (name litany)))
```

Litany reads the compiler's `.cmt` artifacts and requires the same OCaml minor
it was built with (`>= 5.5.0 & < 5.6.0`). The lock pins 5.5.2, so this is
consistent. When litany updates, bump the SHA, relock, run `./dev check`, and
commit the lock diff.

Do not add `litany` to published library packages. It is a development tool;
consumers should not install it.

### 5.5 Compiler and Dune versions

- **Dune:** `(lang dune 3.24)`; CI installs exactly `3.24.2`. The lock format and
  dev-tools behavior are still moving, so a pinned CI Dune version is worth the
  small maintenance cost.
- **OCaml:** 5.5.x. Raven requires 5.5 for Nx; litany requires 5.5 and refuses
  other minors. The lock currently solves 5.5.2.
- **Upgrades:** Dune or OCaml upgrades are their own PRs with a full lock diff
  and a green CI run. Do not combine them with feature work.

### 5.6 What not to commit

```gitignore
_build/
_opam/
.litany-cache/
.DS_Store
.idea/
.vscode/
*.install
runs/
```

Do commit `dune.lock/`, the generated `opam/*.opam` files for published
packages, and the `.ocamlformat` version. Ignore `opam/murder-dev.opam`: it is
a dev-only package that depends on litany, which is not on opam, so it must not
be published. (Raven follows the same pattern with its `dev.opam`.)

---

## 6. Quality gates

### 6.1 Formatting

- One root `.ocamlformat`:

  ```ini
  version = 0.28.1
  profile = default
  parse-docstrings = true
  wrap-comments = true
  ```

- `dune build @fmt` checks formatting and fails with diffs; `dune fmt` applies
  the changes (`dune build @fmt --auto-promote`).
- `./dev fmt` is the friendly local command; CI runs the check, not the
  rewrite.
- Dune's dev-tools solver reads the `.ocamlformat` version and installs the
  matching ocamlformat under `_build`. If it ever picks the wrong version,
  remove `_build/.dev-tools.locks/ocamlformat` and run
  `dune tools install ocamlformat` again. Never commit `_build`.
- Hooks are opt-in and installed by `./dev setup` (`core.hooksPath`).
  `pre-commit` runs `dune build @fmt`; `pre-push` runs `dune build @fmt @lint`.
  CI remains the source of truth.
- Add a `.git-blame-ignore-revs` file when a mass reformat lands, so blame
  stays useful.

### 6.2 Linting with litany

Litany is a typedtree linter: it reads the compiler's `.cmt` artifacts, so a
project must compile first. It has two useful lanes:

- **In the build:** `dune build @lint` runs litany as a dune rule. Findings
  become dune diagnostics and reach editors through dune RPC. This lane
  withholds the cross-module rules (`unused-export`, `dead-code`) because it
  walks the build context rather than the full roster.
- **Outside the build:** `dune exec -- litany check` (from the repo root) runs
  the full roster, including cross-module rules. This is the CI gate.

Root `dune`:

```lisp
; Lint inside the build: findings become dune diagnostics.
(rule
 (alias lint)
 (deps
  (alias_rec check))
 (action
  (run %{bin:litany} check)))

; Optional: apply safe fixes as dune corrections, then `dune promote`.
; Requires (lang dune 3.23) or later.
(rule
 (alias lint-fix)
 (deps
  (alias_rec check))
 (corrections produce)
 (action
  (run %{bin:litany} check --fix)))

; One command for the local pre-push gate.
(alias
 (name ci)
 (deps
  (alias fmt)
  (alias lint)
  (alias_rec runtest)))
```

Root `litany`:

```lisp
(litany-config 1)

(lint
 (select all))

; `all` means every stable rule outside the restriction tier: correctness,
; suspicious, perf, style and pedantic. Restriction rules encode house policy
; and are adopted by exact name, for example:
;   (extend unsafe-partial-stdlib)
; Do not put `nursery` in the gate: nursery rules are expected to churn.
; Run `litany check --select all,restriction,nursery` periodically as an audit.

; If tutorial solutions intentionally keep awkward code, ignore them with a
; reason rather than disabling rules globally:
; (per-path
;  (paths getting-started/**/solutions/**)
;  (ignore all)
;  (reason "teaching solutions sometimes preserve the awkward version"))
```

Adoption policy:

- Start with `(select all)`. The repo is new, so there is no legacy baseline to
  protect; aggressive is cheapest now. `all` includes style and pedantic, but
  not restriction or nursery.
- Fix findings, or suppress with a reason using `[@litany.allow "..."]`.
  `unused-allow` is itself a finding, so suppressions cannot rot.
- Adopt restriction rules one at a time by exact name. They are house policy,
  not universal correctness; some contradict each other.
- Keep nursery rules out of CI. Run them as a non-blocking audit when litany is
  upgraded, and promote a rule to the gate only after the team agrees.
- Litany's `--fix` is safe when it says so; run `litany check --fix` at the
  terminal for the full roster. In the build, `@lint-fix` plus `dune promote`
  applies the safe fixes that the in-build lane can see.
- Pin litany by SHA and update it deliberately. It is a young tool; a new
  version can add stable rules and change the gate. Budget a cleanup commit
  when bumping it.

### 6.3 Testing with windtrap

Every package has a windtrap suite. A suite is an ordinary executable that
`dune runtest` builds and runs:

```lisp
(test
 (name test_greeting)
 (package greeting)
 (libraries greeting windtrap))
```

```ocaml
open Windtrap

let suite =
  group "greeting"
    [
      test "greets by name" (fun () ->
          equal string "hello, ada" (Greeting.greet "ada"));
      prop "never raises on arbitrary names" Gen.string (fun name ->
          ignore (Greeting.greet name));
    ]

let () = exit (run "greeting" [ suite ])
```

Conventions:

- `test/` mirrors `lib/`; shared helpers live in `test/support/`.
- Use `equal`, `less ~than`, `raises`, and typed witnesses that print values on
  failure. Avoid `is_true`.
- Use `prop` for laws (round trips, identities, reference implementations) and
  `stateful` for state across calls. Use `expect` for long text, not for
  numbers.
- Expected values come from the specification, not from running the code. If a
  value is learned from the implementation, write it as an `expect` test and
  say why.
- Keep each package's `runtest` under ~2 minutes warm. If a test is slow, fix
  the cause; do not hide it behind a tag.
- No network access in tests. Download fixtures in a script, with a checksum,
  and cache them outside the source tree; or generate synthetic data.
- Set `WINDTRAP_TIMEOUT` (e.g. 60 locally, 120 in CI) so a hung test fails
  instead of blocking the run. Property count: default locally, 200 in CI,
  1000 for a nightly run.
- GPU tests run on real hardware and skip when it is absent. Do not mock a
  device.

Coverage and mutation:

- Add a `@cover` alias per package or at the root using
  `ppx_windtrap.coverage`; run it on `main`/nightly first, with a low threshold
  or report-only.
- Mutation testing (`ppx_windtrap.mutate`) is expensive. Run it periodically
  as a reading list, not as a PR gate.
- Raise coverage thresholds only when the suite is stable; never lower a
  threshold to land a feature.

### 6.4 Warnings and docs

- Dev profile treats warnings as errors:

  ```lisp
  (env
   (dev
    (flags
     (:standard %{dune-warnings}))
    (odoc
     (warnings fatal)))
   (_
    (env-vars
     (WINDTRAP_TIMEOUT 60))))
  ```

- Public modules have `.mli` files with odoc comments. `dune build @doc` must
  be clean once docs exist.
- Markdown docs with OCaml snippets use `mdx`, so `dune runtest` catches stale
  examples.
- Add `@unused-libs` to CI once the package graph is stable; it catches a
  library that is declared but not used.

---

## 7. CI

CI is the enforcement layer, not the local loop. The local loop is
`./dev check`; CI re-runs the same commands on a clean machine and adds the
platform matrix.

### 7.1 What runs where

| Job | OS | Command | Purpose |
| --- | --- | --- | --- |
| `fmt` | Ubuntu | `dune build @fmt` | formatting is clean |
| `lint` | Ubuntu | `dune exec -- litany check` | full litany roster, GitHub annotations |
| `commit-lint` | Ubuntu | `scripts/check-commit-msg.sh --subject "$PR_TITLE"` | PR title follows the commit convention |
| `test` | Ubuntu, macOS | `dune build @all @install` + `dune runtest` | build and test the libraries and tutorials |
| `docs` (later) | Ubuntu | `dune build @doc` | odoc docs build |
| `coverage` (later, main/nightly) | Ubuntu | `dune build @cover --instrument-with ppx_windtrap.coverage` | report/gate |
| `mutation` (periodic, manual) | Linux | `dune build @mutate --instrument-with ppx_windtrap.mutate` | reading list |

Every PR runs `fmt`, `lint`, `commit-lint`, and `test`. `main` is protected so
these are required checks. Nightly/periodic jobs are allowed to be slower and
noisier.

### 7.2 Initial workflow

This is a working draft; adjust action versions and runner names as the org
requires.

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

env:
  DUNE_CACHE: enabled
  DUNE_CACHE_STORAGE_MODE: copy
  WINDTRAP_TIMEOUT: 120
  WINDTRAP_PROP_COUNT: 200
  LITANY_NO_PROGRESS: 1

jobs:
  fmt:
    name: fmt
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v6
        with:
          persist-credentials: false
      - uses: ocaml-dune/setup-dune@v2
        with:
          version: 3.24.2
          steps: install-dune enable-pkg
      - name: Check the lock is committed
        run: test -f dune.lock/lock.dune
      - name: Check formatting
        run: dune build @fmt

  lint:
    name: lint
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v6
        with:
          persist-credentials: false
      - uses: ocaml-dune/setup-dune@v2
        with:
          version: 3.24.2
          steps: install-dune enable-pkg
      - name: Lint
        run: dune exec -- litany check

  test:
    name: test (${{ matrix.os }})
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v6
        with:
          persist-credentials: false
      - uses: ocaml-dune/setup-dune@v2
        with:
          version: 3.24.2
          steps: install-dune enable-pkg install-depexts
      - name: Build
        run: dune build @all @install
      - name: Test
        run: dune runtest
      - name: Check generated files are committed
        if: runner.os == 'Linux'
        run: git diff --exit-code
```

The PR title is checked by a separate, very cheap workflow so that editing the
title re-runs only this check, not the whole build:

```yaml
name: Commit message

on:
  pull_request:
    types: [opened, edited, synchronize, reopened]

permissions:
  contents: read

jobs:
  commit-lint:
    name: commit-lint
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v6
        with:
          persist-credentials: false
      - name: Check the PR title
        env:
          PR_TITLE: ${{ github.event.pull_request.title }}
        run: ./scripts/check-commit-msg.sh --subject "$PR_TITLE"
```

Notes:

- `ocaml-dune/setup-dune@v2` installs the Dune binary and can cache both
  `_build` and the Dune cache. It does not create an opam switch.
- If a package needs GNU patch on macOS, add `install-gpatch` to the `steps`
  input. Pure OCaml projects usually do not.
- `install-depexts` is a no-op when the lock has no system dependencies. Keep it
  for future packages with C stubs.
- `DUNE_CACHE_STORAGE_MODE=copy` is deliberate: GitHub's cache restore is a
  tarball, and hardlinked cache entries do not survive it reliably.
- `git diff --exit-code` catches stale generated `opam/*.opam` files. Omit it
  until the generated files are committed in the skeleton PR.
- The `lint` job runs litany outside Dune so the cross-module rules run. Under
  `GITHUB_ACTIONS`, litany auto-selects its `github` output format, so findings
  become workflow annotations.
- If GitHub Actions minutes become a problem, demote macOS to `main` plus a
  weekly run. Keep Linux on every PR; it is the fast feedback.
- If the org uses private repositories, check the Actions spending limit and
  cache size. `_build` caches can be large; the setup action restores them by
  key.

### 7.3 Branch protection

Recommended settings for `main`:

- Require a pull request before merging.
- Require at least one approval; dismiss stale approvals on new commits.
- Require status checks: `fmt`, `lint`, `commit-lint`,
  `test (ubuntu-latest)`, and `test (macos-latest)`.
- Require conversation resolution.
- Require linear history; allow squash merge only. Disable merge commits and
  rebase merges, so the PR title is the commit that lands on `main`.
- Block force pushes and deletions.
- Allow administrators to bypass only when necessary; the PI may want this
  during a deadline.

The PR title becomes the squash commit subject. Use a conventional form:

```text
type(scope): Imperative summary
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `bench`, `chore`.
Scope is the package or area: a bare package name (`nx`, `foo`) or one of the
reserved scopes `repo`, `ci`, `docs`, `deps`, `tutorials`, `getting-started`,
`dev`. Keep the subject under ~72 characters (about 50 is better); explain *why*
in the body.

Enforcement, in three layers:

1. **Local `commit-msg` hook.** `./dev setup` sets `core.hooksPath .githooks`;
   `.githooks/commit-msg` calls `scripts/check-commit-msg.sh`. It rejects a bad
   subject before the commit is created. Hooks are bypassable, so they are the
   fast feedback layer, not the authority.
2. **CI PR title check.** A separate `commit-lint` workflow runs
   `scripts/check-commit-msg.sh --subject "$PR_TITLE"`. Since the repo uses
   squash merge, the PR title is exactly the commit that lands on `main`.
3. **Branch protection.** `commit-lint` is a required status check, and merge
   commits/rebase merges are disabled. This makes the convention hold even for
   contributors who never install the hooks.

Rules the checker enforces:

- One of the allowed types, then an optional `(scope)`, then `!` for a breaking
  change, then `: ` and a capitalised imperative summary.
- No trailing period; at most 72 characters.
- Scope is lowercase; one scope per subject. If a change touches several
  packages, choose the main one and list the rest in the body.
- Body: blank line after the subject, wrapped at 72, says why. Describe the
  change, never the process (no reviews, agents, phases, attempts).

The checker is deliberately a small shell script, not a Node/commitlint
dependency: the repo stays OCaml + Dune, and the format is simple enough to
enforce in ~30 lines. Its exit codes are 0 for a good subject, 1 for a bad one,
and 2 for misuse.

### 7.4 Reproducing CI locally

| CI job | Local command |
| --- | --- |
| `fmt` | `dune build @fmt` (or `./dev fmt` to apply) |
| `lint` | `dune exec -- litany check` |
| `test` | `dune build @all @install && dune runtest` |
| `commit-lint` | `./scripts/check-commit-msg.sh --subject "$(git log -1 --pretty=%s)"` |
| everything | `./dev check` |

Troubleshooting:

- **Stale `_build` or cache weirdness:** `rm -rf _build` and rebuild. This is
  safe; `_build` is not source. In CI, bump `cache-prefix` if a cache is cursed.
- **ocamlformat version mismatch:** remove
  `_build/.dev-tools.locks/ocamlformat` and run `dune tools install
  ocamlformat`; Dune reads the version from `.ocamlformat`.
- **Lock conflict:** take one side of `dune.lock/`, run `dune pkg lock`, run
  `./dev check`, commit the regenerated lock.
- **macOS build needs GNU patch:** add `install-gpatch` to the setup-dune
  `steps` input.
- **A test passes locally but fails in CI:** check the compiler version, the
  `WINDTRAP_PROP_COUNT`, and whether the test reads the network or the clock.
  Reproduce with the CI environment variables from §7.2.

### 7.5 Later CI additions

- **Docs job:** `dune build @doc` on Linux. Dune installs odoc as a dev tool.
  Start non-blocking, then make it required.
- **Coverage:** `dune build @cover --instrument-with ppx_windtrap.coverage` on
  `main` and nightly; upload the report as an artifact. Add a threshold only
  after two clean weeks.
- **Mutation:** a scheduled or manual workflow; survivors are a reading list,
  not a failure.
- **Dependency refresh:** a monthly PR from `dune pkg lock` plus
  `dune pkg outdated`. Keep it separate from feature work.
- **Release:** a tag-triggered workflow using `dune-release` if the lab decides
  to publish to opam. Do not build this before the first release is wanted.

---

## 8. Getting started

The `getting-started/` directory is the lab's on-ramp to Raven. It is not a
copy of the Raven docs; it is a guided path with exercises and a capstone, and
it is tested in CI so it cannot rot.

### 8.1 Audience and prerequisites

- Comfortable with OCaml basics: modules, records, variants, functors, `dune`.
- No Raven, NumPy, JAX, or PyTorch experience assumed.
- No GPU required. Everything runs on CPU; GPU notes are optional.
- One to two hours per chapter; the capstone is an afternoon.

### 8.2 Layout

```text
getting-started/
├── README.md                 # start here: setup, order, how to run
├── 00-setup.md               # install Dune, clone, editor, troubleshooting
├── 01-nx/
│   ├── README.md             # arrays, dtypes, shapes, broadcasting, views
│   ├── dune
│   ├── ex01_create.ml
│   ├── ex02_broadcast.ml
│   ├── ex03_slicing.ml
│   ├── ex04_linalg.ml
│   ├── exercises.md
│   └── solutions/
├── 02-rune/
│   ├── README.md             # grad, value_and_grad, pytrees, vmap, jit
│   ├── dune
│   ├── ex01_grad.ml
│   ├── ex02_linear_regression.ml
│   ├── ex03_vmap.ml
│   ├── ex04_jit.ml
│   ├── exercises.md
│   └── solutions/
├── 03-vega/
│   ├── README.md             # optimizers, state, schedules, clipping
│   ├── dune
│   ├── ex01_sgd.ml
│   ├── ex02_adam.ml
│   ├── ex03_schedules.ml
│   ├── ex04_train_mlp.ml
│   ├── exercises.md
│   └── solutions/
└── 04-capstone/
    ├── README.md             # train, evaluate, save, plot, write up
    ├── dune
    ├── train.ml
    └── data/                 # generated or tiny; no downloads in tests
```

### 8.3 Curriculum

**Chapter 1 — Nx.** Tensors as values with a dtype and shape; creating arrays;
elementwise ops; broadcasting; reductions; matrix multiplication; slicing and
views versus copies; reading/writing `.npy`; a first plot with Hugin if the
ecosystem chapter is included. Exercises: numerically stable softmax, one-hot
encoding, layer normalization, a vectorized distance matrix, and a shape-error
debugging exercise.

**Chapter 2 — Rune.** Why autodiff; pure functions; `grad` and
`value_and_grad`; parameter trees (`Nx.Ptree.S`) and typed records; linear
regression from scratch; `check_grads` against finite differences; `vmap` for
per-sample gradients; `jit` and its tracing rules (no side effects, static
shapes); a small MLP forward/backward pass. Exercises: hand-code the MSE
gradient and compare, compute per-sample gradients, jit a training step, and
find the bug in a traced function with a side effect.

**Chapter 3 — Vega.** Optimizer state has the shape of the parameters; `sgd`,
`adam`, `adamw`; a training loop combining Rune and Vega; learning-rate
schedules; gradient clipping and loss scaling; comparing optimizers on the same
model; checkpointing state. Exercises: tune a learning rate, compare SGD and
Adam, add warmup + cosine decay, and train an MLP on a synthetic two-moons or
blobs dataset.

**Chapter 4 — Capstone.** An end-to-end experiment: generate or load a small
dataset, split train/validation, standardize features, train an MLP with
Rune + Vega, track loss/accuracy, evaluate, save the weights, and plot the
result. Optional: use Munin for logging and Hugin for plots. Acceptance:
`dune exec getting-started/04-capstone/train.exe` runs in under a minute on a
laptop CPU, is deterministic with a fixed seed, prints metrics, and has a
README explaining how to reproduce the number.

### 8.4 How the examples stay correct

- Short snippets in `README.md` are mdx code blocks, tested by `dune runtest`.
- Full programs are ordinary executables with windtrap tests for the numerical
  claims (shapes, tolerances, determinism).
- `solutions/` are compiled and tested; the exercise files are complete demos,
  not `failwith "TODO"` stubs. New members write attempts in `scratch/`, which
  is ignored by git.
- No network downloads in tests. Synthetic data or a tiny committed fixture
  with a checksum.
- Total tutorial `runtest` under ~2 minutes warm.
- When Raven changes, the lock bump PR must update the tutorials; CI catches
  broken snippets.

### 8.5 Onboarding path

1. Install Dune 3.24.2 and put it on `PATH`.
2. `git clone git@github.com:hennequin-lab/murder.git && cd murder`.
3. `./dev setup` (installs hooks, builds `.cmt`, downloads dependencies).
4. `dune exec getting-started/01-nx/ex01_create.exe`.
5. Read `getting-started/README.md`, work through Nx, Rune, Vega.
6. Do the capstone; write a short README with the result.
7. Read `CONTRIBUTING.md`, pick a `good-first-issue`, open a PR.

Success criteria: a new member completes steps 1–4 unaided in under 30 minutes;
a lab member can complete the three core chapters in about a day; the tutorial
suite is green in CI on Linux and macOS.

---

## 9. Daily workflow and onboarding

### 9.1 The `./dev` script

A thin wrapper for the commands people run every day. It should print the
underlying `dune` command so it teaches rather than hides.

```sh
./dev setup      # git hooks, dune build @check, first-run dependency download
./dev fmt        # dune fmt (auto-promote)
./dev check      # dune build @ci, then full litany check
./dev test       # dune runtest
./dev lint       # dune exec -- litany check
./dev lock       # dune pkg lock
./dev outdated   # dune pkg outdated
./dev new NAME   # scaffold packages/NAME from the template
```

Direct equivalents should be in `CONTRIBUTING.md`:

```sh
dune build @fmt          # check formatting
dune fmt                 # apply formatting
dune build @lint         # in-build lint
dune exec -- litany check  # full lint
dune build @ci           # fmt + lint + tests
dune runtest             # tests only
```

### 9.2 Hooks

- `commit-msg`: `scripts/check-commit-msg.sh`. Rejects a commit subject that
  does not follow the convention before the commit is created.
- `pre-commit`: `dune build @fmt`. Fast; catches the most common "dirty tree"
  mistake.
- `pre-push`: `dune build @fmt @lint`. Still local; does not run the full test
  suite unless you use `./dev check`.
- `./dev setup` runs `git config core.hooksPath .githooks`.
- Hooks are a convenience, not a gate. CI is the gate; `git push --no-verify`
  is allowed for emergencies, but the PR will fail if the tree is not clean or
  the PR title is malformed.

### 9.3 Editor setup

- Install the OCaml platform extension for VS Code, or `ocaml-eglot`/`ocaml-lsp`
  for Emacs/Vim.
- Install the language server with Dune's dev-tools mechanism:

  ```sh
  dune tools install ocamllsp
  eval $(dune tools env)
  ```

- `dune build @lint` in watch mode (`dune build @default @lint -w`) surfaces
  litany findings as ordinary diagnostics.
- Commit an `.editorconfig` for whitespace and final newlines, but do not commit
  personal editor settings.

### 9.4 First-week checklist for a new member

- [ ] Clone and `./dev setup`; first example runs.
- [ ] Read `getting-started/README.md` and `CONTRIBUTING.md`.
- [ ] Finish Nx and Rune chapters; run the tutorial tests.
- [ ] Make a small docs/test PR and watch CI.
- [ ] Add a test to an existing package with a maintainer.
- [ ] Pick a `good-first-issue`.

### 9.5 Definition of done for a change

- [ ] `./dev fmt` has been run; `dune build @fmt` is clean.
- [ ] `./dev check` passes (format, lint, tests).
- [ ] Commit subject and PR title follow `type(scope): Imperative summary`.
- [ ] Public API changes have `.mli` docs and a `CHANGES.md` entry.
- [ ] New behavior has a windtrap test; bug fixes have a regression test that
  failed before the fix.
- [ ] If dependencies changed, `dune pkg lock` was run and the lock diff is in
  the PR.
- [ ] If docs/examples changed, mdx tests pass.

---

## 10. Documentation map

| File | Audience | Contents |
| --- | --- | --- |
| `README.md` | everyone | What Murder is, package table, 3-command quickstart, links |
| `MONOREPO.md` | maintainers | This plan: decisions, architecture, roadmap |
| `CONTRIBUTING.md` | contributors | Setup, workflow, formatting, linting, tests, PRs, dependencies |
| `AGENTS.md` | coding assistants | Hard rules, commands, conventions, review and commit rules |
| `getting-started/` | new lab members | Guided Raven/Nx/Rune/Vega course + capstone |
| `doc/dev/design.md` | maintainers | Design principles, API and error-message conventions |
| `doc/dev/testing.md` | contributors | Windtrap patterns, property/stateful tests, coverage |
| `doc/dev/ci.md` | maintainers | CI jobs, caching, reproducing failures locally |
| `doc/dev/dependencies.md` | maintainers | Adding deps, pins, lock updates, Raven upgrades |
| `doc/dev/release.md` | maintainers | Versioning, changelog, tags, opam release (when wanted) |
| `packages/*/README.md` | users | What the package does, status, example, tests |
| `CHANGES.md` | users/maintainers | Unreleased and released changes, per package |

`README.md` should answer in one screen: what is this, what is in it, how do I
build it, where do I start. It should link to `getting-started/` before it
links to API docs.

---

## 11. Roadmap

### Phase 0 — Decisions (1–2 days)

- [ ] Confirm publication plans: bare package names now; add a `murder-`
  public prefix only if/when a package goes to opam.
- [ ] Confirm license (ISC).
- [ ] Confirm platforms (Linux + macOS recommended; Windows out of scope).
- [ ] Confirm Raven policy (released alpha3 + lock recommended).
- [ ] Confirm CI budget and whether macOS runs on every PR.
- [ ] Record the decisions in this file.

**Done when:** the open questions in §12 have answers or explicit defaults.

### Phase 1 — Skeleton (1–2 days)

- [ ] Root `dune-project`, `dune-workspace`, `.ocamlformat`, `.gitignore`,
  `.gitattributes`.
- [ ] Root `dune` with `@lint`, `@lint-fix`, `@ci`.
- [ ] `litany` config, `murder-dev` package with the litany pin.
- [ ] `packages/_template` and `packages/README.md`.
- [ ] `scripts/new-package.sh` and `scripts/check-commit-msg.sh`.
- [ ] `./dev` script and `.githooks/` (`commit-msg`, `pre-commit`, `pre-push`).
- [ ] `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE`,
  `CHANGES.md`.
- [ ] `dune pkg lock`; commit `dune.lock/` and the generated `opam/*.opam`
  files for published packages (`opam/murder-dev.opam` is intentionally
  ignored).
- [ ] `./dev check` green locally.
- [ ] `.github/workflows/ci.yml` and `.github/workflows/commit-lint.yml`;
  first green run.
- [ ] Branch protection on `main`; PR template.

**Done when:** a fresh clone on Linux and macOS builds, formats, lints, and
tests with the committed lock.

Suggested first PRs (small, reviewable):

1. `chore(repo): Add root Dune project, workspace, and lock`
2. `chore(tooling): Add formatting, litany config, and dev package`
3. `chore(tooling): Add package template, dev script, and hooks`
4. `chore(ci): Add format, lint, and build/test workflow`
5. `docs: Add README, CONTRIBUTING, and onboarding notes`
6. `feat(foo): Add the first real library`

### Phase 2 — Getting started (about a week)

- [ ] `getting-started/README.md` and `00-setup.md`.
- [ ] Nx chapter with runnable demos, exercises, solutions; mdx.
- [ ] Rune chapter with gradient descent, `vmap`, `jit`; mdx.
- [ ] Vega chapter with optimizer comparison and schedules; mdx.
- [ ] Capstone with a deterministic, sub-minute training run.
- [ ] `dune runtest getting-started` in CI.
- [ ] Pilot the path with one lab member; fix every point of friction.

**Done when:** a new member reaches the first example in under 30 minutes and
the tutorial suite is green on both platforms.

### Phase 3 — Quality and growth (1–2 weeks, then ongoing)

- [ ] Clean the litany baseline; adopt restriction rules one at a time;
  keep nursery as a non-blocking audit.
- [ ] Add `@cover` and a report-only coverage job.
- [ ] Add `@doc` and a docs CI job.
- [ ] Add `@unused-libs` and warning hygiene.
- [ ] Add `CHANGES.md`/PR conventions and a release checklist.
- [ ] Add the first real package(s) via the template.
- [ ] Add `doc/dev/testing.md`, `ci.md`, `dependencies.md`, `release.md`.
- [ ] Schedule a monthly lock refresh and a quarterly mutation review.

**Done when:** two or three real libraries are in `packages/`, the gates have
been stable for two weeks, and a second person can onboard from the docs alone.

### Phase 4 — Later

- [ ] Benchmarks with `thumper` if performance claims matter.
- [ ] Nightly property tests with a higher `WINDTRAP_PROP_COUNT`.
- [ ] Opam release with `dune-release`, if the lab wants external users.
- [ ] Docs site deployment, if useful.
- [ ] Revisit each decision annually; remove process that no longer earns its
  keep.

---

## 12. Open questions

These need a decision from the PI/team. The recommended default is in
parentheses.

1. **Publication plans.** Will any package be published to opam? Default: no,
   bare names; add a `murder-` public prefix at the first publication. If yes,
   decide whether an umbrella `murder` package is worth maintaining.
2. **Project-repo pinning.** Project repos pin a murder commit by default. Do we
   also want a released opam version for stable libraries, and on what cadence?
3. **License.** ISC is settled; keep the `LICENSE` and package metadata in sync.
4. **Raven version.** Released `1.0.0~alpha3` + lock (recommended), or pin
   Raven `main` from day one? Pinning `main` gives the newest APIs but makes
   every lock refresh a compatibility project.
5. **Windtrap version.** Released `0.2.x` (recommended) or pin a commit for
   unreleased features?
6. **Litany restriction rules.** Start with `(select all)`; which restriction
   policies do we adopt by exact name, and when? Default: none on day one;
   add one per week after the baseline is clean.
7. **CI budget.** macOS on every PR (recommended) or Linux PRs + macOS on
   `main`/weekly?
8. **Coverage.** Report-only first (recommended) or a hard threshold from day
   one?
9. **Hooks.** Opt-in via `./dev setup` (recommended) or mandatory? Git hooks
   are local and can always be bypassed; CI is the real gate.
10. **Tutorial package.** Keep `getting-started` unpublished (recommended) or
    publish it as a package? Unpublished avoids opam metadata for teaching
    code.
11. **Capstone ecosystem.** Nx/Rune/Vega only (recommended for focus), or add
    Munin/Hugin to teach logging and plotting?
12. **Releases.** Tag-only/internal (recommended initially) or publish to opam
    from the first release?
13. **Dune floor.** Require 3.24.2 everywhere (recommended) or support older
    Dune? The multi-platform lock and in-build litany fixes want 3.24.
14. **Data policy.** Synthetic/small fixtures in git (recommended); larger data
    via a download script with checksums, never in `runtest`.
15. **Agent rules.** `AGENTS.md` is in the repo. Keep it in sync when the
    workflow or a hard rule changes.

---

## Appendix A — Skeleton files

### A.1 `dune-project`

See §4.1. Keep comments short; the file is read by humans and tools.

### A.2 `dune-workspace`

```lisp
(lang dune 3.24)

; Dune package management is the only package manager for this repo.
(pkg enabled)

(env
 (dev
  (flags
   (:standard %{dune-warnings}))
  (odoc
   (warnings fatal))
  (env-vars
   (WINDTRAP_TIMEOUT 60)))
 (_
  (env-vars
   (WINDTRAP_TIMEOUT 60))))
```

If Dune rejects `(pkg enabled)` in a workspace file in your version, put it in
`~/.config/dune/config` via `dune pkg enabled` or the `setup-dune` action; the
rest of the file is unaffected.

### A.3 Root `dune`

```lisp
(rule
 (alias lint)
 (deps
  (alias_rec check))
 (action
  (run %{bin:litany} check)))

(rule
 (alias lint-fix)
 (deps
  (alias_rec check))
 (corrections produce)
 (action
  (run %{bin:litany} check --fix)))

(alias
 (name ci)
 (deps
  (alias fmt)
  (alias lint)
  (alias_rec runtest)))
```

### A.4 `.ocamlformat`

```ini
version = 0.28.1
profile = default
parse-docstrings = true
wrap-comments = true
```

### A.5 `.gitignore`

```gitignore
_build/
_opam/
.litany-cache/
.DS_Store
.idea/
.vscode/
*.install
runs/

# Dev-only package: depends on litany, which is not on opam
opam/murder-dev.opam

# Local tutorial scratch space
getting-started/**/scratch/
```

Do not ignore `dune.lock/` or the opam files of published packages.

### A.6 `.gitattributes`

```gitattributes
* text=auto eol=lf

*.png binary
*.jpg binary
*.jpeg binary
*.npy binary
*.npz binary
*.ico binary
```

### A.7 Hooks

`.githooks/commit-msg`:

```sh
#!/bin/sh
set -eu
root=$(git rev-parse --show-toplevel)
exec "$root/scripts/check-commit-msg.sh" "$1"
```

`.githooks/pre-commit`:

```sh
#!/bin/sh
set -eu
exec dune build @fmt
```

`.githooks/pre-push`:

```sh
#!/bin/sh
set -eu
exec dune build @fmt @lint
```

Make all three executable (`chmod +x`) and install with
`git config core.hooksPath .githooks`.

### A.7.1 `scripts/check-commit-msg.sh`

Make it executable (`chmod +x scripts/check-commit-msg.sh`).

```sh
#!/bin/sh
# Check a Conventional Commit subject; used by the commit-msg hook and CI.
set -eu

usage() {
  echo "usage: $0 COMMIT_MSG_FILE | $0 --subject SUBJECT" >&2
  exit 2
}

subject=
case "${1:-}" in
  --subject)
    [ "$#" -eq 2 ] || usage
    subject=$2
    ;;
  *)
    [ "$#" -eq 1 ] || usage
    subject=$(sed -n '1p' "$1")
    ;;
esac

types='feat|fix|perf|refactor|test|docs|bench|chore'
pattern="^($types)(\([a-z0-9][a-z0-9_-]*\))?!?: [A-Z].*$"

if [ -z "$subject" ] || ! printf '%s\n' "$subject" | grep -Eq "$pattern"; then
  cat >&2 <<EOF
Bad commit subject: $subject

Expected:
  type(scope): Imperative summary

Types: feat, fix, perf, refactor, test, docs, bench, chore
Scope: a package name, or repo, ci, docs, deps, tutorials,
       getting-started, dev

Example:
  fix(parser): Reject empty input
EOF
  exit 1
fi

case "$subject" in
  *.)
    echo "Bad commit subject: do not end the subject with a period." >&2
    exit 1
    ;;
esac

len=$(printf '%s' "$subject" | wc -c | tr -d ' ')
if [ "$len" -gt 72 ]; then
  echo "Bad commit subject: $len characters; keep it to 72 or fewer." >&2
  exit 1
fi

exit 0
```

### A.8 `scripts/new-package.sh`

```sh
#!/bin/sh
set -eu

name=${1:?usage: new-package.sh NAME}
dest="packages/$name"

if [ -e "$dest" ]; then
  echo "$dest already exists" >&2
  exit 1
fi

cp -R packages/_template "$dest"
find "$dest" -type f -print0 |
  xargs -0 perl -pi -e "s/\@NAME\@/$name/g"

echo "Created $dest"
echo "Add its (package ...) stanza to dune-project, then run ./dev fmt."
```

Use `perl -pi` rather than `sed -i` so the script works on macOS and Linux.

### A.9 Package template

`packages/_template/lib/dune`:

```lisp
(library
 (name @NAME@)
 (public_name @NAME@))
```

`packages/_template/lib/@NAME@.mli`:

```ocaml
(** @NAME@.

    Replace this with the package's contract. *)
```

`packages/_template/lib/@NAME@.ml`:

```ocaml
let hello = "hello from @NAME@"
```

`packages/_template/test/dune`:

```lisp
(test
 (name test_@NAME@)
 (package @NAME@)
 (libraries @NAME@ windtrap))
```

`packages/_template/test/test_@NAME@.ml`:

```ocaml
open Windtrap

let () =
  exit
    (run "@NAME@"
       [ test "placeholder" (fun () -> equal int 1 1) ])
```

The name passed to `./dev new` must be a valid OCaml library name
(`snake_case`, lowercase). The package is bare (`@NAME@`). If it is later
published to opam, keep `(name @NAME@)` and change `(public_name murder-@NAME@)`
and the package stanza to `(name murder-@NAME@)`; the module stays `@NAME@`.

`packages/_template/doc/dune`:

```lisp
(mdx
 (files index.md)
 (packages @NAME@))
```

`packages/_template/doc/index.md`:

````markdown
# @NAME@

```ocaml
let () = print_endline "@NAME@"
```
````

`packages/_template/README.md`:

````markdown
# @NAME@

One sentence: what problem does this package solve?

## Status

Experimental / alpha / stable.

## Example

```ocaml
(* short, compilable example *)
```

## Tests

```sh
dune runtest packages/@NAME@
```
````

### A.10 `CHANGES.md`

```markdown
# Changelog

All notable user-visible changes. Internal refactors, tests, and formatting
changes do not need an entry.

## Unreleased

### example

- Initial release.
```

### A.11 PR template

```markdown
## What

## Why

## Checklist

- [ ] `./dev fmt` run; `dune build @fmt` clean
- [ ] `./dev check` passes
- [ ] Public API has `.mli` docs
- [ ] Windtrap test covers the new behavior / regression
- [ ] `CHANGES.md` entry for a user-visible change
- [ ] If dependencies changed: `dune pkg lock` diff included
- [ ] If docs/examples changed: mdx tests pass
```

### A.12 README skeleton

````markdown
# Murder

The Hennequin lab's monorepo of Raven-based OCaml libraries.

## Quick start

Requires Dune 3.24.2+ and network access on the first build.

```sh
git clone git@github.com:hennequin-lab/murder.git
cd murder
./dev setup
dune exec getting-started/01-nx/ex01_create.exe
```

## Packages

| Package | What it does |
| --- | --- |
| `example` | Replace with the first real library |

## Learn Raven

Start at [`getting-started/`](getting-started/README.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The plan for the repo is in
[`MONOREPO.md`](MONOREPO.md).
````

### A.13 `CONTRIBUTING.md` outline

```markdown
# Contributing

## Set up
- Install Dune 3.24.2+.
- Clone; run `./dev setup`.
- First command: `dune exec getting-started/01-nx/ex01_create.exe`.

## Daily loop
- `./dev fmt` before committing.
- `./dev check` before pushing.
- `dune runtest` while iterating.

## Formatting
- One `.ocamlformat`; `dune build @fmt` is the check; `dune fmt` applies.
- Do not reformat unrelated code in a feature PR.

## Linting
- `dune build @lint` for editor diagnostics.
- `dune exec -- litany check` for the full roster.
- The gate is aggressive on purpose: `(select all)` stable rules.
- Suppress a finding only with a reason; `unused-allow` keeps it honest.

## Project repos
- This repo is for stable libraries only; applications live in separate repos.
- A project repo pins a murder commit (or uses a released opam version) and
  owns its own lock, tests, and CI.

## Commits
- Subject: `type(scope): Imperative summary`; scope is the package or area.
- The `commit-msg` hook checks it locally; CI checks the PR title.
- Body: blank line, wrapped at 72, says why. No process narration.

## Tests
- Every package has a windtrap suite under `test/`.
- State contracts, not implementation; use `prop`/`stateful` where they fit.
- Bug fixes include a test that failed before the fix.

## Dependencies
- Add the dependency to the package's `dune-project` stanza.
- Run `dune pkg lock`; commit the lock diff with the change.
- Pin git dependencies to a full SHA; never a moving branch.

## Pull requests
- PR title: `type(scope): Imperative summary`; squash merge.
- Fill the PR checklist; keep changes small and reviewable.
- Public API changes need `.mli` docs and a `CHANGES.md` entry.

## Adding a package
- `./dev new <name>`, add the `(package ...)` stanza, fill `lib/`, `test/`, `doc/`.
- Internal package names are bare; add a `murder-` public prefix only at opam
  publication.

## Getting unstuck
- Read the first error, not the last.
- `rm -rf _build` for cache weirdness; regenerate the lock for lock weirdness.
- Ask in the lab channel with the command, the full error, and the commit.
```

---

## Appendix B — References

- Raven: <https://github.com/raven-ml/raven>
  - Nx, Rune, Vega READMEs and per-package examples are the primary API
    references.
- Windtrap: <https://github.com/invariant-hq/windtrap>
  - Manual under `doc/manual/`, especially assertions, property testing,
    stateful testing, baselines, coverage, and mutation.
- Litany: <https://github.com/invariant-hq/litany>
  - `doc/manual/configuration.md`, `build-integration.md`, and `rules.md`.
- Dune package management:
  - <https://dune.readthedocs.io/en/stable/explanation/package-management.html>
  - <https://dune.readthedocs.io/en/stable/tutorials/dune-package-management/locking.html>
  - <https://dune.readthedocs.io/en/stable/reference/dune-workspace/lock_dir.html>
  - <https://dune.readthedocs.io/en/stable/reference/dune-tools.html>
- Dune CI setup action: <https://github.com/ocaml-dune/setup-dune>
- Dune formatting: <https://dune.readthedocs.io/en/stable/howto/formatting.html>

---

## Appendix C — Why this plan, in one page

The lab's scarce resources are attention and reproducibility, not disk space.
A single Dune project with a committed lock gives both: one build graph, one
dependency solution, one place to add a library, and a clone-and-build
onboarding path. Dune 3.24 makes that lock practical across the Linux and macOS
machines the lab actually uses. Windtrap and litany are the quality layer:
windtrap makes tests state contracts and print useful failures; litany makes
style and correctness findings automatic and reviewable. The CI is deliberately
small — format, lint, build, test — because a small lab cannot afford a
Rube Goldberg pipeline. The getting-started directory is the investment that
makes the rest work: if a new member can run Nx, Rune, and Vega examples on day
one, the monorepo has done its job.

---

## Appendix D — Checks performed while writing this plan

These are not guesses; they were tested with Dune 3.24.2 and OCaml 5.5.1 on
Linux while drafting this document:

- A minimal project with `windtrap` as a dependency produced a `dune.lock`
  whose `lock.dune` had `solved_for_platforms` for Linux and macOS on x86_64
  and arm64. `dune exec -- windtrap --help` ran the locked binary.
- Pinning litany by full SHA in `dune-project`, putting it in a `murder-dev`-style
  package, and running `dune pkg lock` produced a lock containing `litany.dev`.
  Both `dune build @lint` (in-build, `%{bin:litany}`) and
  `dune exec -- litany check` (full roster) worked.
- `.ocamlformat` with `version = 0.28.1` made Dune's dev-tools solver install
  and use ocamlformat 0.28.1; `dune build @fmt` failed on unformatted files and
  passed once they were formatted.
- `(pkg enabled)` and the `(env ...)` form in `dune-workspace` parsed and built.
- The `@ci` alias pattern (`(alias fmt)`, `(alias lint)`, `(alias_rec runtest)`)
  parsed and ran; using `(alias check)` as a dependency did not, so the plan
  uses `(alias_rec check)` inside the lint rule.
- `ocaml-dune/setup-dune@v2` caches `_build` and `DUNE_CACHE_ROOT`, and installs
  an exact Dune release when `version` is set.

Re-run these checks in the skeleton PR on both Linux and macOS; they are the
assumptions the rest of the plan rests on.
