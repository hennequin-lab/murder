# CI

The workflows live in `.github/workflows/`.

| Job | What it runs |
| --- | --- |
| `fmt` | `dune tools install ocamlformat`, then `dune build @fmt` |
| `lint` | `dune exec -- litany check` |
| `commit-lint` | `scripts/check-commit-msg.sh --subject "$PR_TITLE"` |
| `test` | `dune build @all @install`, then `dune runtest`, on Ubuntu and macOS |

`main` requires the four checks, a pull request, one approval, and squash
merges. Merge commits and rebase merges are disabled, so the PR title is the
commit that lands.

## Reproduce locally

```sh
dune tools install ocamlformat
dune build @fmt
dune exec -- litany check
dune build @all @install && dune runtest
./scripts/check-commit-msg.sh --subject "$(git log -1 --pretty=%s)"
```

Or `./dev check` for everything.

## Notes

- `ocaml-dune/setup-dune@v2` installs Dune 3.24.2, uses the committed lock, and
  caches `_build` and the Dune cache. It does not create an opam switch.
- `ocamlformat` is a dev tool, not a locked dependency, and the runner has no
  OCaml installation. `./dev setup` installs the tool set locally; CI does not
  run setup, so the `fmt` job installs `ocamlformat` itself. Without that step
  `@fmt` fails with `Program ocamlformat not found in the tree or in PATH`.
  The tool lock lives under `_build`, and Dune picks the version named in
  `.ocamlformat`, so bumping that file is enough to move the formatter.
- The lock patch in `scripts/patch-lock.py` is applied when relocking, not in
  CI; CI uses the committed lock.
- Litany on Dune 3.24.2 currently skips ppx-preprocessed units with
  `derived-needs-build`. It still gates the units it admits, but the summary is
  not full coverage. Remove this note when litany's Dune adapter catches up.
- If GitHub Actions minutes become tight, demote macOS to `main` plus a weekly
  run. Keep Linux on every PR.
