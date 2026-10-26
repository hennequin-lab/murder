# CI

The workflows live in `.github/workflows/`.

| Job | What it runs |
| --- | --- |
| `fmt` | `dune build @fmt` |
| `lint` | `dune exec -- litany check` |
| `commit-lint` | `scripts/check-commit-msg.sh --subject "$PR_TITLE"` |
| `test` | `dune build @all @install`, then `dune runtest`, on Ubuntu and macOS |

`main` requires the four checks, a pull request, one approval, and squash
merges. Merge commits and rebase merges are disabled, so the PR title is the
commit that lands.

## Reproduce locally

```sh
dune build @fmt
dune exec -- litany check
dune build @all @install && dune runtest
./scripts/check-commit-msg.sh --subject "$(git log -1 --pretty=%s)"
```

Or `./dev check` for everything.

## Notes

- `ocaml-dune/setup-dune@v2` installs Dune 3.24.2, uses the committed lock, and
  caches `_build` and the Dune cache. It does not create an opam switch.
- The lock patch in `scripts/patch-lock.py` is applied when relocking, not in
  CI; CI uses the committed lock.
- If GitHub Actions minutes become tight, demote macOS to `main` plus a weekly
  run. Keep Linux on every PR.
