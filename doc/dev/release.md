# Release

All packages share the root version in `dune-project`. A release is a tag
`vX.Y.Z` and a `CHANGES.md` section for that version.

## Checklist

1. Move the `CHANGES.md` entries from `Unreleased` to the new version.
2. Run `./dev check` and `dune build @all @install`.
3. Tag: `git tag -s vX.Y.Z -m "vX.Y.Z"`.
4. Push the tag (the human does this).

The repo does not publish to opam. Downstream projects pin a murder commit.

Internal package dependencies use `(= :version)`, so they move together.
