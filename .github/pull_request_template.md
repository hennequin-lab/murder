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
