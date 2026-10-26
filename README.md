# Murder

The Hennequin lab's monorepo of stable, Raven-based OCaml libraries.

Applications and ongoing projects live in their own repositories. This repo is
for shared libraries, their tests and documentation, and the onboarding course.

## Quick start

Requires Dune 3.24.2 or newer and network access on the first build.

```sh
git clone git@github.com:hennequin-lab/murder.git
cd murder
./dev setup
dune exec getting-started/01-nx/ex01_arrays.exe
```

## Packages

| Package | What it does |
| --- | --- |
| [`dilqr`](packages/dilqr) | Differentiable iLQR for optimal control on Nx tensors |
| [`ilqr-vae`](packages/ilqr-vae) | Control-based learning of input-driven dynamics |
| [`symo`](packages/symo) | Weight-space symmetry for curvature estimation |
| [`sofo`](packages/sofo) | Second-order forward-mode optimization |

## Learn Raven

Start at [`getting-started/`](getting-started/README.md). It covers Nx, Rune,
and Vega, then a capstone.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The plan for the repo is in
[`MONOREPO.md`](MONOREPO.md); coding-assistant rules are in
[`AGENTS.md`](AGENTS.md).
