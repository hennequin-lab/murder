# Packages

Shared libraries live here. Applications and ongoing projects live in their own
repositories.

| Package | Status | What it does |
| --- | --- | --- |
| `cmdargs` | stable | Simple parsing of command-line arguments |
| `geom` | alpha | Geometric algebra for robotics on the Raven stack |
| `dilqr` | alpha | Differentiable iLQR on Nx tensors |
| `ilqr-vae` | alpha | iLQR-VAE: control-based learning of input-driven dynamics |
| `symo` | alpha | Weight-space symmetry for curvature estimation |
| `sofo` | alpha | Second-order forward-mode optimization |

A package lives at `packages/<name>/` with `lib/`, `test/`, and `doc/`. Add one
with `./dev new <name>`, then add its `(package ...)` stanza to the root
`dune-project`.
