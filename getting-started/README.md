# Getting started with Raven

This is the lab's on-ramp to the Raven stack. It assumes OCaml basics (modules,
records, variants, dune) and no Raven, NumPy, JAX, or PyTorch experience.

Work through the chapters in order:

1. [Setup](00-setup.md) — install Dune, clone, run the first example.
2. [Nx](01-nx/README.md) — arrays, dtypes, shapes, broadcasting, views.
3. [Rune](02-rune/README.md) — `grad`, parameter trees, `vmap`, `jit`.
4. [Vega](03-vega/README.md) — optimizers, state, schedules, clipping.
5. [Capstone](04-capstone/README.md) — train a small model end to end.

## The loop

```sh
# one example
dune exec getting-started/01-nx/ex01_arrays.exe

# build every example
dune build getting-started
```

Read the chapter, run the example, then do the exercises. Write your attempts in
`getting-started/scratch/` (ignored by git); compare with the chapter when you
are done.

## How to get unstuck

- Read the first error, not the last.
- Print shapes and dtypes before an operation: `Nx.shape`, `Nx.dtype`,
  `Nx.to_string`.
- Check broadcasting from the right: `[2;3]` and `[3]` are compatible;
  `[2;3]` and `[2]` are not.
- Ask in the lab channel with the command, the full error, and the commit.
