# Nx: arrays

Nx is the numerical foundation: n-dimensional arrays with a dtype and shape,
NumPy-like operations, broadcasting, views, and linear algebra.

Run the two examples:

```sh
dune exec getting-started/01-nx/ex01_arrays.exe
dune exec getting-started/01-nx/ex02_slicing.exe
```

## Things to notice

- A tensor carries both an OCaml element type and a buffer dtype. `Nx.float32`
  is a dtype value, not a type.
- `Nx.shape` is an `int array`; wrong shapes fail at the operation, not at a
  silent broadcast.
- `Nx.add` broadcasts a `[1; 3]` row over a `[2; 3]` matrix. Broadcasting
  aligns dimensions from the right.
- `Nx.reshape` and `Nx.transpose` return views when they can; `Nx.item` reads
  one scalar to the host.

## Exercises

1. Build a `[4; 1]` column and a `[1; 3]` row; add them and predict the shape
   before running.
2. Implement numerically stable softmax over the last axis:
   subtract the row maximum, exponentiate, divide by the row sum.
3. Compute a pairwise squared-distance matrix for 100 points without a loop.
4. Deliberately add tensors with incompatible shapes and read the error.
