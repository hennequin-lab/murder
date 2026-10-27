# Rune: differentiation, `vmap`, and `jit`

Rune transforms ordinary OCaml functions over ordinary values. There is no
special tensor type and no encoded parameter tree: a record is a record, and
its `walk` says which fields are leaves.

Run the examples:

```sh
dune exec getting-started/02-rune/ex01_grad.exe
dune exec getting-started/02-rune/ex02_linear_regression.exe
```

## Things to notice

- `Rune.grad'` differentiates a function of one tensor. `Rune.grad` takes a
  structure and returns a gradient of the same type.
- A parameter record needs an `Nx.Ptree.S` module with one `walk`; the compiler
  keeps the record type through the transformation.
- `Rune.value_and_grad` returns `(loss, grads)` in one pass. The update is
  ordinary OCaml: subtract `lr * grad` from each leaf. The training loop is a
  tail-recursive function that threads the parameters; no mutable state is
  needed.
- `Rune.vmap` maps a function over axis 0 of every argument. Composing `vmap`
  with `grad` gives per-example gradients.
- `Rune.jit` traces once and replays. Keep traced functions pure and shapes
  static; host reads and side effects do not trace.

## Exercises

1. Check `grad` against a finite difference for a few inputs.
2. Replace the loop with `Rune.vmap` to compute per-example gradients.
3. Wrap the training step in `Rune.jit`; compare the first loss and the speed.
4. Add a second layer to the model. The parameter tree grows; the loss and the
   update do not change beyond that.
