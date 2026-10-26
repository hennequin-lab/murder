# Vega: optimizers

Vega steps whole parameter structures. An optimizer's state has the shape of
the parameters, so a training step is a pure function of `(params, state)` and
can be checkpointed or compiled.

Run the example:

```sh
dune exec getting-started/03-vega/ex01_optimizers.exe
```

## Things to notice

- `Vega.adam_init p params` builds a state for the parameter structure `p`.
- `Vega.adam_step p ~lr st ~params ~grads` returns `(params', state')`.
- The same loop drives SGD, Adam, and AdamW; only `init` and `step` change.
- `Vega.lr` makes the learning rate explicit. Schedules are values too.
- Gradient clipping is a transformation: `Vega.clip_by_global_norm p ~max_norm`.

## Exercises

1. Compare the three optimizers on the same quadratic. Which reaches the
   minimum first?
2. Replace the constant learning rate with a cosine decay schedule.
3. Clip the gradients of a model whose loss produces a large early gradient.
4. Add a second parameter leaf to a record and run the same optimizer on it.
