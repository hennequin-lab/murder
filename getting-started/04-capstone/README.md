# Capstone: train a small model

This is the end-to-end exercise: make data, build a model, differentiate the
loss, hand the gradients to an optimizer, and read the result.

```sh
dune exec getting-started/04-capstone/train.exe
```

The model is a one-hidden-layer MLP on two Gaussian blobs. It is deliberately
small enough to run on a laptop CPU in seconds.

## What to do next

1. Change the number of hidden units and see how the loss curve moves.
2. Replace `Adam` with `SGD` and then add momentum.
3. Add a validation split and report validation accuracy, not just training.
4. Save the parameters and reload them in a second executable.
5. Plot the decision boundary with Hugin, or log the run with Munin, if you
   want to exercise the rest of the Raven ecosystem.

## Acceptance

- The executable runs in under a minute on a laptop CPU.
- With the fixed keys in the file, the final accuracy is stable across runs.
- The data, model, loss, and step are separate functions; changing one does not
  require rewriting the others.
