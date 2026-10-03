(lang dune 3.24)

(name dilqr)

(license ISC)

(authors "The Dilqr authors")

(package
 (name dilqr)
 (synopsis "Differentiable iLQR for optimal control on Nx tensors")
 (description
  "Dilqr solves discrete-time optimal control problems with iterative LQR on \
  Nx tensors, and differentiates through the solve: the resulting implicit \
  gradients are computed with the LQR machinery itself (a custom VJP), so \
  both the solve and its gradients JIT-compile with Rune to CPU or GPU.")
 (depends
  (ocaml (>= 5.5.0))
  base
  stdio
  nx
  rune))
