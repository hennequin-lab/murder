(lang dune 3.24)

(name symo)

(license ISC)

(authors "Guillaume Hennequin")

(package
 (name symo)
 (allow_empty)
 (synopsis
  "Weight-space symmetry for curvature estimation on the Raven stack")
 (description
  "Orbit averages over parameter-space symmetry groups, a compact basis of\nthe invariant (commutant) subspace, and the Taylor curvature estimator of\nEq. 10, on nx tensors, rune autodiff/JIT and ppx_ptree parameter trees.")
 (depends
  (ocaml
   (>= 5.5.0))
  base
  nx
  ppx_ptree
  rune
  stdio
  ppx_compare
  ppx_sexp_conv))
