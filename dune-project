(lang dune 3.24)

(name murder)
(version 0.1.0)

(source
 (github hennequin-lab/murder))

(homepage "https://github.com/hennequin-lab/murder")
(bug_reports "https://github.com/hennequin-lab/murder/issues")
(authors "Guillaume Hennequin")
(maintainers "Guillaume Hennequin")
(license ISC)

(generate_opam_files true)
(opam_file_location inside_opam_directory)
(implicit_transitive_deps false)
(using mdx 0.4)

(package
 (name dilqr)
 (synopsis "Differentiable iLQR for optimal control on Nx tensors")
 (description
  "Dilqr solves discrete-time optimal control problems with iterative LQR on Nx tensors, and differentiates through the solve: the resulting implicit gradients are computed with the LQR machinery itself (a custom VJP), so both the solve and its gradients JIT-compile with Rune to CPU or GPU.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  base
  stdio
  nx
  rune
  (ppx_ptree :with-test)
  (windtrap :with-test)))

(package
 (name ilqr-vae)
 (synopsis "iLQR-VAE: control-based learning of input-driven dynamics")
 (description
  "iLQR-VAE fits an input-driven latent dynamical system to a time series by variational inference, using iLQR as the encoder: the posterior mean of the Gaussian recognition distribution is the solution of a finite-horizon optimal control problem, and Dilqr differentiates the solve implicitly. Built on Nx tensors and Rune transformations.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  base
  stdio
  nx
  rune
  (dilqr
   (= :version))
  ppx_ptree
  (windtrap :with-test)))

(package
 (name symo)
 (allow_empty)
 (synopsis "Weight-space symmetry for curvature estimation on the Raven stack")
 (description
  "Orbit averages over parameter-space symmetry groups, a compact basis of the invariant (commutant) subspace, and the Taylor curvature estimator of Eq. 10, on Nx tensors, Rune autodiff/JIT and ppx_ptree parameter trees.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  base
  stdio
  nx
  rune
  ppx_ptree
  ppx_compare
  ppx_sexp_conv
  ppx_sexp_message
  (windtrap :with-test)))

(package
 (name sofo)
 (allow_empty)
 (synopsis "Second-order forward-mode optimization on the Raven stack")
 (description
  "Sofo estimates curvature from loss, gradient and Gauss-Newton-vector products along random directions, and trains with a second-order forward-mode optimizer on Nx tensors and Rune transformations.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  base
  cmdargs
  cmdliner
  nx
  ppx_ptree
  rune
  stdio
  (windtrap :with-test)))

(package
 (name murder-dev)
 (allow_empty)
 (synopsis "Development tools and example dependencies for the murder monorepo")
 (description
  "Lint, documentation and the libraries used by the examples and tutorials. This package is not published.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  litany
  bos
  cmdargs
  cmdliner
  hugin
  jera
  ppx_ptree
  rig
  vega))

(executables_implicit_empty_intf false)
