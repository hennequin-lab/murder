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

(implicit_transitive_deps false)
(using mdx 0.4)

; Pins live in dune-project, not dune-workspace, because Dune resolves the pins
; of a pinned project recursively. A project that pins murder inherits the
; Raven commit, windtrap and litany below, without repeating them.

; Raven is an external dependency. Pin its Git repository, never vendor it.
; Move this SHA forward deliberately; dune.lock records the same commit.
(pin
 (url
  "git+https://github.com/raven-ml/raven.git#4a3bc8311990a185a5636f94fa02a69906ff3f1c")
 (package
  (name lz))
 (package
  (name nx))
 (package
  (name rune))
 (package
  (name tolk))
 (package
  (name vega))
 (package
  (name jera))
 (package
  (name hugin))
 (package
  (name rig))
 (package
  (name munin))
 (package
  (name ppx_ptree))
 (package
  (name kaun)))

; Windtrap's releases are on opam, but Raven and litany pin it, and Dune
; resolves a pinned project's pins recursively, so the pin below is what keeps
; our lock on a version we name. It is windtrap's v0.3.0 release.
(pin
 (url
  "git+https://github.com/invariant-hq/windtrap.git#6423db76370e35dfefa0fa2e7a8e6318323b8d81")
 (package
  (name windtrap))
 (package
  (name ppx_windtrap)))

; Litany is not on opam yet. Pin it by full SHA.
(pin
 (url
  "git+https://github.com/invariant-hq/litany.git#4cab5e78fd62d2455638fb76ca030ee6d9ce1d29")
 (package
  (name litany)))

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
  ppx_ptree
  (windtrap :with-test)))

(package
 (name cmdargs)
 (license MIT)
 (synopsis "Simple parsing of command-line arguments")
 (description
  "Small utility for dealing with command-line arguments in OCaml: typed getters with defaults, required values, flags, and input/output directory helpers.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune))

(package
 (name geom)
 (synopsis "Geometric algebra for robotics on the Raven stack")
 (description
  "Geometric algebra on Nx tensors: a small Clifford-algebra core, the E3, PGA and CGA models, motors and rigid motions, robot kinematics, and a Hugin-based visualization layer.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  nx
  rune
  hugin
  (windtrap :with-test)
  (ppx_windtrap :with-test)))

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

; The engine is not published yet (MILESTONES.md: private until M7), but its
; package has to exist for the lock to resolve what the viewer needs: dune pkg
; only resolves what some package depends on, and raylib is the viewer's only
; dependency that is not already in the tree.
(package
 (name mjc)
 (synopsis "A MuJoCo-compatible differentiable physics engine on Nx")
 (depends
  base
  nx
  raylib))

(executables_implicit_empty_intf false)
