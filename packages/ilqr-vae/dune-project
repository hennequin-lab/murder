(lang dune 3.24)

(name ilqr_vae)

(license ISC)

(authors "The Ilqr_vae authors")

(package
 (name ilqr_vae)
 (synopsis "iLQR-VAE: control-based learning of input-driven dynamics")
 (description
  "iLQR-VAE fits an input-driven latent dynamical system to a time series by \
  variational inference, using iLQR as the encoder: the posterior mean of the \
  Gaussian recognition distribution is the solution of a finite-horizon \
  optimal control problem, and Dilqr differentiates the solve implicitly. \
  Built on Nx tensors and Rune transformations.")
 (depends
  (ocaml (>= 5.5.0))
  base
  stdio
  nx
  rune
  dilqr
  ppx_ptree
  (windtrap :with-test)))
