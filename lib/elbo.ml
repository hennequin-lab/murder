(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Types

type 'w t =
  { elbo : 'w value
  ; log_likelihood : 'w value
  ; kl : 'w value
  }

(* [mean_over_batch t n_samples] is the mean of a [[steps; n_samples]] tensor
   of per-sample scalars. *)
let mean_over_batch t n_samples = Nx.div_s (Nx.sum t) (Float.of_int n_samples)

let compute model params ?(beta = 1.0) ~mu_u ~key ~n_samples data =
  if n_samples <= 0 then invalid_arg "Elbo.compute: n_samples must be positive";
  let dtype = model.Model.dtype
  and t_steps = model.Model.n_steps
  and n = model.Model.n in
  let u = Recognition.sample model params ~mu:mu_u ~n_samples ~key in
  let z =
    Dynamics.integrate
      model.Model.dynamics
      params.Model.dynamics
      ~x0:(Nx.zeros dtype [| n |])
      ~u
  in
  (* Observation [k] is emitted from the state reached by [k + 1] inputs. *)
  let z_next = Nx.slice [ R (1, t_steps + 1) ] z in
  let ks = Nx.arange dtype 0 t_steps 1 in
  let _, likelihood_by_step =
    Rune.scan
      Nx.Ptree.unit
      Nx.Ptree.(pair (pair tensor tensor) tensor)
      Nx.Ptree.tensor
      ~f:(fun () ((z_k, o_k), k) ->
        ( ()
        , model.Model.likelihood.Likelihood.logp_t
            params.Model.likelihood
            ~k
            ~x:z_k
            ~o:o_k ))
      ~init:()
      ((z_next, data), ks)
  in
  let log_likelihood = mean_over_batch likelihood_by_step n_samples in
  let kl =
    match model.Model.prior.Prior.kl_to_gaussian with
    | Some kl ->
      kl
        params.Model.prior
        ~mu:mu_u
        ~space_factor:(Covariance.factor params.Model.space_cov)
        ~time_factor:(Covariance.factor params.Model.time_cov)
    | None ->
      let _, prior_by_step =
        Rune.scan
          Nx.Ptree.unit
          Nx.Ptree.(pair tensor tensor)
          Nx.Ptree.tensor
          ~f:(fun () (u_k, k) ->
            (), model.Model.prior.Prior.logp_t params.Model.prior ~k ~u:u_k)
          ~init:()
          (u, ks)
      in
      let mean_log_prior = mean_over_batch prior_by_step n_samples in
      Nx.neg (Nx.add (Recognition.entropy model params) mean_log_prior)
  in
  let elbo = Nx.sub log_likelihood (Nx.mul_s kl beta) in
  { elbo; log_likelihood; kl }

let compute_fixed model params ?beta ~key ~n_samples ~iters ?u_init data =
  let mu_u = Encoder.posterior_mean_fixed model params ?u_init ~iters data in
  compute model params ?beta ~mu_u ~key ~n_samples data

let loss model params ?beta ~mu_u ~key ~n_samples data =
  Nx.neg (compute model params ?beta ~mu_u ~key ~n_samples data).elbo

let loss_fixed model params ?beta ~key ~n_samples ~iters ?u_init data =
  Nx.neg (compute_fixed model params ?beta ~key ~n_samples ~iters ?u_init data).elbo

let batch_loss_fixed model params ?beta ~key ~n_samples ~iters trials =
  let count = Array.length trials in
  if count = 0 then invalid_arg "Elbo.batch_loss_fixed: the batch is empty";
  let keys = Nx.Rng.split ~n:count key in
  let total =
    Array.foldi trials ~init:(Nx.scalar model.Model.dtype 0.0) ~f:(fun i acc data ->
      Nx.add acc (loss_fixed model params ?beta ~key:keys.(i) ~n_samples ~iters data))
  in
  Nx.div_s total (Float.of_int count)
