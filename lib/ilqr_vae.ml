(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Types = Types
module Covariance = Covariance
module Prior = Prior
module Likelihood = Likelihood
module Dynamics = Dynamics
module Model = Model
module Recognition = Recognition
module Encoder = Encoder
module Elbo = Elbo

type 'w value = (float, 'w) Nx.t
type ('w, 'd, 'l, 'p, 'e) model = ('w, 'd, 'l, 'p, 'e) Model.t
type ('w, 'd, 'l, 'p, 'e) params = ('w, 'd, 'l, 'p, 'e) Model.params

let posterior_mean = Encoder.posterior_mean
let posterior_mean_fixed = Encoder.posterior_mean_fixed
let loss = Elbo.loss
let loss_fixed = Elbo.loss_fixed
let batch_loss_fixed = Elbo.batch_loss_fixed

let elbo model params ?beta ~mu_u ~key ~n_samples data =
  (Elbo.compute model params ?beta ~mu_u ~key ~n_samples data).Elbo.elbo

(* [integrate model params ~u] is the deterministic rollout, [z_0 = 0]. *)
let integrate
      (type w d l p e)
      (model : (w, d, l, p, e) Model.t)
      (params : (w, d, l, p, e) Model.params)
      ~(u : w value)
  =
  Dynamics.integrate
    model.Model.dynamics
    params.Model.dynamics
    ~x0:(Nx.zeros model.Model.dtype [| model.Model.n |])
    ~u

(* [decode model params z] is the likelihood's readout of a state sequence:
   observation [k] belongs to state [k + 1]. *)
let decode
      (type w d l p e)
      (model : (w, d, l, p, e) Model.t)
      (params : (w, d, l, p, e) Model.params)
      (z : w value)
  =
  let z_next = Nx.slice [ R (1, model.Model.n_steps + 1) ] z in
  model.Model.likelihood.Likelihood.mean params.Model.likelihood z_next

let sample_generative
      (type w d l p e)
      (model : (w, d, l, p, e) Model.t)
      (params : (w, d, l, p, e) Model.params)
      ~key
  =
  let keys = Nx.Rng.split key in
  let key_u = keys.(0)
  and key_o = keys.(1) in
  let u =
    model.Model.prior.Prior.sample
      params.Model.prior
      ~key:key_u
      ~n_steps:model.Model.n_steps
  in
  let z = integrate model params ~u in
  let o =
    model.Model.likelihood.Likelihood.sample
      params.Model.likelihood
      ~key:key_o
      ~mean:(decode model params z)
  in
  u, z, o

let predictions
      (type w d l p e)
      (model : (w, d, l, p, e) Model.t)
      (params : (w, d, l, p, e) Model.params)
      ~mu
      ~n_samples
      ~key
  =
  let u = Recognition.sample model params ~mu ~n_samples ~key in
  let z = integrate model params ~u in
  u, z, decode model params z

let predictions_deterministic
      (type w d l p e)
      (model : (w, d, l, p, e) Model.t)
      (params : (w, d, l, p, e) Model.params)
      ~mu
  =
  let z = integrate model params ~u:mu in
  z, decode model params z
