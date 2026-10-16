(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Types = Types
module Covariance = Covariance
module Initial = Initial
module Prior = Prior
module Likelihood = Likelihood
module Dynamics = Dynamics
module Recognition = Recognition
module Model = Model
module Elbo = Elbo

type 'w value = (float, 'w) Nx.t
type ('w, 'd, 'l, 'p, 'e, 'r) model = ('w, 'd, 'l, 'p, 'e, 'r) Model.t
type ('w, 'd, 'l, 'p, 'e, 'r) params = ('w, 'd, 'l, 'p, 'e, 'r) Model.params

let posterior_mean = Model.mean
let posterior_draws = Model.infer
let log_prob = Model.log_prob
let loss = Elbo.loss
let loss_fixed = Elbo.loss_fixed
let batch_loss_fixed = Elbo.batch_loss_fixed

let elbo model params ?beta ~posterior data =
  (Elbo.compute model params ?beta ~posterior data).Elbo.elbo

(* [integrate model params ~u] is the deterministic rollout from [x_0 = 0];
   [integrate_from model params ~x0 ~u] starts where the caller says. *)
let integrate
      (type w d l p e r)
      (model : (w, d, l, p, e, r) Model.t)
      (params : (w, d, l, p, e, r) Model.params)
      ~u
  =
  Dynamics.integrate
    model.Model.dynamics
    params.Model.dynamics
    ~x0:(Nx.zeros model.Model.dtype [| model.Model.n |])
    ~u

let integrate_from
      (type w d l p e r)
      (model : (w, d, l, p, e, r) Model.t)
      (params : (w, d, l, p, e, r) Model.params)
      ~x0
      ~u
  =
  Dynamics.integrate model.Model.dynamics params.Model.dynamics ~x0 ~u

(* [decode model params z] is the likelihood's readout of a state sequence:
   observation [k] belongs to state [k + 1]. *)
let decode
      (type w d l p e r)
      (model : (w, d, l, p, e, r) Model.t)
      (params : (w, d, l, p, e, r) Model.params)
      (z : w value)
  =
  let z_next = Nx.slice [ R (1, model.Model.n_steps + 1) ] z in
  model.Model.likelihood.Likelihood.mean params.Model.likelihood z_next

let sample_generative
      (type w d l p e r)
      (model : (w, d, l, p, e, r) Model.t)
      (params : (w, d, l, p, e, r) Model.params)
      ~key
  =
  let keys = Nx.Rng.split ~n:3 key in
  let key_u = keys.(0)
  and key_x0 = keys.(1)
  and key_o = keys.(2) in
  let u =
    model.Model.prior.Prior.sample
      params.Model.prior
      ~key:key_u
      ~n_steps:model.Model.n_steps
  in
  let x0 =
    match params.Model.initial_prior with
    | None -> Nx.zeros model.Model.dtype [| model.Model.n |]
    | Some p -> Nx.reshape [| model.Model.n |] (Initial.sample p ~key:key_x0 ~n_samples:1)
  in
  let z = integrate_from model params ~x0 ~u in
  let o =
    model.Model.likelihood.Likelihood.sample
      params.Model.likelihood
      ~key:key_o
      ~mean:(decode model params z)
  in
  x0, u, z, o

let predictions
      (type w d l p e r)
      (model : (w, d, l, p, e, r) Model.t)
      (params : (w, d, l, p, e, r) Model.params)
      (posterior : w Recognition.posterior)
  =
  let z =
    integrate_from model params ~x0:posterior.Recognition.x0 ~u:posterior.Recognition.u
  in
  posterior.Recognition.u, z, decode model params z

let predictions_deterministic
      (type w d l p e r)
      (model : (w, d, l, p, e, r) Model.t)
      (params : (w, d, l, p, e, r) Model.params)
      (samples : w Recognition.samples)
  =
  let z =
    integrate_from model params ~x0:samples.Recognition.x0 ~u:samples.Recognition.u
  in
  z, decode model params z
