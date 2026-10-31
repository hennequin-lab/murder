(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx
module Types = Types
module Covariance = Covariance
module Initial = Initial
module Prior = Prior
module Likelihood = Likelihood
module Dynamics = Dynamics
module Schedule = Schedule
module Recognition = Recognition
module Model = Model
module Elbo = Elbo

type ('w, 's) value = (float, 'w, 's) Nx.t
type ('w, 'd, 'l, 'p, 'e, 'r, 's) model = ('w, 'd, 'l, 'p, 'e, 'r, 's) Model.t
type ('w, 'd, 'l, 'p, 'e, 'r, 's) params = ('w, 'd, 'l, 'p, 'e, 'r, 's) Model.params

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
      (model : (w, d, l, p, e, r, 's) Model.t)
      (params : (w, d, l, p, e, r, 's) Model.params)
      ~u
  =
  let u_shape = shape u in
  let rank = Array.length u_shape in
  if rank < 2
  then
    invalid_arg
      (Printf.sprintf
         "Ilqr_vae.integrate: u must be [batch…; n_steps; %d], got rank %d"
         model.Model.m
         rank);
  let batch_dims = Array.sub u_shape 0 (rank - 2) in
  let x0 =
    zeros_at (beside u) model.Model.dtype (Array.append batch_dims [| model.Model.n |])
  in
  Dynamics.integrate model.Model.dynamics params.Model.dynamics ~x0 ~u

let integrate_from
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) Model.t)
      (params : (w, d, l, p, e, r, 's) Model.params)
      ~x0
      ~u
  =
  Dynamics.integrate model.Model.dynamics params.Model.dynamics ~x0 ~u

(* [decode model params ~time_axis z] is the likelihood's readout of a state
   sequence: observation [k] belongs to state [k + 1]. *)
let decode
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) Model.t)
      (params : (w, d, l, p, e, r, 's) Model.params)
      ~time_axis
      (z : (w, 's) value)
  =
  let rank = Array.length (shape z) in
  let indices =
    List.init rank (fun axis ->
      if Int.equal axis time_axis
      then (R (1, model.Model.n_steps + 1) : 's Nx.index)
      else A)
  in
  let z_next = slice indices z in
  model.Model.likelihood.Likelihood.mean params.Model.likelihood z_next

let sample_generative
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) Model.t)
      (params : (w, d, l, p, e, r, 's) Model.params)
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
    | None -> zeros_at (beside u) model.Model.dtype [| model.Model.n |]
    | Some p -> reshape [| model.Model.n |] (Initial.sample p ~key:key_x0 ~n_samples:1)
  in
  let z = integrate_from model params ~x0 ~u in
  let time_axis = Array.length (shape z) - 2 in
  let o =
    model.Model.likelihood.Likelihood.sample
      params.Model.likelihood
      ~key:key_o
      ~mean:(decode model params ~time_axis z)
  in
  x0, u, z, o

let predictions
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) Model.t)
      (params : (w, d, l, p, e, r, 's) Model.params)
      (posterior : (w, 's) Recognition.posterior)
  =
  let z =
    Dynamics.integrate_samples
      model.Model.dynamics
      params.Model.dynamics
      ~x0:posterior.Recognition.x0
      ~u:posterior.Recognition.u
  in
  let time_axis = Array.length (shape z) - 3 in
  posterior.Recognition.u, z, decode model params ~time_axis z

let predictions_deterministic
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) Model.t)
      (params : (w, d, l, p, e, r, 's) Model.params)
      (samples : (w, 's) Recognition.samples)
  =
  let z =
    Dynamics.integrate_samples
      model.Model.dynamics
      params.Model.dynamics
      ~x0:samples.Recognition.x0
      ~u:samples.Recognition.u
  in
  let time_axis = Array.length (shape z) - 3 in
  z, decode model params ~time_axis z
