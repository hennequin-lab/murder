(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Types

type ('w, 's) t =
  { elbo : ('w, 's) value
  ; log_likelihood : ('w, 's) value
  ; kl : ('w, 's) value
  }

let shape_string shape =
  String.concat ~sep:"; " (Array.to_list (Array.map shape ~f:Int.to_string))

let append_shape front tail = Array.concat [ front; tail ]

let check_shape ~what expected t =
  let got = Nx.shape t in
  if not (Array.equal Int.equal got expected)
  then
    invalid_arg
      (Printf.sprintf
         "Elbo.compute: %s must be [%s], got [%s]"
         what
         (shape_string expected)
         (shape_string got))

(* [time_major_axes ~time_axis rank] moves [time_axis] to the front, keeping
   every other axis in order. The likelihood and prior scans walk axis 0, so
   a batch-first tensor is transposed for the duration of the scan. *)
let time_major_axes ~time_axis rank =
  time_axis
  :: List.filter (List.init rank ~f:Fn.id) ~f:(fun axis -> not (Int.equal axis time_axis))

let transpose_time ~time_axis t =
  let rank = Array.length (Nx.shape t) in
  transpose ~axes:(time_major_axes ~time_axis rank) t

(* [log_prior model params ~x0 ~u] is the log density of the model's prior
   over the latents at a sample: one value per member of the sample batch. The
   input prior is evaluated step by step; the initial-state prior contributes
   exactly when the model has one. [u] is time-major, so its leading axis is
   scanned and every trial and sample axis is a batch axis. *)
let log_prior
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) Model.t)
      (params : (w, d, l, p, e, r, 's) Model.params)
      ~(x0 : (w, 's) value)
      ~(u : (w, 's) value)
  =
  let t_steps = model.Model.n_steps in
  let u_rank = Array.length (Nx.shape u) in
  let time_axis = Int.max 0 (u_rank - 3) in
  let u_by_time = transpose_time ~time_axis u in
  let ks = arange_at (beside u) model.Model.dtype 0 t_steps 1 in
  let _, by_step =
    Rune.scan
      Nx.Ptree.unit
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      (fun () (u_k, k) -> (), model.Model.prior.Prior.logp_t params.Model.prior ~k ~u:u_k)
      ~init:()
      (u_by_time, ks)
  in
  let u_logp = sum ~axes:[ 0 ] by_step in
  match params.Model.initial_prior with
  | None -> u_logp
  | Some p -> Infix.(u_logp + Initial.logp p x0)

let compute
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) Model.t)
      (params : (w, d, l, p, e, r, 's) Model.params)
      ?beta
      ~(posterior : (w, 's) Recognition.posterior)
      data
  =
  let beta =
    Option.value
      beta
      ~default:(scalar_at (beside posterior.Recognition.x0) model.Model.dtype 1.0)
  in
  let t_steps = model.Model.n_steps
  and n = model.Model.n
  and m = model.Model.m
  and n_output = model.Model.n_output in
  let data_shape = Nx.shape data in
  let data_rank = Array.length data_shape in
  if
    data_rank < 2
    || (not (Int.equal data_shape.(data_rank - 2) t_steps))
    || not (Int.equal data_shape.(data_rank - 1) n_output)
  then
    invalid_arg
      (Printf.sprintf
         "Elbo.compute: data must be [n_steps; n_output] or [batch…; n_steps; n_output] \
          with n_steps = %d and n_output = %d, got [%s]"
         t_steps
         n_output
         (shape_string data_shape));
  let batch_dims = Array.sub data_shape ~pos:0 ~len:(data_rank - 2) in
  if Array.exists batch_dims ~f:(fun d -> d <= 0)
  then
    invalid_arg
      (Printf.sprintf
         "Elbo.compute: every batch axis of the data must be positive, got [%s]"
         (shape_string data_shape));
  let batch_rank = Array.length batch_dims in
  let log_prob_shape = Nx.shape posterior.Recognition.log_prob in
  if Array.length log_prob_shape <> batch_rank + 1
  then
    invalid_arg
      (Printf.sprintf
         "Elbo.compute: log_prob must be [batch…; n_samples] matching data [%s], got [%s]"
         (shape_string data_shape)
         (shape_string log_prob_shape));
  let n_samples = log_prob_shape.(batch_rank) in
  if n_samples <= 0 then invalid_arg "Elbo.compute: the posterior batch is empty";
  check_shape
    ~what:"x0"
    (append_shape batch_dims [| n_samples; n |])
    posterior.Recognition.x0;
  check_shape
    ~what:"u"
    (append_shape batch_dims [| t_steps; n_samples; m |])
    posterior.Recognition.u;
  let z =
    Dynamics.integrate_samples
      model.Model.dynamics
      params.Model.dynamics
      ~x0:posterior.Recognition.x0
      ~u:posterior.Recognition.u
  in
  (* Observation [k] is emitted from the state reached by [k + 1] inputs. *)
  let z_by_time = transpose_time ~time_axis:batch_rank z in
  let z_next = slice [ R (1, t_steps + 1) ] z_by_time in
  let data_by_time = transpose_time ~time_axis:batch_rank data in
  let ks = arange_at (beside data) model.Model.dtype 0 t_steps 1 in
  let _, likelihood_by_step =
    Rune.scan
      Nx.Ptree.unit
      Nx.Ptree.(pair (pair tensor tensor) tensor)
      Nx.Ptree.tensor
      (fun () ((z_k, o_k), k) ->
         (* [o_k] carries the trial axes and the observation axis; the singleton
            added before that axis broadcasts it over the sample axis. *)
         let o_k = unsqueeze ~axes:[ -2 ] o_k in
         ( ()
         , model.Model.likelihood.Likelihood.logp_t
             params.Model.likelihood
             ~k
             ~x:z_k
             ~o:o_k ))
      ~init:()
      ((z_next, data_by_time), ks)
  in
  let log_likelihood_by_sample = sum ~axes:[ 0 ] likelihood_by_step in
  let log_prior_by_sample =
    log_prior model params ~x0:posterior.Recognition.x0 ~u:posterior.Recognition.u
  in
  (* The sampled KL, and the bound at [beta]. Both densities are evaluated at
     the same reparameterized draws; {!Recognition.infer} stopped the
     recognition's parameters inside [log_prob], so differentiating this is
     the path-derivative (sticking-the-landing) estimator of Roeder, Wu &
     Duvenaud (2017): the gradient reaches the recognition through the draws
     only. *)
  let kl_by_sample = Infix.(posterior.Recognition.log_prob - log_prior_by_sample) in
  let elbo_by_sample = Infix.(log_likelihood_by_sample - (kl_by_sample * beta)) in
  (* The bound is the mean over every draw of every trial. *)
  { elbo = mean elbo_by_sample
  ; log_likelihood = mean log_likelihood_by_sample
  ; kl = mean kl_by_sample
  }

let loss model params ?beta ?iters ~key ~n_samples data =
  let posterior = Model.infer model params ?iters ~n_samples ~key data in
  Infix.(-(compute model params ?beta ~posterior data).elbo)

let loss_fixed model params ?beta ~iters ~key ~n_samples data =
  loss model params ?beta ~iters ~key ~n_samples data

let batch_loss_fixed model params ?beta ~key ~n_samples ~iters data =
  let shape = Nx.shape data in
  if Array.length shape < 3
  then
    invalid_arg
      (Printf.sprintf
         "Elbo.batch_loss_fixed: data must be [batch…; n_steps; n_output], got [%s]"
         (shape_string shape));
  loss_fixed model params ?beta ~key ~n_samples ~iters data
