(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Types

(* [controls_of model tau] is the control part of a stacked trajectory: rows
   [0 … n_steps − 1], columns [n … n + m − 1]. *)
let controls_of (type w d l p e) (model : (w, d, l, p, e) Model.t) (tau : w value) =
  let n = model.Model.n
  and m = model.Model.m
  and t_steps = model.Model.n_steps in
  Nx.slice [ R (0, t_steps); R (n, n + m) ] tau

let initial_controls (type w d l p e) (model : (w, d, l, p, e) Model.t) u_init =
  let dtype = model.Model.dtype
  and m = model.Model.m
  and t_steps = model.Model.n_steps in
  match u_init with
  | None -> Array.init t_steps ~f:(fun _ -> Nx.zeros dtype [| m |])
  | Some u ->
    if
      Array.length (Nx.shape u) <> 2
      || Array.get (Nx.shape u) 0 <> t_steps
      || Array.get (Nx.shape u) 1 <> m
    then invalid_arg "Encoder: u_init must be [n_steps; m]";
    Array.init t_steps ~f:(fun k -> Nx.get [ k ] u)

let problem
      (type w d l p e)
      (model : (w, d, l, p, e) Model.t)
      (params : (w, d, l, p, e) Model.params)
      (data : w value)
  : (w, (w, d, l, p, e) Model.params) Dilqr.Problem.t
  =
  let n = model.Model.n
  and m = model.Model.m
  and t_steps = model.Model.n_steps
  and n_output = model.Model.n_output
  and dtype = model.Model.dtype in
  let shape = Nx.shape data in
  if
    Array.length shape <> 2
    || Array.get shape 0 <> t_steps
    || Array.get shape 1 <> n_output
  then
    invalid_arg (Printf.sprintf "Encoder.problem: data must be [%d; %d]" t_steps n_output);
  let zero = Nx.scalar dtype 0.0
  and one = Nx.scalar dtype 1.0 in
  let last = Nx.get [ t_steps - 1 ] data in
  (* The running loss at step [k] carries the recognition prior on [u_k] and
     — from step 1 on — the likelihood of [o_{k−1}] at [x_k]; the terminal
     loss carries the last observation. [o] is selected at the traced step
     index with [Nx.take], clamped so that step 0 reads a valid row and is
     then masked out. *)
  let running_loss
        (theta : (w, d, l, p, e) Model.params)
        ~(k : w value)
        ~(x : w value)
        ~(u : w value)
    =
    let prior_term =
      Nx.neg (model.Model.encoder_prior.Prior.logp_t theta.Model.encoder_prior ~k ~u)
    in
    let index = Nx.maximum (Nx.sub k one) zero in
    (* [take] with a 0-d [indices] removes the axis, so the row is [n_output]. *)
    let o = Nx.take ~axis:0 ~indices:(Nx.cast Nx.int64 index) data in
    let likelihood_term =
      Nx.neg (model.Model.likelihood.Likelihood.logp_t theta.Model.likelihood ~k ~x ~o)
    in
    Nx.add prior_term (Nx.where (Nx.equal k zero) zero likelihood_term)
  in
  let final_loss (theta : (w, d, l, p, e) Model.params) ~(x : w value) =
    Nx.neg
      (model.Model.likelihood.Likelihood.logp_t
         theta.Model.likelihood
         ~k:(Nx.scalar dtype (Float.of_int t_steps))
         ~x
         ~o:last)
  in
  { Dilqr.Problem.dtype
  ; n
  ; m
  ; horizon = t_steps
  ; verbose = false
  ; params = Model.structure model
  ; dynamics =
      (fun theta ~k ~x ~u ->
        model.Model.dynamics.Dynamics.dyn theta.Model.dynamics ~k ~x ~u)
  ; running_loss
  ; final_loss
  }

let posterior_mean model params ?u_init ?(max_iter = 20) ?(conv_threshold = 1e-6) data =
  let p = problem model params data in
  let inputs : (_, _) Dilqr.Problem.Inputs.t =
    { theta = params
    ; x0 = Nx.zeros model.Model.dtype [| model.Model.n |]
    ; us0 = initial_controls model u_init
    }
  in
  controls_of model (Dilqr.solve_ilqr p ~max_iter ~conv_threshold inputs)

let posterior_mean_fixed model params ?u_init ~iters ?alphas ?mu ?c1 data =
  let p = problem model params data in
  let inputs : (_, _) Dilqr.Problem.Inputs.t =
    { theta = params
    ; x0 = Nx.zeros model.Model.dtype [| model.Model.n |]
    ; us0 = initial_controls model u_init
    }
  in
  controls_of model (Dilqr.solve_ilqr_fixed p ~iters ?alphas ?mu ?c1 inputs)
