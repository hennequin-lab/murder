(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Types

type ('w, 'd, 'l, 'p, 'e, 'r) t =
  { dtype : (float, 'w) Nx.dtype
  ; n : int
  ; m : int
  ; n_steps : int
  ; n_output : int
  ; dynamics : ('w, 'd) Dynamics.t
  ; likelihood : ('w, 'l) Likelihood.t
  ; prior : ('w, 'p) Prior.t
  ; encoder_prior : ('w, 'e) Prior.t
  ; initial_prior : 'w value Initial.params option
  ; recognition : ('w, 'd, 'l, 'p, 'e, 'r) Recognition.t
  }

type ('w, 'd, 'l, 'p, 'e, 'r) params =
  { dynamics : 'd
  ; likelihood : 'l
  ; prior : 'p
  ; encoder_prior : 'e
  ; initial_prior : 'w value Initial.params option
  ; recognition : 'r
  }

let require_dim name value = if value <= 0 then invalid_arg (name ^ " must be positive")

let make
      dtype
      ~n
      ~m
      ~n_steps
      ~n_output
      ~dynamics
      ~likelihood
      ~prior
      ~encoder_prior
      ~recognition
      ?initial_prior
      ()
  =
  require_dim "Model.make: n" n;
  require_dim "Model.make: m" m;
  require_dim "Model.make: n_steps" n_steps;
  require_dim "Model.make: n_output" n_output;
  if not (Int.equal dynamics.Dynamics.n n)
  then
    invalid_arg
      (Printf.sprintf "Model.make: dynamics.n is %d but n is %d" dynamics.Dynamics.n n);
  if not (Int.equal dynamics.Dynamics.m m)
  then
    invalid_arg
      (Printf.sprintf "Model.make: dynamics.m is %d but m is %d" dynamics.Dynamics.m m);
  if not (Int.equal likelihood.Likelihood.n n)
  then
    invalid_arg
      (Printf.sprintf
         "Model.make: likelihood.n is %d but n is %d"
         likelihood.Likelihood.n
         n);
  if not (Int.equal likelihood.Likelihood.n_output n_output)
  then
    invalid_arg
      (Printf.sprintf
         "Model.make: likelihood.n_output is %d but n_output is %d"
         likelihood.Likelihood.n_output
         n_output);
  if not (Int.equal prior.Prior.m m)
  then
    invalid_arg (Printf.sprintf "Model.make: prior.m is %d but m is %d" prior.Prior.m m);
  if not (Int.equal encoder_prior.Prior.m m)
  then
    invalid_arg
      (Printf.sprintf
         "Model.make: encoder_prior.m is %d but m is %d"
         encoder_prior.Prior.m
         m);
  (* The recognition is built for a shape of model; a mismatch would fail
     later, inside a solve, with a shape error far from its cause. *)
  let dims = recognition.Recognition.dims in
  if
    not
      (Int.equal dims.Recognition.n n
       && Int.equal dims.Recognition.m m
       && Int.equal dims.Recognition.n_steps n_steps
       && Int.equal dims.Recognition.n_output n_output)
  then
    invalid_arg
      (Printf.sprintf
         "Model.make: recognition is built for n=%d m=%d n_steps=%d n_output=%d, but the \
          model is n=%d m=%d n_steps=%d n_output=%d"
         dims.Recognition.n
         dims.Recognition.m
         dims.Recognition.n_steps
         dims.Recognition.n_output
         n
         m
         n_steps
         n_output);
  (* A recognition that infers the initial state needs a prior over it, and
     one that fixes [x_0 = 0] must not be given one: the ELBO's initial-state
     term would not cancel. *)
  (match initial_prior with
   | None ->
     if recognition.Recognition.infers_initial
     then
       invalid_arg
         (Printf.sprintf
            "Model.make: the %s recognition infers an initial state; pass ~initial_prior"
            recognition.Recognition.name)
   | Some p ->
     if not recognition.Recognition.infers_initial
     then
       invalid_arg
         (Printf.sprintf
            "Model.make: the %s recognition fixes x_0 = 0; it must not be given an \
             initial_prior"
            recognition.Recognition.name);
     let size = Array.get (shape p.Initial.mean) 0 in
     if not (Int.equal size n)
     then
       invalid_arg
         (Printf.sprintf "Model.make: initial_prior has size %d but must be %d" size n));
  { dtype
  ; n
  ; m
  ; n_steps
  ; n_output
  ; dynamics
  ; likelihood
  ; prior
  ; encoder_prior
  ; initial_prior
  ; recognition
  }

let init (type w d l p e r) (model : (w, d, l, p, e, r) t) : (w, d, l, p, e, r) params =
  { dynamics = model.dynamics.Dynamics.params
  ; likelihood = model.likelihood.Likelihood.params
  ; prior = model.prior.Prior.params
  ; encoder_prior = model.encoder_prior.Prior.params
  ; initial_prior = model.initial_prior
  ; recognition = model.recognition.Recognition.params
  }

let structure (type w d l p e r) (model : (w, d, l, p, e, r) t)
  : (w, d, l, p, e, r) params Nx.Ptree.t
  =
  Nx.Ptree.iso
    (fun (d, (l, (p, (e, (i, r))))) ->
       { dynamics = d
       ; likelihood = l
       ; prior = p
       ; encoder_prior = e
       ; initial_prior = i
       ; recognition = r
       })
    (fun r ->
       ( r.dynamics
       , (r.likelihood, (r.prior, (r.encoder_prior, (r.initial_prior, r.recognition)))) ))
    Nx.Ptree.(
      pair
        (field "dynamics" model.dynamics.Dynamics.structure)
        (pair
           (field "likelihood" model.likelihood.Likelihood.structure)
           (pair
              (field "prior" model.prior.Prior.structure)
              (pair
                 (field "encoder_prior" model.encoder_prior.Prior.structure)
                 (pair
                    (option (field "initial_prior" (Initial.structure ())))
                    (field "recognition" model.recognition.Recognition.structure))))))

(* [generative model] is the part of the model a recognition reads. [gparams params] is the matching part of the parameters. *)
let generative (type w d l p e r) (model : (w, d, l, p, e, r) t)
  : (w, d, l, p, e) Recognition.generative
  =
  { Recognition.dtype = model.dtype
  ; Recognition.n = model.n
  ; Recognition.m = model.m
  ; Recognition.n_steps = model.n_steps
  ; Recognition.n_output = model.n_output
  ; Recognition.dynamics = model.dynamics
  ; Recognition.likelihood = model.likelihood
  ; Recognition.prior = model.prior
  ; Recognition.encoder_prior = model.encoder_prior
  }

let gparams (type w d l p e r) (params : (w, d, l, p, e, r) params)
  : (w, d, l, p, e) Recognition.gparams
  =
  { Recognition.dynamics = params.dynamics
  ; Recognition.likelihood = params.likelihood
  ; Recognition.prior = params.prior
  ; Recognition.encoder_prior = params.encoder_prior
  }

let infer
      (type w d l p e r)
      (model : (w, d, l, p, e, r) t)
      (params : (w, d, l, p, e, r) params)
      ?iters
      ~n_samples
      ~key
      data
  =
  model.recognition.Recognition.infer
    (generative model)
    (gparams params)
    ~params:params.recognition
    ?iters
    ~n_samples
    ~key
    data

let mean
      (type w d l p e r)
      (model : (w, d, l, p, e, r) t)
      (params : (w, d, l, p, e, r) params)
      ?iters
      data
  =
  model.recognition.Recognition.mean
    (generative model)
    (gparams params)
    ~params:params.recognition
    ?iters
    data

let log_prob
      (type w d l p e r)
      (model : (w, d, l, p, e, r) t)
      (params : (w, d, l, p, e, r) params)
      ?iters
      data
      samples
  =
  model.recognition.Recognition.log_prob
    (generative model)
    (gparams params)
    ~params:params.recognition
    ?iters
    data
    samples
