(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Types

type ('w, 'd, 'l, 'p, 'e, 'r, 's) t =
  { dtype : (float, 'w) Nx.dtype
  ; n : int
  ; m : int
  ; n_steps : int
  ; n_output : int
  ; dynamics : ('w, 'd, 's) Dynamics.t
  ; likelihood : ('w, 'l, 's) Likelihood.t
  ; prior : ('w, 'p, 's) Prior.t
  ; encoder_prior : ('w, 'e, 's) Prior.t
  ; initial_prior : (('w, 's) value, 's) Initial.params option
  ; recognition : ('w, 'd, 'l, 'p, 'e, 'r, 's) Recognition.t
  }

type ('w, 'd, 'l, 'p, 'e, 'r, 's) params =
  { dynamics : 'd
  ; likelihood : 'l
  ; prior : 'p
  ; encoder_prior : 'e
  ; initial_prior : (('w, 's) value, 's) Initial.params option
  ; recognition : 'r
  }

let require_dim name value = if value <= 0 then invalid_arg (name ^ " must be positive")

(* [what] is the dimension the model takes as its own, [who] the component
   field that must agree with it. *)
let require_agree ~who ~who_is what expected =
  if not (Int.equal who_is expected)
  then
    invalid_arg
      (Printf.sprintf "Model.make: %s is %d but %s is %d" who who_is what expected)

let make dtype ~dynamics ~likelihood ~prior ~encoder_prior ~recognition ?initial_prior () =
  (* Every dimension is already a field of a component: the state and input
     dimensions of the dynamics, the observation dimension of the likelihood,
     the horizon of the recognition. The model takes them from there and only
     checks that the components agree with each other. *)
  let n = dynamics.Dynamics.n in
  let m = dynamics.Dynamics.m in
  let n_output = likelihood.Likelihood.n_output in
  let n_steps = recognition.Recognition.dims.Recognition.n_steps in
  require_dim "Model.make: dynamics.n" n;
  require_dim "Model.make: dynamics.m" m;
  require_dim "Model.make: likelihood.n_output" n_output;
  require_dim "Model.make: recognition n_steps" n_steps;
  require_agree ~who:"likelihood.n" ~who_is:likelihood.Likelihood.n "dynamics.n" n;
  require_agree ~who:"prior.m" ~who_is:prior.Prior.m "dynamics.m" m;
  require_agree ~who:"encoder_prior.m" ~who_is:encoder_prior.Prior.m "dynamics.m" m;
  (* The recognition is built for a shape of model; a mismatch would fail
     later, inside a solve, with a shape error far from its cause. *)
  let dims = recognition.Recognition.dims in
  require_agree ~who:"recognition.n" ~who_is:dims.Recognition.n "dynamics.n" n;
  require_agree ~who:"recognition.m" ~who_is:dims.Recognition.m "dynamics.m" m;
  require_agree
    ~who:"recognition.n_output"
    ~who_is:dims.Recognition.n_output
    "likelihood.n_output"
    n_output;
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
         (Printf.sprintf
            "Model.make: initial_prior has size %d but dynamics.n is %d"
            size
            n));
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

let params (type w d l p e r) (model : (w, d, l, p, e, r, 's) t)
  : (w, d, l, p, e, r, 's) params
  =
  { dynamics = model.dynamics.Dynamics.params
  ; likelihood = model.likelihood.Likelihood.params
  ; prior = model.prior.Prior.params
  ; encoder_prior = model.encoder_prior.Prior.params
  ; initial_prior = model.initial_prior
  ; recognition = model.recognition.Recognition.params
  }

let structure (type w d l p e r) (model : (w, d, l, p, e, r, 's) t)
  : ((w, d, l, p, e, r, 's) params, 's) Nx.Ptree.t
  =
  Nx.Ptree.v (fun c r ->
    let open Nx.Ptree.Walk in
    let dynamics =
      field c "dynamics" (structure model.dynamics.Dynamics.structure) r.dynamics
    in
    let likelihood =
      field c "likelihood" (structure model.likelihood.Likelihood.structure) r.likelihood
    in
    let prior = field c "prior" (structure model.prior.Prior.structure) r.prior in
    let encoder_prior =
      field
        c
        "encoder_prior"
        (structure model.encoder_prior.Prior.structure)
        r.encoder_prior
    in
    let initial_prior =
      field c "initial_prior" (option (structure (Initial.structure ()))) r.initial_prior
    in
    let recognition =
      field
        c
        "recognition"
        (structure model.recognition.Recognition.structure)
        r.recognition
    in
    { dynamics; likelihood; prior; encoder_prior; initial_prior; recognition })

(* [generative model] is the part of the model a recognition reads. [gparams params] is the matching part of the parameters. *)
let generative (type w d l p e r) (model : (w, d, l, p, e, r, 's) t)
  : (w, d, l, p, e, 's) Recognition.generative
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
  ; Recognition.initial_prior = model.initial_prior
  }

let gparams (type w d l p e r) (params : (w, d, l, p, e, r, 's) params)
  : (w, d, l, p, e, 's) Recognition.gparams
  =
  { Recognition.dynamics = params.dynamics
  ; Recognition.likelihood = params.likelihood
  ; Recognition.prior = params.prior
  ; Recognition.encoder_prior = params.encoder_prior
  ; Recognition.initial_prior = params.initial_prior
  }

let infer
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) t)
      (params : (w, d, l, p, e, r, 's) params)
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
      (model : (w, d, l, p, e, r, 's) t)
      (params : (w, d, l, p, e, r, 's) params)
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
      (model : (w, d, l, p, e, r, 's) t)
      (params : (w, d, l, p, e, r, 's) params)
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
