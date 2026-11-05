(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
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
  ; initial_prior : Initial.t option
  ; recognition : ('w, 'd, 'l, 'e, 'r, 's) Recognition.t
  }

type ('w, 'd, 'l, 'p, 'e, 'r, 's) params =
  { dynamics : 'd
  ; likelihood : 'l
  ; prior : 'p
  ; encoder_prior : 'e
  ; initial_prior : (('w, 's) value, 's) Initial.params option
  ; recognition : 'r
  }

let require_dim field value =
  if value <= 0 then invalid_arg ("Model.make: " ^ field ^ " must be positive")

(* [require_agree ~left ~right actual expected] requires the dimension named
   [left] to equal the one named [right], and says both in the error. *)
let require_agree ~left ~right actual expected =
  if not (Int.equal actual expected)
  then
    invalid_arg
      (Printf.sprintf "Model.make: %s is %d but %s is %d" left actual right expected)

(* [check_initial_prior ~n p] requires the prior's dimension to be the model's. *)
let check_initial_prior ~n (p : Initial.t) =
  if not (Int.equal p.Initial.n n)
  then
    invalid_arg
      (Printf.sprintf
         "Model.make: initial_prior is over %d dimensions but dynamics.n is %d"
         p.Initial.n
         n)

let dtype (m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t) = m.dtype
let n (m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t) = m.n
let m (m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t) = m.m
let n_steps (m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t) = m.n_steps
let n_output (m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t) = m.n_output
let dynamics (m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t) = m.dynamics
let likelihood (m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t) = m.likelihood
let prior (m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t) = m.prior
let encoder_prior (m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t) = m.encoder_prior
let initial_prior (m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t) = m.initial_prior
let recognition (m : ('w, 'd, 'l, 'p, 'e, 'r, 's) t) = m.recognition

let make
      (type w d l p e r)
      (dtype : (float, w) Nx.dtype)
      ~(dynamics : (w, d, 's) Dynamics.t)
      ~(likelihood : (w, l, 's) Likelihood.t)
      ~(prior : (w, p, 's) Prior.t)
      ~(encoder_prior : (w, e, 's) Prior.t)
      ~(recognition : (w, d, l, e, r, 's) Recognition.t)
      ?(initial_prior : Initial.t option)
      ()
  : (w, d, l, p, e, r, 's) t
  =
  (* Every dimension is already a field of a component: the state and input
     dimensions of the dynamics, the observation dimension of the likelihood,
     the horizon of the recognition. The model takes them from there and only
     checks that the components agree with each other. *)
  let n = Dynamics.n dynamics in
  let m = Dynamics.m dynamics in
  let n_output = Likelihood.n_output likelihood in
  let n_steps = Recognition.dims recognition |> fun d -> d.n_steps in
  require_dim "dynamics.n" n;
  require_dim "dynamics.m" m;
  require_dim "likelihood.n_output" n_output;
  require_dim "recognition n_steps" n_steps;
  require_agree ~left:"likelihood.n" ~right:"dynamics.n" (Likelihood.n likelihood) n;
  require_agree ~left:"prior.m" ~right:"dynamics.m" (Prior.m prior) m;
  require_agree ~left:"encoder_prior.m" ~right:"dynamics.m" (Prior.m encoder_prior) m;
  (* The recognition is built for a shape of model; a mismatch would fail
     later, inside a solve, with a shape error far from its cause. *)
  let dims = Recognition.dims recognition in
  require_agree ~left:"recognition.n" ~right:"dynamics.n" dims.Recognition.n n;
  require_agree ~left:"recognition.m" ~right:"dynamics.m" dims.Recognition.m m;
  require_agree
    ~left:"recognition.n_output"
    ~right:"likelihood.n_output"
    dims.Recognition.n_output
    n_output;
  (* A recognition that infers the initial state needs a prior over it, and
     one that fixes [x_0 = 0] must not be given one: the ELBO's initial-state
     term would not cancel. *)
  (match initial_prior with
   | None ->
     if Recognition.infers_initial recognition
     then
       invalid_arg
         (Printf.sprintf
            "Model.make: the %s recognition infers an initial state; pass ~initial_prior"
            (Recognition.name recognition))
   | Some p ->
     if not (Recognition.infers_initial recognition)
     then
       invalid_arg
         (Printf.sprintf
            "Model.make: the %s recognition fixes x_0 = 0; it must not be given an \
             initial_prior"
            (Recognition.name recognition));
     check_initial_prior ~n p);
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

(* [Model.init ~at ?key model] draws every component's parameters at [at],
   one subkey each, and the initial-state prior's -- which is deterministic --
   at the same placement. *)
let init
      (type w d l p e r)
      ~(at : 's Nx.Placement.t)
      ?key
      (model : (w, d, l, p, e, r, 's) t)
  : (w, d, l, p, e, r, 's) params
  =
  let keys = Option.map key ~f:(fun key -> Nx.Rng.split ~n:5 key) in
  let sub i = Option.map keys ~f:(fun keys -> keys.(i)) in
  let dtype = model.dtype in
  { dynamics = Dynamics.init ?key:(sub 0) dtype model.dynamics
  ; likelihood = Likelihood.init ?key:(sub 1) dtype model.likelihood
  ; prior = Prior.init ?key:(sub 2) dtype model.prior
  ; encoder_prior = Prior.init ?key:(sub 3) dtype model.encoder_prior
  ; initial_prior = Option.map model.initial_prior ~f:(Initial.init ~at dtype)
  ; recognition = Recognition.init ?key:(sub 4) dtype model.recognition
  }

let structure (type w d l p e r) (model : (w, d, l, p, e, r, 's) t)
  : ((w, d, l, p, e, r, 's) params, 's) Nx.Ptree.t
  =
  Nx.Ptree.v (fun c r ->
    let open Nx.Ptree.Walk in
    let dynamics =
      field c "dynamics" (structure (Dynamics.structure model.dynamics)) r.dynamics
    in
    let likelihood =
      field
        c
        "likelihood"
        (structure (Likelihood.structure model.likelihood))
        r.likelihood
    in
    let prior = field c "prior" (structure (Prior.structure model.prior)) r.prior in
    let encoder_prior =
      field
        c
        "encoder_prior"
        (structure (Prior.structure model.encoder_prior))
        r.encoder_prior
    in
    let initial_prior =
      field c "initial_prior" (option (structure (Initial.structure ()))) r.initial_prior
    in
    let recognition =
      field
        c
        "recognition"
        (structure (Recognition.structure model.recognition))
        r.recognition
    in
    { dynamics; likelihood; prior; encoder_prior; initial_prior; recognition })

(* [generative model] is the part of the model a recognition reads. [gparams params] is the matching part of the parameters. *)
let generative (type w d l p e r) (model : (w, d, l, p, e, r, 's) t)
  : (w, d, l, e, 's) Recognition.generative
  =
  { Recognition.dtype = model.dtype
  ; Recognition.n = model.n
  ; Recognition.m = model.m
  ; Recognition.n_steps = model.n_steps
  ; Recognition.n_output = model.n_output
  ; Recognition.dynamics = model.dynamics
  ; Recognition.likelihood = model.likelihood
  ; Recognition.encoder_prior = model.encoder_prior
  }

let gparams (type w d l p e r) (params : (w, d, l, p, e, r, 's) params)
  : (w, d, l, e, 's) Recognition.gparams
  =
  { Recognition.dynamics = params.dynamics
  ; Recognition.likelihood = params.likelihood
  ; Recognition.encoder_prior = params.encoder_prior
  ; Recognition.initial_prior = params.initial_prior
  }

let infer
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) t)
      (params : (w, d, l, p, e, r, 's) params)
      ?solve
      ~n_samples
      ~key
      data
  =
  Recognition.infer
    model.recognition
    (generative model)
    (gparams params)
    ~params:params.recognition
    ?solve
    ~n_samples
    ~key
    data

let mean
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) t)
      (params : (w, d, l, p, e, r, 's) params)
      ?solve
      data
  =
  Recognition.mean
    model.recognition
    (generative model)
    (gparams params)
    ~params:params.recognition
    ?solve
    data

let log_prob
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) t)
      (params : (w, d, l, p, e, r, 's) params)
      ?solve
      data
      samples
  =
  Recognition.log_prob
    model.recognition
    (generative model)
    (gparams params)
    ~params:params.recognition
    ?solve
    data
    samples
