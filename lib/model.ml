(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Types

type ('w, 'd, 'l, 'p, 'e) t =
  { dtype : (float, 'w) Nx.dtype
  ; n : int
  ; m : int
  ; n_steps : int
  ; n_output : int
  ; dynamics : ('w, 'd) Dynamics.t
  ; likelihood : ('w, 'l) Likelihood.t
  ; prior : ('w, 'p) Prior.t
  ; encoder_prior : ('w, 'e) Prior.t
  ; space_cov : 'w value Covariance.params
  ; time_cov : 'w value Covariance.params
  }

type ('w, 'd, 'l, 'p, 'e) params =
  { dynamics : 'd
  ; likelihood : 'l
  ; prior : 'p
  ; encoder_prior : 'e
  ; space_cov : 'w value Covariance.params
  ; time_cov : 'w value Covariance.params
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
      ?space_cov
      ?time_cov
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
  let covariance name value expected =
    let size = Array.get (Nx.shape value.Covariance.d) 0 in
    if not (Int.equal size expected)
    then
      invalid_arg
        (Printf.sprintf "Model.make: %s has size %d but must be %d" name size expected)
  in
  let space_cov = Option.value space_cov ~default:(Covariance.make ~n:m dtype) in
  let time_cov = Option.value time_cov ~default:(Covariance.make ~n:n_steps dtype) in
  covariance "space_cov" space_cov m;
  covariance "time_cov" time_cov n_steps;
  { dtype
  ; n
  ; m
  ; n_steps
  ; n_output
  ; dynamics
  ; likelihood
  ; prior
  ; encoder_prior
  ; space_cov
  ; time_cov
  }

let init (type w d l p e) (model : (w, d, l, p, e) t) : (w, d, l, p, e) params =
  { dynamics = model.dynamics.Dynamics.params
  ; likelihood = model.likelihood.Likelihood.params
  ; prior = model.prior.Prior.params
  ; encoder_prior = model.encoder_prior.Prior.params
  ; space_cov = model.space_cov
  ; time_cov = model.time_cov
  }

let structure (type w d l p e) (model : (w, d, l, p, e) t)
  : (w, d, l, p, e) params Nx.Ptree.t
  =
  Nx.Ptree.iso
    (fun (d, (l, (p, (e, (sc, tc))))) ->
       { dynamics = d
       ; likelihood = l
       ; prior = p
       ; encoder_prior = e
       ; space_cov = sc
       ; time_cov = tc
       })
    (fun r ->
       r.dynamics, (r.likelihood, (r.prior, (r.encoder_prior, (r.space_cov, r.time_cov)))))
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
                    (field "space_cov" (Covariance.structure ()))
                    (field "time_cov" (Covariance.structure ())))))))
