(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Types
(* The generative prior over the initial state [x_0].

   The library's own generative model starts every trial at [x_0 = 0] and
   leaves the initial state out of the latent variables: iLQR-style
   recognition infers only the inputs and a model built without an
   {!Initial.t} keeps that convention. An LFADS-style recognition infers an
   initial state as well, and then a prior over it is needed to weigh the
   posterior against; {!Model} takes one as [~initial_prior].

   The distribution is a factorized Gaussian, [p(x_0) = N(mu, diag (exp (2 log_stds)))], with both the mean and the log standard deviations
   learned like any other parameter. The paper initializes [p(x_0)] at
   [N(0, kappa I)]; the reference implementation makes it learnable, and so
   does this one. *)

(* The prior as a description: what it is, not the value it starts at. It
   holds no tensor, so it needs no dtype and no placement. *)
type t =
  { n : int
  ; mean : float
  ; std : float
  }

type ('a, _) params =
  { mean : 'a
  ; log_stds : 'a
  }
[@@deriving ptree]

let make ?(mean = 0.0) ?(std = 1.0) ~n () =
  if n <= 0 then invalid_arg "Initial.make: n must be positive";
  if not (Float.( > ) std 0.0) then invalid_arg "Initial.make: std must be positive";
  { n; mean; std }

let init ~at (type w) (dtype : (float, w) Nx.dtype) (p : t) : ((w, 's) value, 's) params =
  { mean = full_at at dtype [| p.n |] p.mean
  ; log_stds = full_at at dtype [| p.n |] (Float.log p.std)
  }

let structure () =
  Nx.Ptree.instantiate
    (module struct
      type ('a, 's) t = ('a, 's) params

      let walk = walk_params
    end)

let place ~at p =
  Nx.Ptree.place
    (module struct
      type ('a, 's) t = ('a, 's) params

      let walk = walk_params
    end)
    at
    p

(* [logp p x] is [log p(x)] summed over the last axis: one value per row of a
   batched [x], a scalar for a single state. *)
let logp p x =
  Density.logp ~mean:p.mean ~covariance:(Density.Diagonal { log_stds = p.log_stds }) x

(* [sample p key ~n_samples] draws [n_samples] states, of shape
   [[n_samples; n]]. Reparameterized and differentiable. *)
let sample p ~key ~n_samples =
  if n_samples <= 0 then invalid_arg "Initial.sample: n_samples must be positive";
  let n = Array.get (shape p.mean) 0 in
  let dtype = Nx.dtype p.mean in
  let eps = Nx.Rng.normal_at (beside p.mean) ~key dtype [| n_samples; n |] in
  Infix.(reshape [| 1; n |] p.mean + (eps * exp p.log_stds))
