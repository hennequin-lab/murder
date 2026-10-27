(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
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

type 'a params =
  { mean : 'a
  ; log_stds : 'a
  }
[@@deriving ptree]

let structure () =
  Nx.Ptree.instantiate
    (module struct
      type 'a t = 'a params

      let walk = walk_params
    end)

let make ?(mean = 0.0) ?(std = 1.0) ~n dtype =
  if n <= 0 then invalid_arg "Initial.make: n must be positive";
  if not (Float.( > ) std 0.0) then invalid_arg "Initial.make: std must be positive";
  { mean = full dtype [| n |] mean; log_stds = full dtype [| n |] (Float.log std) }

let log_2pi = Float.log (2.0 *. Float.pi)

(* [logp p x] is [log p(x)] summed over the last axis: one value per row of a
   batched [x], a scalar for a single state. *)
let logp p x =
  let open Infix in
  let scaled = (x - p.mean) / exp p.log_stds in
  (* [log (2 pi sigma^2)] is [log 2 pi + 2 log sigma]. *)
  let log_scale = (p.log_stds *$ 2.0) + scalar_like p.log_stds log_2pi in
  sum ~axes:[ -1 ] ((scaled * scaled) + log_scale) *$ -0.5

(* [sample p key ~n_samples] draws [n_samples] states, of shape
   [[n_samples; n]]. Reparameterized and differentiable. *)
let sample p ~key ~n_samples =
  if n_samples <= 0 then invalid_arg "Initial.sample: n_samples must be positive";
  let n = Array.get (shape p.mean) 0 in
  let dtype = Nx.dtype p.mean in
  let eps = Nx.Rng.normal key dtype [| n_samples; n |] in
  Infix.(reshape [| 1; n |] p.mean + (eps * exp p.log_stds))
