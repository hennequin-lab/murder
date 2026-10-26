(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Types

type ('w, 'p) t =
  { n : int
  ; n_output : int
  ; structure : 'p Nx.Ptree.t
  ; params : 'p
  ; logp_t : 'p -> k:'w value -> x:'w value -> o:'w value -> 'w value
  ; mean : 'p -> 'w value -> 'w value
  ; sample : 'p -> key:Nx.Rng.t -> mean:'w value -> 'w value
  }

let log_2pi = Float.log (2.0 *. Float.pi)

let randn ?key dtype shape =
  match key with
  | Some key -> Nx.Rng.normal key dtype shape
  | None -> Nx.randn dtype shape

(* [readout c x] is [x @ cᵀ] for a state row or a batch of rows: the last axis
   of [x] is contracted with the second axis of [c], whatever leading axes [x]
   carries. *)
let readout c x = Nx.matmul x (Nx.matrix_transpose c)

(* {1 Gaussian} *)

module Gaussian = struct
  type 'a params =
    { c : 'a
    ; bias : 'a
    ; log_variances : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type 'a t = 'a params

        let walk = walk_params
      end)

  let make ?key ?c_mask ?(scale = 1.0) ?(sigma2 = 1.0) ?(bias = 0.0) ~n ~n_output dtype =
    if n <= 0 then invalid_arg "Likelihood.Gaussian.make: n must be positive";
    if n_output <= 0
    then invalid_arg "Likelihood.Gaussian.make: n_output must be positive";
    if not (Float.( > ) scale 0.0)
    then invalid_arg "Likelihood.Gaussian.make: scale must be positive";
    if not (Float.( > ) sigma2 0.0)
    then invalid_arg "Likelihood.Gaussian.make: sigma2 must be positive";
    let c =
      Nx.mul_s (randn ?key dtype [| n_output; n |]) (scale /. Float.sqrt (Float.of_int n))
    in
    let c =
      match c_mask with
      | None -> c
      | Some mask -> Nx.mul c mask
    in
    let mean p x = Nx.add (readout p.c x) p.bias in
    (* The residual is [mean − o] rather than [o − mean]. Squaring makes the
       two identical, but the constant-first subtraction makes tolk's
       bottom-up rewrite cycle on the graph the encoder's scans build. *)
    let logp_t p ~k:_ ~x ~o =
      let diff = Nx.sub (mean p x) o in
      let variances = Nx.exp p.log_variances in
      let term =
        Nx.add (Nx.div (Nx.mul diff diff) variances) (Nx.add_s (Nx.log variances) log_2pi)
      in
      Nx.mul_s (Nx.sum ~axes:[ -1 ] term) (-0.5)
    in
    let sample p ~key ~mean =
      let noise = Nx.Rng.normal key dtype (Nx.shape mean) in
      Nx.add mean (Nx.mul noise (Nx.sqrt (Nx.exp p.log_variances)))
    in
    { n
    ; n_output
    ; structure = structure ()
    ; params =
        { c
        ; bias = Nx.full dtype [| n_output |] bias
        ; log_variances = Nx.full dtype [| n_output |] (Float.log sigma2)
        }
    ; logp_t
    ; mean
    ; sample
    }
end

(* {1 Poisson} *)

module Poisson = struct
  type 'a params =
    { c : 'a
    ; bias : 'a
    ; log_gain : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type 'a t = 'a params

        let walk = walk_params
      end)

  let default_link x = Nx.add_s (Nx.exp x) 1e-3

  let make ?key ?c_mask ?link ?(dt = 1.0) ~n ~n_output dtype =
    if n <= 0 then invalid_arg "Likelihood.Poisson.make: n must be positive";
    if n_output <= 0 then invalid_arg "Likelihood.Poisson.make: n_output must be positive";
    if not (Float.( > ) dt 0.0)
    then invalid_arg "Likelihood.Poisson.make: dt must be positive";
    let link = Option.value link ~default:default_link in
    let dt = Nx.scalar dtype dt in
    let c =
      Nx.mul_s (randn ?key dtype [| n_output; n |]) (1.0 /. Float.sqrt (Float.of_int n))
    in
    let c =
      match c_mask with
      | None -> c
      | Some mask -> Nx.mul c mask
    in
    let mean p x =
      let drive = Nx.add (readout p.c x) p.bias in
      Nx.mul (Nx.mul (link drive) (Nx.exp p.log_gain)) dt
    in
    let logp_t p ~k:_ ~x ~o =
      let rate = mean p x in
      let log_rate = Nx.log rate in
      let ll = Nx.sum ~axes:[ -1 ] (Nx.sub (Nx.mul o log_rate) rate) in
      let log_factorial = Nx.sum ~axes:[ -1 ] (Special.lgamma (Nx.add_s o 1.0)) in
      Nx.sub ll log_factorial
    in
    let sample _p ~key ~mean = Nx.cast dtype (Nx.Rng.poisson key mean) in
    { n
    ; n_output
    ; structure = structure ()
    ; params =
        { c
        ; bias = Nx.zeros dtype [| n_output |]
        ; log_gain = Nx.zeros dtype [| n_output |]
        }
    ; logp_t
    ; mean
    ; sample
    }
end
