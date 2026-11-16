(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Types

type ('w, 'p, 's) t =
  { n : int
  ; n_output : int
  ; structure : ('p, 's) Nx.Ptree.t
  ; init : ?key:'s Nx.Rng.key -> (float, 'w) Nx.dtype -> 'p
  ; logp_t :
      'p -> k:('w, 's) value -> x:('w, 's) value -> o:('w, 's) value -> ('w, 's) value
  ; mean : 'p -> ('w, 's) value -> ('w, 's) value
  ; sample : 'p -> key:'s Nx.Rng.key -> mean:('w, 's) value -> ('w, 's) value
  }

let make ~n ~n_output ~structure ~init ~logp_t ~mean ~sample () =
  if n <= 0 then invalid_arg "Likelihood.make: n must be positive";
  if n_output <= 0 then invalid_arg "Likelihood.make: n_output must be positive";
  { n; n_output; structure; init; logp_t; mean; sample }

let n (l : ('w, 'p, 's) t) = l.n
let n_output (l : ('w, 'p, 's) t) = l.n_output
let structure (l : ('w, 'p, 's) t) = l.structure
let init ?key dtype (l : ('w, 'p, 's) t) = l.init ?key dtype
let logp_t (l : ('w, 'p, 's) t) = l.logp_t
let mean (l : ('w, 'p, 's) t) = l.mean
let sample (l : ('w, 'p, 's) t) = l.sample

(* [randn ~at ?key dtype shape] is a standard normal draw at [at], from [key]
   when given and from the implicit random scope otherwise. *)
let randn ~at ?key dtype shape =
  match key with
  | Some key -> Nx.Rng.normal_at at ~key dtype shape
  | None -> Nx.Rng.normal_at at dtype shape

(* [readout c x] is [x @ cᵀ] for a state row or a batch of rows: the last axis
   of [x] is contracted with the second axis of [c], whatever leading axes [x]
   carries. Both likelihoods here are stationary, so [logp_t] ignores the step
   index [k], which names the state's step and not the datum's. *)
let readout c x = Infix.(x *@ matrix_transpose c)

(* {1 Gaussian} *)

module Gaussian = struct
  type ('a, _) params =
    { c : 'a
    ; bias : 'a
    ; log_variances : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)

  let make
        (type w)
        ~at
        ?c_mask
        ?(scale = 1.0)
        ?(sigma2 = 1.0)
        ?(bias = 0.0)
        ~n
        ~n_output
        ()
    =
    if n <= 0 then invalid_arg "Likelihood.Gaussian.make: n must be positive";
    if n_output <= 0
    then invalid_arg "Likelihood.Gaussian.make: n_output must be positive";
    if not (Float.( > ) scale 0.0)
    then invalid_arg "Likelihood.Gaussian.make: scale must be positive";
    if not (Float.( > ) sigma2 0.0)
    then invalid_arg "Likelihood.Gaussian.make: sigma2 must be positive";
    let init ?key (dtype : (float, w) Nx.dtype) =
      let c =
        Infix.(
          randn ~at ?key dtype [| n_output; n |] *$ (scale /. Float.sqrt (Float.of_int n)))
      in
      let c =
        match c_mask with
        | None -> c
        | Some mask -> Infix.(c * mask)
      in
      { c
      ; bias = full_at at dtype [| n_output |] bias
      ; log_variances = full_at at dtype [| n_output |] (Float.log sigma2)
      }
    in
    let mean (p : (_, _) params) x = Infix.(readout p.c x + p.bias) in
    let logp_t (p : (_, _) params) ~k:_ ~x ~o =
      Density.logp
        ~mean:(mean p x)
        ~covariance:(Density.Diagonal { log_stds = Infix.(p.log_variances *$ 0.5) })
        o
    in
    let sample (p : (_, _) params) ~key ~mean =
      let noise = Nx.Rng.normal_at (beside mean) ~key (Nx.dtype mean) (shape mean) in
      Infix.(mean + (noise * sqrt (exp p.log_variances)))
    in
    make ~n ~n_output ~structure:(structure ()) ~init ~logp_t ~mean ~sample ()

  let place ~at p =
    Nx.Ptree.place
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)
      at
      p
end

(* {1 Poisson} *)

module Poisson = struct
  type ('a, _) params =
    { c : 'a
    ; bias : 'a
    ; log_gain : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)

  let default_link x = Infix.(exp x +$ 1e-3)

  let make
        (type w)
        ~at
        ?c_mask
        ?link
        ?(dt = 1.0)
        ?(bias = 0.)
        ?(scale = 1.)
        ?(log_gain = 0.)
        ~n
        ~n_output
        ()
    =
    if n <= 0 then invalid_arg "Likelihood.Poisson.make: n must be positive";
    if n_output <= 0 then invalid_arg "Likelihood.Poisson.make: n_output must be positive";
    if not (Float.( > ) dt 0.0)
    then invalid_arg "Likelihood.Poisson.make: dt must be positive";
    let link = Option.value link ~default:default_link in
    let init ?key (dtype : (float, w) Nx.dtype) =
      let c =
        Infix.(
          randn ~at ?key dtype [| n_output; n |] *$ (scale /. Float.sqrt (Float.of_int n)))
      in
      let c =
        match c_mask with
        | None -> c
        | Some mask -> Infix.(c * mask)
      in
      { c
      ; bias = full_at at dtype [| n_output |] bias
      ; log_gain = full_at at dtype [| n_output |] log_gain
      }
    in
    let mean (p : (_, _) params) x =
      let open Infix in
      let drive = readout p.c x + p.bias in
      link drive * exp p.log_gain *$ dt
    in
    let logp_t (p : (_, _) params) ~k:_ ~x ~o =
      let open Infix in
      let rate = mean p x in
      let log_rate = log rate in
      let ll = sum ~axes:[ -1 ] ((o * log_rate) - rate) in
      let log_factorial = sum ~axes:[ -1 ] (Special.lgamma (o +$ 1.0)) in
      ll - log_factorial
    in
    let sample _p ~key ~mean = cast (Nx.dtype mean) (Nx.Rng.poisson ~key mean) in
    make ~n ~n_output ~structure:(structure ()) ~init ~logp_t ~mean ~sample ()

  let place ~at p =
    Nx.Ptree.place
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)
      at
      p
end
