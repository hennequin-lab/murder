(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Types

type ('w, 'p, 's) t =
  { m : int
  ; structure : ('p, 's) Nx.Ptree.t
  ; logp_t : 'p -> k:('w, 's) value -> u:('w, 's) value -> ('w, 's) value
  ; sample : 'p -> key:'s Nx.Rng.key -> n_steps:int -> ('w, 's) value
  ; kl_to_gaussian :
      ('p
       -> mu:('w, 's) value
       -> space_factor:('w, 's) value
       -> time_factor:('w, 's) value
       -> ('w, 's) value)
        option
  ; init : ?key:'s Nx.Rng.key -> (float, 'w) Nx.dtype -> 'p
  }

let make ~m ~structure ~init ~logp_t ~sample ?kl_to_gaussian () =
  if m <= 0 then invalid_arg "Prior.make: m must be positive";
  { m; structure; init; logp_t; sample; kl_to_gaussian }

let m (p : ('w, 'p, 's) t) = p.m
let structure (p : ('w, 'p, 's) t) = p.structure
let init ?key dtype (p : ('w, 'p, 's) t) = p.init ?key dtype
let logp_t (p : ('w, 'p, 's) t) = p.logp_t
let sample (p : ('w, 'p, 's) t) = p.sample
let kl_to_gaussian (p : ('w, 'p, 's) t) = p.kl_to_gaussian
let log_2 = Float.log 2.0

(* {1 Gaussian} *)

module Gaussian = struct
  type ('a, _) params = { log_spatial_stds : 'a } [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)

  let logp_t p ~k:_ ~u =
    Density.logp ~covariance:(Density.Diagonal { log_stds = p.log_spatial_stds }) u

  let sample p ~key ~n_steps =
    if n_steps <= 0 then invalid_arg "Prior.Gaussian.sample: n_steps must be positive";
    let m = Array.get (shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype p.log_spatial_stds in
    let open Infix in
    Nx.Rng.normal_at (beside p.log_spatial_stds) ~key dtype [| n_steps; m |]
    * exp p.log_spatial_stds

  (* [KL(q ‖ p)] with [q = N(mu, Σ_t ⊗ Σ_s)] and a factorized Gaussian [p]:
     the quadratic form of [mu], the trace [tr (Σ_p⁻¹ Σ_q)], and the two log
     determinants. Σ_p's scales are [I_t ⊗ σ_s], so the trace factorizes. *)
  let kl_to_gaussian p ~mu ~space_factor ~time_factor =
    let t_steps = Array.get (shape mu) 0 in
    let m = Array.get (shape mu) 1 in
    let spatial = exp p.log_spatial_stds in
    let quadratic =
      let open Infix in
      let scaled = mu / reshape [| 1; m |] spatial in
      sum (scaled * scaled)
    in
    let dim = Float.of_int (t_steps * m) in
    let open Infix in
    let trace_time = sum (time_factor * time_factor) in
    let scaled_space = space_factor / reshape [| m; 1 |] spatial in
    let trace_space = sum (scaled_space * scaled_space) in
    let trace = trace_time * trace_space in
    let log_det factor = sum (log (diag factor)) *$ 2.0 in
    let log_det_p = sum p.log_spatial_stds *$ (2.0 *. Float.of_int t_steps) in
    let log_det_q =
      (log_det time_factor *$ Float.of_int m)
      + (log_det space_factor *$ Float.of_int t_steps)
    in
    (trace + log_det_p - log_det_q + quadratic + scalar_like mu (-.dim)) *$ 0.5

  let place ~at p =
    Nx.Ptree.place
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)
      at
      p

  let make ~at ?(spatial_std = 1.0) ~m () =
    if m <= 0 then invalid_arg "Prior.Gaussian.make: m must be positive";
    if not (Float.( > ) spatial_std 0.0)
    then invalid_arg "Prior.Gaussian.make: spatial_std must be positive";
    let init (type w) ?key:(_ : 's Nx.Rng.key option) (dtype : (float, w) Nx.dtype) =
      { log_spatial_stds = full_at at dtype [| m |] (Float.log spatial_std) }
    in
    make ~m ~structure:(structure ()) ~init ~logp_t ~sample ~kl_to_gaussian ()
end

(* {1 Student-t} *)

module Student = struct
  type ('a, _) params =
    { log_spatial_stds : 'a
    ; log_nu_minus_two : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)

  (* [nu] is a scalar; the density is
     [∝ (1 + ‖u / σ‖² / ν) ^ (−(ν + m) / 2)] with [σ] itself scaled by
     [sqrt ((ν − 2) / ν)]. *)
  let t_params p =
    let open Infix in
    let nu = exp p.log_nu_minus_two +$ 2.0 in
    let log_sigma = p.log_spatial_stds + (log ((nu -$ 2.0) / nu) *$ 0.5) in
    nu, exp log_sigma, log_sigma

  let logp_t p ~k:_ ~u =
    let m = Array.get (shape p.log_spatial_stds) 0 in
    let m_f = Float.of_int m in
    let nu, sigma, log_sigma = t_params p in
    let open Infix in
    let scaled = u / sigma in
    let nu_half = nu *$ 0.5 in
    let nu_plus_m_half = (nu *$ 0.5) +$ (0.5 *. m_f) in
    let cst =
      Special.lgamma nu_half
      - Special.lgamma nu_plus_m_half
      + ((log (nu *$ Float.pi) *$ (0.5 *. m_f)) + sum log_sigma)
    in
    let radius_sq = sum ~axes:[ -1 ] (scaled * scaled) in
    -(cst + (log1p (radius_sq / nu) * nu_plus_m_half))

  let sample p ~key ~n_steps =
    if n_steps <= 0 then invalid_arg "Prior.Student.sample: n_steps must be positive";
    let m = Array.get (shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype p.log_spatial_stds in
    let k1, k2 = Nx.Rng.split ~n:2 key |> fun a -> a.(0), a.(1) in
    let nu, sigma, _ = t_params p in
    (* [u = σ · z / sqrt (χ²_ν / ν)] is a multivariate t with [ν] degrees of
       freedom, which is the density [logp_t] writes. One independent [χ²_ν]
       per step. *)
    let z = Nx.Rng.normal_at (beside p.log_spatial_stds) ~key:k1 dtype [| n_steps; m |] in
    let open Infix in
    let chi2 = Nx.Rng.gamma ~key:k2 (broadcast_to [| n_steps; 1 |] (nu *$ 0.5)) *$ 2.0 in
    let t = z / sqrt (chi2 / nu) in
    t * sigma

  let place ~at p =
    Nx.Ptree.place
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)
      at
      p

  let make ~at ?(spatial_std = 1.0) ?(nu = 10.0) ~m () =
    if m <= 0 then invalid_arg "Prior.Student.make: m must be positive";
    if not (Float.( > ) spatial_std 0.0)
    then invalid_arg "Prior.Student.make: spatial_std must be positive";
    if not (Float.( > ) nu 2.0)
    then invalid_arg "Prior.Student.make: nu must be greater than 2";
    let init (type w) ?key:(_ : 's Nx.Rng.key option) (dtype : (float, w) Nx.dtype) =
      { log_spatial_stds = full_at at dtype [| m |] (Float.log spatial_std)
      ; log_nu_minus_two = scalar_at at dtype (Float.log (nu -. 2.0))
      }
    in
    make ~m ~structure:(structure ()) ~init ~logp_t ~sample ()
end

(* {1 Laplacian} *)

module Laplacian = struct
  type ('a, _) params = { log_spatial_stds : 'a } [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)

  let logp_t p ~k:_ ~u =
    let m = Array.get (shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype u in
    let sigma = exp p.log_spatial_stds in
    Infix.(
      -(sum ~axes:[ -1 ] (abs u / sigma)
        + (sum p.log_spatial_stds + scalar_at (beside u) dtype (Float.of_int m *. log_2))
       ))

  let sample p ~key ~n_steps =
    if n_steps <= 0 then invalid_arg "Prior.Laplacian.sample: n_steps must be positive";
    let m = Array.get (shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype p.log_spatial_stds in
    let k1, k2 = Nx.Rng.split ~n:2 key |> fun a -> a.(0), a.(1) in
    let open Infix in
    let scale = exp p.log_spatial_stds in
    let magnitude =
      Nx.Rng.exponential_at (beside scale) ~key:k1 dtype [| n_steps; m |] * scale
    in
    let signs = sign (Nx.Rng.normal_at (beside scale) ~key:k2 dtype [| n_steps; m |]) in
    signs * magnitude

  let place ~at p =
    Nx.Ptree.place
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)
      at
      p

  let make ~at ?(spatial_std = 1.0) ~m () =
    if m <= 0 then invalid_arg "Prior.Laplacian.make: m must be positive";
    if not (Float.( > ) spatial_std 0.0)
    then invalid_arg "Prior.Laplacian.make: spatial_std must be positive";
    let init (type w) ?key:(_ : 's Nx.Rng.key option) (dtype : (float, w) Nx.dtype) =
      { log_spatial_stds = full_at at dtype [| m |] (Float.log spatial_std) }
    in
    make ~m ~structure:(structure ()) ~init ~logp_t ~sample ()
end
