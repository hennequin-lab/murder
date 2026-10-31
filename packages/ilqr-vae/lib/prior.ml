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
  ; params : 'p
  ; logp_t : 'p -> k:('w, 's) value -> u:('w, 's) value -> ('w, 's) value
  ; sample : 'p -> key:'s Nx.Rng.key -> n_steps:int -> ('w, 's) value
  ; kl_to_gaussian :
      ('p
       -> mu:('w, 's) value
       -> space_factor:('w, 's) value
       -> time_factor:('w, 's) value
       -> ('w, 's) value)
        option
  }

let log_2pi = Float.log (2.0 *. Float.pi)
let log_2 = Float.log 2.0

(* [first_step k] is a boolean scalar that is [true] at step 0. *)
let first_step (k : ('w, 's) value) = Nx.equal k (scalar_like k 0.0)

(* [gaussian_logp ~log_scale ~log_scale_first ~k ~u] is the log density of a
   factorized zero-mean Gaussian, summed over the last axis: per-row scalars
   for a batch of rows, a scalar for one row. The first step uses
   [log_scale_first], every other step [log_scale]. *)
let gaussian_logp ~log_scale ~log_scale_first ~k ~u =
  let open Infix in
  let log_scale_k = where (first_step k) log_scale_first log_scale in
  let scaled = u / exp log_scale_k in
  let term = (scaled * scaled) + ((log_scale_k *$ 2.0) +$ log_2pi) in
  sum ~axes:[ -1 ] term *$ -0.5

(* [steps_mask at dtype ~n_steps] is a boolean [[n_steps; 1]] tensor at [at]
   that is [true] at the first step. Samplers use it to give that step its own
   distribution. *)
let steps_mask at (dtype : (float, 'w) Nx.dtype) ~n_steps =
  let k = arange_at at dtype 0 n_steps 1 in
  unsqueeze ~axes:[ 1 ] (Nx.equal k (scalar_at at dtype 0.0))

(* {1 Gaussian} *)

module Gaussian = struct
  type ('a, _) params =
    { log_spatial_stds : 'a
    ; log_first_bin : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)

  let logp_t p ~k ~u =
    gaussian_logp
      ~log_scale:p.log_spatial_stds
      ~log_scale_first:Infix.(p.log_spatial_stds + p.log_first_bin)
      ~k
      ~u

  let sample p ~key ~n_steps =
    if n_steps <= 0 then invalid_arg "Prior.Gaussian.sample: n_steps must be positive";
    let m = Array.get (shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype p.log_spatial_stds in
    let open Infix in
    let scale = broadcast_to [| n_steps; m |] p.log_spatial_stds in
    let log_scale =
      where (steps_mask (beside scale) dtype ~n_steps) (scale + p.log_first_bin) scale
    in
    Nx.Rng.normal_at (beside p.log_spatial_stds) ~key dtype [| n_steps; m |]
    * exp log_scale

  (* [KL(q ‖ p)] with [q = N(mu, Σ_t ⊗ Σ_s)] and a factorized Gaussian [p]:
     the quadratic form of [mu], the trace [tr (Σ_p⁻¹ Σ_q)], and the two log
     determinants. Σ_p's scales are [σ_t ⊗ σ_s], so the trace factorizes and
     each factor's rows are scaled by their own scales. *)
  let kl_to_gaussian p ~mu ~space_factor ~time_factor =
    let t_steps = Array.get (shape mu) 0 in
    let m = Array.get (shape mu) 1 in
    let dtype = Nx.dtype mu in
    let spatial = exp p.log_spatial_stds in
    let sig_t =
      where
        (steps_mask (beside mu) dtype ~n_steps:t_steps)
        (broadcast_to [| t_steps; 1 |] (exp p.log_first_bin))
        (ones_at (beside mu) dtype [| t_steps; 1 |])
    in
    let quadratic =
      let open Infix in
      let denom = sig_t * reshape [| 1; m |] spatial in
      let scaled = mu / denom in
      sum (scaled * scaled)
    in
    let trace_of factor scales =
      let open Infix in
      let scaled = factor / scales in
      sum (scaled * scaled)
    in
    let dim = Float.of_int (t_steps * m) in
    let open Infix in
    let trace =
      trace_of time_factor sig_t * trace_of space_factor (reshape [| m; 1 |] spatial)
    in
    let log_det factor = sum (log (diag factor)) *$ 2.0 in
    let log_det_p =
      (sum (log sig_t) *$ (2.0 *. Float.of_int m))
      + (sum (log spatial) *$ (2.0 *. Float.of_int t_steps))
    in
    let log_det_q =
      (log_det time_factor *$ Float.of_int m)
      + (log_det space_factor *$ Float.of_int t_steps)
    in
    (trace + log_det_p - log_det_q + quadratic + scalar_like mu (-.dim)) *$ 0.5

  let make ?(spatial_std = 1.0) ?(first_bin = 1.0) ~m dtype =
    if m <= 0 then invalid_arg "Prior.Gaussian.make: m must be positive";
    if not (Float.( > ) spatial_std 0.0)
    then invalid_arg "Prior.Gaussian.make: spatial_std must be positive";
    if not (Float.( > ) first_bin 0.0)
    then invalid_arg "Prior.Gaussian.make: first_bin must be positive";
    { m
    ; structure = structure ()
    ; params =
        { log_spatial_stds = full dtype [| m |] (Float.log spatial_std)
        ; log_first_bin = scalar dtype (Float.log first_bin)
        }
    ; logp_t
    ; sample
    ; kl_to_gaussian = Some kl_to_gaussian
    }
end

(* {1 Student-t} *)

module Student = struct
  type ('a, _) params =
    { log_spatial_stds : 'a
    ; log_nu_minus_two : 'a
    ; log_first_stds : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)

  (* [nu] and the scale of the t are scalars; the density is
     [∝ (1 + ‖u / σ‖² / ν) ^ (−(ν + m) / 2)] with [σ] itself scaled by
     [sqrt ((ν − 2) / ν)], exactly as the reference implementation writes
     it. *)
  let t_params p =
    let open Infix in
    let nu = exp p.log_nu_minus_two +$ 2.0 in
    let log_sigma = p.log_spatial_stds + (log ((nu -$ 2.0) / nu) *$ 0.5) in
    nu, exp log_sigma, log_sigma

  let logp_t p ~k ~u =
    let m = Array.get (shape p.log_spatial_stds) 0 in
    let m_f = Float.of_int m in
    let nu, sigma, log_sigma = t_params p in
    let gaussian =
      gaussian_logp ~log_scale:p.log_spatial_stds ~log_scale_first:p.log_first_stds ~k ~u
    in
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
    let student = -(cst + (log1p (radius_sq / nu) * nu_plus_m_half)) in
    where (first_step k) gaussian student

  let sample p ~key ~n_steps =
    if n_steps <= 0 then invalid_arg "Prior.Student.sample: n_steps must be positive";
    let m = Array.get (shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype p.log_spatial_stds in
    let k1, k2, k3 = Nx.Rng.split ~n:3 key |> fun a -> a.(0), a.(1), a.(2) in
    let nu, sigma, _ = t_params p in
    (* [u = σ · z / sqrt (χ²_ν / ν)] is a multivariate t with [ν] degrees of
       freedom, which is the density [logp_t] writes. *)
    let z = Nx.Rng.normal_at (beside p.log_spatial_stds) ~key:k1 dtype [| n_steps; m |] in
    let open Infix in
    let chi2 = Nx.Rng.gamma ~key:k2 (nu *$ 0.5) *$ 2.0 in
    let t = z / sqrt (chi2 / nu) in
    let first =
      Nx.Rng.normal_at (beside p.log_first_stds) ~key:k3 dtype [| n_steps; m |]
      * broadcast_to [| n_steps; m |] (exp p.log_first_stds)
    in
    where (steps_mask (beside t) dtype ~n_steps) first (t * sigma)

  let make ?(spatial_std = 1.0) ?(nu = 10.0) ?(first_std = 1.0) ~m dtype =
    if m <= 0 then invalid_arg "Prior.Student.make: m must be positive";
    if not (Float.( > ) spatial_std 0.0)
    then invalid_arg "Prior.Student.make: spatial_std must be positive";
    if not (Float.( > ) first_std 0.0)
    then invalid_arg "Prior.Student.make: first_std must be positive";
    if not (Float.( > ) nu 2.0)
    then invalid_arg "Prior.Student.make: nu must be greater than 2";
    { m
    ; structure = structure ()
    ; params =
        { log_spatial_stds = full dtype [| m |] (Float.log spatial_std)
        ; log_nu_minus_two = scalar dtype (Float.log (nu -. 2.0))
        ; log_first_stds = full dtype [| m |] (Float.log first_std)
        }
    ; logp_t
    ; sample
    ; kl_to_gaussian = None
    }
end

(* {1 Laplacian} *)

module Laplacian = struct
  type ('a, _) params =
    { log_spatial_stds : 'a
    ; log_first_stds : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 's) t = ('a, 's) params

        let walk = walk_params
      end)

  let logp_t p ~k ~u =
    let m = Array.get (shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype u in
    let sigma = exp p.log_spatial_stds in
    let gaussian =
      gaussian_logp ~log_scale:p.log_spatial_stds ~log_scale_first:p.log_first_stds ~k ~u
    in
    let laplace =
      Infix.(
        -(sum ~axes:[ -1 ] (abs u / sigma)
          + (sum p.log_spatial_stds + scalar_at (beside u) dtype (Float.of_int m *. log_2))
         ))
    in
    where (first_step k) gaussian laplace

  let sample p ~key ~n_steps =
    if n_steps <= 0 then invalid_arg "Prior.Laplacian.sample: n_steps must be positive";
    let m = Array.get (shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype p.log_spatial_stds in
    let k1, k2, k3 = Nx.Rng.split ~n:3 key |> fun a -> a.(0), a.(1), a.(2) in
    let open Infix in
    let scale = exp p.log_spatial_stds in
    let magnitude =
      Nx.Rng.exponential_at (beside scale) ~key:k1 dtype [| n_steps; m |] * scale
    in
    let signs = sign (Nx.Rng.normal_at (beside scale) ~key:k2 dtype [| n_steps; m |]) in
    let laplace = signs * magnitude in
    let first =
      Nx.Rng.normal_at (beside p.log_first_stds) ~key:k3 dtype [| n_steps; m |]
      * broadcast_to [| n_steps; m |] (exp p.log_first_stds)
    in
    where (steps_mask (beside laplace) dtype ~n_steps) first laplace

  let make ?(spatial_std = 1.0) ?(first_std = 1.0) ~m dtype =
    if m <= 0 then invalid_arg "Prior.Laplacian.make: m must be positive";
    if not (Float.( > ) spatial_std 0.0)
    then invalid_arg "Prior.Laplacian.make: spatial_std must be positive";
    if not (Float.( > ) first_std 0.0)
    then invalid_arg "Prior.Laplacian.make: first_std must be positive";
    { m
    ; structure = structure ()
    ; params =
        { log_spatial_stds = full dtype [| m |] (Float.log spatial_std)
        ; log_first_stds = full dtype [| m |] (Float.log first_std)
        }
    ; logp_t
    ; sample
    ; kl_to_gaussian = None
    }
end
