(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Types

type ('w, 'p) t =
  { m : int
  ; structure : 'p Nx.Ptree.t
  ; params : 'p
  ; logp_t : 'p -> k:'w value -> u:'w value -> 'w value
  ; sample : 'p -> key:Nx.Rng.t -> n_steps:int -> 'w value
  ; kl_to_gaussian :
      ('p -> mu:'w value -> space_factor:'w value -> time_factor:'w value -> 'w value)
        option
  }

let log_2pi = Float.log (2.0 *. Float.pi)
let log_2 = Float.log 2.0

(* [first_step k] is a boolean scalar that is [true] at step 0. *)
let first_step (k : 'w value) = Nx.equal k (Nx.scalar_like k 0.0)

(* [gaussian_logp ~log_scale ~log_scale_first ~k ~u] is the log density of a
   factorized zero-mean Gaussian, summed over the last axis: per-row scalars
   for a batch of rows, a scalar for one row. The first step uses
   [log_scale_first], every other step [log_scale]. *)
let gaussian_logp ~log_scale ~log_scale_first ~k ~u =
  let log_scale_k = Nx.where (first_step k) log_scale_first log_scale in
  let scaled = Nx.div u (Nx.exp log_scale_k) in
  let term =
    Nx.add (Nx.mul scaled scaled) (Nx.add_s (Nx.mul_s log_scale_k 2.0) log_2pi)
  in
  Nx.mul_s (Nx.sum ~axes:[ -1 ] term) (-0.5)

(* [steps_mask dtype ~n_steps] is a boolean [[n_steps; 1]] tensor that is
   [true] at the first step. Samplers use it to give that step its own
   distribution. *)
let steps_mask (dtype : (float, 'w) Nx.dtype) ~n_steps =
  let k = Nx.arange dtype 0 n_steps 1 in
  Nx.unsqueeze ~axes:[ 1 ] (Nx.equal k (Nx.scalar dtype 0.0))

(* {1 Gaussian} *)

module Gaussian = struct
  type 'a params =
    { log_spatial_stds : 'a
    ; log_first_bin : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type 'a t = 'a params

        let walk = walk_params
      end)

  let logp_t p ~k ~u =
    gaussian_logp
      ~log_scale:p.log_spatial_stds
      ~log_scale_first:(Nx.add p.log_spatial_stds p.log_first_bin)
      ~k
      ~u

  let sample p ~key ~n_steps =
    if n_steps <= 0 then invalid_arg "Prior.Gaussian.sample: n_steps must be positive";
    let m = Array.get (Nx.shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype p.log_spatial_stds in
    let scale = Nx.broadcast_to [| n_steps; m |] p.log_spatial_stds in
    let log_scale =
      Nx.where (steps_mask dtype ~n_steps) (Nx.add scale p.log_first_bin) scale
    in
    Nx.mul (Nx.Rng.normal key dtype [| n_steps; m |]) (Nx.exp log_scale)

  (* [KL(q ‖ p)] with [q = N(mu, Σ_t ⊗ Σ_s)] and a factorized Gaussian [p]:
     the quadratic form of [mu], the trace [tr (Σ_p⁻¹ Σ_q)], and the two log
     determinants. Σ_p's scales are [σ_t ⊗ σ_s], so the trace factorizes and
     each factor's rows are scaled by their own scales. *)
  let kl_to_gaussian p ~mu ~space_factor ~time_factor =
    let t_steps = Array.get (Nx.shape mu) 0 in
    let m = Array.get (Nx.shape mu) 1 in
    let dtype = Nx.dtype mu in
    let spatial = Nx.exp p.log_spatial_stds in
    let sig_t =
      Nx.where
        (steps_mask dtype ~n_steps:t_steps)
        (Nx.broadcast_to [| t_steps; 1 |] (Nx.exp p.log_first_bin))
        (Nx.ones dtype [| t_steps; 1 |])
    in
    let quadratic =
      let denom = Nx.mul sig_t (Nx.reshape [| 1; m |] spatial) in
      let scaled = Nx.div mu denom in
      Nx.sum (Nx.mul scaled scaled)
    in
    let trace_of factor scales =
      let scaled = Nx.div factor scales in
      Nx.sum (Nx.mul scaled scaled)
    in
    let trace =
      Nx.mul
        (trace_of time_factor sig_t)
        (trace_of space_factor (Nx.reshape [| m; 1 |] spatial))
    in
    let log_det factor = Nx.mul_s (Nx.sum (Nx.log (Nx.diag factor))) 2.0 in
    let log_det_p =
      Nx.add
        (Nx.mul_s (Nx.sum (Nx.log sig_t)) (2.0 *. Float.of_int m))
        (Nx.mul_s (Nx.sum (Nx.log spatial)) (2.0 *. Float.of_int t_steps))
    in
    let log_det_q =
      Nx.add
        (Nx.mul_s (log_det time_factor) (Float.of_int m))
        (Nx.mul_s (log_det space_factor) (Float.of_int t_steps))
    in
    let dim = Float.of_int (t_steps * m) in
    Nx.mul_s
      (Nx.add
         (Nx.add (Nx.sub (Nx.add trace log_det_p) log_det_q) quadratic)
         (Nx.scalar_like mu (-.dim)))
      0.5

  let make ?(spatial_std = 1.0) ?(first_bin = 1.0) ~m dtype =
    if m <= 0 then invalid_arg "Prior.Gaussian.make: m must be positive";
    if not (Float.( > ) spatial_std 0.0)
    then invalid_arg "Prior.Gaussian.make: spatial_std must be positive";
    if not (Float.( > ) first_bin 0.0)
    then invalid_arg "Prior.Gaussian.make: first_bin must be positive";
    { m
    ; structure = structure ()
    ; params =
        { log_spatial_stds = Nx.full dtype [| m |] (Float.log spatial_std)
        ; log_first_bin = Nx.scalar dtype (Float.log first_bin)
        }
    ; logp_t
    ; sample
    ; kl_to_gaussian = Some kl_to_gaussian
    }
end

(* {1 Student-t} *)

module Student = struct
  type 'a params =
    { log_spatial_stds : 'a
    ; log_nu_minus_two : 'a
    ; log_first_stds : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type 'a t = 'a params

        let walk = walk_params
      end)

  (* [nu] and the scale of the t are scalars; the density is
     [∝ (1 + ‖u / σ‖² / ν) ^ (−(ν + m) / 2)] with [σ] itself scaled by
     [sqrt ((ν − 2) / ν)], exactly as the reference implementation writes
     it. *)
  let t_params p =
    let nu = Nx.add_s (Nx.exp p.log_nu_minus_two) 2.0 in
    let log_sigma =
      Nx.add p.log_spatial_stds (Nx.mul_s (Nx.log (Nx.div (Nx.sub_s nu 2.0) nu)) 0.5)
    in
    nu, Nx.exp log_sigma, log_sigma

  let logp_t p ~k ~u =
    let m = Array.get (Nx.shape p.log_spatial_stds) 0 in
    let m_f = Float.of_int m in
    let nu, sigma, log_sigma = t_params p in
    let gaussian =
      gaussian_logp ~log_scale:p.log_spatial_stds ~log_scale_first:p.log_first_stds ~k ~u
    in
    let scaled = Nx.div u sigma in
    let nu_half = Nx.mul_s nu 0.5 in
    let nu_plus_m_half = Nx.add_s (Nx.mul_s nu 0.5) (0.5 *. m_f) in
    let cst =
      Nx.add
        (Nx.sub (Special.lgamma nu_half) (Special.lgamma nu_plus_m_half))
        (Nx.add
           (Nx.mul_s (Nx.log (Nx.mul_s nu Float.pi)) (0.5 *. m_f))
           (Nx.sum log_sigma))
    in
    let radius_sq = Nx.sum ~axes:[ -1 ] (Nx.mul scaled scaled) in
    let student =
      Nx.neg (Nx.add cst (Nx.mul (Nx.log1p (Nx.div radius_sq nu)) nu_plus_m_half))
    in
    Nx.where (first_step k) gaussian student

  let sample p ~key ~n_steps =
    if n_steps <= 0 then invalid_arg "Prior.Student.sample: n_steps must be positive";
    let m = Array.get (Nx.shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype p.log_spatial_stds in
    let k1, k2, k3 = Nx.Rng.split ~n:3 key |> fun a -> a.(0), a.(1), a.(2) in
    let nu, sigma, _ = t_params p in
    (* [u = σ · z / sqrt (χ²_ν / ν)] is a multivariate t with [ν] degrees of
       freedom, which is the density [logp_t] writes. *)
    let z = Nx.Rng.normal k1 dtype [| n_steps; m |] in
    let chi2 = Nx.mul_s (Nx.Rng.gamma k2 (Nx.mul_s nu 0.5)) 2.0 in
    let t = Nx.div z (Nx.sqrt (Nx.div chi2 nu)) in
    let first =
      Nx.mul
        (Nx.Rng.normal k3 dtype [| n_steps; m |])
        (Nx.broadcast_to [| n_steps; m |] (Nx.exp p.log_first_stds))
    in
    Nx.where (steps_mask dtype ~n_steps) first (Nx.mul t sigma)

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
        { log_spatial_stds = Nx.full dtype [| m |] (Float.log spatial_std)
        ; log_nu_minus_two = Nx.scalar dtype (Float.log (nu -. 2.0))
        ; log_first_stds = Nx.full dtype [| m |] (Float.log first_std)
        }
    ; logp_t
    ; sample
    ; kl_to_gaussian = None
    }
end

(* {1 Laplacian} *)

module Laplacian = struct
  type 'a params =
    { log_spatial_stds : 'a
    ; log_first_stds : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type 'a t = 'a params

        let walk = walk_params
      end)

  let logp_t p ~k ~u =
    let m = Array.get (Nx.shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype u in
    let sigma = Nx.exp p.log_spatial_stds in
    let gaussian =
      gaussian_logp ~log_scale:p.log_spatial_stds ~log_scale_first:p.log_first_stds ~k ~u
    in
    let laplace =
      Nx.neg
        (Nx.add
           (Nx.sum ~axes:[ -1 ] (Nx.div (Nx.abs u) sigma))
           (Nx.add
              (Nx.sum p.log_spatial_stds)
              (Nx.scalar dtype (Float.of_int m *. log_2))))
    in
    Nx.where (first_step k) gaussian laplace

  let sample p ~key ~n_steps =
    if n_steps <= 0 then invalid_arg "Prior.Laplacian.sample: n_steps must be positive";
    let m = Array.get (Nx.shape p.log_spatial_stds) 0 in
    let dtype = Nx.dtype p.log_spatial_stds in
    let k1, k2, k3 = Nx.Rng.split ~n:3 key |> fun a -> a.(0), a.(1), a.(2) in
    let scale = Nx.exp p.log_spatial_stds in
    let magnitude = Nx.mul (Nx.Rng.exponential k1 dtype [| n_steps; m |]) scale in
    let signs = Nx.sign (Nx.Rng.normal k2 dtype [| n_steps; m |]) in
    let laplace = Nx.mul signs magnitude in
    let first =
      Nx.mul
        (Nx.Rng.normal k3 dtype [| n_steps; m |])
        (Nx.broadcast_to [| n_steps; m |] (Nx.exp p.log_first_stds))
    in
    Nx.where (steps_mask dtype ~n_steps) first laplace

  let make ?(spatial_std = 1.0) ?(first_std = 1.0) ~m dtype =
    if m <= 0 then invalid_arg "Prior.Laplacian.make: m must be positive";
    if not (Float.( > ) spatial_std 0.0)
    then invalid_arg "Prior.Laplacian.make: spatial_std must be positive";
    if not (Float.( > ) first_std 0.0)
    then invalid_arg "Prior.Laplacian.make: first_std must be positive";
    { m
    ; structure = structure ()
    ; params =
        { log_spatial_stds = Nx.full dtype [| m |] (Float.log spatial_std)
        ; log_first_stds = Nx.full dtype [| m |] (Float.log first_std)
        }
    ; logp_t
    ; sample
    ; kl_to_gaussian = None
    }
end
