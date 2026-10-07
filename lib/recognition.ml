(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Types

let log_2pi = Float.log (2.0 *. Float.pi)

let sample model params ~mu ~n_samples ~key =
  if n_samples <= 0 then invalid_arg "Recognition.sample: n_samples must be positive";
  let t_steps = model.Model.n_steps in
  let m = model.Model.m in
  let l_time = Covariance.factor params.Model.time_cov in
  let l_space = Covariance.factor params.Model.space_cov in
  (* [u = mu + L_time xi L_spaceᵀ] with [xi] standard normal: [L_time] acts on
     the time axis and [L_spaceᵀ] on the input axis, so the draws have
     covariance [Σ_time ⊗ Σ_space]. *)
  let xi = Nx.Rng.normal key model.Model.dtype [| t_steps; n_samples; m |] in
  let along_time = Nx.matmul l_time (Nx.reshape [| t_steps; n_samples * m |] xi) in
  let noise =
    Nx.matmul
      (Nx.reshape [| t_steps; n_samples; m |] along_time)
      (Nx.matrix_transpose l_space)
  in
  Nx.add (Nx.unsqueeze ~axes:[ 1 ] mu) noise

let log_prob model params ~mu u =
  let t_steps = model.Model.n_steps in
  let m = model.Model.m in
  let dtype = model.Model.dtype in
  let shape = Nx.shape u in
  let rank = Array.length shape in
  if
    not (Int.equal (Array.get shape 0) t_steps && Int.equal (Array.get shape (rank - 1)) m)
  then
    invalid_arg
      (Printf.sprintf
         "Recognition.log_prob: expected [n_steps; …; m] = [%d; …; %d]"
         t_steps
         m);
  let batch_dims = Array.sub shape ~pos:1 ~len:(rank - 2) in
  let batch = Array.fold batch_dims ~init:1 ~f:( * ) in
  let l_time = Covariance.factor params.Model.time_cov in
  let l_space = Covariance.factor params.Model.space_cov in
  let u3 = Nx.reshape (Array.append [| t_steps |] (Array.append [| batch |] [| m |])) u in
  let diff = Nx.sub u3 (Nx.reshape [| t_steps; 1; m |] mu) in
  (* [(u − μ)ᵀ Σ⁻¹ (u − μ) = ‖L_time⁻¹ (u − μ) L_space⁻ᵀ‖²_F], two triangular
     solves: one down the time axis, one down the input axis. *)
  let along_time = Nx.reshape [| t_steps; batch * m |] diff in
  let solved_time = Nx.solve_triangular ~upper:true l_time along_time in
  let rows = Nx.reshape [| t_steps * batch; m |] solved_time in
  let solved_space =
    Nx.matrix_transpose
      (Nx.solve_triangular ~upper:true l_space (Nx.matrix_transpose rows))
  in
  let quadratic =
    Nx.reshape
      (Array.append [| t_steps |] batch_dims)
      (Nx.sum ~axes:[ -1 ] (Nx.mul solved_space solved_space))
    |> Nx.sum ~axes:[ 0 ]
  in
  let log_det =
    Nx.add
      (Nx.mul_s (Covariance.log_det params.Model.time_cov) (Float.of_int m))
      (Nx.mul_s (Covariance.log_det params.Model.space_cov) (Float.of_int t_steps))
  in
  let dim = Float.of_int (t_steps * m) in
  Nx.mul_s (Nx.add quadratic (Nx.add log_det (Nx.scalar dtype (dim *. log_2pi)))) (-0.5)

let entropy model params =
  let t_steps = model.Model.n_steps in
  let m = model.Model.m in
  let log_det =
    Nx.add
      (Nx.mul_s (Covariance.log_det params.Model.time_cov) (Float.of_int m))
      (Nx.mul_s (Covariance.log_det params.Model.space_cov) (Float.of_int t_steps))
  in
  let dim = Float.of_int (t_steps * m) in
  Nx.mul_s (Nx.add (Nx.scalar model.Model.dtype (dim *. (1.0 +. log_2pi))) log_det) 0.5
