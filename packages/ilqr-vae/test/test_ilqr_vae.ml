(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* iLQR-VAE test suite:
   - covariance factors, log-determinants and the Kronecker recognition
     covariance (sampling covariance, log density and entropy against dense
     computations),
   - the prior log densities (normalization by numerical integration and
     sampling moments), the Gaussian KL, and the initial-state prior,
   - the likelihoods (log densities against hand-computed values),
   - the dynamics (hand-computed steps, batched rollouts),
   - the iLQR encoder: stationarity on an LQR problem,
   - the LFADS recognition: shapes, the replay of its own density, the
     bidirectional encoder and the controller against hand-rolled rollouts,
     the exact-zero KL when the posterior matches the prior (and the
     path-derivative gradient that sticks to it), gradient checks and JIT,
   - the ELBO: gradient check of the whole inference path (implicit gradient
     included) against finite differences,
   - JIT round-trips of the fixed solve and of its gradient,
   - swapping recognitions, model validation, and end-to-end training smoke
     tests. *)

open Base
open Windtrap
module Nx = Nx
module C = Ilqr_vae.Covariance
module I = Ilqr_vae.Initial
module P = Ilqr_vae.Prior
module L = Ilqr_vae.Likelihood
module D = Ilqr_vae.Dynamics
module Schedule = Ilqr_vae.Schedule
module R = Ilqr_vae.Recognition
module Il = R.Iilqr
module Lf = R.Lfads
module M = Ilqr_vae.Model
module El = Ilqr_vae.Elbo

let dtype = Nx.float64
let phi = Nx.tanh
let require ~msg b = if not b then failwith ("assert failed: " ^ msg)

let check_close ~msg ~tol got expected =
  let g = Nx.to_array got
  and e = Nx.to_array expected in
  require ~msg (Array.length g = Array.length e);
  Array.iteri g ~f:(fun i a ->
    let b = e.(i) in
    let within = Float.(abs (a -. b) <= tol +. (tol *. abs b)) in
    if not within
    then
      Stdio.printf
        "MISMATCH %s: got %.8g expected %.8g (diff %.3g, tol %g)\n%!"
        msg
        a
        b
        (a -. b)
        tol;
    require ~msg Float.(abs (a -. b) <= tol +. (tol *. abs b)))

let vec values = Nx.create dtype [| List.length values |] (Array.of_list values)
let scalar v = Nx.scalar dtype v
let item t = Nx.item [] t
let max_abs t = item (Nx.max (Nx.abs t))

(* ──────────────────────────────────────────────────────────────────────────
   Concrete dtypes for the test models.
   ────────────────────────────────────────────────────────────────────────── *)

type linear_params = Nx.float64_t D.Linear.params
type mgu_params = Nx.float64_t D.Mgu.params
type lik_params = Nx.float64_t L.Gaussian.params
type poisson_params = Nx.float64_t L.Poisson.params
type pri_params = Nx.float64_t P.Gaussian.params
type ilqr_rec = Nx.float64_t Il.params
type lfads_rec = Nx.float64_t Lf.params

type iilqr_model =
  (Nx.float64_elt, linear_params, lik_params, pri_params, pri_params, ilqr_rec) M.t

type iilqr_params =
  (Nx.float64_elt, linear_params, lik_params, pri_params, pri_params, ilqr_rec) M.params

type lfads_model =
  (Nx.float64_elt, mgu_params, lik_params, pri_params, pri_params, lfads_rec) M.t

type lfads_params =
  (Nx.float64_elt, mgu_params, lik_params, pri_params, pri_params, lfads_rec) M.params

let setup ?(key = Nx.Rng.key 0) ~n ~m ~t_steps ~n_output () : iilqr_model * iilqr_params =
  let keys = Nx.Rng.split ~n:2 key in
  let dynamics = D.Linear.make ~key:keys.(0) ~n ~m dtype in
  let likelihood = L.Gaussian.make ~key:keys.(1) ~n ~n_output dtype in
  let prior = P.Gaussian.make ~m ~spatial_std:1.0 dtype in
  let encoder_prior = P.Gaussian.make ~m ~spatial_std:1.0 dtype in
  let recognition = Il.make ~n ~m ~n_steps:t_steps ~n_output dtype in
  let model =
    M.make
      dtype
      ~n
      ~m
      ~n_steps:t_steps
      ~n_output
      ~dynamics
      ~likelihood
      ~prior
      ~encoder_prior
      ~recognition
      ()
  in
  model, M.init model

let setup_lfads
      ?(key = Nx.Rng.key 1)
      ?(n_ic = 4)
      ?(n_ci = 4)
      ?(n_ctrl = 3)
      ?(with_inputs = true)
      ~n
      ~m
      ~t_steps
      ~n_output
      ()
  : lfads_model * lfads_params
  =
  let keys = Nx.Rng.split ~n:4 key in
  let dynamics = D.Mgu.make ~key:keys.(0) ~n ~m ~phi dtype in
  let likelihood = L.Gaussian.make ~key:keys.(1) ~n ~n_output dtype in
  let prior = P.Gaussian.make ~m ~spatial_std:1.0 dtype in
  let encoder_prior = P.Gaussian.make ~m ~spatial_std:1.0 dtype in
  let initial_prior = I.make ~std:1.0 ~n dtype in
  let recognition =
    Lf.make
      ~key:keys.(2)
      ~n_ic
      ~n_ci
      ~n_ctrl
      ~with_inputs
      ~phi
      ~n
      ~m
      ~n_steps:t_steps
      ~n_output
      dtype
  in
  let model =
    M.make
      dtype
      ~n
      ~m
      ~n_steps:t_steps
      ~n_output
      ~dynamics
      ~likelihood
      ~prior
      ~encoder_prior
      ~recognition
      ~initial_prior
      ()
  in
  model, M.init model

let data_of ~n_output ~t_steps values =
  Nx.create dtype [| t_steps; n_output |] (Array.of_list values)

(* A hand-rolled MGU rollout: the reference for the encoder tests. *)
let run_mgu ~phi (p : mgu_params) ~init inputs =
  let t_steps = (Nx.shape inputs).(0) in
  let x = ref init in
  for k = 0 to t_steps - 1 do
    x := D.Mgu.dyn ~phi p ~k:(scalar (Float.of_int k)) ~x:!x ~u:(Nx.get [ k ] inputs)
  done;
  !x

(* ──────────────────────────────────────────────────────────────────────────
   Covariance
   ────────────────────────────────────────────────────────────────────────── *)

let test_covariance () =
  let n = 3 in
  let p = C.make ~sigma2:2.0 ~n dtype in
  let l = C.factor p in
  let arr = Nx.to_array l in
  for i = 0 to n - 1 do
    for j = 0 to i - 1 do
      let index = (i * n) + j in
      require ~msg:"factor is upper triangular" (Float.equal arr.(index) 0.0)
    done
  done;
  let sigma = Nx.matmul l (Nx.matrix_transpose l) in
  check_close ~msg:"sigma diagonal" ~tol:1e-12 (Nx.diag sigma) (vec [ 2.0; 2.0; 2.0 ]);
  check_close
    ~msg:"log det"
    ~tol:1e-12
    (C.log_det p)
    (scalar (Float.of_int n *. Float.log 2.0));
  (* a non-diagonal factor: check Σ = L Lᵀ elementwise and its log det *)
  let p =
    { C.d = vec [ 0.5; 2.0 ]; t = Nx.create dtype [| 2; 2 |] [| 0.0; 0.3; 0.0; 0.0 |] }
  in
  let l = C.factor p in
  let sigma = Nx.matmul l (Nx.matrix_transpose l) in
  check_close
    ~msg:"sigma from factor"
    ~tol:1e-12
    (Nx.reshape [| 4 |] sigma)
    (vec [ 0.34; 0.6; 0.6; 4.0 ]);
  check_close
    ~msg:"non-diagonal log det"
    ~tol:1e-12
    (C.log_det p)
    (scalar (Float.log ((0.34 *. 4.0) -. (0.6 *. 0.6))))

(* ──────────────────────────────────────────────────────────────────────────
   Initial-state prior
   ────────────────────────────────────────────────────────────────────────── *)

let test_initial_prior () =
  let n = 2 in
  let p = I.make ~mean:0.5 ~std:2.0 ~n dtype in
  let density x =
    let sq = (x -. 0.5) *. (x -. 0.5) in
    -0.5 *. ((sq /. 4.0) +. (2.0 *. Float.log 2.0) +. Float.log (2.0 *. Float.pi))
  in
  check_close
    ~msg:"initial log density, one state"
    ~tol:1e-12
    (I.logp p (vec [ 1.0; -0.5 ]))
    (scalar (density 1.0 +. density (-0.5)));
  let batch = Nx.create dtype [| 2; 2 |] [| 1.0; -0.5; 0.0; 0.5 |] in
  check_close
    ~msg:"initial log density, batched"
    ~tol:1e-12
    (I.logp p batch)
    (vec [ density 1.0 +. density (-0.5); density 0.0 +. density 0.5 ]);
  (* the sampler reproduces the mean and standard deviation *)
  let n_samples = 40_000 in
  let s = I.sample p ~key:(Nx.Rng.key 3) ~n_samples in
  require ~msg:"initial sample shape" Poly.(Nx.shape s = [| n_samples; n |]);
  let mean = Nx.mean ~axes:[ 0 ] s in
  check_close ~msg:"initial sample mean" ~tol:0.05 mean (vec [ 0.5; 0.5 ]);
  let sd =
    let centered = Nx.sub s (Nx.reshape [| 1; n |] (Nx.mean ~axes:[ 0 ] s)) in
    Nx.sqrt (Nx.mean ~axes:[ 0 ] (Nx.mul centered centered))
  in
  check_close ~msg:"initial sample sd" ~tol:0.05 sd (vec [ 2.0; 2.0 ])

(* ──────────────────────────────────────────────────────────────────────────
   iLQR recognition
   ────────────────────────────────────────────────────────────────────────── *)

let dense_covariance ~t_steps ~m time_params space_params =
  let lt = C.factor time_params
  and ls = C.factor space_params in
  let st = Nx.to_array (Nx.matmul lt (Nx.matrix_transpose lt))
  and ss = Nx.to_array (Nx.matmul ls (Nx.matrix_transpose ls)) in
  let size = t_steps * m in
  let out = Array.create ~len:(size * size) 0.0 in
  for t = 0 to t_steps - 1 do
    for j = 0 to m - 1 do
      for t' = 0 to t_steps - 1 do
        for j' = 0 to m - 1 do
          out.((((t * m) + j) * size) + ((t' * m) + j'))
          <- st.((t * t_steps) + t') *. ss.((j * m) + j')
        done
      done
    done
  done;
  Nx.create dtype [| size; size |] out

let test_recognition_sampling () =
  let t_steps = 4
  and m = 2 in
  let model, params = setup ~n:2 ~m ~t_steps ~n_output:2 () in
  let time_cov =
    { C.d = vec [ 0.5; 1.0; 1.5; 2.0 ]
    ; t =
        Nx.create
          dtype
          [| 4; 4 |]
          [| 0.; 0.4; 0.; 0.; 0.; 0.; 0.3; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
    }
  in
  let space_cov =
    { C.d = vec [ 0.7; 1.3 ]; t = Nx.create dtype [| 2; 2 |] [| 0.; -0.2; 0.; 0. |] }
  in
  let params =
    { params with M.recognition = ({ Il.space_cov; time_cov } : Nx.float64_t Il.params) }
  in
  let n_samples = 40_000 in
  let data = Nx.zeros dtype [| t_steps; 2 |] in
  let posterior = M.infer model params ~iters:4 ~n_samples ~key:(Nx.Rng.key 17) data in
  require
    ~msg:"posterior u shape"
    Poly.(Nx.shape posterior.R.u = [| t_steps; n_samples; m |]);
  (* empirical covariance over samples, laid out like the dense one *)
  let u2 =
    Nx.reshape [| t_steps; m; n_samples |] (Nx.transpose ~axes:[ 0; 2; 1 ] posterior.R.u)
    |> Nx.reshape [| t_steps * m; n_samples |]
  in
  let empirical =
    Nx.mul_s (Nx.matmul u2 (Nx.matrix_transpose u2)) (1.0 /. Float.of_int (n_samples - 1))
  in
  let dense = dense_covariance ~t_steps ~m time_cov space_cov in
  check_close ~msg:"sampling covariance" ~tol:0.05 empirical dense

let test_recognition_log_prob () =
  let t_steps = 3
  and m = 2 in
  let model, params = setup ~n:2 ~m ~t_steps ~n_output:2 () in
  let time_cov =
    { C.d = vec [ 1.0; 0.8; 1.2 ]
    ; t = Nx.create dtype [| 3; 3 |] [| 0.; 0.2; 0.3; 0.; 0.; -0.1; 0.; 0.; 0. |]
    }
  in
  let space_cov =
    { C.d = vec [ 1.1; 0.9 ]; t = Nx.create dtype [| 2; 2 |] [| 0.; 0.4; 0.; 0. |] }
  in
  let params =
    { params with M.recognition = ({ Il.space_cov; time_cov } : Nx.float64_t Il.params) }
  in
  let data = Nx.create dtype [| t_steps; 2 |] [| 0.1; -0.2; 0.3; 0.0; -0.4; 0.2 |] in
  let mu = (M.mean model params ~iters:60 data).R.u in
  let u = Nx.create dtype [| 3; 2 |] [| 0.0; 0.1; -0.2; 0.3; 0.5; -0.1 |] in
  let log_q =
    M.log_prob model params ~iters:60 data { R.x0 = Nx.zeros dtype [| 1; 2 |]; u }
  in
  (* dense Gaussian log density, same covariance and mean *)
  let mu2 = Nx.reshape [| t_steps; m |] mu in
  let dense = dense_covariance ~t_steps ~m time_cov space_cov in
  let diff = Nx.reshape [| t_steps * m |] (Nx.sub u mu2) in
  let factor = Nx.cholesky dense in
  let solved = Nx.solve factor diff in
  let quadratic = item (Nx.sum (Nx.mul solved solved)) in
  let log_det = item (Nx.mul_s (Nx.sum (Nx.log (Nx.diag factor))) 2.0) in
  let dim = Float.of_int (t_steps * m) in
  let expected = -0.5 *. (quadratic +. log_det +. (dim *. Float.log (2.0 *. Float.pi))) in
  check_close ~msg:"log q" ~tol:1e-9 log_q (scalar expected);
  (* a batch of two sequences gives one value per sequence *)
  let u_batch =
    Nx.concatenate
      ~axis:1
      [ Nx.unsqueeze ~axes:[ 1 ] u; Nx.unsqueeze ~axes:[ 1 ] (Nx.add_s u 0.1) ]
  in
  let batch =
    M.log_prob
      model
      params
      ~iters:60
      data
      { R.x0 = Nx.zeros dtype [| 2; 2 |]; u = u_batch }
  in
  require ~msg:"batched log q shape" Poly.(Nx.shape batch = [| 2 |]);
  check_close
    ~msg:"batched log q"
    ~tol:1e-9
    batch
    (Nx.stack
       ~axis:0
       [ log_q
       ; M.log_prob
           model
           params
           ~iters:60
           data
           { R.x0 = Nx.zeros dtype [| 1; 2 |]; u = Nx.add_s u 0.1 }
       ])

(* ──────────────────────────────────────────────────────────────────────────
   Priors
   ────────────────────────────────────────────────────────────────────────── *)

let integrate_density f lo hi n =
  let h = (hi -. lo) /. Float.of_int n in
  let total = ref 0.0 in
  for i = 0 to n do
    let x = lo +. (h *. Float.of_int i) in
    let w = if i = 0 || i = n then 0.5 else 1.0 in
    total := !total +. (w *. Float.exp (item (f (scalar x))))
  done;
  !total *. h

let test_prior_densities () =
  let m = 1 in
  (* The exact densities of the reference implementation, normalized. *)
  let gaussian = P.Gaussian.make ~spatial_std:1.5 ~first_bin:0.25 ~m dtype in
  let student = P.Student.make ~spatial_std:1.5 ~nu:7.0 ~first_std:0.5 ~m dtype in
  let laplacian = P.Laplacian.make ~spatial_std:1.5 ~first_std:0.5 ~m dtype in
  let mass prior k =
    integrate_density
      (fun u -> prior.P.logp_t prior.P.params ~k:(scalar (Float.of_int k)) ~u)
      (-25.0)
      25.0
      20_000
  in
  check_close
    ~msg:"gaussian density k=0"
    ~tol:1e-4
    (scalar (mass gaussian 0))
    (scalar 1.0);
  check_close
    ~msg:"gaussian density k=1"
    ~tol:1e-4
    (scalar (mass gaussian 1))
    (scalar 1.0);
  check_close ~msg:"student density k=0" ~tol:1e-4 (scalar (mass student 0)) (scalar 1.0);
  check_close ~msg:"student density k=1" ~tol:1e-3 (scalar (mass student 1)) (scalar 1.0);
  check_close
    ~msg:"laplacian density k=0"
    ~tol:1e-4
    (scalar (mass laplacian 0))
    (scalar 1.0);
  check_close
    ~msg:"laplacian density k=1"
    ~tol:1e-4
    (scalar (mass laplacian 1))
    (scalar 1.0);
  (* the Student's scale makes its variance spatial_std²; the Laplacian's is
     2 spatial_std² *)
  let sample_var prior =
    let n_steps = 2_000 in
    let n_keys = 64 in
    let total = ref 0.0 in
    for i = 1 to n_keys do
      let u = prior.P.sample prior.P.params ~key:(Nx.Rng.key i) ~n_steps in
      let x = Nx.slice [ R (1, n_steps) ] u |> Nx.reshape [| n_steps - 1 |] in
      total
      := !total
         +. item (Nx.mul_s (Nx.sum (Nx.mul x x)) (1.0 /. Float.of_int (n_steps - 1)))
    done;
    !total /. Float.of_int n_keys
  in
  check_close
    ~msg:"student sample variance"
    ~tol:0.6
    (scalar (sample_var student))
    (scalar (1.5 *. 1.5));
  check_close
    ~msg:"laplacian sample variance"
    ~tol:0.05
    (scalar (sample_var laplacian))
    (scalar (2.0 *. 1.5 *. 1.5))

(* The Gaussian prior's closed-form KL, against a dense computation of
   [½ (tr (Σ_p⁻¹ Σ_q) + muᵀ Σ_p⁻¹ mu + log det Σ_p − log det Σ_q − d)] over the
   full [T·m] covariance. *)
let test_kl () =
  let t_steps = 3
  and m = 2 in
  let prior = P.Gaussian.make ~spatial_std:1.5 ~first_bin:0.5 ~m dtype in
  let time_cov =
    { C.d = vec [ 0.7; 1.0; 1.3 ]
    ; t = Nx.create dtype [| 3; 3 |] [| 0.; 0.2; 0.1; 0.; 0.; -0.3; 0.; 0.; 0. |]
    }
  in
  let space_cov =
    { C.d = vec [ 1.1; 0.9 ]; t = Nx.create dtype [| 2; 2 |] [| 0.; 0.4; 0.; 0. |] }
  in
  let mu = Nx.create dtype [| 3; 2 |] [| 0.2; -0.1; 0.4; 0.0; -0.3; 0.5 |] in
  let got =
    (Option.value_exn prior.P.kl_to_gaussian)
      prior.P.params
      ~mu
      ~space_factor:(C.factor space_cov)
      ~time_factor:(C.factor time_cov)
  in
  let dense_q = dense_covariance ~t_steps ~m time_cov space_cov in
  let spatial = Nx.to_array (Nx.exp prior.P.params.P.Gaussian.log_spatial_stds) in
  let first_bin = Float.exp (item prior.P.params.P.Gaussian.log_first_bin) in
  let d = t_steps * m in
  let prior_var r =
    let t = r / m
    and j = Int.rem r m in
    let s = spatial.(j) in
    let sigma_t = if t = 0 then first_bin else 1.0 in
    sigma_t *. s *. sigma_t *. s
  in
  let q = Nx.to_array dense_q
  and mu_arr = Nx.to_array mu in
  let trace = ref 0.0
  and quadratic = ref 0.0
  and log_det_p = ref 0.0 in
  for r = 0 to d - 1 do
    let v = prior_var r in
    trace := !trace +. (q.((r * d) + r) /. v);
    quadratic := !quadratic +. (mu_arr.(r) *. mu_arr.(r) /. v);
    log_det_p := !log_det_p +. Float.log v
  done;
  let log_det_q = item (Nx.mul_s (Nx.sum (Nx.log (Nx.diag (Nx.cholesky dense_q)))) 2.0) in
  let expected =
    0.5 *. (!trace +. !quadratic +. !log_det_p -. log_det_q -. Float.of_int d)
  in
  check_close ~msg:"KL against a dense computation" ~tol:1e-9 got (scalar expected);
  (* The KL is non-negative and grows as the prior narrows. *)
  require ~msg:"KL is non-negative" Float.(item got > -1e-9);
  let narrow =
    { prior with
      P.params =
        { prior.P.params with
          P.Gaussian.log_spatial_stds =
            Nx.sub_s prior.P.params.P.Gaussian.log_spatial_stds 1.0
        }
    }
  in
  let narrow_kl =
    (Option.value_exn narrow.P.kl_to_gaussian)
      narrow.P.params
      ~mu
      ~space_factor:(C.factor space_cov)
      ~time_factor:(C.factor time_cov)
  in
  require ~msg:"narrower prior, larger KL" Float.(item narrow_kl > item got);
  (* mu = 0 and q matching the prior factor gives no KL at all. *)
  let matching_prior = P.Gaussian.make ~spatial_std:1.0 ~first_bin:1.0 ~m dtype in
  let identity_space = { C.d = vec [ 1.0; 1.0 ]; t = Nx.zeros dtype [| 2; 2 |] } in
  let identity_time = { C.d = vec [ 1.0; 1.0; 1.0 ]; t = Nx.zeros dtype [| 3; 3 |] } in
  let zero_kl =
    (Option.value_exn matching_prior.P.kl_to_gaussian)
      matching_prior.P.params
      ~mu:(Nx.zeros dtype [| t_steps; m |])
      ~space_factor:(C.factor identity_space)
      ~time_factor:(C.factor identity_time)
  in
  check_close ~msg:"KL of a matching Gaussian" ~tol:1e-10 zero_kl (scalar 0.0)

(* ──────────────────────────────────────────────────────────────────────────
   Likelihoods
   ────────────────────────────────────────────────────────────────────────── *)

let test_likelihoods () =
  let n = 2
  and n_output = 2 in
  let likelihood = L.Gaussian.make ~n ~n_output dtype in
  let params : lik_params =
    { L.Gaussian.c = Nx.eye dtype n
    ; bias = Nx.zeros dtype [| n_output |]
    ; log_variances = vec [ Float.log 0.5; Float.log 2.0 ]
    }
  in
  let x = vec [ 1.0; -1.0 ]
  and o = vec [ 0.5; 0.5 ] in
  let logp = likelihood.L.logp_t params ~k:(scalar 0.0) ~x ~o in
  let expected =
    let sq x = x *. x in
    -0.5
    *. ((sq (0.5 -. 1.0) /. 0.5)
        +. Float.log 0.5
        +. Float.log (2.0 *. Float.pi)
        +. ((sq (0.5 +. 1.0) /. 2.0) +. Float.log 2.0 +. Float.log (2.0 *. Float.pi)))
  in
  check_close ~msg:"gaussian log p" ~tol:1e-12 logp (scalar expected);
  let mean = likelihood.L.mean params (vec [ 2.0; 3.0 ]) in
  check_close ~msg:"gaussian mean" ~tol:1e-12 mean (vec [ 2.0; 3.0 ]);
  let poisson = L.Poisson.make ~n ~n_output ~dt:0.1 dtype in
  let poisson_params : poisson_params =
    { L.Poisson.c = Nx.eye dtype n
    ; bias = vec [ 0.0; 0.0 ]
    ; log_gain = vec [ 0.0; Float.log 2.0 ]
    }
  in
  let x = vec [ 1.0; 0.5 ] in
  let o = vec [ 3.0; 2.0 ] in
  let logp = poisson.L.logp_t poisson_params ~k:(scalar 0.0) ~x ~o in
  let rate _j xj g = 0.1 *. g *. (Float.exp xj +. 1e-3) in
  let lgamma_exact z =
    (* log Γ of a small integer + 1, by definition *)
    let z = Int.of_float (z +. 0.5) in
    let s = ref 0.0 in
    for i = 2 to z do
      s := !s +. Float.log (Float.of_int i)
    done;
    !s
  in
  let expected =
    let r0 = rate 0 1.0 1.0
    and r1 = rate 1 0.5 2.0 in
    (3.0 *. Float.log r0)
    -. r0
    -. lgamma_exact 3.0
    +. ((2.0 *. Float.log r1) -. r1 -. lgamma_exact 2.0)
  in
  check_close ~msg:"poisson log p" ~tol:1e-9 logp (scalar expected);
  let counts =
    poisson.L.sample poisson_params ~key:(Nx.Rng.key 5) ~mean:(Nx.full dtype [| 5 |] 3.0)
  in
  require ~msg:"poisson sample shape" Poly.(Nx.shape counts = [| 5 |]);
  let arr = Nx.to_array counts in
  Array.iter arr ~f:(fun v -> require ~msg:"poisson sample non-negative" Float.(v >= 0.0));
  let mean = item (Nx.mul_s (Nx.sum counts) (1.0 /. 5.0)) in
  require ~msg:"poisson sample mean near rate" Float.(Float.abs (mean -. 3.0) < 2.5)

(* ──────────────────────────────────────────────────────────────────────────
   Dynamics
   ────────────────────────────────────────────────────────────────────────── *)

let test_dynamics () =
  let n = 2
  and m = 2 in
  let dynamics = D.Linear.make ~n ~m dtype in
  let params =
    { dynamics.D.params with
      w = Nx.create dtype [| 2; 2 |] [| 0.8; 0.2; 0.0; 0.9 |]
    ; dt_over_tau = 1.0
    }
  in
  let matrix = D.Linear.matrix params in
  check_close
    ~msg:"linear matrix"
    ~tol:1e-12
    (Nx.reshape [| 4 |] matrix)
    (vec [ 0.8; 0.2; 0.0; 0.9 ]);
  let x = vec [ 1.0; 2.0 ]
  and u = vec [ 0.5; -0.5 ] in
  check_close
    ~msg:"linear step k=0"
    ~tol:1e-12
    (dynamics.D.dyn params ~k:(scalar 0.0) ~x ~u)
    (vec [ 1.3; 1.5 ]);
  check_close
    ~msg:"linear step k=1"
    ~tol:1e-12
    (dynamics.D.dyn params ~k:(scalar 1.0) ~x ~u)
    (vec [ 1.3; 1.5 ]);
  (* a rollout matches a hand-written loop *)
  let t_steps = 5 in
  let u =
    Nx.create
      dtype
      [| t_steps; m |]
      [| 0.1; 0.0; 0.2; -0.1; 0.0; 0.3; -0.2; 0.1; 0.05; 0.0 |]
  in
  let z = D.integrate dynamics params ~x0:(Nx.zeros dtype [| n |]) ~u in
  require ~msg:"rollout shape" Poly.(Nx.shape z = [| t_steps + 1; n |]);
  let expected =
    let xs = Array.create ~len:(t_steps + 1) (Nx.zeros dtype [| n |]) in
    for k = 0 to t_steps - 1 do
      xs.(k + 1)
      <- dynamics.D.dyn params ~k:(scalar (Float.of_int k)) ~x:xs.(k) ~u:(Nx.get [ k ] u)
    done;
    Nx.stack ~axis:0 (Array.to_list xs)
  in
  check_close ~msg:"rollout" ~tol:1e-12 z expected;
  (* the tanh RNN rolls out, batched and unbatched, with the same values *)
  let rnn = D.Linear_nonlinear.make ~n ~nh:4 ~m ~dt:0.1 ~tau:1.0 ~phi:Nx.tanh dtype in
  let batched =
    D.integrate
      rnn
      rnn.D.params
      ~x0:(Nx.zeros dtype [| 2; n |])
      ~u:(Nx.broadcast_to [| t_steps; 2; m |] (Nx.unsqueeze ~axes:[ 1 ] u))
  in
  require ~msg:"batched rollout shape" Poly.(Nx.shape batched = [| t_steps + 1; 2; n |]);
  let first = Nx.slice [ A; I 0 ] batched in
  check_close
    ~msg:"batched rollout member"
    ~tol:1e-10
    first
    (D.integrate rnn rnn.D.params ~x0:(Nx.zeros dtype [| n |]) ~u)

(* The paper's MGU (Appendix B.2, with the [z_{t-1}] subscript typo of (S9)
   corrected): the input is additive in the candidate and does not enter the
   forget gate. The expected step is computed with host floats from the same
   equations, not from the implementation. *)
let test_mgu () =
  let n = 2
  and m = 2 in
  let dynamics = D.Mgu.make ~n ~m ~key:(Nx.Rng.key 4) ~phi dtype in
  let params : mgu_params =
    { uf = Nx.create dtype [| n; n |] [| 0.5; 0.0; 0.0; -0.5 |]
    ; uh = Nx.create dtype [| n; n |] [| 0.25; 0.0; 0.0; 0.25 |]
    ; w = Nx.create dtype [| m; n |] [| 1.0; 0.5; -0.5; 1.0 |]
    ; bf = vec [ 0.3; -0.1 ]
    ; bh = vec [ 0.0; 0.2 ]
    }
  in
  let x = vec [ 1.0; -2.0 ]
  and u = vec [ 0.5; -1.0 ]
  and k = scalar 0.0 in
  (* f = tanh (U_f z + b_f); ẑ = tanh (U_h (f ⊙ z) + b_h) + u W;
     z' = (1 − f) ⊙ z + f ⊙ ẑ. *)
  let f1 = Float.tanh ((0.5 *. 1.0) +. 0.3)
  and f2 = Float.tanh ((-0.5 *. -2.0) -. 0.1) in
  let h1 = 0.25 *. f1
  and h2 = (0.25 *. f2 *. -2.0) +. 0.2 in
  let uh1 = (0.5 *. 1.0) +. (-1.0 *. -0.5)
  and uh2 = (0.5 *. 0.5) +. (-1.0 *. 1.0) in
  let z1 = ((1.0 -. f1) *. 1.0) +. (f1 *. (Float.tanh h1 +. uh1))
  and z2 = ((1.0 -. f2) *. -2.0) +. (f2 *. (Float.tanh h2 +. uh2)) in
  check_close ~msg:"mgu step" ~tol:1e-12 (dynamics.D.dyn params ~k ~x ~u) (vec [ z1; z2 ]);
  (* the same step under the ReLU activation: the paper's gate and candidate
     with [phi = relu] *)
  let relu x = Nx.maximum x (Nx.zeros_like x) in
  let r = Float.max 0.0 in
  let f1r = r ((0.5 *. 1.0) +. 0.3)
  and f2r = r ((-0.5 *. -2.0) -. 0.1) in
  let h1r = r (0.25 *. f1r)
  and h2r = r ((0.25 *. f2r *. -2.0) +. 0.2) in
  let z1r = ((1.0 -. f1r) *. 1.0) +. (f1r *. (h1r +. uh1))
  and z2r = ((1.0 -. f2r) *. -2.0) +. (f2r *. (h2r +. uh2)) in
  check_close
    ~msg:"mgu relu step"
    ~tol:1e-12
    (D.Mgu.dyn ~phi:relu params ~k ~x ~u)
    (vec [ z1r; z2r ]);
  let u' = vec [ -0.25; 0.75 ] in
  let z_u = dynamics.D.dyn params ~k ~x ~u in
  let z_u' = dynamics.D.dyn params ~k ~x ~u:u' in
  let z_zero = dynamics.D.dyn params ~k ~x ~u:(Nx.zeros dtype [| m |]) in
  let z_sum = dynamics.D.dyn params ~k ~x ~u:(Nx.add u u') in
  check_close
    ~msg:"mgu input additivity"
    ~tol:1e-12
    (Nx.sub z_sum z_zero)
    (Nx.add (Nx.sub z_u z_zero) (Nx.sub z_u' z_zero));
  (* a rollout matches a hand-written loop, and a batched rollout its
     members *)
  let t_steps = 4 in
  let us =
    Nx.create dtype [| t_steps; m |] [| 0.5; -1.0; 0.25; 0.0; -0.75; 0.5; 0.0; 0.25 |]
  in
  let x0 = vec [ 0.3; -0.4 ] in
  let z = D.integrate dynamics params ~x0 ~u:us in
  require ~msg:"mgu rollout shape" Poly.(Nx.shape z = [| t_steps + 1; n |]);
  let expected =
    let xs = Array.create ~len:(t_steps + 1) x0 in
    for k = 0 to t_steps - 1 do
      xs.(k + 1)
      <- dynamics.D.dyn params ~k:(scalar (Float.of_int k)) ~x:xs.(k) ~u:(Nx.get [ k ] us)
    done;
    Nx.stack ~axis:0 (Array.to_list xs)
  in
  check_close ~msg:"mgu rollout" ~tol:1e-12 z expected;
  let batched =
    D.integrate
      dynamics
      params
      ~x0:(Nx.broadcast_to [| 2; n |] x0)
      ~u:(Nx.broadcast_to [| t_steps; 2; m |] (Nx.unsqueeze ~axes:[ 1 ] us))
  in
  require
    ~msg:"mgu batched rollout shape"
    Poly.(Nx.shape batched = [| t_steps + 1; 2; n |]);
  check_close ~msg:"mgu batched rollout member" ~tol:1e-10 (Nx.slice [ A; I 0 ] batched) z;
  (* the dynamics is differentiable in every parameter *)
  match
    Rune.check_grads
      (D.Mgu.structure ())
      (fun p -> Nx.sum (D.integrate dynamics p ~x0 ~u:us))
      params
  with
  | Ok () -> ()
  | Error msg -> failwith ("mgu gradient check failed: " ^ msg)

(* ──────────────────────────────────────────────────────────────────────────
   iLQR encoder
   ────────────────────────────────────────────────────────────────────────── *)

(* The cost the encoder minimizes for a linear dynamics, a Gaussian
   likelihood of unit variance and an isotropic Gaussian encoder prior:
   [½ Σ_k ‖u_k‖² + ½ Σ_{k≥1} ‖x_k − o_{k−1}‖² + ½ ‖x_T − o_{T−1}‖²]. *)
let lqr_cost ~a ~o ~t_steps u =
  let x = ref (Nx.zeros dtype [| (Nx.shape a).(0) |]) in
  let total = ref (scalar 0.0) in
  for k = 0 to t_steps - 1 do
    if k >= 1
    then (
      let diff = Nx.sub !x (Nx.get [ k - 1 ] o) in
      total := Nx.add !total (Nx.mul_s (Nx.sum (Nx.mul diff diff)) 0.5));
    let u_k = Nx.get [ k ] u in
    total := Nx.add !total (Nx.mul_s (Nx.sum (Nx.mul u_k u_k)) 0.5);
    x := Nx.add (Nx.matmul !x a) u_k
  done;
  let diff = Nx.sub !x (Nx.get [ t_steps - 1 ] o) in
  Nx.add !total (Nx.mul_s (Nx.sum (Nx.mul diff diff)) 0.5)

let lqr_setup () =
  let n = 2
  and m = 2
  and t_steps = 5 in
  let model, params = setup ~n ~m ~t_steps ~n_output:2 () in
  let w = Nx.create dtype [| 2; 2 |] [| 0.8; 0.2; 0.0; 0.9 |] in
  let params : iilqr_params =
    { params with
      M.dynamics = { params.M.dynamics with w; dt_over_tau = 1.0 }
    ; likelihood =
        { L.Gaussian.c = Nx.eye dtype n
        ; bias = Nx.zeros dtype [| 2 |]
        ; log_variances = Nx.zeros dtype [| 2 |]
        }
    }
  in
  let o =
    Nx.create
      dtype
      [| t_steps; 2 |]
      [| 1.0; 0.0; 0.5; -0.5; -0.2; 0.7; 0.3; 0.1; -0.6; 0.4 |]
  in
  model, params, o

let test_encoder_stationarity () =
  let model, params, o = lqr_setup () in
  let generative = M.generative model
  and gparams = M.gparams params in
  let mu = Il.posterior_mean generative gparams ~max_iter:100 ~conv_threshold:1e-14 o in
  require ~msg:"posterior mean shape" Poly.(Nx.shape mu = [| 5; 2 |]);
  let a = D.Linear.matrix params.M.dynamics in
  let grad = Rune.grad' (fun u -> lqr_cost ~a ~o ~t_steps:5 u) mu in
  let max_grad = item (Nx.max (Nx.abs grad)) in
  if not Float.(max_grad < 1e-6)
  then Stdio.printf "encoder stationarity: max |grad| = %.3g\n%!" max_grad;
  require ~msg:"iLQR solves the LQR problem" Float.(max_grad < 1e-6);
  (* the fixed solver finds the same point, and the generic mean agrees *)
  let mu_fixed = Il.posterior_mean_fixed generative gparams ~iters:60 o in
  check_close ~msg:"fixed encoder matches adaptive" ~tol:1e-6 mu_fixed mu;
  let via_model = (M.mean model params ~iters:60 o).R.u in
  check_close
    ~msg:"model mean matches the encoder"
    ~tol:1e-6
    (Nx.reshape [| 5; 2 |] via_model)
    mu

(* ──────────────────────────────────────────────────────────────────────────
   LFADS recognition
   ────────────────────────────────────────────────────────────────────────── *)

let lfads_data ~t_steps ~n_output () =
  let values = [ 0.4; -0.3; 0.7; 0.2; -0.5; 0.1; 0.3; -0.2; 0.6; -0.4; 0.15; 0.25 ] in
  let values = List.take values (t_steps * n_output) in
  data_of ~n_output ~t_steps values

let test_lfads_shapes_and_density () =
  let t_steps = 4
  and n = 2
  and m = 2 in
  let model, params = setup_lfads ~n ~m ~t_steps ~n_output:2 () in
  let data = lfads_data ~t_steps ~n_output:2 () in
  let posterior = M.infer model params ~n_samples:3 ~key:(Nx.Rng.key 11) data in
  require ~msg:"lfads x0 shape" Poly.(Nx.shape posterior.R.x0 = [| 3; n |]);
  require ~msg:"lfads u shape" Poly.(Nx.shape posterior.R.u = [| t_steps; 3; m |]);
  require ~msg:"lfads log_prob shape" Poly.(Nx.shape posterior.R.log_prob = [| 3 |]);
  (* the standalone density replays the same trajectory: same value *)
  let replay =
    M.log_prob model params data { R.x0 = posterior.R.x0; u = posterior.R.u }
  in
  check_close ~msg:"lfads density replay" ~tol:1e-10 replay posterior.R.log_prob;
  (* the same key gives the same draws, a different key different ones *)
  let again = M.infer model params ~n_samples:3 ~key:(Nx.Rng.key 11) data in
  check_close ~msg:"lfads draws are keyed" ~tol:1e-12 again.R.u posterior.R.u;
  let other = M.infer model params ~n_samples:3 ~key:(Nx.Rng.key 12) data in
  require
    ~msg:"different keys draw differently"
    Float.(max_abs (Nx.sub other.R.u posterior.R.u) > 1e-6);
  (* without inferred inputs the inputs are the point mass at zero *)
  let model0, params0 = setup_lfads ~n ~m ~t_steps ~n_output:2 ~with_inputs:false () in
  let posterior0 = M.infer model0 params0 ~n_samples:3 ~key:(Nx.Rng.key 13) data in
  check_close
    ~msg:"lfads without inputs, u = 0"
    ~tol:1e-12
    posterior0.R.u
    (Nx.zeros dtype [| t_steps; 3; m |]);
  (* the mean is a batch of one *)
  let mean = M.mean model params data in
  require ~msg:"lfads mean x0 shape" Poly.(Nx.shape mean.R.x0 = [| 1; n |]);
  require ~msg:"lfads mean u shape" Poly.(Nx.shape mean.R.u = [| t_steps; 1; m |])

(* The initial-condition encoder against a hand-rolled MGU rollout: a readout
   that selects the forward (resp. backward) encoder's final state must give
   that state back, and the backward encoder must have read the data in
   reverse. *)
let test_lfads_encoder () =
  let t_steps = 4
  and n = 2
  and n_output = 2 in
  let model, params = setup_lfads ~n ~m:2 ~t_steps ~n_output ~n_ic:4 () in
  let data = lfads_data ~t_steps ~n_output () in
  let p = params.M.recognition in
  let selector rows =
    (* [h = e_ic · readout]; map [e_ic] entries [rows] onto the mean head. *)
    let r = Nx.zeros dtype [| 2 * 4; 2 * n |] in
    let built = ref r in
    List.iteri rows ~f:(fun j i ->
      built := Nx.set [ I i; I j ] (Nx.ones dtype [||]) !built);
    !built
  in
  let with_readout rows =
    { params with
      M.recognition =
        { p with
          Lf.ic_readout = selector rows
        ; Lf.ic_readout_bias = Nx.zeros dtype [| 2 * n |]
        }
    }
  in
  let forward_expected =
    Nx.slice [ R (0, n) ] (run_mgu ~phi p.Lf.ic_fwd ~init:p.Lf.ic_fwd_init data)
  in
  let fwd_params = with_readout [ 0; 1 ] in
  let fwd = M.mean model fwd_params data in
  check_close
    ~msg:"forward initial-condition encoder"
    ~tol:1e-12
    (Nx.reshape [| n |] fwd.R.x0)
    forward_expected;
  let backward_expected =
    Nx.slice
      [ R (0, n) ]
      (run_mgu ~phi p.Lf.ic_bwd ~init:p.Lf.ic_bwd_init (Nx.flip ~axes:[ 0 ] data))
  in
  let bwd_params = with_readout [ 4; 5 ] in
  let bwd = M.mean model bwd_params data in
  check_close
    ~msg:"backward initial-condition encoder"
    ~tol:1e-12
    (Nx.reshape [| n |] bwd.R.x0)
    backward_expected

(* The controller against a hand-rolled generator rollout: a zero input
   readout with a constant bias makes every conditional mean that constant,
   so the push-mean trajectory is the generator driven by a constant input. *)
let test_lfads_controller () =
  let t_steps = 4
  and n = 2
  and m = 2 in
  let model, params = setup_lfads ~n ~m ~t_steps ~n_output:2 () in
  let data = lfads_data ~t_steps ~n_output:2 () in
  let p = params.M.recognition in
  let c = vec [ 0.2; -0.3 ] in
  let params =
    { params with
      M.recognition =
        { p with
          Lf.u_readout = Nx.zeros dtype [| (Nx.shape p.Lf.u_readout).(0); 2 * m |]
        ; Lf.u_readout_bias = Nx.concatenate ~axis:0 [ c; Nx.zeros dtype [| m |] ]
        }
    }
  in
  let mean = M.mean model params data in
  let expected_u = Nx.broadcast_to [| t_steps; 1; m |] (Nx.reshape [| 1; 1; m |] c) in
  check_close ~msg:"push-mean inputs" ~tol:1e-12 mean.R.u expected_u;
  let z, _ = Ilqr_vae.predictions_deterministic model params mean in
  let x = ref (Nx.reshape [| n |] mean.R.x0) in
  let states = ref [ !x ] in
  for k = 0 to t_steps - 1 do
    x := D.Mgu.dyn ~phi params.M.dynamics ~k:(scalar (Float.of_int k)) ~x:!x ~u:c;
    states := !x :: !states
  done;
  let expected_z = Nx.stack ~axis:0 (List.rev !states) in
  check_close
    ~msg:"push-mean rollout"
    ~tol:1e-12
    (Nx.reshape [| t_steps + 1; n |] z)
    expected_z

(* When the posterior is exactly the prior, the sampled KL is identically
   zero, and the path-derivative (sticking-the-landing) gradient of the KL
   with respect to the density's parameters is zero at every draw. The
   gradient through the live density is not: that is the score-function part
   the estimator removes. *)
let test_lfads_sticking_the_landing () =
  let t_steps = 3
  and n = 2
  and m = 2
  and spatial_std = 0.8 in
  let initial_prior = I.make ~mean:0.25 ~std:0.6 ~n dtype in
  let keys = Nx.Rng.split ~n:5 (Nx.Rng.key 42) in
  let dynamics = D.Mgu.make ~key:keys.(0) ~n ~m ~phi dtype in
  let likelihood = L.Gaussian.make ~key:keys.(1) ~n ~n_output:2 dtype in
  let prior = P.Gaussian.make ~m ~spatial_std ~first_bin:1.0 dtype in
  let encoder_prior = P.Gaussian.make ~m ~spatial_std:1.0 dtype in
  let recognition =
    Lf.make
      ~key:keys.(2)
      ~n_ic:4
      ~n_ci:4
      ~n_ctrl:3
      ~phi
      ~n
      ~m
      ~n_steps:t_steps
      ~n_output:2
      dtype
  in
  let model =
    M.make
      dtype
      ~n
      ~m
      ~n_steps:t_steps
      ~n_output:2
      ~dynamics
      ~likelihood
      ~prior
      ~encoder_prior
      ~recognition
      ~initial_prior
      ()
  in
  let params = M.init model in
  let p = params.M.recognition in
  (* q(x0) = p(x0) and q(u_t) = p(u_t) for every t. *)
  let recognition_params =
    { p with
      Lf.ic_readout = Nx.zeros dtype [| 2 * 4; 2 * n |]
    ; Lf.ic_readout_bias =
        Nx.concatenate
          ~axis:0
          [ initial_prior.I.mean; Nx.mul_s initial_prior.I.log_stds 2.0 ]
    ; Lf.u_readout = Nx.zeros dtype [| (Nx.shape p.Lf.u_readout).(0); 2 * m |]
    ; Lf.u_readout_bias =
        Nx.concatenate
          ~axis:0
          [ Nx.zeros dtype [| m |]; Nx.full dtype [| m |] (2.0 *. Float.log spatial_std) ]
    }
  in
  let params = { params with M.recognition = recognition_params } in
  let data = lfads_data ~t_steps ~n_output:2 () in
  let key = Nx.Rng.key 5 in
  let kl params =
    let posterior = M.infer model params ~n_samples:4 ~key data in
    (El.compute model params ~posterior data).El.kl
  in
  check_close ~msg:"KL is exactly zero" ~tol:1e-10 (kl params) (scalar 0.0);
  let with_bias bias =
    { params with M.recognition = { params.M.recognition with Lf.u_readout_bias = bias } }
  in
  let stl_grad =
    Rune.grad' (fun bias -> kl (with_bias bias)) params.M.recognition.Lf.u_readout_bias
  in
  require
    ~msg:"path-derivative gradient sticks to the landing"
    Float.(max_abs stl_grad < 1e-10);
  (* the live-parameter density, by contrast, has a nonzero sampled gradient *)
  let live_kl params =
    let posterior = M.infer model params ~n_samples:4 ~key data in
    Nx.mean
      (Nx.sub
         (M.log_prob model params data { R.x0 = posterior.R.x0; u = posterior.R.u })
         (El.log_prior model params ~x0:posterior.R.x0 ~u:posterior.R.u))
  in
  let live_grad =
    Rune.grad'
      (fun bias -> live_kl (with_bias bias))
      params.M.recognition.Lf.u_readout_bias
  in
  if Float.(max_abs live_grad < 1e-8)
  then Stdio.printf "note: the live-density gradient happened to vanish\n%!";
  require ~msg:"live density does not stick" Float.(max_abs live_grad > 1e-8)

(* A plain-pathwise objective: the same value as {!Elbo.loss_fixed}, but its
   density is evaluated with live parameters, so its autodiff gradient is the
   exact derivative of the traced function and {!Rune.check_grads} applies.
   The library's own objective removes the score-function part of that
   derivative (Roeder, Wu & Duvenaud, "sticking the landing"), which is an
   unbiased estimator of the gradient rather than the derivative of any single
   trace. *)
let plain_objective
      (type w d l p e r)
      (model : (w, d, l, p, e, r) M.t)
      (params : (w, d, l, p, e, r) M.params)
      ~key
      ~n_samples
      ~iters
      data
  =
  let posterior = M.infer model params ~iters ~n_samples ~key data in
  let samples : w R.samples = { R.x0 = posterior.R.x0; u = posterior.R.u } in
  let log_prob = M.log_prob model params ~iters data samples in
  let posterior = { posterior with R.log_prob } in
  Nx.neg (El.compute model params ~posterior data).El.elbo

let test_lfads_gradients () =
  let t_steps = 3
  and n = 2
  and m = 2 in
  let model, params = setup_lfads ~n ~m ~t_steps ~n_output:2 () in
  let data = lfads_data ~t_steps ~n_output:2 () in
  let structure = M.structure model in
  let key = Nx.Rng.key 21 in
  (match
     Rune.check_grads
       structure
       (fun p -> plain_objective model p ~key ~n_samples:3 ~iters:4 data)
       params
   with
   | Ok () -> ()
   | Error msg -> failwith ("lfads gradient check failed: " ^ msg));
  check_close
    ~msg:"lfads path-derivative objective has the same value"
    ~tol:1e-12
    (El.loss_fixed model params ~key ~n_samples:3 ~iters:4 data)
    (plain_objective model params ~key ~n_samples:3 ~iters:4 data)

let test_lfads_jit () =
  let t_steps = 3
  and n = 2
  and m = 2 in
  let model, params = setup_lfads ~n ~m ~t_steps ~n_output:2 () in
  let data = lfads_data ~t_steps ~n_output:2 () in
  let key = Nx.Rng.key 22 in
  let structure = M.structure model in
  let eager = El.loss_fixed model params ~key ~n_samples:3 ~iters:4 data in
  let jitted =
    Rune.jit
      Nx.Ptree.(structure @-> Nx.Rng.ptree @-> returns Nx.Ptree.tensor)
      (fun p k -> El.loss_fixed model p ~key:k ~n_samples:3 ~iters:4 data)
  in
  check_close ~msg:"lfads jit loss" ~tol:1e-10 (jitted params key) eager;
  let eager_grad =
    Rune.value_and_grad
      structure
      (fun p -> El.loss_fixed model p ~key ~n_samples:3 ~iters:4 data)
      params
  in
  let jitted_grad =
    Rune.jit
      Nx.Ptree.(structure @-> Nx.Rng.ptree @-> returns (pair tensor structure))
      (fun p k ->
         Rune.value_and_grad
           structure
           (fun p -> El.loss_fixed model p ~key:k ~n_samples:3 ~iters:4 data)
           p)
  in
  let loss', grads' = jitted_grad params key in
  check_close ~msg:"lfads jit grad loss" ~tol:1e-10 loss' (fst eager_grad);
  let grads = snd eager_grad in
  check_close
    ~msg:"lfads jit grad controller"
    ~tol:1e-9
    grads'.M.recognition.Lf.con.uf
    grads.M.recognition.Lf.con.uf;
  check_close
    ~msg:"lfads jit grad ic readout"
    ~tol:1e-9
    grads'.M.recognition.Lf.ic_readout
    grads.M.recognition.Lf.ic_readout

(* ──────────────────────────────────────────────────────────────────────────
   ELBO
   ────────────────────────────────────────────────────────────────────────── *)

let test_elbo_gradient () =
  let n = 2
  and m = 2
  and t_steps = 4 in
  let model, params = setup ~n ~m ~t_steps ~n_output:2 () in
  let o =
    Nx.create dtype [| t_steps; 2 |] [| 1.0; 0.0; 0.5; -0.5; -0.2; 0.7; 0.3; 0.1 |]
  in
  let structure = M.structure model in
  let key = Nx.Rng.key 1 in
  (* the whole objective, with its live density: the gradient of the encoder
     is implicit, so this checks the solve's VJP end to end *)
  (match
     Rune.check_grads
       structure
       (fun p -> plain_objective model p ~key ~n_samples:2 ~iters:12 o)
       params
   with
   | Ok () -> ()
   | Error msg -> failwith ("implicit-gradient check failed: " ^ msg));
  (* the library objective differs only in the backward pass: same value *)
  check_close
    ~msg:"the path-derivative objective has the same value"
    ~tol:1e-12
    (El.loss_fixed model params ~key ~n_samples:2 ~iters:12 o)
    (plain_objective model params ~key ~n_samples:2 ~iters:12 o)

let test_elbo_terms () =
  let n = 2
  and m = 2
  and t_steps = 4 in
  let model, params = setup ~n ~m ~t_steps ~n_output:2 () in
  let o =
    Nx.create dtype [| t_steps; 2 |] [| 1.0; 0.0; 0.5; -0.5; -0.2; 0.7; 0.3; 0.1 |]
  in
  let structure = M.structure model in
  let posterior = M.infer model params ~iters:12 ~n_samples:4 ~key:(Nx.Rng.key 2) o in
  let terms = El.compute model params ~posterior o in
  check_close
    ~msg:"elbo = ll − beta kl"
    ~tol:1e-10
    terms.El.elbo
    (Nx.sub terms.El.log_likelihood terms.El.kl);
  let loss = El.loss model params ~iters:12 ~key:(Nx.Rng.key 2) ~n_samples:4 o in
  check_close ~msg:"loss is the negated elbo" ~tol:1e-12 loss (Nx.neg terms.El.elbo);
  (* beta scales the KL term *)
  let half = El.compute model params ~beta:(scalar 0.5) ~posterior o in
  check_close
    ~msg:"beta scales the KL"
    ~tol:1e-10
    half.El.elbo
    (Nx.sub terms.El.log_likelihood (Nx.mul_s terms.El.kl 0.5));
  (* the prior enters the KL: a wider prior has a larger KL *)
  let wide =
    { params with
      M.prior =
        { params.M.prior with
          log_spatial_stds = Nx.add_s params.M.prior.log_spatial_stds 1.0
        }
    }
  in
  let wide_posterior = M.infer model wide ~iters:12 ~n_samples:4 ~key:(Nx.Rng.key 2) o in
  let wide_terms = El.compute model wide ~posterior:wide_posterior o in
  require ~msg:"wider prior, larger KL" Float.(item wide_terms.El.kl > item terms.El.kl);
  (* the parameters carry gradients through the implicit solve *)
  let _, grads =
    Rune.value_and_grad
      structure
      (fun p -> El.loss_fixed model p ~key:(Nx.Rng.key 2) ~n_samples:2 ~iters:8 o)
      params
  in
  require
    ~msg:"the covariance parameters are learned"
    Float.(max_abs grads.M.recognition.Il.time_cov.t > 1e-10)

let test_kl_schedule () =
  let n = 2
  and m = 2
  and t_steps = 3 in
  let model, params = setup ~n ~m ~t_steps ~n_output:2 () in
  let o = lfads_data ~t_steps ~n_output:2 () in
  let sched = Schedule.linear_warmup ~steps:10 dtype in
  check_close
    ~msg:"warm-up starts at zero"
    ~tol:1e-12
    (scalar (Schedule.value sched 0))
    (scalar 0.0);
  check_close
    ~msg:"warm-up midpoint"
    ~tol:1e-12
    (scalar (Schedule.value sched 5))
    (scalar 0.5);
  check_close
    ~msg:"warm-up reaches one"
    ~tol:1e-12
    (scalar (Schedule.value sched 10))
    (scalar 1.0);
  check_close
    ~msg:"warm-up stays at one"
    ~tol:1e-12
    (scalar (Schedule.value sched 99))
    (scalar 1.0);
  let delayed =
    Schedule.linear_warmup ~start:0.1 ~stop:0.9 ~start_step:4 ~steps:2 dtype
  in
  check_close
    ~msg:"delayed warm-up, before"
    ~tol:1e-12
    (scalar (Schedule.value delayed 4))
    (scalar 0.1);
  check_close
    ~msg:"delayed warm-up, middle"
    ~tol:1e-12
    (scalar (Schedule.value delayed 5))
    (scalar 0.5);
  check_close
    ~msg:"delayed warm-up, after"
    ~tol:1e-12
    (scalar (Schedule.value delayed 6))
    (scalar 0.9);
  check_close
    ~msg:"constant schedule"
    ~tol:1e-12
    (scalar (Schedule.value (Schedule.constant 0.3 dtype) 7))
    (scalar 0.3);
  let custom = Schedule.custom (fun step -> Float.of_int step *. 0.25) dtype in
  check_close
    ~msg:"custom schedule"
    ~tol:1e-12
    (scalar (Schedule.value custom 3))
    (scalar 0.75);
  let beta = Schedule.at sched 5 in
  require ~msg:"beta is a scalar tensor" Poly.(Nx.shape beta = [||]);
  (* beta = 0 leaves the log likelihood alone *)
  let posterior = M.infer model params ~iters:6 ~n_samples:3 ~key:(Nx.Rng.key 30) o in
  let zero = El.compute model params ~beta:(Schedule.at sched 0) ~posterior o in
  check_close
    ~msg:"beta 0 gives the log likelihood"
    ~tol:1e-12
    zero.El.elbo
    zero.El.log_likelihood;
  (* the weight is an argument of the compiled step: it follows the schedule
     without a retrace, and the compiled value matches the eager one *)
  let structure = M.structure model in
  let key = Nx.Rng.key 31 in
  let compiled =
    Rune.jit
      Nx.Ptree.(structure @-> tensor @-> Nx.Rng.ptree @-> returns tensor)
      (fun p beta k -> El.loss_fixed model p ~beta ~iters:6 ~key:k ~n_samples:3 o)
  in
  let eager beta =
    Nx.item
      []
      (El.loss_fixed
         model
         params
         ~beta:(Schedule.at sched beta)
         ~iters:6
         ~key
         ~n_samples:3
         o)
  in
  let via_jit step = Nx.item [] (compiled params (Schedule.at sched step) key) in
  check_close
    ~msg:"compiled step at beta = 0"
    ~tol:1e-10
    (scalar (via_jit 0))
    (scalar (eager 0));
  check_close
    ~msg:"compiled step at beta = 1"
    ~tol:1e-10
    (scalar (via_jit 10))
    (scalar (eager 10));
  require
    ~msg:"the weight changes the loss"
    Float.(Float.abs (via_jit 0 -. via_jit 10) > 1e-6)

let test_jit () =
  let n = 2
  and m = 2
  and t_steps = 4 in
  let model, params = setup ~n ~m ~t_steps ~n_output:2 () in
  let o =
    Nx.create dtype [| t_steps; 2 |] [| 1.0; 0.0; 0.5; -0.5; -0.2; 0.7; 0.3; 0.1 |]
  in
  let key = Nx.Rng.key 3 in
  let structure = M.structure model in
  let eager = El.loss_fixed model params ~key ~n_samples:4 ~iters:4 o in
  let jitted =
    Rune.jit
      Nx.Ptree.(structure @-> Nx.Rng.ptree @-> returns Nx.Ptree.tensor)
      (fun p k -> El.loss_fixed model p ~key:k ~n_samples:4 ~iters:4 o)
  in
  check_close ~msg:"jit loss matches eager" ~tol:1e-12 (jitted params key) eager;
  let eager_grad =
    Rune.value_and_grad
      structure
      (fun p -> El.loss_fixed model p ~key ~n_samples:4 ~iters:4 o)
      params
  in
  let jitted_grad =
    Rune.jit
      Nx.Ptree.(structure @-> Nx.Rng.ptree @-> returns (pair tensor structure))
      (fun p k ->
         Rune.value_and_grad
           structure
           (fun p -> El.loss_fixed model p ~key:k ~n_samples:4 ~iters:4 o)
           p)
  in
  let loss', grads' = jitted_grad params key in
  check_close ~msg:"jit grad loss" ~tol:1e-12 loss' (fst eager_grad);
  let grads = snd eager_grad in
  check_close ~msg:"jit grad dynamics" ~tol:1e-10 grads'.M.dynamics.w grads.M.dynamics.w;
  check_close
    ~msg:"jit grad likelihood"
    ~tol:1e-10
    grads'.M.likelihood.c
    grads.M.likelihood.c;
  check_close
    ~msg:"jit grad covariance"
    ~tol:1e-10
    grads'.M.recognition.Il.time_cov.t
    grads.M.recognition.Il.time_cov.t

(* ──────────────────────────────────────────────────────────────────────────
   Swapping recognitions and validation
   ────────────────────────────────────────────────────────────────────────── *)

let test_swapping () =
  let n = 2
  and m = 2
  and t_steps = 3 in
  let keys = Nx.Rng.split ~n:2 (Nx.Rng.key 31) in
  let dynamics = D.Linear.make ~key:keys.(0) ~n ~m dtype in
  let likelihood = L.Gaussian.make ~key:keys.(1) ~n ~n_output:2 dtype in
  let prior = P.Gaussian.make ~m ~spatial_std:1.0 dtype in
  let encoder_prior = P.Gaussian.make ~m ~spatial_std:1.0 dtype in
  let data = lfads_data ~t_steps ~n_output:2 () in
  let make recognition initial_prior =
    M.make
      dtype
      ~n
      ~m
      ~n_steps:t_steps
      ~n_output:2
      ~dynamics
      ~likelihood
      ~prior
      ~encoder_prior
      ~recognition
      ?initial_prior
      ()
  in
  let iilqr = make (Il.make ~n ~m ~n_steps:t_steps ~n_output:2 dtype) None in
  let lfads =
    make
      (Lf.make
         ~key:(Nx.Rng.key 32)
         ~n_ic:4
         ~n_ctrl:3
         ~phi
         ~n
         ~m
         ~n_steps:t_steps
         ~n_output:2
         dtype)
      (Some (I.make ~std:1.0 ~n dtype))
  in
  let loss_of model =
    let params = M.init model in
    El.loss_fixed model params ~key:(Nx.Rng.key 33) ~n_samples:2 ~iters:4 data
  in
  let l_iilqr = loss_of iilqr
  and l_lfads = loss_of lfads in
  require
    ~msg:"both recognitions give finite losses"
    (Float.is_finite (item l_iilqr) && Float.is_finite (item l_lfads));
  (* both recognition parameter blocks receive gradients *)
  let params = M.init iilqr in
  let _, grads =
    Rune.value_and_grad
      (M.structure iilqr)
      (fun p -> El.loss_fixed iilqr p ~key:(Nx.Rng.key 33) ~n_samples:2 ~iters:4 data)
      params
  in
  require
    ~msg:"the iLQR covariance receives gradients"
    Float.(max_abs grads.M.recognition.Il.space_cov.d > 1e-10);
  let params = M.init lfads in
  let _, grads =
    Rune.value_and_grad
      (M.structure lfads)
      (fun p -> El.loss_fixed lfads p ~key:(Nx.Rng.key 33) ~n_samples:2 ~iters:4 data)
      params
  in
  require
    ~msg:"the LFADS encoder receives gradients"
    Float.(max_abs grads.M.recognition.Lf.ic_readout > 1e-10);
  (* mismatched initial-state prior and recognition are rejected *)
  let rejects f =
    match f () with
    | _ -> false
    | exception Invalid_argument _ -> true
  in
  require
    ~msg:"iLQR rejects an initial-state prior"
    (rejects (fun () ->
       make (Il.make ~n ~m ~n_steps:t_steps ~n_output:2 dtype) (Some (I.make ~n dtype))));
  require
    ~msg:"LFADS requires an initial-state prior"
    (rejects (fun () ->
       make
         (Lf.make
            ~key:(Nx.Rng.key 32)
            ~n_ic:4
            ~n_ctrl:3
            ~phi
            ~n
            ~m
            ~n_steps:t_steps
            ~n_output:2
            dtype)
         None))

(* ──────────────────────────────────────────────────────────────────────────
   End to end
   ────────────────────────────────────────────────────────────────────────── *)

let test_training_smoke () =
  let n = 2
  and m = 2
  and t_steps = 6 in
  let model, params = setup ~n ~m ~t_steps ~n_output:2 () in
  (* synthetic trials from the model itself *)
  let true_params = params in
  let trials =
    Array.init 4 ~f:(fun i ->
      let key = Nx.Rng.key (100 + i) in
      let _, _, _, o = Ilqr_vae.sample_generative model true_params ~key in
      o)
  in
  let structure = M.structure model in
  let key = Nx.Rng.key 7 in
  let objective p = El.batch_loss_fixed model p ~key ~n_samples:4 ~iters:4 trials in
  let initial = item (objective params) in
  let lr = 0.02 in
  let params = ref params in
  for _ = 1 to 40 do
    let _, grads = Rune.value_and_grad structure objective !params in
    let step : 'a 'b. Nx.Ptree.Path.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t =
      fun _ p g -> Nx.sub p (Nx.mul (Nx.cast (Nx.dtype p) (Nx.scalar Nx.float64 lr)) g)
    in
    params := Nx.Ptree.map2 structure step !params grads
  done;
  let final = item (objective !params) in
  Stdio.printf "training smoke: %.6g -> %.6g\n%!" initial final;
  require ~msg:"training decreases the loss" Float.(final < initial);
  (* generative and predictive shapes *)
  let x0, u, z, o = Ilqr_vae.sample_generative model !params ~key:(Nx.Rng.key 9) in
  require ~msg:"generative x0 shape" Poly.(Nx.shape x0 = [| n |]);
  require ~msg:"generative u shape" Poly.(Nx.shape u = [| t_steps; m |]);
  require ~msg:"generative z shape" Poly.(Nx.shape z = [| t_steps + 1; n |]);
  require ~msg:"generative o shape" Poly.(Nx.shape o = [| t_steps; 2 |]);
  let posterior =
    M.infer model !params ~iters:4 ~n_samples:5 ~key:(Nx.Rng.key 10) (Array.get trials 0)
  in
  let u, z, o = Ilqr_vae.predictions model !params posterior in
  require ~msg:"predictions u shape" Poly.(Nx.shape u = [| t_steps; 5; m |]);
  require ~msg:"predictions z shape" Poly.(Nx.shape z = [| t_steps + 1; 5; n |]);
  require ~msg:"predictions o shape" Poly.(Nx.shape o = [| t_steps; 5; 2 |]);
  let mean = M.mean model !params (Array.get trials 0) in
  let z, o = Ilqr_vae.predictions_deterministic model !params mean in
  require ~msg:"deterministic z shape" Poly.(Nx.shape z = [| t_steps + 1; 1; n |]);
  require ~msg:"deterministic o shape" Poly.(Nx.shape o = [| t_steps; 1; 2 |])

let test_lfads_training_smoke () =
  let n = 2
  and m = 2
  and t_steps = 5 in
  let model, params = setup_lfads ~n ~m ~t_steps ~n_output:2 () in
  let true_params = params in
  let trials =
    Array.init 3 ~f:(fun i ->
      let key = Nx.Rng.key (200 + i) in
      let _, _, _, o = Ilqr_vae.sample_generative model true_params ~key in
      o)
  in
  let structure = M.structure model in
  let key = Nx.Rng.key 17 in
  let objective p = El.batch_loss_fixed model p ~key ~n_samples:3 ~iters:4 trials in
  let initial = item (objective params) in
  let lr = 0.002 in
  let params = ref params in
  for _ = 1 to 25 do
    let _, grads = Rune.value_and_grad structure objective !params in
    let step : 'a 'b. Nx.Ptree.Path.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t =
      fun _ p g -> Nx.sub p (Nx.mul (Nx.cast (Nx.dtype p) (Nx.scalar Nx.float64 lr)) g)
    in
    params := Nx.Ptree.map2 structure step !params grads
  done;
  let final = item (objective !params) in
  Stdio.printf "lfads training smoke: %.6g -> %.6g\n%!" initial final;
  require ~msg:"lfads training decreases the loss" Float.(final < initial)

(* ────────────────────────────────────────────────────────────────────────── *)

let () =
  Stdlib.exit
    (run
       "ilqr_vae"
       [ test "covariance factors" test_covariance
       ; test "initial-state prior" test_initial_prior
       ; test "iLQR sampling covariance" test_recognition_sampling
       ; test "iLQR log density" test_recognition_log_prob
       ; test "prior densities" test_prior_densities
       ; test "gaussian KL" test_kl
       ; test "likelihoods" test_likelihoods
       ; test "dynamics" test_dynamics
       ; test "mgu dynamics" test_mgu
       ; test "encoder stationarity" test_encoder_stationarity
       ; test "lfads shapes and density" test_lfads_shapes_and_density
       ; test "lfads encoder" test_lfads_encoder
       ; test "lfads controller" test_lfads_controller
       ; test "lfads sticking the landing" test_lfads_sticking_the_landing
       ; test "lfads gradients" test_lfads_gradients
       ; test "lfads jit" test_lfads_jit
       ; test "elbo gradient check" test_elbo_gradient
       ; test "elbo terms" test_elbo_terms
       ; test "kl schedule" test_kl_schedule
       ; test "jit round-trips" test_jit
       ; test "recognition swapping" test_swapping
       ; test "training smoke test" test_training_smoke
       ; test "lfads training smoke test" test_lfads_training_smoke
       ])
