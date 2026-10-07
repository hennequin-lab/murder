(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* iLQR-VAE test suite:
   - covariance factors, log-determinants and the Kronecker recognition
     covariance (sampling covariance, log density and entropy against dense
     computations),
   - the prior log densities (normalization by numerical integration and
     sampling moments),
   - the likelihoods (log densities against hand-computed values),
   - the dynamics (hand-computed steps, batched rollouts),
   - the encoder: iLQR stationarity on an LQR problem,
   - the ELBO: gradient check of the whole inference path (implicit gradient
     included) against finite differences,
   - JIT round-trips of the fixed solve and of its gradient,
   - an end-to-end training smoke test. *)

open Base
open Windtrap
open Stdio
module Nx = Nx
module C = Ilqr_vae.Covariance
module P = Ilqr_vae.Prior
module L = Ilqr_vae.Likelihood
module D = Ilqr_vae.Dynamics
module M = Ilqr_vae.Model
module R = Ilqr_vae.Recognition
module E = Ilqr_vae.Encoder
module El = Ilqr_vae.Elbo

let dtype = Nx.float64
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

(* Concrete dtypes for the test model: linear dynamics, Gaussian likelihood
   and Gaussian priors. *)
type dyn_params = Nx.float64_t D.Linear.params
type lik_params = Nx.float64_t L.Gaussian.params
type poisson_params = Nx.float64_t L.Poisson.params
type pri_params = Nx.float64_t P.Gaussian.params
type model = (Nx.float64_elt, dyn_params, lik_params, pri_params, pri_params) M.t
type params = (Nx.float64_elt, dyn_params, lik_params, pri_params, pri_params) M.params

let setup ?(key = Nx.Rng.key 0) ~n ~m ~t_steps ~n_output () : model * params =
  let dynamics = D.Linear.make ~key ~n ~m dtype in
  let likelihood = L.Gaussian.make ~key ~n ~n_output dtype in
  let prior = P.Gaussian.make ~m ~spatial_std:1.0 dtype in
  let encoder_prior = P.Gaussian.make ~m ~spatial_std:1.0 dtype in
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
      ()
  in
  model, M.init model

let data_of ~n_output ~t_steps values =
  Nx.create dtype [| t_steps; n_output |] (Array.of_list values)

(* Central finite differences of a scalar function of one tensor. *)
let fd_grad ~eps f (x : Nx.float64_t) : Nx.float64_t =
  let n = (Nx.shape x).(0) in
  let out = Array.create ~len:n 0.0 in
  for i = 0 to n - 1 do
    let xi = Nx.item [ i ] x in
    let xp = Nx.set [ I i ] (Nx.scalar dtype (xi +. eps)) x in
    let xm = Nx.set [ I i ] (Nx.scalar dtype (xi -. eps)) x in
    out.(i) <- (f xp -. f xm) /. (2.0 *. eps)
  done;
  Nx.create dtype [| n |] out

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
   Recognition
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
  let params = { params with M.space_cov; time_cov } in
  let n_samples = 40_000 in
  let mu = Nx.zeros dtype [| t_steps; m |] in
  let u = R.sample model params ~mu ~n_samples ~key:(Nx.Rng.key 17) in
  require ~msg:"sample shape" Poly.(Nx.shape u = [| t_steps; n_samples; m |]);
  (* empirical covariance over samples, laid out like the dense one *)
  let u2 =
    Nx.reshape [| t_steps; m; n_samples |] (Nx.transpose ~axes:[ 0; 2; 1 ] u)
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
  let params = { params with M.space_cov; time_cov } in
  let mu = vec [ 0.1; -0.2; 0.3; 0.0; -0.4; 0.2 ] |> Nx.reshape [| 3; 2 |] in
  let u = Nx.create dtype [| 3; 2 |] [| 0.0; 0.1; -0.2; 0.3; 0.5; -0.1 |] in
  let log_q = R.log_prob model params ~mu u in
  (* dense Gaussian log density, same covariance *)
  let dense = dense_covariance ~t_steps ~m time_cov space_cov in
  let diff = Nx.reshape [| t_steps * m |] (Nx.sub u mu) in
  let factor = Nx.cholesky dense in
  let solved = Nx.solve factor diff in
  let quadratic = item (Nx.sum (Nx.mul solved solved)) in
  let log_det = item (Nx.mul_s (Nx.sum (Nx.log (Nx.diag factor))) 2.0) in
  let dim = Float.of_int (t_steps * m) in
  let expected = -0.5 *. (quadratic +. log_det +. (dim *. Float.log (2.0 *. Float.pi))) in
  check_close ~msg:"log q" ~tol:1e-9 log_q (scalar expected);
  let entropy = R.entropy model params in
  let expected_entropy =
    0.5 *. (log_det +. (dim *. (1.0 +. Float.log (2.0 *. Float.pi))))
  in
  check_close ~msg:"entropy" ~tol:1e-9 entropy (scalar expected_entropy);
  (* a batch of two sequences gives one value per sequence *)
  let u_batch =
    Nx.concatenate
      ~axis:1
      [ Nx.unsqueeze ~axes:[ 1 ] u; Nx.unsqueeze ~axes:[ 1 ] (Nx.add_s u 0.1) ]
  in
  let batch = R.log_prob model params ~mu u_batch in
  require ~msg:"batched log q shape" Poly.(Nx.shape batch = [| 2 |]);
  check_close
    ~msg:"batched log q"
    ~tol:1e-9
    batch
    (Nx.stack
       ~axis:0
       [ log_q; R.log_prob model params ~mu:(Nx.add_s mu 0.0) (Nx.add_s u 0.1) ])

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
  let rate j xj g = 0.1 *. g *. (Float.exp xj +. 1e-3) in
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

(* ──────────────────────────────────────────────────────────────────────────
   Encoder
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
  let params : params =
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
  let mu = E.posterior_mean model params ~max_iter:100 ~conv_threshold:1e-14 o in
  require ~msg:"posterior mean shape" Poly.(Nx.shape mu = [| 5; 2 |]);
  let a = D.Linear.matrix params.M.dynamics in
  let grad = Rune.grad' (fun u -> lqr_cost ~a ~o ~t_steps:5 u) mu in
  let max_grad = item (Nx.max (Nx.abs grad)) in
  if not Float.(max_grad < 1e-6)
  then Stdio.printf "encoder stationarity: max |grad| = %.3g\n%!" max_grad;
  require ~msg:"iLQR solves the LQR problem" Float.(max_grad < 1e-6);
  (* the fixed solver finds the same point *)
  let mu_fixed = E.posterior_mean_fixed model params ~iters:60 o in
  check_close ~msg:"fixed encoder matches adaptive" ~tol:1e-6 mu_fixed mu

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
  (* the ELBO alone, at a fixed mean *)
  let mu_u = E.posterior_mean model params o in
  (match
     Rune.check_grads
       structure
       (fun p -> El.loss model p ~mu_u ~key:(Nx.Rng.key 1) ~n_samples:2 o)
       params
   with
   | Ok () -> ()
   | Error msg -> failwith ("ELBO gradient check failed: " ^ msg));
  (* the whole objective: the gradient of the encoder is implicit *)
  match
    Rune.check_grads
      structure
      (fun p -> El.loss_fixed model p ~key:(Nx.Rng.key 1) ~n_samples:2 ~iters:12 o)
      params
  with
  | Ok () -> ()
  | Error msg -> failwith ("implicit-gradient check failed: " ^ msg)

let test_elbo_terms () =
  let n = 2
  and m = 2
  and t_steps = 4 in
  let model, params = setup ~n ~m ~t_steps ~n_output:2 () in
  let o =
    Nx.create dtype [| t_steps; 2 |] [| 1.0; 0.0; 0.5; -0.5; -0.2; 0.7; 0.3; 0.1 |]
  in
  let mu_u = E.posterior_mean model params o in
  let terms = El.compute model params ~mu_u ~key:(Nx.Rng.key 2) ~n_samples:4 o in
  check_close
    ~msg:"elbo = ll − beta kl"
    ~tol:1e-10
    terms.El.elbo
    (Nx.sub terms.El.log_likelihood terms.El.kl);
  (* the same samples give the same numbers on a second key-dependent call *)
  let loss = El.loss model params ~mu_u ~key:(Nx.Rng.key 2) ~n_samples:4 o in
  check_close ~msg:"loss is the negated elbo" ~tol:1e-12 loss (Nx.neg terms.El.elbo);
  (* the prior enters the KL: a wider prior has a larger KL *)
  let wide =
    { params with
      M.prior =
        { params.M.prior with
          log_spatial_stds = Nx.add_s params.M.prior.log_spatial_stds 1.0
        }
    }
  in
  let wide_kl = (El.compute model wide ~mu_u ~key:(Nx.Rng.key 2) ~n_samples:4 o).El.kl in
  require ~msg:"wider prior, larger KL" Float.(item wide_kl > item terms.El.kl)

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
  check_close ~msg:"jit grad covariance" ~tol:1e-10 grads'.M.time_cov.t grads.M.time_cov.t

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
      let _, _, o = Ilqr_vae.sample_generative model true_params ~key in
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
  let u, z, o = Ilqr_vae.sample_generative model !params ~key:(Nx.Rng.key 9) in
  require ~msg:"generative u shape" Poly.(Nx.shape u = [| t_steps; m |]);
  require ~msg:"generative z shape" Poly.(Nx.shape z = [| t_steps + 1; n |]);
  require ~msg:"generative o shape" Poly.(Nx.shape o = [| t_steps; 2 |]);
  let mu = E.posterior_mean model !params (Array.get trials 0) in
  let u, z, o =
    Ilqr_vae.predictions model !params ~mu ~n_samples:5 ~key:(Nx.Rng.key 10)
  in
  require ~msg:"predictions u shape" Poly.(Nx.shape u = [| t_steps; 5; m |]);
  require ~msg:"predictions z shape" Poly.(Nx.shape z = [| t_steps + 1; 5; n |]);
  require ~msg:"predictions o shape" Poly.(Nx.shape o = [| t_steps; 5; 2 |]);
  let z, o = Ilqr_vae.predictions_deterministic model !params ~mu in
  require ~msg:"deterministic z shape" Poly.(Nx.shape z = [| t_steps + 1; n |]);
  require ~msg:"deterministic o shape" Poly.(Nx.shape o = [| t_steps; 2 |])

(* ────────────────────────────────────────────────────────────────────────── *)

let () =
  Stdlib.exit
    (run
       "ilqr_vae"
       [ test "covariance factors" test_covariance
       ; test "recognition sampling covariance" test_recognition_sampling
       ; test "recognition log density and entropy" test_recognition_log_prob
       ; test "prior densities" test_prior_densities
       ; test "gaussian KL" test_kl
       ; test "likelihoods" test_likelihoods
       ; test "dynamics" test_dynamics
       ; test "encoder stationarity" test_encoder_stationarity
       ; test "elbo gradient check" test_elbo_gradient
       ; test "elbo terms" test_elbo_terms
       ; test "jit round-trips" test_jit
       ; test "training smoke test" test_training_smoke
       ])
