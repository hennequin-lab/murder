(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* iLQR-VAE test suite:
   - covariance factors and log-determinants, the Gaussian density in each of
     its covariance forms, and the Kronecker recognition covariance (sampling
     covariance and log density against dense computations),
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
   - tensor mini-batches: the recognition's batch shapes, the batched bound
     against the per-trial mean, and a compiled step that takes the batch as
     an argument,
   - swapping recognitions, model validation, and end-to-end training smoke
     tests. *)

open Base
open Nx
open Windtrap
module Nx = Nx
module C = Ilqr_vae.Covariance
module Density = Ilqr_vae.Density
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

let dtype = float64
let phi = tanh

(* windtrap's tolerant float witness: [close tol] compares within [tol],
   relative or absolute, and reports the largest error it saw. *)
let close tol = float_rel ~rel:tol ~abs:tol

(* The iLQR recognition's inner solver, as the tests ask for it: a fixed
   number of iterations, which is what makes the whole path traceable. *)
let fixed ?conv_threshold iters = R.fixed ~iters ?conv_threshold ()

let check_close ?msg ~tol got expected =
  equal ?msg (array (close tol)) (to_array expected) (to_array got)

let check_shape ?msg expected t = equal ?msg (array int) expected (Nx.shape t)

(* windtrap has no finiteness verb: [fail] names the value. *)
let check_finite ?msg x =
  if not (Float.is_finite x)
  then
    fail
      (Printf.sprintf
         "%s: %s is not finite"
         (Option.value msg ~default:"value")
         (Float.to_string x))

let vec values = create dtype [| List.length values |] (Array.of_list values)
let scalar v = Nx.scalar dtype v
let item t = Nx.item [] t
let max_abs t = item (max (abs t))

(* ──────────────────────────────────────────────────────────────────────────
   Concrete dtypes for the test models.
   ────────────────────────────────────────────────────────────────────────── *)

type linear_params = (Nx.host Nx.float64_t, Nx.host) D.Linear.params
type mgu_params = (Nx.host Nx.float64_t, Nx.host) D.Mgu.params
type lik_params = (Nx.host Nx.float64_t, Nx.host) L.Gaussian.params
type poisson_params = (Nx.host Nx.float64_t, Nx.host) L.Poisson.params
type pri_params = (Nx.host Nx.float64_t, Nx.host) P.Gaussian.params
type ilqr_rec = (Nx.host Nx.float64_t, Nx.host) Il.params
type lfads_rec = (Nx.host Nx.float64_t, Nx.host) Lf.params

type iilqr_model =
  ( Nx.float64_elt
    , linear_params
    , lik_params
    , pri_params
    , pri_params
    , ilqr_rec
    , Nx.host )
    M.t

type iilqr_params =
  ( Nx.float64_elt
    , linear_params
    , lik_params
    , pri_params
    , pri_params
    , ilqr_rec
    , Nx.host )
    M.params

type lfads_model =
  (Nx.float64_elt, mgu_params, lik_params, pri_params, pri_params, lfads_rec, Nx.host) M.t

type lfads_params =
  ( Nx.float64_elt
    , mgu_params
    , lik_params
    , pri_params
    , pri_params
    , lfads_rec
    , Nx.host )
    M.params

let setup ?(key = Nx.Rng.key 0) ~n ~m ~t_steps ~n_output () : iilqr_model * iilqr_params =
  let at = Nx.Host.on in
  let dynamics = D.Linear.make ~at ~n ~m () in
  let likelihood = L.Gaussian.make ~at:Nx.Host.on ~n ~n_output () in
  let prior = P.Gaussian.make ~at:Nx.Host.on ~m ~spatial_std:1.0 () in
  let encoder_prior = P.Gaussian.make ~at:Nx.Host.on ~m ~spatial_std:1.0 () in
  let recognition = Il.make ~at:Nx.Host.on ~n ~m ~n_steps:t_steps ~n_output () in
  let model = M.make dtype ~dynamics ~likelihood ~prior ~encoder_prior ~recognition () in
  model, M.init ~at ~key model

(* The same model with a free initial state: the encoder prices it with the
   initial-state prior, and the recognition carries the covariance scale. *)
let setup_initial ?(key = Nx.Rng.key 2) ~n ~m ~t_steps ~n_output ()
  : iilqr_model * iilqr_params
  =
  let at = Nx.Host.on in
  let dynamics = D.Linear.make ~at ~n ~m () in
  let likelihood = L.Gaussian.make ~at:Nx.Host.on ~n ~n_output () in
  let prior = P.Gaussian.make ~at:Nx.Host.on ~m ~spatial_std:1.0 () in
  let encoder_prior = P.Gaussian.make ~at:Nx.Host.on ~m ~spatial_std:1.0 () in
  let initial_prior = I.make ~mean:0.5 ~std:1.5 ~n () in
  let recognition =
    Il.make ~at:Nx.Host.on ~infer_initial:true ~n ~m ~n_steps:t_steps ~n_output ()
  in
  let model =
    M.make
      dtype
      ~dynamics
      ~likelihood
      ~prior
      ~encoder_prior
      ~recognition
      ~initial_prior
      ()
  in
  model, M.init ~at ~key model

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
  let at = Nx.Host.on in
  let dynamics = D.Mgu.make ~at ~n ~m ~phi () in
  let likelihood = L.Gaussian.make ~at:Nx.Host.on ~n ~n_output () in
  let prior = P.Gaussian.make ~at:Nx.Host.on ~m ~spatial_std:1.0 () in
  let encoder_prior = P.Gaussian.make ~at:Nx.Host.on ~m ~spatial_std:1.0 () in
  let initial_prior = I.make ~std:1.0 ~n () in
  let recognition =
    Lf.make
      ~at:Nx.Host.on
      ~n_ic
      ~n_ci
      ~n_ctrl
      ~with_inputs
      ~phi
      ~n
      ~m
      ~n_steps:t_steps
      ~n_output
      ()
  in
  let model =
    M.make
      dtype
      ~dynamics
      ~likelihood
      ~prior
      ~encoder_prior
      ~recognition
      ~initial_prior
      ()
  in
  model, M.init ~at ~key model

let data_of ~n_output ~t_steps values =
  create dtype [| t_steps; n_output |] (Array.of_list values)

(* A hand-rolled MGU rollout: the reference for the encoder tests. *)
let run_mgu ~phi (p : mgu_params) ~init inputs =
  let t_steps = (shape inputs).(0) in
  let x = ref init in
  for k = 0 to t_steps - 1 do
    x := D.Mgu.dyn ~phi p ~k:(scalar (Float.of_int k)) ~x:!x ~u:(get [ k ] inputs)
  done;
  !x

(* ──────────────────────────────────────────────────────────────────────────
   Covariance
   ────────────────────────────────────────────────────────────────────────── *)

(* The four forms of a covariance are one density: each is checked against a
   dense computation, and the two factor forms are checked against their own
   factor, so a wrong triangular convention would show. *)
let test_density () =
  let d = 3 in
  let mean = vec [ 0.2; -0.4; 0.1 ] in
  let x = create dtype [| 2; 3 |] [| 0.1; 0.2; 0.3; -0.5; 0.0; 0.7 |] in
  let identity = eye dtype d in
  let inverse sigma = solve sigma identity in
  (* the dense log density of [N (mean, sigma)] at every row of [x] *)
  let reference sigma =
    let diff = reshape [| 2; d |] Infix.(x - reshape [| 1; d |] mean) in
    let quadratic =
      sum ~axes:[ -1 ] Infix.(diff * (diff *@ matrix_transpose (inverse sigma)))
    in
    let log_det = item Infix.(sum (log (diag (cholesky sigma))) *$ 2.0) in
    let dim = Float.of_int d in
    let normalizer = 0.5 *. (log_det +. (dim *. Float.log (2.0 *. Float.pi))) in
    Infix.((quadratic *$ -0.5) - scalar normalizer)
  in
  let sigma =
    create dtype [| d; d |] [| 1.2; 0.3; -0.2; 0.3; 0.8; 0.15; -0.2; 0.15; 1.5 |]
  in
  check_close
    ~msg:"cholesky of the covariance"
    ~tol:1e-10
    (Density.logp ~mean ~covariance:(Density.Cholesky { factor = cholesky sigma }) x)
    (reference sigma);
  check_close
    ~msg:"cholesky of the precision"
    ~tol:1e-10
    (Density.logp
       ~mean
       ~covariance:
         (Density.Precision_cholesky { factor = cholesky ~triangle:Upper (inverse sigma) })
       x)
    (reference sigma);
  (* a scaled precision is a scaled factor: the Laplace block's form *)
  let log_scale = 0.3 in
  let scaled = Infix.(sigma *$ Float.exp (2.0 *. log_scale)) in
  check_close
    ~msg:"scaled precision factor"
    ~tol:1e-10
    (Density.logp
       ~mean
       ~covariance:
         (Density.Precision_cholesky
            { factor =
                Infix.(
                  cholesky ~triangle:Upper (inverse sigma) *$ Float.exp (-.log_scale))
            })
       x)
    (reference scaled);
  let variances = vec [ 0.5; 2.0; 0.25 ] in
  check_close
    ~msg:"diagonal"
    ~tol:1e-12
    (Density.logp
       ~mean
       ~covariance:(Density.Diagonal { log_stds = Infix.(log variances *$ 0.5) })
       x)
    (reference (diag variances));
  (* one scale per row: the shape a per-step prior has *)
  let per_row = create dtype [| 2; 3 |] [| 0.0; 0.0; 0.0; 0.1; -0.2; 0.0 |] in
  let rows = Density.logp ~mean ~covariance:(Density.Diagonal { log_stds = per_row }) x in
  check_shape ~msg:"per-row diagonal shape" [| 2 |] rows;
  for i = 0 to 1 do
    check_close
      ~msg:"per-row diagonal"
      ~tol:1e-12
      (get [ i ] rows)
      (get [ i ] (reference (diag Infix.(exp (get [ i ] per_row *$ 2.0)))))
  done;
  let variance = 0.7 in
  check_close
    ~msg:"isotropic"
    ~tol:1e-12
    (Density.logp ~mean ~covariance:(Density.Isotropic (scalar variance)) x)
    (reference Infix.(identity *$ variance))

let test_covariance () =
  let n = 3 in
  let p = C.make ~at:Nx.Host.on ~sigma2:2.0 ~n dtype in
  let l = C.factor p in
  let arr = to_array l in
  for i = 0 to n - 1 do
    for j = 0 to i - 1 do
      let index = (i * n) + j in
      equal ~msg:"factor is upper triangular" float_exact 0.0 arr.(index)
    done
  done;
  let sigma = Infix.(l *@ matrix_transpose l) in
  check_close ~msg:"sigma diagonal" ~tol:1e-12 (diag sigma) (vec [ 2.0; 2.0; 2.0 ]);
  check_close
    ~msg:"log det"
    ~tol:1e-12
    (C.log_det p)
    (scalar (Float.of_int n *. Float.log 2.0));
  (* a non-diagonal factor: check Σ = L Lᵀ elementwise and its log det *)
  let p =
    { C.d = vec [ 0.5; 2.0 ]; t = create dtype [| 2; 2 |] [| 0.0; 0.3; 0.0; 0.0 |] }
  in
  let l = C.factor p in
  let sigma = Infix.(l *@ matrix_transpose l) in
  check_close
    ~msg:"sigma from factor"
    ~tol:1e-12
    (reshape [| 4 |] sigma)
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
  let prior = I.make ~mean:0.5 ~std:2.0 ~n () in
  let p = I.init ~at:Nx.Host.on dtype prior in
  let density x =
    let sq = (x -. 0.5) *. (x -. 0.5) in
    -0.5 *. ((sq /. 4.0) +. (2.0 *. Float.log 2.0) +. Float.log (2.0 *. Float.pi))
  in
  check_close
    ~msg:"initial log density, one state"
    ~tol:1e-12
    (I.logp p (vec [ 1.0; -0.5 ]))
    (scalar (density 1.0 +. density (-0.5)));
  let batch = create dtype [| 2; 2 |] [| 1.0; -0.5; 0.0; 0.5 |] in
  check_close
    ~msg:"initial log density, batched"
    ~tol:1e-12
    (I.logp p batch)
    (vec [ density 1.0 +. density (-0.5); density 0.0 +. density 0.5 ]);
  (* the sampler reproduces the mean and standard deviation *)
  let n_samples = 40_000 in
  let s = I.sample p ~key:(Nx.Rng.key 3) ~n_samples in
  check_shape ~msg:"initial sample shape" [| n_samples; n |] s;
  let mean = Nx.mean ~axes:[ 0 ] s in
  check_close ~msg:"initial sample mean" ~tol:0.05 mean (vec [ 0.5; 0.5 ]);
  let sd =
    let open Infix in
    let centered = s - reshape [| 1; n |] (Nx.mean ~axes:[ 0 ] s) in
    sqrt (Nx.mean ~axes:[ 0 ] (centered * centered))
  in
  check_close ~msg:"initial sample sd" ~tol:0.05 sd (vec [ 2.0; 2.0 ])

(* ──────────────────────────────────────────────────────────────────────────
   iLQR recognition
   ────────────────────────────────────────────────────────────────────────── *)

let dense_covariance ~t_steps ~m time_params space_params =
  let lt = C.factor time_params
  and ls = C.factor space_params in
  let st = to_array Infix.(lt *@ matrix_transpose lt)
  and ss = to_array Infix.(ls *@ matrix_transpose ls) in
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
  create dtype [| size; size |] out

let test_recognition_sampling () =
  let t_steps = 4
  and m = 2 in
  let model, params = setup ~n:2 ~m ~t_steps ~n_output:2 () in
  let time_cov =
    { C.d = vec [ 0.5; 1.0; 1.5; 2.0 ]
    ; t =
        create
          dtype
          [| 4; 4 |]
          [| 0.; 0.4; 0.; 0.; 0.; 0.; 0.3; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
    }
  in
  let space_cov =
    { C.d = vec [ 0.7; 1.3 ]; t = create dtype [| 2; 2 |] [| 0.; -0.2; 0.; 0. |] }
  in
  let params =
    { params with
      M.recognition =
        ({ Il.space_cov; time_cov; initial_log_scale = None }
         : (Nx.host Nx.float64_t, Nx.host) Il.params)
    }
  in
  let n_samples = 40_000 in
  let data = zeros dtype [| t_steps; 2 |] in
  let posterior =
    M.infer model params ~solve:(fixed 4) ~n_samples ~key:(Nx.Rng.key 17) data
  in
  check_shape ~msg:"posterior u shape" [| t_steps; n_samples; m |] posterior.R.u;
  (* empirical covariance over samples, laid out like the dense one *)
  let u2 =
    reshape [| t_steps; m; n_samples |] (transpose ~axes:[ 0; 2; 1 ] posterior.R.u)
    |> reshape [| t_steps * m; n_samples |]
  in
  let scale = 1.0 /. Float.of_int (n_samples - 1) in
  let empirical = Infix.(u2 *@ matrix_transpose u2 *$ scale) in
  let dense = dense_covariance ~t_steps ~m time_cov space_cov in
  (* The second moment of [n] draws of a zero-mean Gaussian of covariance [Σ]
     estimates each entry [Σ_ij] with standard error
     [sqrt ((Σ_ii Σ_jj + Σ_ij²) / n)]: every entry lies within four of its
     standard errors. *)
  let variances = diagonal dense in
  let stderr =
    Infix.(
      sqrt
        (((reshape [| -1; 1 |] variances * reshape [| 1; -1 |] variances) + (dense * dense)
         )
         /$ Float.of_int n_samples))
  in
  let z = Infix.((empirical - dense) / stderr) in
  less ~msg:"sampling covariance, in standard errors" float_exact ~than:4.0 (max_abs z)

let test_recognition_log_prob () =
  let t_steps = 3
  and m = 2 in
  let model, params = setup ~n:2 ~m ~t_steps ~n_output:2 () in
  let time_cov =
    { C.d = vec [ 1.0; 0.8; 1.2 ]
    ; t = create dtype [| 3; 3 |] [| 0.; 0.2; 0.3; 0.; 0.; -0.1; 0.; 0.; 0. |]
    }
  in
  let space_cov =
    { C.d = vec [ 1.1; 0.9 ]; t = create dtype [| 2; 2 |] [| 0.; 0.4; 0.; 0. |] }
  in
  let params =
    { params with
      M.recognition =
        ({ Il.space_cov; time_cov; initial_log_scale = None }
         : (Nx.host Nx.float64_t, Nx.host) Il.params)
    }
  in
  let data = create dtype [| t_steps; 2 |] [| 0.1; -0.2; 0.3; 0.0; -0.4; 0.2 |] in
  let mu = (M.mean model params ~solve:(fixed 60) data).R.u in
  let u = create dtype [| 3; 2 |] [| 0.0; 0.1; -0.2; 0.3; 0.5; -0.1 |] in
  let one = unsqueeze ~axes:[ 1 ] u in
  let log_q =
    M.log_prob
      model
      params
      ~solve:(fixed 60)
      data
      { R.x0 = zeros dtype [| 1; 2 |]; u = one }
  in
  (* dense Gaussian log density, same covariance and mean *)
  let mu2 = reshape [| t_steps; m |] mu in
  let dense = dense_covariance ~t_steps ~m time_cov space_cov in
  let diff = reshape [| t_steps * m |] Infix.(u - mu2) in
  let factor = cholesky dense in
  let solved = solve factor diff in
  let quadratic = item (sum Infix.(solved * solved)) in
  let log_det = item Infix.(sum (log (diag factor)) *$ 2.0) in
  let dim = Float.of_int (t_steps * m) in
  let expected = -0.5 *. (quadratic +. log_det +. (dim *. Float.log (2.0 *. Float.pi))) in
  check_close ~msg:"log q" ~tol:1e-9 log_q (vec [ expected ]);
  (* a batch of two sequences gives one value per sequence *)
  let u_batch =
    concatenate
      ~axis:1
      [ unsqueeze ~axes:[ 1 ] u; unsqueeze ~axes:[ 1 ] Infix.(u +$ 0.1) ]
  in
  let batch =
    M.log_prob
      model
      params
      ~solve:(fixed 60)
      data
      { R.x0 = zeros dtype [| 2; 2 |]; u = u_batch }
  in
  check_shape ~msg:"batched log q shape" [| 2 |] batch;
  check_close
    ~msg:"batched log q"
    ~tol:1e-9
    batch
    (stack
       ~axis:0
       [ log_q
       ; M.log_prob
           model
           params
           ~solve:(fixed 60)
           data
           { R.x0 = zeros dtype [| 1; 2 |]; u = unsqueeze ~axes:[ 1 ] Infix.(u +$ 0.1) }
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
  let gaussian = P.Gaussian.make ~at:Nx.Host.on ~spatial_std:1.5 ~first_bin:0.25 ~m () in
  let gaussian_params = P.init dtype gaussian in
  let student =
    P.Student.make ~at:Nx.Host.on ~spatial_std:1.5 ~nu:7.0 ~first_std:0.5 ~m ()
  in
  let student_params = P.init dtype student in
  let laplacian = P.Laplacian.make ~at:Nx.Host.on ~spatial_std:1.5 ~first_std:0.5 ~m () in
  let laplacian_params = P.init dtype laplacian in
  let mass (prior, params) k =
    integrate_density
      (fun u -> P.logp_t prior params ~k:(scalar (Float.of_int k)) ~u)
      (-25.0)
      25.0
      20_000
  in
  check_close
    ~msg:"gaussian density k=0"
    ~tol:1e-4
    (scalar (mass (gaussian, gaussian_params) 0))
    (scalar 1.0);
  check_close
    ~msg:"gaussian density k=1"
    ~tol:1e-4
    (scalar (mass (gaussian, gaussian_params) 1))
    (scalar 1.0);
  check_close
    ~msg:"student density k=0"
    ~tol:1e-4
    (scalar (mass (student, student_params) 0))
    (scalar 1.0);
  check_close
    ~msg:"student density k=1"
    ~tol:1e-3
    (scalar (mass (student, student_params) 1))
    (scalar 1.0);
  check_close
    ~msg:"laplacian density k=0"
    ~tol:1e-4
    (scalar (mass (laplacian, laplacian_params) 0))
    (scalar 1.0);
  check_close
    ~msg:"laplacian density k=1"
    ~tol:1e-4
    (scalar (mass (laplacian, laplacian_params) 1))
    (scalar 1.0);
  (* the Student's scale makes its variance spatial_std²; the Laplacian's is
     2 spatial_std² *)
  let sample_var (prior, params) =
    let n_steps = 2_000 in
    let n_keys = 64 in
    let total = ref 0.0 in
    for i = 1 to n_keys do
      let u = P.sample prior params ~key:(Nx.Rng.key i) ~n_steps in
      let x = slice [ R (1, n_steps) ] u |> reshape [| n_steps - 1 |] in
      let scale = 1.0 /. Float.of_int (n_steps - 1) in
      total := !total +. item Infix.(sum (x * x) *$ scale)
    done;
    !total /. Float.of_int n_keys
  in
  check_close
    ~msg:"student sample variance"
    ~tol:0.6
    (scalar (sample_var (student, student_params)))
    (scalar (1.5 *. 1.5));
  check_close
    ~msg:"laplacian sample variance"
    ~tol:0.05
    (scalar (sample_var (laplacian, laplacian_params)))
    (scalar (2.0 *. 1.5 *. 1.5))

(* The Gaussian prior's closed-form KL, against a dense computation of
   [½ (tr (Σ_p⁻¹ Σ_q) + muᵀ Σ_p⁻¹ mu + log det Σ_p − log det Σ_q − d)] over the
   full [T·m] covariance. *)
let test_kl () =
  let t_steps = 3
  and m = 2 in
  let prior = P.Gaussian.make ~at:Nx.Host.on ~spatial_std:1.5 ~first_bin:0.5 ~m () in
  let prior_params = P.init dtype prior in
  let time_cov =
    { C.d = vec [ 0.7; 1.0; 1.3 ]
    ; t = create dtype [| 3; 3 |] [| 0.; 0.2; 0.1; 0.; 0.; -0.3; 0.; 0.; 0. |]
    }
  in
  let space_cov =
    { C.d = vec [ 1.1; 0.9 ]; t = create dtype [| 2; 2 |] [| 0.; 0.4; 0.; 0. |] }
  in
  let mu = create dtype [| 3; 2 |] [| 0.2; -0.1; 0.4; 0.0; -0.3; 0.5 |] in
  let got =
    (Option.value_exn (P.kl_to_gaussian prior))
      prior_params
      ~mu
      ~space_factor:(C.factor space_cov)
      ~time_factor:(C.factor time_cov)
  in
  let dense_q = dense_covariance ~t_steps ~m time_cov space_cov in
  let spatial = to_array (exp prior_params.P.Gaussian.log_spatial_stds) in
  let first_bin = Float.exp (item prior_params.P.Gaussian.log_first_bin) in
  let d = t_steps * m in
  let prior_var r =
    let t = r / m
    and j = Int.rem r m in
    let s = spatial.(j) in
    let sigma_t = if t = 0 then first_bin else 1.0 in
    sigma_t *. s *. sigma_t *. s
  in
  let q = to_array dense_q
  and mu_arr = to_array mu in
  let trace = ref 0.0
  and quadratic = ref 0.0
  and log_det_p = ref 0.0 in
  for r = 0 to d - 1 do
    let v = prior_var r in
    trace := !trace +. (q.((r * d) + r) /. v);
    quadratic := !quadratic +. (mu_arr.(r) *. mu_arr.(r) /. v);
    log_det_p := !log_det_p +. Float.log v
  done;
  let log_det_q = item Infix.(sum (log (diag (cholesky dense_q))) *$ 2.0) in
  let expected =
    0.5 *. (!trace +. !quadratic +. !log_det_p -. log_det_q -. Float.of_int d)
  in
  check_close ~msg:"KL against a dense computation" ~tol:1e-9 got (scalar expected);
  (* The KL is non-negative and grows as the prior narrows. *)
  greater ~msg:"KL is non-negative" float_exact ~than:(-1e-9) (item got);
  let narrow_params =
    { prior_params with
      P.Gaussian.log_spatial_stds =
        Infix.(prior_params.P.Gaussian.log_spatial_stds -$ 1.0)
    }
  in
  let narrow_kl =
    (Option.value_exn (P.kl_to_gaussian prior))
      narrow_params
      ~mu
      ~space_factor:(C.factor space_cov)
      ~time_factor:(C.factor time_cov)
  in
  greater ~msg:"narrower prior, larger KL" float_exact ~than:(item got) (item narrow_kl);
  (* mu = 0 and q matching the prior factor gives no KL at all. *)
  let matching_prior =
    P.Gaussian.make ~at:Nx.Host.on ~spatial_std:1.0 ~first_bin:1.0 ~m ()
  in
  let matching_params = P.init dtype matching_prior in
  let identity_space = { C.d = vec [ 1.0; 1.0 ]; t = zeros dtype [| 2; 2 |] } in
  let identity_time = { C.d = vec [ 1.0; 1.0; 1.0 ]; t = zeros dtype [| 3; 3 |] } in
  let zero_kl =
    (Option.value_exn (P.kl_to_gaussian matching_prior))
      matching_params
      ~mu:(zeros dtype [| t_steps; m |])
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
  let likelihood = L.Gaussian.make ~at:Nx.Host.on ~n ~n_output () in
  let params : lik_params =
    { L.Gaussian.c = eye dtype n
    ; bias = zeros dtype [| n_output |]
    ; log_variances = vec [ Float.log 0.5; Float.log 2.0 ]
    }
  in
  let x = vec [ 1.0; -1.0 ]
  and o = vec [ 0.5; 0.5 ] in
  let logp = L.logp_t likelihood params ~k:(scalar 0.0) ~x ~o in
  let expected =
    let sq x = x *. x in
    -0.5
    *. ((sq (0.5 -. 1.0) /. 0.5)
        +. Float.log 0.5
        +. Float.log (2.0 *. Float.pi)
        +. ((sq (0.5 +. 1.0) /. 2.0) +. Float.log 2.0 +. Float.log (2.0 *. Float.pi)))
  in
  check_close ~msg:"gaussian log p" ~tol:1e-12 logp (scalar expected);
  let mean = L.mean likelihood params (vec [ 2.0; 3.0 ]) in
  check_close ~msg:"gaussian mean" ~tol:1e-12 mean (vec [ 2.0; 3.0 ]);
  let poisson = L.Poisson.make ~at:Nx.Host.on ~n ~n_output ~dt:0.1 () in
  let poisson_params : poisson_params =
    { L.Poisson.c = eye dtype n
    ; bias = vec [ 0.0; 0.0 ]
    ; log_gain = vec [ 0.0; Float.log 2.0 ]
    }
  in
  let x = vec [ 1.0; 0.5 ] in
  let o = vec [ 3.0; 2.0 ] in
  let logp = L.logp_t poisson poisson_params ~k:(scalar 0.0) ~x ~o in
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
    L.sample poisson poisson_params ~key:(Nx.Rng.key 5) ~mean:(full dtype [| 5 |] 3.0)
  in
  check_shape ~msg:"poisson sample shape" [| 5 |] counts;
  let arr = to_array counts in
  Array.iter arr ~f:(fun v ->
    greater ~msg:"poisson sample non-negative" float_exact ~than:(-1e-9) v);
  let mean = item Infix.(sum counts *$ (1.0 /. 5.0)) in
  less
    ~msg:"poisson sample mean near rate"
    float_exact
    ~than:2.5
    (Float.abs (mean -. 3.0))

(* ──────────────────────────────────────────────────────────────────────────
   Dynamics
   ────────────────────────────────────────────────────────────────────────── *)

let test_dynamics () =
  let n = 2
  and m = 2 in
  let dynamics = D.Linear.make ~at:Nx.Host.on ~n ~m () in
  let initial = D.init dtype dynamics in
  let params =
    { initial with
      w = create dtype [| 2; 2 |] [| 0.8; 0.2; 0.0; 0.9 |]
    ; dt_over_tau = 1.0
    }
  in
  let matrix = D.Linear.matrix params in
  check_close
    ~msg:"linear matrix"
    ~tol:1e-12
    (reshape [| 4 |] matrix)
    (vec [ 0.8; 0.2; 0.0; 0.9 ]);
  let x = vec [ 1.0; 2.0 ]
  and u = vec [ 0.5; -0.5 ] in
  check_close
    ~msg:"linear step k=0"
    ~tol:1e-12
    (D.dyn dynamics params ~k:(scalar 0.0) ~x ~u)
    (vec [ 1.3; 1.5 ]);
  check_close
    ~msg:"linear step k=1"
    ~tol:1e-12
    (D.dyn dynamics params ~k:(scalar 1.0) ~x ~u)
    (vec [ 1.3; 1.5 ]);
  (* a rollout matches a hand-written loop *)
  let t_steps = 5 in
  let u =
    create
      dtype
      [| t_steps; m |]
      [| 0.1; 0.0; 0.2; -0.1; 0.0; 0.3; -0.2; 0.1; 0.05; 0.0 |]
  in
  let z = D.integrate dynamics params ~x0:(zeros dtype [| n |]) ~u in
  check_shape ~msg:"rollout shape" [| t_steps + 1; n |] z;
  let expected =
    let xs = Array.create ~len:(t_steps + 1) (zeros dtype [| n |]) in
    for k = 0 to t_steps - 1 do
      xs.(k + 1)
      <- D.dyn dynamics params ~k:(scalar (Float.of_int k)) ~x:xs.(k) ~u:(get [ k ] u)
    done;
    stack ~axis:0 (Array.to_list xs)
  in
  check_close ~msg:"rollout" ~tol:1e-12 z expected;
  (* the tanh RNN rolls out, batched and unbatched, with the same values *)
  let rnn =
    D.Linear_nonlinear.make ~at:Nx.Host.on ~n ~nh:4 ~m ~dt:0.1 ~tau:1.0 ~phi:tanh ()
  in
  let rnn_params = D.init dtype rnn in
  let batched =
    D.integrate
      rnn
      rnn_params
      ~x0:(zeros dtype [| 2; n |])
      ~u:(broadcast_to [| 2; t_steps; m |] (unsqueeze ~axes:[ 0 ] u))
  in
  check_shape ~msg:"batched rollout shape" [| 2; t_steps + 1; n |] batched;
  let first = slice [ I 0 ] batched in
  check_close
    ~msg:"batched rollout member"
    ~tol:1e-10
    first
    (D.integrate rnn rnn_params ~x0:(zeros dtype [| n |]) ~u)

(* The paper's MGU (Appendix B.2, with the [z_{t-1}] subscript typo of (S9)
   corrected): the input is additive in the candidate and does not enter the
   forget gate. The expected step is computed with host floats from the same
   equations, not from the implementation. *)
let test_mgu () =
  let n = 2
  and m = 2 in
  let dynamics = D.Mgu.make ~at:Nx.Host.on ~n ~m ~phi () in
  let params : mgu_params =
    { uf = create dtype [| n; n |] [| 0.5; 0.0; 0.0; -0.5 |]
    ; uh = create dtype [| n; n |] [| 0.25; 0.0; 0.0; 0.25 |]
    ; w = create dtype [| m; n |] [| 1.0; 0.5; -0.5; 1.0 |]
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
  check_close ~msg:"mgu step" ~tol:1e-12 (D.dyn dynamics params ~k ~x ~u) (vec [ z1; z2 ]);
  (* the same step under the ReLU activation: the paper's gate and candidate
     with [phi = relu] *)
  let relu x = maximum x (zeros_like x) in
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
  let z_u = D.dyn dynamics params ~k ~x ~u in
  let z_u' = D.dyn dynamics params ~k ~x ~u:u' in
  let z_zero = D.dyn dynamics params ~k ~x ~u:(zeros dtype [| m |]) in
  let z_sum = D.dyn dynamics params ~k ~x ~u:Infix.(u + u') in
  check_close
    ~msg:"mgu input additivity"
    ~tol:1e-12
    Infix.(z_sum - z_zero)
    Infix.(z_u - z_zero + (z_u' - z_zero));
  (* a rollout matches a hand-written loop, and a batched rollout its
     members *)
  let t_steps = 4 in
  let us =
    create dtype [| t_steps; m |] [| 0.5; -1.0; 0.25; 0.0; -0.75; 0.5; 0.0; 0.25 |]
  in
  let x0 = vec [ 0.3; -0.4 ] in
  let z = D.integrate dynamics params ~x0 ~u:us in
  check_shape ~msg:"mgu rollout shape" [| t_steps + 1; n |] z;
  let expected =
    let xs = Array.create ~len:(t_steps + 1) x0 in
    for k = 0 to t_steps - 1 do
      xs.(k + 1)
      <- D.dyn dynamics params ~k:(scalar (Float.of_int k)) ~x:xs.(k) ~u:(get [ k ] us)
    done;
    stack ~axis:0 (Array.to_list xs)
  in
  check_close ~msg:"mgu rollout" ~tol:1e-12 z expected;
  let batched =
    D.integrate
      dynamics
      params
      ~x0:(broadcast_to [| 2; n |] x0)
      ~u:(broadcast_to [| 2; t_steps; m |] (unsqueeze ~axes:[ 0 ] us))
  in
  check_shape ~msg:"mgu batched rollout shape" [| 2; t_steps + 1; n |] batched;
  check_close ~msg:"mgu batched rollout member" ~tol:1e-10 (slice [ I 0 ] batched) z;
  (* the dynamics is differentiable in every parameter *)
  match
    Rune.check_grads
      (D.Mgu.structure ())
      (fun p -> sum (D.integrate dynamics p ~x0 ~u:us))
      params
  with
  | Ok () -> ()
  | Error msg -> fail ("mgu gradient check failed: " ^ msg)

(* ──────────────────────────────────────────────────────────────────────────
   iLQR encoder
   ────────────────────────────────────────────────────────────────────────── *)

(* The cost the encoder minimizes for a linear dynamics, a Gaussian
   likelihood of unit variance and an isotropic Gaussian encoder prior:
   [½ Σ_k ‖u_k‖² + ½ Σ_{k≥1} ‖x_k − o_{k−1}‖² + ½ ‖x_T − o_{T−1}‖²]. *)
let lqr_cost ~a ~o ~t_steps u =
  let x = ref (zeros dtype [| (shape a).(0) |]) in
  let total = ref (scalar 0.0) in
  for k = 0 to t_steps - 1 do
    if k >= 1
    then (
      let prev = k - 1 in
      let diff = Infix.(!x - get [ prev ] o) in
      total := Infix.(!total + (sum (diff * diff) *$ 0.5)));
    let u_k = get [ k ] u in
    (total := Infix.(!total + (sum (u_k * u_k) *$ 0.5)));
    x := Infix.((!x *@ a) + u_k)
  done;
  let last = t_steps - 1 in
  let diff = Infix.(!x - get [ last ] o) in
  Infix.(!total + (sum (diff * diff) *$ 0.5))

let lqr_setup () =
  let n = 2
  and m = 2
  and t_steps = 5 in
  let model, params = setup ~n ~m ~t_steps ~n_output:2 () in
  let w = create dtype [| 2; 2 |] [| 0.8; 0.2; 0.0; 0.9 |] in
  let params : iilqr_params =
    { params with
      M.dynamics = { params.M.dynamics with w; dt_over_tau = 1.0 }
    ; likelihood =
        { L.Gaussian.c = eye dtype n
        ; bias = zeros dtype [| 2 |]
        ; log_variances = zeros dtype [| 2 |]
        }
    }
  in
  let o =
    create
      dtype
      [| t_steps; 2 |]
      [| 1.0; 0.0; 0.5; -0.5; -0.2; 0.7; 0.3; 0.1; -0.6; 0.4 |]
  in
  model, params, o

let test_encoder_stationarity () =
  let model, params, o = lqr_setup () in
  let generative = M.generative model
  and gparams = M.gparams params in
  let mu =
    Il.posterior_mean
      generative
      gparams
      ~solve:(R.adaptive ~max_iter:100 ~conv_threshold:1e-14 ())
      o
  in
  check_shape ~msg:"posterior mean shape" [| 5; 2 |] mu;
  let a = D.Linear.matrix params.M.dynamics in
  let grad = Rune.grad' (fun u -> lqr_cost ~a ~o ~t_steps:5 u) mu in
  let max_grad = item (max (abs grad)) in
  less ~msg:"iLQR solves the LQR problem" float_exact ~than:1e-6 max_grad;
  (* the fixed solver finds the same point, and the generic mean agrees *)
  let mu_fixed =
    Il.posterior_mean generative gparams ~solve:(fixed ~conv_threshold:1e-14 60) o
  in
  check_close ~msg:"fixed encoder matches adaptive" ~tol:1e-6 mu_fixed mu;
  let via_model = (M.mean model params ~solve:(fixed ~conv_threshold:1e-14 60) o).R.u in
  check_close
    ~msg:"model mean matches the encoder"
    ~tol:1e-6
    (reshape [| 5; 2 |] via_model)
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
  check_shape ~msg:"lfads x0 shape" [| 3; n |] posterior.R.x0;
  check_shape ~msg:"lfads u shape" [| t_steps; 3; m |] posterior.R.u;
  check_shape ~msg:"lfads log_prob shape" [| 3 |] posterior.R.log_prob;
  (* the standalone density replays the same trajectory: same value *)
  let replay =
    M.log_prob model params data { R.x0 = posterior.R.x0; u = posterior.R.u }
  in
  check_close ~msg:"lfads density replay" ~tol:1e-10 replay posterior.R.log_prob;
  (* the same key gives the same draws, a different key different ones *)
  let again = M.infer model params ~n_samples:3 ~key:(Nx.Rng.key 11) data in
  check_close ~msg:"lfads draws are keyed" ~tol:1e-12 again.R.u posterior.R.u;
  let other = M.infer model params ~n_samples:3 ~key:(Nx.Rng.key 12) data in
  greater
    ~msg:"different keys draw differently"
    float_exact
    ~than:1e-6
    (max_abs Infix.(other.R.u - posterior.R.u));
  (* without inferred inputs the inputs are the point mass at zero *)
  let model0, params0 = setup_lfads ~n ~m ~t_steps ~n_output:2 ~with_inputs:false () in
  let posterior0 = M.infer model0 params0 ~n_samples:3 ~key:(Nx.Rng.key 13) data in
  check_close
    ~msg:"lfads without inputs, u = 0"
    ~tol:1e-12
    posterior0.R.u
    (zeros dtype [| t_steps; 3; m |]);
  (* the mean is a batch of one *)
  let mean = M.mean model params data in
  check_shape ~msg:"lfads mean x0 shape" [| 1; n |] mean.R.x0;
  check_shape ~msg:"lfads mean u shape" [| t_steps; 1; m |] mean.R.u

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
    let r = zeros dtype [| 2 * 4; 2 * n |] in
    let built = ref r in
    List.iteri rows ~f:(fun j i -> built := set [ I i; I j ] (ones dtype [||]) !built);
    !built
  in
  let with_readout rows =
    { params with
      M.recognition =
        { p with
          Lf.ic_readout = selector rows
        ; Lf.ic_readout_bias = zeros dtype [| 2 * n |]
        }
    }
  in
  let forward_expected =
    slice [ R (0, n) ] (run_mgu ~phi p.Lf.ic_fwd ~init:p.Lf.ic_fwd_init data)
  in
  let fwd_params = with_readout [ 0; 1 ] in
  let fwd = M.mean model fwd_params data in
  check_close
    ~msg:"forward initial-condition encoder"
    ~tol:1e-12
    (reshape [| n |] fwd.R.x0)
    forward_expected;
  let backward_expected =
    slice
      [ R (0, n) ]
      (run_mgu ~phi p.Lf.ic_bwd ~init:p.Lf.ic_bwd_init (flip ~axes:[ 0 ] data))
  in
  let bwd_params = with_readout [ 4; 5 ] in
  let bwd = M.mean model bwd_params data in
  check_close
    ~msg:"backward initial-condition encoder"
    ~tol:1e-12
    (reshape [| n |] bwd.R.x0)
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
          Lf.u_readout = zeros dtype [| (shape p.Lf.u_readout).(0); 2 * m |]
        ; Lf.u_readout_bias = concatenate ~axis:0 [ c; zeros dtype [| m |] ]
        }
    }
  in
  let mean = M.mean model params data in
  let expected_u = broadcast_to [| t_steps; 1; m |] (reshape [| 1; 1; m |] c) in
  check_close ~msg:"push-mean inputs" ~tol:1e-12 mean.R.u expected_u;
  let z = (Ilqr_vae.predictions_deterministic model params mean).Ilqr_vae.z in
  let x = ref (reshape [| n |] mean.R.x0) in
  let states = ref [ !x ] in
  for k = 0 to t_steps - 1 do
    x := D.Mgu.dyn ~phi params.M.dynamics ~k:(scalar (Float.of_int k)) ~x:!x ~u:c;
    states := !x :: !states
  done;
  let expected_z = stack ~axis:0 (List.rev !states) in
  check_close
    ~msg:"push-mean rollout"
    ~tol:1e-12
    (reshape [| t_steps + 1; n |] z)
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
  let initial_prior = I.make ~mean:0.25 ~std:0.6 ~n () in
  let at = Nx.Host.on in
  let dynamics = D.Mgu.make ~at ~n ~m ~phi () in
  let likelihood = L.Gaussian.make ~at:Nx.Host.on ~n ~n_output:2 () in
  let prior = P.Gaussian.make ~at:Nx.Host.on ~m ~spatial_std ~first_bin:1.0 () in
  let encoder_prior = P.Gaussian.make ~at:Nx.Host.on ~m ~spatial_std:1.0 () in
  let recognition =
    Lf.make
      ~at:Nx.Host.on
      ~n_ic:4
      ~n_ci:4
      ~n_ctrl:3
      ~phi
      ~n
      ~m
      ~n_steps:t_steps
      ~n_output:2
      ()
  in
  let model =
    M.make
      dtype
      ~dynamics
      ~likelihood
      ~prior
      ~encoder_prior
      ~recognition
      ~initial_prior
      ()
  in
  let params = M.init ~at ~key:(Nx.Rng.key 42) model in
  let p = params.M.recognition in
  let initial_params = Option.value_exn params.M.initial_prior in
  (* q(x0) = p(x0) and q(u_t) = p(u_t) for every t. *)
  let recognition_params =
    { p with
      Lf.ic_readout = zeros dtype [| 2 * 4; 2 * n |]
    ; Lf.ic_readout_bias =
        concatenate
          ~axis:0
          [ initial_params.I.mean; Infix.(initial_params.I.log_stds *$ 2.0) ]
    ; Lf.u_readout = zeros dtype [| (shape p.Lf.u_readout).(0); 2 * m |]
    ; Lf.u_readout_bias =
        concatenate
          ~axis:0
          [ zeros dtype [| m |]; full dtype [| m |] (2.0 *. Float.log spatial_std) ]
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
  less
    ~msg:"path-derivative gradient sticks to the landing"
    float_exact
    ~than:1e-10
    (max_abs stl_grad);
  (* the live-parameter density, by contrast, has a nonzero sampled gradient *)
  let live_kl params =
    let posterior = M.infer model params ~n_samples:4 ~key data in
    Nx.mean
      (sub
         (M.log_prob model params data { R.x0 = posterior.R.x0; u = posterior.R.u })
         (El.log_prior model params ~x0:posterior.R.x0 ~u:posterior.R.u))
  in
  let live_grad =
    Rune.grad'
      (fun bias -> live_kl (with_bias bias))
      params.M.recognition.Lf.u_readout_bias
  in
  greater ~msg:"live density does not stick" float_exact ~than:1e-8 (max_abs live_grad)

(* A plain-pathwise objective: the same value as {!Elbo.loss_fixed}, but its
   density is evaluated with live parameters, so its autodiff gradient is the
   exact derivative of the traced function and {!Rune.check_grads} applies.
   The library's own objective removes the score-function part of that
   derivative (Roeder, Wu & Duvenaud, "sticking the landing"), which is an
   unbiased estimator of the gradient rather than the derivative of any single
   trace. *)
let plain_objective
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) M.t)
      (params : (w, d, l, p, e, r, 's) M.params)
      ~key
      ~n_samples
      ~iters
      data
  =
  let posterior = M.infer model params ~solve:(fixed iters) ~n_samples ~key data in
  let samples : (w, 's) R.samples = { R.x0 = posterior.R.x0; u = posterior.R.u } in
  let log_prob = M.log_prob model params ~solve:(fixed iters) data samples in
  let posterior = { posterior with R.log_prob } in
  Infix.(-(El.compute model params ~posterior data).El.elbo)

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
   | Error msg -> fail ("lfads gradient check failed: " ^ msg));
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
  let o = create dtype [| t_steps; 2 |] [| 1.0; 0.0; 0.5; -0.5; -0.2; 0.7; 0.3; 0.1 |] in
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
   | Error msg -> fail ("implicit-gradient check failed: " ^ msg));
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
  let o = create dtype [| t_steps; 2 |] [| 1.0; 0.0; 0.5; -0.5; -0.2; 0.7; 0.3; 0.1 |] in
  let structure = M.structure model in
  let posterior =
    M.infer model params ~solve:(fixed 12) ~n_samples:4 ~key:(Nx.Rng.key 2) o
  in
  let terms = El.compute model params ~posterior o in
  check_close
    ~msg:"elbo = ll − beta kl"
    ~tol:1e-10
    terms.El.elbo
    Infix.(terms.El.log_likelihood - terms.El.kl);
  let loss = El.loss model params ~solve:(fixed 12) ~key:(Nx.Rng.key 2) ~n_samples:4 o in
  check_close ~msg:"loss is the negated elbo" ~tol:1e-12 loss Infix.(-terms.El.elbo);
  (* beta scales the KL term *)
  let half = El.compute model params ~beta:(scalar 0.5) ~posterior o in
  check_close
    ~msg:"beta scales the KL"
    ~tol:1e-10
    half.El.elbo
    Infix.(terms.El.log_likelihood - (terms.El.kl *$ 0.5));
  (* the prior enters the KL: a wider prior has a larger KL *)
  let wide =
    { params with
      M.prior =
        { params.M.prior with
          log_spatial_stds = Infix.(params.M.prior.log_spatial_stds +$ 1.0)
        }
    }
  in
  let wide_posterior =
    M.infer model wide ~solve:(fixed 12) ~n_samples:4 ~key:(Nx.Rng.key 2) o
  in
  let wide_terms = El.compute model wide ~posterior:wide_posterior o in
  greater
    ~msg:"wider prior, larger KL"
    float_exact
    ~than:(item terms.El.kl)
    (item wide_terms.El.kl);
  (* the parameters carry gradients through the implicit solve *)
  let _, grads =
    Rune.value_and_grad
      structure
      (fun p -> El.loss_fixed model p ~key:(Nx.Rng.key 2) ~n_samples:2 ~iters:8 o)
      params
  in
  greater
    ~msg:"the covariance parameters are learned"
    float_exact
    ~than:1e-10
    (max_abs grads.M.recognition.Il.time_cov.t)

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
  check_shape ~msg:"beta is a scalar tensor" [||] beta;
  (* beta = 0 leaves the log likelihood alone *)
  let posterior =
    M.infer model params ~solve:(fixed 6) ~n_samples:3 ~key:(Nx.Rng.key 30) o
  in
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
  greater
    ~msg:"the weight changes the loss"
    float_exact
    ~than:1e-6
    (Float.abs (via_jit 0 -. via_jit 10))

let test_jit () =
  let n = 2
  and m = 2
  and t_steps = 4 in
  let model, params = setup ~n ~m ~t_steps ~n_output:2 () in
  let o = create dtype [| t_steps; 2 |] [| 1.0; 0.0; 0.5; -0.5; -0.2; 0.7; 0.3; 0.1 |] in
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
   Mini-batches
   ────────────────────────────────────────────────────────────────────────── *)

let test_batched_iilqr () =
  let n = 2
  and m = 2
  and t_steps = 3 in
  let model, params = setup ~n ~m ~t_steps ~n_output:2 () in
  let o1 = data_of ~n_output:2 ~t_steps [ 1.0; 0.0; 0.5; -0.5; -0.2; 0.7 ] in
  let o2 = data_of ~n_output:2 ~t_steps [ 0.3; -0.1; -0.4; 0.8; 0.2; -0.6 ] in
  let batch = stack ~axis:0 [ o1; o2 ] in
  let key = Nx.Rng.key 40 in
  let n_samples = 3
  and iters = 4 in
  (* the recognition maps one trial's solve over the leading batch axis *)
  let posterior = M.infer model params ~solve:(fixed iters) ~n_samples ~key batch in
  check_shape ~msg:"batched x0 shape" [| 2; n_samples; n |] posterior.R.x0;
  check_shape ~msg:"batched u shape" [| 2; t_steps; n_samples; m |] posterior.R.u;
  check_shape ~msg:"batched log_prob shape" [| 2; n_samples |] posterior.R.log_prob;
  let posterior_mean = M.mean model params ~solve:(fixed iters) batch in
  check_shape ~msg:"batched mean x0 shape" [| 2; 1; n |] posterior_mean.R.x0;
  check_shape ~msg:"batched mean u shape" [| 2; t_steps; 1; m |] posterior_mean.R.u;
  let prediction = Ilqr_vae.predictions model params posterior in
  let pu = prediction.Ilqr_vae.u
  and pz = prediction.Ilqr_vae.z
  and po = prediction.Ilqr_vae.observations in
  check_shape ~msg:"batched predictions u shape" [| 2; t_steps; n_samples; m |] pu;
  check_shape ~msg:"batched predictions z shape" [| 2; t_steps + 1; n_samples; n |] pz;
  check_shape ~msg:"batched predictions o shape" [| 2; t_steps; n_samples; 2 |] po;
  let deterministic = Ilqr_vae.predictions_deterministic model params posterior_mean in
  let dz = deterministic.Ilqr_vae.z
  and do_ = deterministic.Ilqr_vae.observations in
  check_shape ~msg:"batched deterministic z shape" [| 2; t_steps + 1; 1; n |] dz;
  check_shape ~msg:"batched deterministic o shape" [| 2; t_steps; 1; 2 |] do_;
  let pz0 = Ilqr_vae.integrate model params ~u:(zeros dtype [| 2; t_steps; m |]) in
  check_shape ~msg:"batched integrate shape" [| 2; t_steps + 1; n |] pz0;
  (* the density at the draws is the same one the recognition reports *)
  let log_prob =
    M.log_prob
      model
      params
      ~solve:(fixed iters)
      batch
      { R.x0 = posterior.R.x0; u = posterior.R.u }
  in
  check_close ~msg:"batched log q replays" ~tol:1e-10 log_prob posterior.R.log_prob;
  (* compute averages the per-trial bounds of the same draws *)
  let terms = El.compute model params ~posterior batch in
  let per_trial =
    Array.init 2 ~f:(fun i ->
      let posterior_i =
        { R.x0 = slice [ I i ] posterior.R.x0
        ; u = slice [ I i ] posterior.R.u
        ; log_prob = slice [ I i ] posterior.R.log_prob
        }
      in
      El.compute model params ~posterior:posterior_i (slice [ I i ] batch))
  in
  let mean_of field =
    mean (stack ~axis:0 (Array.to_list (Array.map per_trial ~f:field)))
  in
  check_close
    ~msg:"batched elbo is the trial mean"
    ~tol:1e-12
    terms.El.elbo
    (mean_of (fun t -> t.El.elbo));
  check_close
    ~msg:"batched likelihood is the trial mean"
    ~tol:1e-12
    terms.El.log_likelihood
    (mean_of (fun t -> t.El.log_likelihood));
  check_close
    ~msg:"batched kl is the trial mean"
    ~tol:1e-12
    terms.El.kl
    (mean_of (fun t -> t.El.kl));
  (* the key is split into one subkey per trial, as for separate losses *)
  let keys = Nx.Rng.split ~n:2 key in
  let separate =
    stack
      ~axis:0
      [ El.loss_fixed model params ~key:keys.(0) ~n_samples ~iters o1
      ; El.loss_fixed model params ~key:keys.(1) ~n_samples ~iters o2
      ]
  in
  check_close
    ~msg:"batched loss is the trial mean"
    ~tol:1e-12
    (El.loss_fixed model params ~key ~n_samples ~iters batch)
    (mean separate);
  (* the mini-batch is an argument of the compiled step, not a closed array *)
  let structure = M.structure model in
  let eager =
    Rune.value_and_grad
      structure
      (fun p -> El.loss_fixed model p ~key ~n_samples ~iters batch)
      params
  in
  let compiled =
    Rune.jit
      Nx.Ptree.(structure @-> Nx.Rng.ptree @-> tensor @-> returns (pair tensor structure))
      (fun p k data ->
         Rune.value_and_grad
           structure
           (fun p -> El.loss_fixed model p ~key:k ~n_samples ~iters data)
           p)
  in
  let loss, grads = compiled params key batch in
  check_close ~msg:"batched jit loss" ~tol:1e-10 loss (fst eager);
  let eager_grads = snd eager in
  check_close
    ~msg:"batched jit dynamics gradient"
    ~tol:1e-9
    grads.M.dynamics.w
    eager_grads.M.dynamics.w;
  check_close
    ~msg:"batched jit covariance gradient"
    ~tol:1e-9
    grads.M.recognition.Il.time_cov.t
    eager_grads.M.recognition.Il.time_cov.t;
  (* the adaptive solve is eager and runs one trial at a time *)
  let adaptive = M.infer model params ~n_samples:2 ~key:(Nx.Rng.key 43) batch in
  check_shape ~msg:"adaptive batched x0 shape" [| 2; 2; n |] adaptive.R.x0;
  check_shape ~msg:"adaptive batched u shape" [| 2; t_steps; 2; m |] adaptive.R.u;
  let adaptive_log_prob =
    M.log_prob model params batch { R.x0 = adaptive.R.x0; u = adaptive.R.u }
  in
  check_shape ~msg:"adaptive batched log q shape" [| 2; 2 |] adaptive_log_prob;
  let adaptive_mean = M.mean model params batch in
  check_shape
    ~msg:"adaptive batched mean u shape"
    [| 2; t_steps; 1; m |]
    adaptive_mean.R.u;
  let adaptive_loss = El.loss model params ~key:(Nx.Rng.key 44) ~n_samples:2 batch in
  check_finite ~msg:"adaptive batched loss" (item adaptive_loss);
  (* more than one leading trial axis behaves as one flattened batch *)
  let batch2 = stack ~axis:0 [ batch; batch ] in
  let posterior2 =
    M.infer model params ~solve:(fixed iters) ~n_samples ~key:(Nx.Rng.key 42) batch2
  in
  check_shape ~msg:"two-axis batch x0 shape" [| 2; 2; n_samples; n |] posterior2.R.x0;
  check_shape
    ~msg:"two-axis batch u shape"
    [| 2; 2; t_steps; n_samples; m |]
    posterior2.R.u;
  check_shape
    ~msg:"two-axis batch log_prob shape"
    [| 2; 2; n_samples |]
    posterior2.R.log_prob;
  let terms2 = El.compute model params ~posterior:posterior2 batch2 in
  check_shape ~msg:"two-axis batch bound is scalar" [||] terms2.El.elbo

let test_batched_lfads () =
  let n = 2
  and m = 2
  and t_steps = 3 in
  let model, params = setup_lfads ~n ~m ~t_steps ~n_output:2 () in
  let o = lfads_data ~t_steps ~n_output:2 () in
  let batch = stack ~axis:0 [ o; Infix.(o *$ 0.5) ] in
  let key = Nx.Rng.key 41 in
  let n_samples = 3
  and iters = 4 in
  let posterior = M.infer model params ~solve:(fixed iters) ~n_samples ~key batch in
  check_shape ~msg:"lfads batched x0 shape" [| 2; n_samples; n |] posterior.R.x0;
  check_shape ~msg:"lfads batched u shape" [| 2; t_steps; n_samples; m |] posterior.R.u;
  check_shape ~msg:"lfads batched log_prob shape" [| 2; n_samples |] posterior.R.log_prob;
  let log_prob =
    M.log_prob
      model
      params
      ~solve:(fixed iters)
      batch
      { R.x0 = posterior.R.x0; u = posterior.R.u }
  in
  check_close ~msg:"lfads batched log q replays" ~tol:1e-10 log_prob posterior.R.log_prob;
  (* the bound is the mean over the trials, from the same draws *)
  let terms = El.compute model params ~posterior batch in
  let per_trial =
    Array.init 2 ~f:(fun i ->
      let posterior_i =
        { R.x0 = slice [ I i ] posterior.R.x0
        ; u = slice [ I i ] posterior.R.u
        ; log_prob = slice [ I i ] posterior.R.log_prob
        }
      in
      El.compute model params ~posterior:posterior_i (slice [ I i ] batch))
  in
  let mean_of field =
    mean (stack ~axis:0 (Array.to_list (Array.map per_trial ~f:field)))
  in
  check_close
    ~msg:"lfads batched elbo is the trial mean"
    ~tol:1e-12
    terms.El.elbo
    (mean_of (fun t -> t.El.elbo));
  (* eager and compiled batch losses are one function of the data argument *)
  let structure = M.structure model in
  let eager = El.loss_fixed model params ~key ~n_samples ~iters batch in
  let compiled =
    Rune.jit
      Nx.Ptree.(structure @-> Nx.Rng.ptree @-> tensor @-> returns tensor)
      (fun p k data -> El.loss_fixed model p ~key:k ~n_samples ~iters data)
  in
  check_close ~msg:"lfads batched jit loss" ~tol:1e-10 (compiled params key batch) eager

(* ──────────────────────────────────────────────────────────────────────────
   Swapping recognitions and validation
   ────────────────────────────────────────────────────────────────────────── *)

let test_swapping () =
  let n = 2
  and m = 2
  and t_steps = 3 in
  let at = Nx.Host.on in
  let dynamics = D.Linear.make ~at ~n ~m () in
  let likelihood = L.Gaussian.make ~at ~n ~n_output:2 () in
  let prior = P.Gaussian.make ~at ~m ~spatial_std:1.0 () in
  let encoder_prior = P.Gaussian.make ~at ~m ~spatial_std:1.0 () in
  let data = lfads_data ~t_steps ~n_output:2 () in
  let make recognition initial_prior =
    M.make
      dtype
      ~dynamics
      ~likelihood
      ~prior
      ~encoder_prior
      ~recognition
      ?initial_prior
      ()
  in
  let iilqr = make (Il.make ~at ~n ~m ~n_steps:t_steps ~n_output:2 ()) None in
  let iilqr_params = M.init ~at ~key:(Nx.Rng.key 31) iilqr in
  let lfads =
    make
      (Lf.make ~at ~n_ic:4 ~n_ctrl:3 ~phi ~n ~m ~n_steps:t_steps ~n_output:2 ())
      (Some (I.make ~std:1.0 ~n ()))
  in
  let lfads_params = M.init ~at ~key:(Nx.Rng.key 32) lfads in
  let loss_of model params =
    El.loss_fixed model params ~key:(Nx.Rng.key 33) ~n_samples:2 ~iters:4 data
  in
  let l_iilqr = loss_of iilqr iilqr_params
  and l_lfads = loss_of lfads lfads_params in
  check_finite ~msg:"the iLQR bound" (item l_iilqr);
  check_finite ~msg:"the LFADS bound" (item l_lfads);
  (* both recognition parameter blocks receive gradients *)
  let _, grads =
    Rune.value_and_grad
      (M.structure iilqr)
      (fun p -> El.loss_fixed iilqr p ~key:(Nx.Rng.key 33) ~n_samples:2 ~iters:4 data)
      iilqr_params
  in
  greater
    ~msg:"the iLQR covariance receives gradients"
    float_exact
    ~than:1e-10
    (max_abs grads.M.recognition.Il.space_cov.d);
  let _, grads =
    Rune.value_and_grad
      (M.structure lfads)
      (fun p -> El.loss_fixed lfads p ~key:(Nx.Rng.key 33) ~n_samples:2 ~iters:4 data)
      lfads_params
  in
  greater
    ~msg:"the LFADS encoder receives gradients"
    float_exact
    ~than:1e-10
    (max_abs grads.M.recognition.Lf.ic_readout);
  (* mismatched initial-state prior and recognition are rejected *)
  let rejects ?msg f = raises_match ?msg (Exn.invalid_arg ~sub:"Model.make") f in
  rejects ~msg:"iLQR rejects an initial-state prior" (fun () ->
    make (Il.make ~at ~n ~m ~n_steps:t_steps ~n_output:2 ()) (Some (I.make ~n ())));
  rejects ~msg:"LFADS requires an initial-state prior" (fun () ->
    make (Lf.make ~at ~n_ic:4 ~n_ctrl:3 ~phi ~n ~m ~n_steps:t_steps ~n_output:2 ()) None);
  (* the model takes its dimensions from the components, so it checks that
     they agree with each other *)
  equal ~msg:"the model takes its state dimension from the dynamics" int n (M.n iilqr);
  equal ~msg:"the model takes its input dimension from the dynamics" int m (M.m iilqr);
  equal
    ~msg:"the model takes its horizon from the recognition"
    int
    t_steps
    (M.n_steps iilqr);
  equal
    ~msg:"the model takes its observation dimension from the likelihood"
    int
    2
    (M.n_output iilqr);
  let rejects_model
        ?msg
        ?(dynamics = dynamics)
        ?(likelihood = likelihood)
        ?(prior = prior)
        ?(encoder_prior = encoder_prior)
        ?(recognition = Il.make ~at ~n ~m ~n_steps:t_steps ~n_output:2 ())
        ()
    =
    rejects ?msg (fun () ->
      M.make dtype ~dynamics ~likelihood ~prior ~encoder_prior ~recognition ())
  in
  rejects_model
    ~msg:"a recognition built for another state dimension is rejected"
    ~recognition:(Il.make ~at ~n:(n + 1) ~m ~n_steps:t_steps ~n_output:2 ())
    ();
  rejects_model
    ~msg:"a likelihood built for another state dimension is rejected"
    ~likelihood:(L.Gaussian.make ~at ~n:(n + 1) ~n_output:2 ())
    ();
  rejects_model
    ~msg:"a likelihood with another observation dimension is rejected"
    ~likelihood:(L.Gaussian.make ~at ~n ~n_output:3 ())
    ();
  rejects_model
    ~msg:"a prior over other inputs is rejected"
    ~prior:(P.Gaussian.make ~at ~m:(m + 1) ~spatial_std:1.0 ())
    ()

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
  let trial_data =
    Array.init 4 ~f:(fun i ->
      let key = Nx.Rng.key (100 + i) in
      let o = (Ilqr_vae.sample_generative model true_params ~key).Ilqr_vae.observations in
      o)
  in
  let trials = stack ~axis:0 (Array.to_list trial_data) in
  let structure = M.structure model in
  let key = Nx.Rng.key 7 in
  let objective p = El.loss_fixed model p ~key ~n_samples:4 ~iters:4 trials in
  let initial = item (objective params) in
  let lr = 0.02 in
  let params = ref params in
  for _ = 1 to 40 do
    let _, grads = Rune.value_and_grad structure objective !params in
    let step
      :  'a 'b.
         Nx.Ptree.Path.t
      -> ('a, 'b, Nx.host) Nx.t
      -> ('a, 'b, Nx.host) Nx.t
      -> ('a, 'b, Nx.host) Nx.t
      =
      fun _ p g -> Infix.(p - (cast (Nx.dtype p) (Nx.scalar float64 lr) * g))
    in
    params := Nx.Ptree.map2 structure step !params grads
  done;
  let final = item (objective !params) in
  less ~msg:"training decreases the loss" float_exact ~than:initial final;
  (* generative and predictive shapes *)
  let draw = Ilqr_vae.sample_generative model !params ~key:(Nx.Rng.key 9) in
  let x0 = draw.Ilqr_vae.x0
  and u = draw.Ilqr_vae.u
  and z = draw.Ilqr_vae.z
  and o = draw.Ilqr_vae.observations in
  check_shape ~msg:"generative x0 shape" [| n |] x0;
  check_shape ~msg:"generative u shape" [| t_steps; m |] u;
  check_shape ~msg:"generative z shape" [| t_steps + 1; n |] z;
  check_shape ~msg:"generative o shape" [| t_steps; 2 |] o;
  let posterior =
    M.infer
      model
      !params
      ~solve:(fixed 4)
      ~n_samples:5
      ~key:(Nx.Rng.key 10)
      trial_data.(0)
  in
  let prediction = Ilqr_vae.predictions model !params posterior in
  let u = prediction.Ilqr_vae.u
  and z = prediction.Ilqr_vae.z
  and o = prediction.Ilqr_vae.observations in
  check_shape ~msg:"predictions u shape" [| t_steps; 5; m |] u;
  check_shape ~msg:"predictions z shape" [| t_steps + 1; 5; n |] z;
  check_shape ~msg:"predictions o shape" [| t_steps; 5; 2 |] o;
  let mean = M.mean model !params trial_data.(0) in
  let deterministic = Ilqr_vae.predictions_deterministic model !params mean in
  let z = deterministic.Ilqr_vae.z
  and o = deterministic.Ilqr_vae.observations in
  check_shape ~msg:"deterministic z shape" [| t_steps + 1; 1; n |] z;
  check_shape ~msg:"deterministic o shape" [| t_steps; 1; 2 |] o

let test_lfads_training_smoke () =
  let n = 2
  and m = 2
  and t_steps = 5 in
  let model, params = setup_lfads ~n ~m ~t_steps ~n_output:2 () in
  let true_params = params in
  let trial_data =
    Array.init 3 ~f:(fun i ->
      let key = Nx.Rng.key (200 + i) in
      let o = (Ilqr_vae.sample_generative model true_params ~key).Ilqr_vae.observations in
      o)
  in
  let trials = stack ~axis:0 (Array.to_list trial_data) in
  let structure = M.structure model in
  let key = Nx.Rng.key 17 in
  let objective p = El.loss_fixed model p ~key ~n_samples:3 ~iters:4 trials in
  let initial = item (objective params) in
  let lr = 0.002 in
  let params = ref params in
  for _ = 1 to 25 do
    let _, grads = Rune.value_and_grad structure objective !params in
    let step
      :  'a 'b.
         Nx.Ptree.Path.t
      -> ('a, 'b, Nx.host) Nx.t
      -> ('a, 'b, Nx.host) Nx.t
      -> ('a, 'b, Nx.host) Nx.t
      =
      fun _ p g -> Infix.(p - (cast (Nx.dtype p) (Nx.scalar float64 lr) * g))
    in
    params := Nx.Ptree.map2 structure step !params grads
  done;
  let final = item (objective !params) in
  less ~msg:"lfads training decreases the loss" float_exact ~than:initial final

(* [visits s v] is the steps of [s]'s walk of [v], one per line. *)
let visits s v =
  String.concat
    (List.map (Nx.Ptree.visits s v) ~f:(fun x ->
       Stdlib.Format.asprintf "%a\n" Nx.Ptree.pp_visit x))

(* The model's and the generative parameters' structures walk their
   components' structures by hand, so their visits name every part: each
   component under its field, the initial-state prior's presence, and every
   array of each component. *)
(* ──────────────────────────────────────────────────────────────────────────
   The free initial state
   ────────────────────────────────────────────────────────────────────────── *)

(* A model whose encoder infers the initial state requires the prior and gets
   a genuine free value: the posterior has the documented shapes, its initial
   state is not the fixed zero, and its scale parameter widens the draws. *)
let test_initial_state_recognition () =
  let n = 2
  and m = 2
  and t_steps = 5 in
  let model, params = setup_initial ~n ~m ~t_steps ~n_output:2 () in
  raises_match (Exn.invalid_arg ~sub:"pass ~initial_prior") (fun () ->
    M.make
      dtype
      ~dynamics:(M.dynamics model)
      ~likelihood:(M.likelihood model)
      ~prior:(M.prior model)
      ~encoder_prior:(M.encoder_prior model)
      ~recognition:(M.recognition model)
      ());
  let data =
    let o =
      (Ilqr_vae.sample_generative model params ~key:(Nx.Rng.key 31)).Ilqr_vae.observations
    in
    o
  in
  let gen = M.generative model
  and gp = M.gparams params in
  let initial = Option.value_exn (Il.posterior_initial gen gp ~solve:(fixed 8) data) in
  check_shape ~msg:"initial mean shape" [| n |] initial.R.mean;
  check_shape ~msg:"initial precision shape" [| n; n |] initial.R.precision;
  greater
    ~msg:"the initial state is not the fixed zero"
    float_exact
    ~than:1e-3
    (max_abs initial.R.mean);
  let posterior =
    M.infer model params ~solve:(fixed 8) ~n_samples:4 ~key:(Nx.Rng.key 32) data
  in
  check_shape ~msg:"posterior x0 shape" [| 4; n |] posterior.R.x0;
  check_shape ~msg:"posterior u shape" [| t_steps; 4; m |] posterior.R.u;
  check_shape ~msg:"posterior log_prob shape" [| 4 |] posterior.R.log_prob;
  equal ~msg:"posterior is finite" bool true (item (all (isfinite posterior.R.log_prob)));
  let mean = M.mean model params ~solve:(fixed 8) data in
  check_shape ~msg:"mean x0 shape" [| 1; n |] mean.R.x0;
  check_shape ~msg:"mean u shape" [| t_steps; 1; m |] mean.R.u;
  equal
    ~msg:"the mean scores finitely under its own density"
    bool
    true
    (Nx.item [ 0 ] (isfinite (M.log_prob model params ~solve:(fixed 8) data mean)));
  (* a mini-batch keeps the trial axes leading *)
  let batch =
    concatenate ~axis:0 [ unsqueeze ~axes:[ 0 ] data; unsqueeze ~axes:[ 0 ] data ]
  in
  let batched =
    M.infer model params ~solve:(fixed 4) ~n_samples:2 ~key:(Nx.Rng.key 33) batch
  in
  check_shape ~msg:"batched x0 shape" [| 2; 2; n |] batched.R.x0;
  check_shape ~msg:"batched u shape" [| 2; t_steps; 2; m |] batched.R.u;
  (* the scale is what widens the draws *)
  let spread scale =
    let params =
      { params with
        M.recognition =
          { params.M.recognition with Il.initial_log_scale = Some (scalar scale) }
      }
    in
    let draws =
      M.infer model params ~solve:(fixed 8) ~n_samples:400 ~key:(Nx.Rng.key 34) data
    in
    let centered =
      Infix.(draws.R.x0 - reshape [| 1; n |] (Nx.mean ~axes:[ 0 ] draws.R.x0))
    in
    item (Nx.mean Infix.(centered * centered))
  in
  check_close
    ~msg:"the scale sets the spread"
    ~tol:0.05
    (scalar (spread (Float.log 2.0)))
    (scalar (4.0 *. spread 0.0))

(* The encoder's initial state is the mode of its own cost, not a by-product of
   the first input: at the returned mean the value of the control problem is
   stationary in [x_0], and its curvature there is the returned precision. The
   input is low-rank ([m < n]) and cannot set the state alone. *)
let test_initial_state_is_the_mode () =
  let n = 3
  and m = 1
  and t_steps = 5 in
  let model, params = setup_initial ~n ~m ~t_steps ~n_output:3 () in
  let data =
    let o =
      (Ilqr_vae.sample_generative model params ~key:(Nx.Rng.key 41)).Ilqr_vae.observations
    in
    o
  in
  let gen = M.generative model
  and gp = M.gparams params in
  let iters = 40 in
  let initial =
    Option.value_exn (Il.posterior_initial gen gp ~solve:(fixed iters) data)
  in
  let problem = Il.problem gen data in
  let us0 = Array.init t_steps ~f:(fun _ -> zeros dtype [| m |]) in
  (* [value x0] is the encoder's own cost with the initial state held at [x0]:
     the value the local model approximates. *)
  let value x0 =
    let inputs : (_, _) Dilqr.Inputs.t = { theta = gp; x0; us0 } in
    Dilqr.loss_of_taus problem gp (Dilqr.solve_fixed problem ~max_iters:iters inputs)
  in
  let structure = Dilqr.inputs problem in
  let gradient x0 =
    (Rune.grad
       structure
       (fun i -> value i.Dilqr.Inputs.x0)
       { Dilqr.Inputs.theta = gp; x0; us0 })
      .Dilqr.Inputs.x0
  in
  (* The point of the free initial state, in one number: the part of it the
     first input cannot produce. [m < n] and [b] is the input weight, so the
     reachable set of [x_1] is the span of [bᵀ] and the projection below
     removes it. *)
  let b = Option.value_exn params.M.dynamics.D.Linear.b in
  let mean = initial.R.mean in
  let reachable =
    Infix.(matrix_transpose b *@ (b *@ mean) *$ (1.0 /. item (sum (b * b))))
  in
  let outside = Infix.(mean - reachable) in
  greater
    ~msg:"the mean has a component the first input cannot produce"
    float_exact
    ~than:1e-2
    (max_abs outside);
  less
    ~msg:"the initial state is stationary"
    float_exact
    ~than:1e-5
    (max_abs (gradient initial.R.mean));
  (* The value's gradient is affine in [x_0] with slope its curvature, so one
     central difference of it gives the precision a column at a time — through
     the implicit VJP rather than through the Riccati recursion the precision
     came from. *)
  let h = 1e-2 in
  let scale i s =
    create dtype [| n |] (Array.init n ~f:(fun k -> if Int.equal k i then s else 0.0))
  in
  for j = 0 to n - 1 do
    let at s = gradient Infix.(initial.R.mean + scale j s) in
    let column = Infix.((at h - at (-.h)) /$ (2.0 *. h)) in
    for i = 0 to n - 1 do
      check_close
        ~msg:(Printf.sprintf "curvature %d %d is the precision" i j)
        ~tol:1e-6
        (get [ i; j ] initial.R.precision)
        (get [ i ] column)
    done
  done

(* The bound trains with a free initial state: the prior is a term of the ELBO
   and of the encoder's cost, and the gradient reaches both through the solve. *)
let test_initial_state_training_smoke () =
  let n = 2
  and m = 2
  and t_steps = 5 in
  let model, params = setup_initial ~n ~m ~t_steps ~n_output:2 () in
  let trial_data =
    Array.init 3 ~f:(fun i ->
      let o =
        (Ilqr_vae.sample_generative model params ~key:(Nx.Rng.key (300 + i)))
          .Ilqr_vae.observations
      in
      o)
  in
  let trials = stack ~axis:0 (Array.to_list trial_data) in
  let structure = M.structure model in
  let key = Nx.Rng.key 21 in
  let objective p = El.loss_fixed model p ~key ~n_samples:3 ~iters:4 trials in
  let initial = item (objective params) in
  check_finite ~msg:"the bound" initial;
  let lr = 0.01 in
  let params = ref params in
  for _ = 1 to 12 do
    let _, grads = Rune.value_and_grad structure objective !params in
    let step
      :  'a 'b.
         Nx.Ptree.Path.t
      -> ('a, 'b, Nx.host) Nx.t
      -> ('a, 'b, Nx.host) Nx.t
      -> ('a, 'b, Nx.host) Nx.t
      =
      fun _ p g -> Infix.(p - (cast (Nx.dtype p) (Nx.scalar float64 lr) * g))
    in
    params := Nx.Ptree.map2 structure step !params grads
  done;
  let final = item (objective !params) in
  less
    ~msg:"training with a free initial state decreases the loss"
    float_exact
    ~than:initial
    final

(* The input prior is a term of the bound exactly when the recognition infers
   the inputs: a recognition that leaves them at the point mass at zero has no
   input factor, so widening the generative input prior cannot move the bound.
   With recorded inputs it must. *)
let test_input_prior_terms () =
  let n = 2
  and m = 2
  and t_steps = 3 in
  let o = lfads_data ~t_steps ~n_output:2 () in
  let widened params =
    { params with
      M.prior =
        { params.M.prior with
          P.Gaussian.log_spatial_stds =
            Infix.(params.M.prior.P.Gaussian.log_spatial_stds +$ 1.0)
        }
    }
  in
  let bound ~with_inputs =
    let model, params =
      setup_lfads ~key:(Nx.Rng.key 60) ~with_inputs ~n ~m ~t_steps ~n_output:2 ()
    in
    let key = Nx.Rng.key 61 in
    let posterior = M.infer model params ~n_samples:4 ~key o in
    let narrow = (El.compute model params ~posterior o).El.elbo in
    let wide =
      let posterior = M.infer model (widened params) ~n_samples:4 ~key o in
      (El.compute model (widened params) ~posterior o).El.elbo
    in
    narrow, wide
  in
  let narrow, wide = bound ~with_inputs:false in
  check_close ~msg:"an uninferred input has no prior term" ~tol:1e-12 wide narrow;
  let narrow, wide = bound ~with_inputs:true in
  greater
    ~msg:"an inferred input brings the prior into the bound"
    float_exact
    ~than:1e-3
    (Float.abs (item Infix.(wide - narrow)))

(* The shapes the interfaces state, refused rather than broadcast: a wrong
   latent or batch shape would otherwise reach the arithmetic and answer a
   different question. *)
let test_shape_errors () =
  let n = 2
  and m = 2
  and t_steps = 3 in
  let model, params = setup ~n ~m ~t_steps ~n_output:2 () in
  let data = zeros dtype [| t_steps; 2 |] in
  let batch = stack ~axis:0 [ data; data ] in
  let solve = fixed 3 in
  let posterior = M.infer model params ~solve ~n_samples:2 ~key:(Nx.Rng.key 70) batch in
  raises_match (Exn.invalid_arg ~sub:"log_prob must be") (fun () ->
    El.compute
      model
      params
      ~posterior:
        { posterior with
          R.log_prob = reshape [| 1; 2 |] (slice [ I 0 ] posterior.R.log_prob)
        }
      batch);
  raises_match (Exn.invalid_arg ~sub:"x0 must be [n_samples; 2]") (fun () ->
    M.log_prob
      model
      params
      ~solve
      data
      { R.x0 = zeros dtype [| 1; 3 |]; u = zeros dtype [| t_steps; 1; 2 |] });
  raises_match (Exn.invalid_arg ~sub:"u must be [3; 1; 2]") (fun () ->
    M.log_prob
      model
      params
      ~solve
      data
      { R.x0 = zeros dtype [| 1; 2 |]; u = zeros dtype [| t_steps; 2 |] });
  raises_match (Exn.invalid_arg ~sub:"input dimension 2") (fun () ->
    D.integrate
      (M.dynamics model)
      params.M.dynamics
      ~x0:(zeros dtype [| 2 |])
      ~u:(zeros dtype [| t_steps; 3 |]));
  raises_match (Exn.invalid_arg ~sub:"Elbo.log_prior") (fun () ->
    El.log_prior model params ~x0:(zeros dtype [| 4; 2 |]) ~u:(zeros dtype [| 3; 2 |]));
  let lfads, lfads_params = setup_lfads ~n ~m ~t_steps ~n_output:2 () in
  raises_match (Exn.invalid_arg ~sub:"Recognition.Lfads.log_prob") (fun () ->
    M.log_prob
      lfads
      lfads_params
      data
      { R.x0 = zeros dtype [| 1; 2 |]; u = zeros dtype [| t_steps; 2 |] });
  (* the constructors refuse the arguments that cannot mean anything *)
  raises_match (Exn.invalid_arg ~sub:"iters") (fun () -> R.fixed ~iters:0 ());
  raises_match (Exn.invalid_arg ~sub:"max_iter") (fun () -> R.adaptive ~max_iter:0 ());
  raises_match (Exn.invalid_arg ~sub:"initial_prior is over 3 dimensions") (fun () ->
    M.make
      dtype
      ~dynamics:(M.dynamics model)
      ~likelihood:(M.likelihood model)
      ~prior:(M.prior model)
      ~encoder_prior:(M.encoder_prior model)
      ~recognition:
        (R.Iilqr.make
           ~at:Nx.Host.on
           ~n
           ~m
           ~n_steps:t_steps
           ~n_output:2
           ~infer_initial:true
           ())
      ~initial_prior:(I.make ~n:3 ())
      ())

(* A placement is a real parameter of a model: built at a brand that is not
   [Nx.host], its parameters are made there and inference, the encoder's solve
   and the bound run there. A minted set over the host device exercises the
   path without a GPU; every tensor it creates, and the key, must be at that
   brand, which the types enforce. *)
let test_placement () =
  let module Brand = (val Nx.devices [ Rig.host ] : Nx.Devices) in
  let at = Brand.on in
  let n = 2
  and m = 2
  and t_steps = 3 in
  let dynamics = D.Linear.make ~at ~n ~m () in
  let likelihood = L.Gaussian.make ~at ~n ~n_output:2 () in
  let prior = P.Gaussian.make ~at ~m () in
  let encoder_prior = P.Gaussian.make ~at ~m () in
  let recognition = Il.make ~at ~n ~m ~n_steps:t_steps ~n_output:2 () in
  let model = M.make dtype ~dynamics ~likelihood ~prior ~encoder_prior ~recognition () in
  let key = Nx.Rng.place at (Nx.Rng.key 1) in
  let params = M.init ~at ~key model in
  let data = Nx.zeros_at at dtype [| t_steps; 2 |] in
  let posterior = M.infer model params ~solve:(fixed 4) ~n_samples:2 ~key data in
  check_shape ~msg:"draws at the placement" [| 2; n |] posterior.R.x0;
  check_shape ~msg:"inputs at the placement" [| t_steps; 2; m |] posterior.R.u;
  let terms = El.compute model params ~posterior data in
  check_finite ~msg:"the bound at the placement" (item terms.El.elbo);
  let mean = M.mean model params ~solve:(fixed 4) data in
  check_shape ~msg:"the mean at the placement" [| t_steps; 1; m |] mean.R.u

(* A component's parameters can be moved to another placement: the values are
   the same, and the type says where they live. *)
let test_place () =
  let module Brand = (val Nx.devices [ Rig.host ] : Nx.Devices) in
  let at = Brand.on in
  let prior = P.Gaussian.make ~at:Nx.Host.on ~spatial_std:1.5 ~first_bin:0.5 ~m:2 () in
  let params = P.init dtype prior in
  let moved = P.Gaussian.place ~at params in
  equal
    ~msg:"a placed prior keeps its scales"
    (array float_exact)
    (to_array params.P.Gaussian.log_spatial_stds)
    (to_array moved.P.Gaussian.log_spatial_stds);
  equal
    ~msg:"a placed prior keeps its first-step multiplier"
    float_exact
    (item params.P.Gaussian.log_first_bin)
    (item moved.P.Gaussian.log_first_bin);
  (* an optional field is a subtree, and moves with the rest *)
  let dynamics = D.Linear.make ~at:Nx.Host.on ~n:2 ~m:3 () in
  let params = D.init dtype dynamics in
  let moved = D.Linear.place ~at params in
  equal
    ~msg:"a placed dynamics keeps its readout"
    (array float_exact)
    (to_array params.D.Linear.w)
    (to_array moved.D.Linear.w);
  (match params.D.Linear.b, moved.D.Linear.b with
   | Some b, Some moved_b ->
     equal ~msg:"the input projection" (array float_exact) (to_array b) (to_array moved_b)
   | None, None -> ()
   | _ -> fail "a placed dynamics changed its optional input projection");
  (* the initial-state prior and the recognition's blocks too *)
  let initial = I.init ~at dtype (I.make ~n:2 ()) in
  let moved_initial = I.place ~at initial in
  equal
    ~msg:"a placed initial-state prior keeps its mean"
    (array float_exact)
    (to_array initial.I.mean)
    (to_array moved_initial.I.mean)

let test_structures () =
  let model, params = setup ~n:2 ~m:1 ~t_steps:3 ~n_output:2 () in
  expect (visits (M.structure model) params)
  @@ Stdlib.__POS_OF__
       {|
    dynamics.w: a leaf
    dynamics.b: Some
    dynamics.b: a leaf
    likelihood.c: a leaf
    likelihood.bias: a leaf
    likelihood.log_variances: a leaf
    prior.log_spatial_stds: a leaf
    prior.log_first_bin: a leaf
    encoder_prior.log_spatial_stds: a leaf
    encoder_prior.log_first_bin: a leaf
    initial_prior: None
    recognition.space_cov.d: a leaf
    recognition.space_cov.t: a leaf
    recognition.time_cov.d: a leaf
    recognition.time_cov.t: a leaf
    recognition.initial_log_scale: None
  |};
  expect (visits (R.gparams_structure (M.generative model)) (M.gparams params))
  @@ Stdlib.__POS_OF__
       {|
    dynamics.w: a leaf
    dynamics.b: Some
    dynamics.b: a leaf
    likelihood.c: a leaf
    likelihood.bias: a leaf
    likelihood.log_variances: a leaf
    encoder_prior.log_spatial_stds: a leaf
    encoder_prior.log_first_bin: a leaf
    initial_prior: None
  |};
  let model, params = setup_lfads ~n:2 ~m:2 ~t_steps:3 ~n_output:2 () in
  expect (visits (M.structure model) params)
  @@ Stdlib.__POS_OF__
       {|
    dynamics.uf: a leaf
    dynamics.uh: a leaf
    dynamics.w: a leaf
    dynamics.bf: a leaf
    dynamics.bh: a leaf
    likelihood.c: a leaf
    likelihood.bias: a leaf
    likelihood.log_variances: a leaf
    prior.log_spatial_stds: a leaf
    prior.log_first_bin: a leaf
    encoder_prior.log_spatial_stds: a leaf
    encoder_prior.log_first_bin: a leaf
    initial_prior: Some
    initial_prior.mean: a leaf
    initial_prior.log_stds: a leaf
    recognition.ic_fwd.uf: a leaf
    recognition.ic_fwd.uh: a leaf
    recognition.ic_fwd.w: a leaf
    recognition.ic_fwd.bf: a leaf
    recognition.ic_fwd.bh: a leaf
    recognition.ic_bwd.uf: a leaf
    recognition.ic_bwd.uh: a leaf
    recognition.ic_bwd.w: a leaf
    recognition.ic_bwd.bf: a leaf
    recognition.ic_bwd.bh: a leaf
    recognition.ci_fwd.uf: a leaf
    recognition.ci_fwd.uh: a leaf
    recognition.ci_fwd.w: a leaf
    recognition.ci_fwd.bf: a leaf
    recognition.ci_fwd.bh: a leaf
    recognition.ci_bwd.uf: a leaf
    recognition.ci_bwd.uh: a leaf
    recognition.ci_bwd.w: a leaf
    recognition.ci_bwd.bf: a leaf
    recognition.ci_bwd.bh: a leaf
    recognition.con.uf: a leaf
    recognition.con.uh: a leaf
    recognition.con.w: a leaf
    recognition.con.bf: a leaf
    recognition.con.bh: a leaf
    recognition.ic_fwd_init: a leaf
    recognition.ic_bwd_init: a leaf
    recognition.ci_fwd_init: a leaf
    recognition.ci_bwd_init: a leaf
    recognition.con_init: a leaf
    recognition.ic_readout: a leaf
    recognition.ic_readout_bias: a leaf
    recognition.u_readout: a leaf
    recognition.u_readout_bias: a leaf
  |}

(* ────────────────────────────────────────────────────────────────────────── *)

let () =
  Stdlib.exit
    (run
       "ilqr_vae"
       [ test "covariance factors" test_covariance
       ; test "gaussian densities" test_density
       ; test "initial-state prior" test_initial_prior
       ; test "initial-state recognition" test_initial_state_recognition
       ; test "initial state is the mode" test_initial_state_is_the_mode
       ; test "initial-state training smoke" test_initial_state_training_smoke
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
       ; test "batched iLQR elbo" test_batched_iilqr
       ; test "batched LFADS elbo" test_batched_lfads
       ; test "recognition swapping" test_swapping
       ; test "training smoke test" test_training_smoke
       ; test "lfads training smoke test" test_lfads_training_smoke
       ; test "input prior terms" test_input_prior_terms
       ; test "shape errors" test_shape_errors
       ; test "placement" test_placement
       ; test "placing parameters" test_place
       ; test "structures" test_structures
       ])
