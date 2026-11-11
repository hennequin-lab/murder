(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The MPI driver's suite.

   The driver is an MPI program: run by `dune runtest` it is a single rank of
   `MPI_COMM_WORLD`, and every test below then checks the single-process case.
   The distributed case needs a launcher, so it is not part of the gate:

   {v
     dune build packages/ilqr-vae/test/mpi/test_mpi.exe
     mpirun -np 4 ./_build/default/packages/ilqr-vae/test/mpi/test_mpi.exe
   v}

   Every rank runs the whole suite, and the tests that compare a distributed
   step with the sequential batch objective are the real multi-rank check; the
   rest hold on every rank. The models are two-dimensional systems over three
   steps, so a multi-rank run fits on a login node. *)

open Base
open Nx
open Windtrap
module M = Ilqr_vae.Model
module El = Ilqr_vae.Elbo
module R = Ilqr_vae.Recognition
module D = Ilqr_vae.Dynamics
module L = Ilqr_vae.Likelihood
module P = Ilqr_vae.Prior
module Schedule = Ilqr_vae.Schedule
module I = Ilqr_vae_mpi

let dtype = float64

type linear_params = (Nx.host Nx.float64_t, Nx.host) D.Linear.params
type lik_params = (Nx.host Nx.float64_t, Nx.host) L.Gaussian.params
type pri_params = (Nx.host Nx.float64_t, Nx.host) P.Gaussian.params
type rec_params = (Nx.host Nx.float64_t, Nx.host) R.Iilqr.params

type model =
  ( Nx.float64_elt
    , linear_params
    , lik_params
    , pri_params
    , pri_params
    , rec_params
    , Nx.host )
    M.t

type params =
  ( Nx.float64_elt
    , linear_params
    , lik_params
    , pri_params
    , pri_params
    , rec_params
    , Nx.host )
    M.params

(* The smallest model the driver can be useful on: a linear dynamics, a
   Gaussian likelihood and an iLQR recognition, all of them trainable. *)
let setup ?(key = Nx.Rng.key 0) ~n ~m ~t_steps ~n_output () : model * params =
  let at = Nx.Host.on in
  let dynamics = D.Linear.make ~at ~n ~m () in
  let likelihood = L.Gaussian.make ~at ~n ~n_output () in
  let prior = P.Gaussian.make ~at ~m ~spatial_std:1.0 () in
  let encoder_prior = P.Gaussian.make ~at ~m ~spatial_std:1.0 () in
  let recognition = R.Iilqr.make ~at ~n ~m ~n_steps:t_steps ~n_output () in
  let model = M.make dtype ~dynamics ~likelihood ~prior ~encoder_prior ~recognition () in
  model, M.init ~at ~key model

let data ~key shape = Nx.Rng.normal_at Nx.Host.on ~key dtype shape

(* A comparison of two parameter trees, leaf by leaf. Values are read at
   float64, which is exact for the float64 leaves the test models have.
   [exact] is for the claims that the driver must not round at all: a
   parameter that crosses the ranks unchanged. *)
let check_leaf ~msg ~exact ~tol expected got =
  let witness =
    if exact then array float_exact else array (float_rel ~rel:tol ~abs:tol)
  in
  equal
    ~msg
    witness
    (Nx.to_array (Nx.cast float64 expected))
    (Nx.to_array (Nx.cast float64 got))

let check_params ~msg ~exact ~tol structure expected got =
  ignore
    (Nx.Ptree.map2
       structure
       (fun path expected got ->
          check_leaf
            ~msg:(msg ^ " at " ^ Nx.Ptree.Path.to_string path)
            ~exact
            ~tol
            expected
            got;
          got)
       expected
       got)

let check_float ~msg ~tol expected got =
  equal ~msg (float_rel ~rel:tol ~abs:tol) expected got

(* The sequential objective the distributed one must reproduce: the batch's
   mean negative ELBO and its gradient, from one process. *)
let batch_objective model params ?beta ~key ~iters ~n_samples data =
  Rune.value_and_grad
    (M.structure model)
    (fun p -> El.loss_fixed model p ?beta ~iters ~key ~n_samples data)
    params

let step_capture driver params ~key ?beta ~data update =
  let grads = ref None in
  let guarded params grads' =
    if not (I.is_root driver) then fail "update was called on a rank that is not the root";
    grads := Some grads';
    update params grads'
  in
  let params', loss = I.step driver params ~key ?beta ~data guarded in
  params', loss, !grads

(* ──────────────────────────────────────────────────────────────────────────
   The split
   ────────────────────────────────────────────────────────────────────────── *)

let test_partition =
  let covers ~trials ~ranges =
    let rec go expected = function
      | [] -> Int.equal expected trials
      | range :: rest ->
        Int.equal range.I.first expected && go (expected + range.I.count) rest
    in
    go 0 (Array.to_list ranges)
  in
  group
    "the split"
    [ prop
        "the shares are contiguous, ordered and cover every trial"
        Gen.(pair (int_range 1 12) (int_range 0 60))
        (fun (ranks, trials) ->
           let ranges =
             Array.init ranks ~f:(fun rank -> I.partition ~ranks ~trials ~rank)
           in
           let total =
             Array.fold ranges ~init:0 ~f:(fun total range -> total + range.I.count)
           in
           equal int trials total;
           if not (covers ~trials ~ranges)
           then
             fail
               (Printf.sprintf
                  "the shares of %d trials over %d ranks are not contiguous"
                  trials
                  ranks);
           (* the shares differ by at most one trial *)
           let smallest =
             Array.fold ranges ~init:Int.max_value ~f:(fun smallest range ->
               Int.min smallest range.I.count)
           in
           Array.iter ranges ~f:(fun range ->
             if range.I.count - smallest > 1
             then
               fail
                 (Printf.sprintf
                    "rank shares of %d trials over %d ranks differ by more than one: %d \
                     and %d"
                    trials
                    ranks
                    range.I.count
                    smallest)))
    ; test "rejects bad arguments" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"ranks must be positive") (fun () ->
          I.partition ~ranks:0 ~trials:3 ~rank:0);
        raises_match (Exn.invalid_arg ~sub:"rank must be in") (fun () ->
          I.partition ~ranks:2 ~trials:3 ~rank:2);
        raises_match (Exn.invalid_arg ~sub:"trials must be non-negative") (fun () ->
          I.partition ~ranks:1 ~trials:(-1) ~rank:0))
    ]

(* ──────────────────────────────────────────────────────────────────────────
   One step
   ────────────────────────────────────────────────────────────────────────── *)

let iters = 4
let n_samples = 3

let test_step_is_the_batch_objective () =
  let n = 2
  and m = 2
  and t_steps = 3
  and n_output = 2 in
  let model, params = setup ~n ~m ~t_steps ~n_output () in
  let driver = I.make model ~iters ~n_samples () in
  let ranks = I.size driver in
  (* One more trial than a multiple of the ranks, so the split is uneven
     whenever there is more than one rank. *)
  let trials = (3 * ranks) + 1 in
  let data = data ~key:(Nx.Rng.key 11) [| trials; t_steps; n_output |] in
  let key = Nx.Rng.key 12 in
  let beta = Nx.scalar dtype 0.5 in
  let expected_loss, expected_grads =
    batch_objective model params ~key ~beta ~iters ~n_samples data
  in
  let identity params _ = params in
  let params', loss, grads = step_capture driver params ~key ~beta ~data identity in
  check_float
    ~msg:"the distributed loss is the batch's"
    ~tol:1e-12
    loss
    (Nx.item [] expected_loss);
  check_params
    ~msg:"an identity update leaves the parameters unchanged"
    ~exact:true
    ~tol:0.0
    (M.structure model)
    params
    params';
  match grads with
  | None -> if I.is_root driver then fail "the root did not receive the gradient"
  | Some grads ->
    check_params
      ~msg:"the averaged gradient is the batch's"
      ~exact:false
      ~tol:1e-10
      (M.structure model)
      expected_grads
      grads

(* Trials spread over leading axes are trials; the driver flattens them. *)
let test_step_flattens_leading_axes () =
  let t_steps = 3
  and n_output = 2 in
  let model, params = setup ~n:2 ~m:2 ~t_steps ~n_output () in
  let driver = I.make model ~iters ~n_samples () in
  let data = data ~key:(Nx.Rng.key 13) [| 2; 3; t_steps; n_output |] in
  let key = Nx.Rng.key 14 in
  let expected_loss, _ = batch_objective model params ~key ~iters ~n_samples data in
  let _, loss, _ = step_capture driver params ~key ~data (fun p _ -> p) in
  check_float
    ~msg:"a batch with two leading trial axes is one batch"
    ~tol:1e-12
    loss
    (Nx.item [] expected_loss)

(* A rank can own no trial at all, and must still take part in the reductions. *)
let test_step_with_fewer_trials_than_ranks () =
  let t_steps = 3
  and n_output = 2 in
  let model, params = setup ~n:2 ~m:2 ~t_steps ~n_output () in
  let driver = I.make model ~iters ~n_samples () in
  let trials = Int.max 1 (I.size driver - 1) in
  let data = data ~key:(Nx.Rng.key 15) [| trials; t_steps; n_output |] in
  let key = Nx.Rng.key 16 in
  let expected_loss, expected_grads =
    batch_objective model params ~key ~iters ~n_samples data
  in
  let _, loss, grads = step_capture driver params ~key ~data (fun p _ -> p) in
  check_float
    ~msg:"fewer trials than ranks still average the batch"
    ~tol:1e-12
    loss
    (Nx.item [] expected_loss);
  match grads with
  | None -> if I.is_root driver then fail "the root did not receive the gradient"
  | Some grads ->
    check_params
      ~msg:"the gradient of a short batch"
      ~exact:false
      ~tol:1e-10
      (M.structure model)
      expected_grads
      grads

(* The root's update is what every rank holds afterwards, and the callback is
   called nowhere else. *)
let test_step_broadcasts_the_update () =
  let model, params = setup ~n:2 ~m:2 ~t_steps:3 ~n_output:2 () in
  let driver = I.make model ~iters ~n_samples () in
  let data = data ~key:(Nx.Rng.key 17) [| 2; 3; 2 |] in
  let key = Nx.Rng.key 18 in
  let shift = 0.25 in
  let shift_leaf : type v s. (v, s, Nx.host) Nx.t -> (v, s, Nx.host) Nx.t =
    fun leaf ->
    match Nx.dtype leaf with
    | Nx.Dtype.Float64 -> Nx.Infix.(leaf +$ shift)
    | _ -> leaf
  in
  let update params _ =
    Nx.Ptree.map (M.structure model) (fun _ leaf -> shift_leaf leaf) params
  in
  let expected = update params params in
  let params', _loss, _grads = step_capture driver params ~key ~data update in
  check_params
    ~msg:"every rank returns the root's parameters"
    ~exact:true
    ~tol:0.0
    (M.structure model)
    expected
    params'

(* ──────────────────────────────────────────────────────────────────────────
   The loop
   ────────────────────────────────────────────────────────────────────────── *)

let test_run () =
  let t_steps = 3
  and n_output = 2 in
  let model, params = setup ~n:2 ~m:2 ~t_steps ~n_output () in
  let driver = I.make model ~iters ~n_samples () in
  let steps = 3 in
  let batches =
    Array.init steps ~f:(fun i ->
      data ~key:(Nx.Rng.key (100 + i)) [| 2; t_steps; n_output |])
  in
  let key = Nx.Rng.key 19 in
  let schedule = Schedule.linear_warmup ~steps:2 dtype in
  let params', losses =
    I.run
      driver
      params
      ~steps
      ~key
      ~beta:(Schedule.at schedule)
      ~data:(fun i -> batches.(i))
      (fun p _ -> p)
  in
  check_params
    ~msg:"the loop's parameters are the last step's"
    ~exact:true
    ~tol:0.0
    (M.structure model)
    params
    params';
  equal int steps (Array.length losses);
  Array.iteri losses ~f:(fun i loss ->
    let beta = Schedule.at schedule i in
    let expected =
      El.loss_fixed
        model
        params
        ~beta
        ~iters
        ~key:(Nx.Rng.fold_in key i)
        ~n_samples
        batches.(i)
    in
    check_float
      ~msg:(Printf.sprintf "step %d's loss" i)
      ~tol:1e-12
      loss
      (Nx.item [] expected));
  (* no beta means the ELBO's default weight of one *)
  let _, plain =
    I.run driver params ~steps:1 ~key ~data:(fun _ -> batches.(0)) (fun p _ -> p)
  in
  let expected =
    El.loss_fixed model params ~iters ~key:(Nx.Rng.fold_in key 0) ~n_samples batches.(0)
  in
  check_float
    ~msg:"a run without beta weighs the KL by one"
    ~tol:1e-12
    plain.(0)
    (Nx.item [] expected);
  raises_match (Exn.invalid_arg ~sub:"steps must be non-negative") (fun () ->
    I.run driver params ~steps:(-1) ~key ~data:(fun _ -> batches.(0)) (fun p _ -> p))

(* A float32 model crosses the ranks at float64 and comes back at float32;
   the round trip is exact for every value float32 can hold. *)
let test_float32_moves_parameters () =
  let at = Nx.Host.on in
  let n = 2
  and m = 2
  and t_steps = 3
  and n_output = 2 in
  let model =
    M.make
      Nx.float32
      ~dynamics:(D.Linear.make ~at ~n ~m ())
      ~likelihood:(L.Gaussian.make ~at ~n ~n_output ())
      ~prior:(P.Gaussian.make ~at ~m ~spatial_std:1.0 ())
      ~encoder_prior:(P.Gaussian.make ~at ~m ~spatial_std:1.0 ())
      ~recognition:(R.Iilqr.make ~at ~n ~m ~n_steps:t_steps ~n_output ())
      ()
  in
  let params = M.init ~at ~key:(Nx.Rng.key 25) model in
  let driver = I.make model ~iters:2 ~n_samples:2 () in
  let data =
    Nx.Rng.normal_at at ~key:(Nx.Rng.key 26) Nx.float32 [| 2; t_steps; n_output |]
  in
  let params', _loss = I.step driver params ~key:(Nx.Rng.key 27) ~data (fun p _ -> p) in
  check_params
    ~msg:"a float32 model's parameters survive the exchange"
    ~exact:true
    ~tol:0.0
    (M.structure model)
    params
    params'

(* A training run: the driver's step, an optimizer of the caller's, and a
   loss that falls. The update is what the root runs, so the loop is the same
   on every rank and the parameters stay in step. *)
let test_training () =
  let model, params = setup ~n:2 ~m:2 ~t_steps:4 ~n_output:2 () in
  let driver = I.make model ~iters ~n_samples () in
  let trials =
    stack
      ~axis:0
      (List.init 3 ~f:(fun i ->
         (Ilqr_vae.sample_generative model params ~key:(Nx.Rng.key (200 + i)))
           .Ilqr_vae.observations))
  in
  let key = Nx.Rng.key 21 in
  let objective p = El.loss_fixed model p ~key ~n_samples:4 ~iters:4 trials in
  let initial = Nx.item [] (objective params) in
  (* the same draw every step: the comparison is between two values of one
     objective, not of a noisy estimate *)
  let sgd lr params grads =
    Nx.Ptree.map2
      (M.structure model)
      (fun _ p g -> Nx.Infix.(p - (Nx.cast (Nx.dtype p) (Nx.scalar Nx.float64 lr) * g)))
      params
      grads
  in
  let params = ref params in
  for _ = 1 to 30 do
    let next, _ = I.step driver !params ~key ~data:trials (sgd 0.02) in
    params := next
  done;
  let final = Nx.item [] (objective !params) in
  less ~msg:"training with the driver decreases the loss" float_exact ~than:initial final

(* ──────────────────────────────────────────────────────────────────────────
   Bad arguments
   ────────────────────────────────────────────────────────────────────────── *)

let test_errors () =
  let model, params = setup ~n:2 ~m:2 ~t_steps:3 ~n_output:2 () in
  raises_match (Exn.invalid_arg ~sub:"iters must be positive") (fun () ->
    I.make model ~iters:0 ~n_samples:1 ());
  raises_match (Exn.invalid_arg ~sub:"n_samples must be positive") (fun () ->
    I.make model ~iters:1 ~n_samples:0 ());
  let driver = I.make model ~iters ~n_samples () in
  let key = Nx.Rng.key 20 in
  let identity params _ = params in
  (* one trial is not a mini-batch: the batch axis is what the ranks split *)
  raises_match (Exn.invalid_arg ~sub:"data must be") (fun () ->
    I.step driver params ~key ~data:(zeros dtype [| 3; 2 |]) identity);
  raises_match (Exn.invalid_arg ~sub:"data must be") (fun () ->
    I.step driver params ~key ~data:(zeros dtype [| 2; 4; 2 |]) identity);
  raises_match (Exn.invalid_arg ~sub:"data must be") (fun () ->
    I.step driver params ~key ~data:(zeros dtype [| 2; 3; 3 |]) identity);
  raises_match (Exn.invalid_arg ~sub:"every trial axis") (fun () ->
    I.step driver params ~key ~data:(zeros dtype [| 0; 3; 2 |]) identity)

(* ────────────────────────────────────────────────────────────────────────── *)

let () =
  Stdlib.exit
    (run
       "ilqr_vae_mpi"
       [ test_partition
       ; test "a step is the batch objective" test_step_is_the_batch_objective
       ; test "a step flattens leading trial axes" test_step_flattens_leading_axes
       ; test "a step with fewer trials than ranks" test_step_with_fewer_trials_than_ranks
       ; test "a step broadcasts the update" test_step_broadcasts_the_update
       ; test "the loop" test_run
       ; test "a float32 model's parameters" test_float32_moves_parameters
       ; test "training" test_training
       ; test "bad arguments" test_errors
       ])
