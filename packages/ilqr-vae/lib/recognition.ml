(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Types

(* A recognition model is the [q] of the VAE: a distribution over a trial's
   latent variables, conditioned on its observations. What varies between
   iLQR-VAE and LFADS is only how that distribution is built, so the two live
   behind one interface here and a model carries whichever of them the user
   chose.

   {1 The latents}

   A trial's latents are the initial state [x0] (shape [n]) and the inputs
   [u_0 … u_{T-1}] (shape [T; m]). A model built without an initial-state
   prior fixes [x_0 = 0] and iLQR-style recognition infers only the inputs;
   an LFADS-style recognition infers the initial state as well. A sample
   batch is {!type-samples}: [x0] of shape [[n_samples; n]] and [u] of shape
   [[n_steps; n_samples; m]].

   {1 What a recognition provides}

   - {!val-infer}: reparameterized draws of the latents together with the
     value of [log q] at those draws;
   - {!val-mean}: the posterior mean, for prediction and diagnostics;
   - {!val-log_prob}: [log q] at given latents, for diagnostics, importance
     weighting, or a KL estimated from a fixed set of draws.

   {!val-log_prob} takes the trial's data because both recognitions condition
   on it: iLQR's mean solves a control problem on it, and LFADS's encoder
   reads it.

   {1 Parameters}

   A recognition owns its parameters, in its own record type: the covariance
   factors for iLQR, the encoder and controller networks for LFADS. They are
   a field of {!Model.params} and are learned together with the rest of the
   model. The recognition is a value, not a module: {!module-Iilqr} and
   {!module-Lfads} are constructors of such values, and a new architecture is
   a record literal. *)

(* {1 Interface types} *)

(** The generative model as a recognition sees it: the dimensions, the
    dynamics, the likelihood and the priors. A subset of {!Model.t}, built by
    {!Model.generative}, so that this module does not depend on the model. *)
type ('w, 'd, 'l, 'p, 'e) generative =
  { dtype : (float, 'w) Nx.dtype
  ; n : int
  ; m : int
  ; n_steps : int
  ; n_output : int
  ; dynamics : ('w, 'd) Dynamics.t
  ; likelihood : ('w, 'l) Likelihood.t
  ; prior : ('w, 'p) Prior.t (** the generative prior over inputs *)
  ; encoder_prior : ('w, 'e) Prior.t (** the prior of the iLQR cost, [p̃] *)
  }

(** The corresponding parameters, a subset of {!Model.params}. The type
    parameter ['w] is phantom; it keeps the shape of the type parallel to
    {!type-generative}. *)
type ('w, 'd, 'l, 'p, 'e) gparams =
  { dynamics : 'd
  ; likelihood : 'l
  ; prior : 'p
  ; encoder_prior : 'e
  }

(** One reparameterized draw of a trial's latents: the initial state, of shape
    [[n_samples; n]], and the inputs, of shape [[n_steps; n_samples; m]]. *)
type 'w samples =
  { x0 : 'w value
  ; u : 'w value
  }

(** A draw together with the value of [log q] at it, of shape
    [[n_samples]] — one scalar per draw. This is what {!Elbo} consumes. *)
type 'w posterior =
  { x0 : 'w value
  ; u : 'w value
  ; log_prob : 'w value
  }

(** The dimensions a recognition is built for, checked against the model's by
    {!Model.make}. *)
type dims =
  { n : int
  ; m : int
  ; n_steps : int
  ; n_output : int
  }

(** A recognition model over parameters ['r]. Everything but {!type-dims},
    {!field-name} and {!field-infers_initial} is a function of the current
    parameters, the way the constructors of {!Dynamics}, {!Prior} and
    {!Likelihood} are. *)
type ('w, 'd, 'l, 'p, 'e, 'r) t =
  { dims : dims
  ; infers_initial : bool
    (** [true] when [q] carries an initial-state factor, so that
        {!Model.make} can require a matching initial-state prior. *)
  ; name : string (** for diagnostics *)
  ; structure : 'r Nx.Ptree.t (** the structure of the parameters *)
  ; params : 'r (** the parameters a fresh model starts from *)
  ; infer :
      ('w, 'd, 'l, 'p, 'e) generative
      -> ('w, 'd, 'l, 'p, 'e) gparams
      -> params:'r
      -> ?iters:int
      -> n_samples:int
      -> key:Nx.Rng.t
      -> 'w value
      -> 'w posterior
    (** [infer gen gp ~params ~n_samples ~key data] draws [n_samples] latent
        trajectories from [q(· | data)] and evaluates [log q] at each of them.
        [iters] is a hint for recognitions with an iterative inner solver:
        [Some n] asks for [n] fixed iterations (traceable), [None] for the
        default complete solve. Recognitions without one ignore it. *)
  ; mean :
      ('w, 'd, 'l, 'p, 'e) generative
      -> ('w, 'd, 'l, 'p, 'e) gparams
      -> params:'r
      -> ?iters:int
      -> 'w value
      -> 'w samples
    (** [mean gen gp ~params data] is the posterior mean — a single
        deterministic latent trajectory, shaped like one member of a sample
        batch ([n_samples = 1]). *)
  ; log_prob :
      ('w, 'd, 'l, 'p, 'e) generative
      -> ('w, 'd, 'l, 'p, 'e) gparams
      -> params:'r
      -> ?iters:int
      -> 'w value
      -> 'w samples
      -> 'w value
    (** [log_prob gen gp ~params data s] is [log q(s | data)], summed over
        the latents: one value per member of a batch. *)
  }

(* {1 Helpers} *)

let log_2pi = Float.log (2.0 *. Float.pi)
let infer (type w d l p e r) (recog : (w, d, l, p, e, r) t) = recog.infer
let mean (type w d l p e r) (recog : (w, d, l, p, e, r) t) = recog.mean
let log_prob (type w d l p e r) (recog : (w, d, l, p, e, r) t) = recog.log_prob

(** [gparams_structure gen] is the [Nx.Ptree.t] of {!type-gparams} at [gen]:
    the same structure the model gives its own parameters, with only the
    paths a recognition reads. *)
let gparams_structure (type w d l p e) (gen : (w, d, l, p, e) generative)
  : (w, d, l, p, e) gparams Nx.Ptree.t
  =
  Nx.Ptree.iso
    (fun (d, (l, (p, e))) ->
       { dynamics = d; likelihood = l; prior = p; encoder_prior = e })
    (fun r -> r.dynamics, (r.likelihood, (r.prior, r.encoder_prior)))
    Nx.Ptree.(
      pair
        (field "dynamics" gen.dynamics.Dynamics.structure)
        (pair
           (field "likelihood" gen.likelihood.Likelihood.structure)
           (pair
              (field "prior" gen.prior.Prior.structure)
              (field "encoder_prior" gen.encoder_prior.Prior.structure))))

let check_data ~n_steps ~n_output data =
  let shape = Nx.shape data in
  if
    Array.length shape <> 2
    || (not (Int.equal shape.(0) n_steps))
    || not (Int.equal shape.(1) n_output)
  then invalid_arg (Printf.sprintf "Recognition: data must be [%d; %d]" n_steps n_output)

(* [scaled_randn ?key ~sigma dtype shape] is a normal draw of standard
   deviation [sigma], from [key] when given and from the current random scope
   otherwise. *)
let scaled_randn ?key ~sigma dtype shape =
  let draw =
    match key with
    | Some key -> Nx.Rng.normal key dtype shape
    | None -> randn dtype shape
  in
  Infix.(draw *$ sigma)

(* [split2 ~axis t] splits [t] into two equal halves along [axis]. The
   readouts below always emit a mean and a log-variance head side by side. *)
let split2 ~axis t =
  match split ~axis 2 t with
  | [ a; b ] -> a, b
  | _ -> assert false

(* [gaussian_logp ~x ~mu ~logvar] is the diagonal Gaussian log density of [x]
   summed over the last axis: one value per row of a batch. *)
let gaussian_logp ~x ~mu ~logvar =
  let open Infix in
  let scaled = (x - mu) / exp (logvar *$ 0.5) in
  sum ~axes:[ -1 ] ((scaled * scaled) + (logvar + scalar_like logvar log_2pi)) *$ -0.5

(* {1 iLQR recognition} *)

(** The iLQR-VAE recognition: the mean is the solution of a finite-horizon
    optimal control problem, and the covariance is the Kronecker product of a
    factor over time and a factor over input dimensions. The initial state is
    fixed at zero, so {!Model} is built without an initial-state prior. *)
module Iilqr = struct
  type 'a params =
    { space_cov : 'a Covariance.params (** the factor over input dimensions *)
    ; time_cov : 'a Covariance.params (** the factor over the horizon *)
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type 'a t = 'a params

        let walk = walk_params
      end)

  (** [init ?space_sigma2 ?time_sigma2 ~n ~m ~n_steps ~n_output dtype] is the
      recognition's initial parameters: two covariance factors of
      [space_sigma2 · I] and [time_sigma2 · I], both [1.] by default. The
      dimensions only say what the factors must multiply; they are checked
      against the model's by {!make}. *)
  let init ?space_sigma2 ?time_sigma2 ~n ~m ~n_steps ~n_output dtype =
    if n <= 0 then invalid_arg "Recognition.Iilqr.init: n must be positive";
    if m <= 0 then invalid_arg "Recognition.Iilqr.init: m must be positive";
    if n_steps <= 0 then invalid_arg "Recognition.Iilqr.init: n_steps must be positive";
    if n_output <= 0 then invalid_arg "Recognition.Iilqr.init: n_output must be positive";
    { space_cov = Covariance.make ?sigma2:space_sigma2 ~n:m dtype
    ; time_cov = Covariance.make ?sigma2:time_sigma2 ~n:n_steps dtype
    }

  (* [controls_of gen tau] is the control part of a stacked trajectory: rows
     [0 … n_steps − 1], columns [n … n + m − 1]. *)
  let controls_of (type w) (gen : (w, _, _, _, _) generative) (tau : w value) =
    slice [ R (0, gen.n_steps); R (gen.n, gen.n + gen.m) ] tau

  let initial_controls (type w) (gen : (w, _, _, _, _) generative) u_init =
    match u_init with
    | None -> Array.init gen.n_steps ~f:(fun _ -> zeros gen.dtype [| gen.m |])
    | Some u ->
      if
        Array.length (Nx.shape u) <> 2
        || (not (Int.equal (Nx.shape u).(0) gen.n_steps))
        || not (Int.equal (Nx.shape u).(1) gen.m)
      then invalid_arg "Recognition.Iilqr: u_init must be [n_steps; m]";
      Array.init gen.n_steps ~f:(fun k -> get [ k ] u)

  (** [problem gen gp data] is the encoder's control problem for one trial:
      horizon [n_steps], state [n], control [m], dynamics from
      [gen.dynamics], running loss the encoder prior minus the likelihood and
      terminal loss the last observation's likelihood. The trial's data is
      captured by the losses. The problem's parameters are the generative
      share of the model's, so {!Dilqr} differentiates the solve with respect
      to the dynamics, the likelihood and the encoder prior. *)
  let problem
        (type w d l p e)
        (gen : (w, d, l, p, e) generative)
        (_gp : (w, d, l, p, e) gparams)
        (data : w value)
    : (w, (w, d, l, p, e) gparams) Dilqr.Problem.t
    =
    let n = gen.n
    and m = gen.m
    and t_steps = gen.n_steps
    and n_output = gen.n_output
    and dtype = gen.dtype in
    check_data ~n_steps:t_steps ~n_output data;
    let zero = scalar dtype 0.0
    and one = scalar dtype 1.0 in
    let last = get [ t_steps - 1 ] data in
    (* The running loss at step [k] carries the encoder prior on [u_k] and —
       from step 1 on — the likelihood of [o_{k−1}] at [x_k]; the terminal
       loss carries the last observation. [o] is selected at the traced step
       index with [Nx.take], clamped so that step 0 reads a valid row and is
       then masked out. *)
    let running_loss
          (theta : (w, d, l, p, e) gparams)
          ~(k : w value)
          ~(x : w value)
          ~(u : w value)
      =
      let prior_term =
        Infix.(-gen.encoder_prior.Prior.logp_t theta.encoder_prior ~k ~u)
      in
      let index = maximum Infix.(k - one) zero in
      (* [take] with a 0-d [indices] removes the axis, so the row is
         [n_output]. *)
      let o = take ~axis:0 ~indices:(cast int64 index) data in
      let likelihood_term =
        Infix.(-gen.likelihood.Likelihood.logp_t theta.likelihood ~k ~x ~o)
      in
      Infix.(prior_term + where (Nx.equal k zero) zero likelihood_term)
    in
    let final_loss (theta : (w, d, l, p, e) gparams) ~(x : w value) =
      neg
        (gen.likelihood.Likelihood.logp_t
           theta.likelihood
           ~k:(scalar dtype (Float.of_int t_steps))
           ~x
           ~o:last)
    in
    { Dilqr.Problem.dtype
    ; n
    ; m
    ; horizon = t_steps
    ; verbose = false
    ; params = gparams_structure gen
    ; dynamics = (fun theta ~k ~x ~u -> gen.dynamics.Dynamics.dyn theta.dynamics ~k ~x ~u)
    ; running_loss
    ; final_loss
    }

  (** [posterior_mean gen gp ?u_init ?max_iter ?conv_threshold data] is the
      adaptive iLQR solution's controls, of shape [[n_steps; m]] — the
      recognition mean. [u_init] ([n_steps; m], zeros when omitted) is the
      initial guess and carries no gradient: the idealized solve does not
      depend on it. Eager only. *)
  let posterior_mean gen gp ?u_init ?max_iter ?conv_threshold data =
    let p = problem gen gp data in
    let inputs : (_, _) Dilqr.Problem.Inputs.t =
      { theta = gp; x0 = zeros gen.dtype [| gen.n |]; us0 = initial_controls gen u_init }
    in
    controls_of gen (Dilqr.solve_adaptive p ?max_iter ?conv_threshold inputs)

  (** [posterior_mean_fixed gen gp ?u_init ~iters ?alphas ?mu ?c1 data] is
      the same mean with [iters] fixed iterations: tensor-only control flow,
      traceable and JIT-compilable together with its implicit VJP. *)
  let posterior_mean_fixed gen gp ?u_init ~iters ?alphas ?mu ?c1 data =
    let p = problem gen gp data in
    let inputs : (_, _) Dilqr.Problem.Inputs.t =
      { theta = gp; x0 = zeros gen.dtype [| gen.n |]; us0 = initial_controls gen u_init }
    in
    controls_of gen (Dilqr.solve_fixed p ~iters ?alphas ?mu ?c1 inputs)

  (* [sample_u gen p ~mu ~n_samples ~key] draws [u = mu + L_time xi L_spaceᵀ] with [xi] standard normal: [L_time] acts on the time axis and
     [L_spaceᵀ] on the input axis, so the draws have covariance
     [Σ_time ⊗ Σ_space]. *)
  let sample_u
        (type w)
        (gen : (w, _, _, _, _) generative)
        (p : w value params)
        ~mu
        ~n_samples
        ~key
    =
    if n_samples <= 0
    then invalid_arg "Recognition.Iilqr.sample_u: n_samples must be positive";
    let t_steps = gen.n_steps
    and m = gen.m in
    let l_time = Covariance.factor p.time_cov in
    let l_space = Covariance.factor p.space_cov in
    let xi = Nx.Rng.normal key gen.dtype [| t_steps; n_samples; m |] in
    let cols = n_samples * m in
    let along_time = Infix.(l_time *@ reshape [| t_steps; cols |] xi) in
    let noise =
      Infix.(reshape [| t_steps; n_samples; m |] along_time *@ matrix_transpose l_space)
    in
    Infix.(unsqueeze ~axes:[ 1 ] mu + noise)

  (* [log_prob_u gen p ~mu u] is log q(u) for an input sequence of shape
     [[n_steps; m]], or one value per sequence for a batch
     [[n_steps; batch…; m]]. The quadratic form is evaluated with two
     triangular solves against the Cholesky factors. *)
  let log_prob_u (type w) (gen : (w, _, _, _, _) generative) (p : w value params) ~mu u =
    let t_steps = gen.n_steps
    and m = gen.m
    and dtype = gen.dtype in
    let shape = Nx.shape u in
    let rank = Array.length shape in
    if not (Int.equal shape.(0) t_steps && Int.equal shape.(rank - 1) m)
    then
      invalid_arg
        (Printf.sprintf
           "Recognition.Iilqr.log_prob: expected [n_steps; …; m] = [%d; …; %d]"
           t_steps
           m);
    let batch_dims = Array.sub shape ~pos:1 ~len:(rank - 2) in
    let batch = Array.fold batch_dims ~init:1 ~f:( * ) in
    let l_time = Covariance.factor p.time_cov in
    let l_space = Covariance.factor p.space_cov in
    let u3 = reshape (Array.append [| t_steps |] (Array.append [| batch |] [| m |])) u in
    let diff = Infix.(u3 - reshape [| t_steps; 1; m |] mu) in
    (* [(u − μ)ᵀ Σ⁻¹ (u − μ) = ‖L_time⁻¹ (u − μ) L_space⁻ᵀ‖²_F], two triangular
       solves: one down the time axis, one down the input axis. *)
    let along_time = reshape [| t_steps; batch * m |] diff in
    let solved_time = solve_triangular ~upper:true l_time along_time in
    let rows = reshape [| t_steps * batch; m |] solved_time in
    let solved_space =
      matrix_transpose (solve_triangular ~upper:true l_space (matrix_transpose rows))
    in
    let quadratic =
      reshape
        (Array.append [| t_steps |] batch_dims)
        (sum ~axes:[ -1 ] Infix.(solved_space * solved_space))
      |> sum ~axes:[ 0 ]
    in
    let log_det =
      Infix.(
        (Covariance.log_det p.time_cov *$ Float.of_int m)
        + (Covariance.log_det p.space_cov *$ Float.of_int t_steps))
    in
    let dim = Float.of_int (t_steps * m) in
    Infix.((quadratic + (log_det + scalar dtype (dim *. log_2pi))) *$ -0.5)

  let solve
        (type w d l p e)
        (gen : (w, d, l, p, e) generative)
        (gp : (w, d, l, p, e) gparams)
        ?iters
        data
    =
    match iters with
    | None -> posterior_mean gen gp data
    | Some iters -> posterior_mean_fixed gen gp ~iters data

  (* [stopped params] is the same parameters with every tensor detached. The
     density of [infer] is evaluated at it, so that the gradient of the bound
     reaches the recognition only through the sampled latents: the
     path-derivative (sticking-the-landing) estimator of the KL gradient. *)
  let stopped (type w) (params : w value params) : w value params =
    { space_cov =
        { Covariance.d = Rune.detach params.space_cov.d
        ; Covariance.t = Rune.detach params.space_cov.t
        }
    ; time_cov =
        { Covariance.d = Rune.detach params.time_cov.d
        ; Covariance.t = Rune.detach params.time_cov.t
        }
    }

  let infer
        (type w d l p e)
        (gen : (w, d, l, p, e) generative)
        (gp : (w, d, l, p, e) gparams)
        ~(params : w value params)
        ?iters
        ~n_samples
        ~key
        data
    =
    check_data ~n_steps:gen.n_steps ~n_output:gen.n_output data;
    let mu_u = solve gen gp ?iters data in
    let x0 = zeros gen.dtype [| gen.n |] in
    let u = sample_u gen params ~mu:mu_u ~n_samples ~key in
    let log_prob = log_prob_u gen (stopped params) ~mu:(Rune.detach mu_u) u in
    { x0 = broadcast_to [| n_samples; gen.n |] x0; u; log_prob }

  let mean
        (type w d l p e)
        (gen : (w, d, l, p, e) generative)
        (gp : (w, d, l, p, e) gparams)
        ~(params : w value params)
        ?iters
        data
    =
    let _ = params in
    check_data ~n_steps:gen.n_steps ~n_output:gen.n_output data;
    let mu_u = solve gen gp ?iters data in
    { x0 = reshape [| 1; gen.n |] (zeros gen.dtype [| gen.n |])
    ; u = reshape [| gen.n_steps; 1; gen.m |] mu_u
    }

  let log_prob
        (type w d l p e)
        (gen : (w, d, l, p, e) generative)
        (gp : (w, d, l, p, e) gparams)
        ~(params : w value params)
        ?iters
        data
        (samples : w samples)
    =
    check_data ~n_steps:gen.n_steps ~n_output:gen.n_output data;
    let mu_u = solve gen gp ?iters data in
    log_prob_u gen params ~mu:mu_u samples.u

  (** [make ?space_sigma2 ?time_sigma2 ~n ~m ~n_steps ~n_output dtype] is the
      iLQR recognition with covariance factors initialized to
      [space_sigma2 · I] and [time_sigma2 · I]. The dimensions are the
      model's; {!Model.make} checks them. *)
  let make ?space_sigma2 ?time_sigma2 ~n ~m ~n_steps ~n_output dtype =
    { dims = { n; m; n_steps; n_output }
    ; infers_initial = false
    ; name = "ilqr"
    ; structure = structure ()
    ; params = init ?space_sigma2 ?time_sigma2 ~n ~m ~n_steps ~n_output dtype
    ; infer
    ; mean
    ; log_prob
    }
end

(* {1 LFADS recognition} *)

(** The LFADS recognition (Pandarinath et al., 2018): the initial state is
    inferred by two bidirectional encoders and the inputs are produced by a
    controller RNN that runs alongside the generator.

    The architecture is the paper's, with the primitive recurrent cell of
    this library ({!Dynamics.Mgu}) wherever the paper has an RNN, and the
    generative model of this library ({!Model.dynamics}, {!Model.likelihood}
    and {!Model.prior}) in place of LFADS's own generator, factors and rate
    model:

    - {b Initial condition.} Two encoder RNNs read the observations, one in
      time order and one in reverse. Their final states are concatenated and
      an affine readout gives the mean and log-variance of a diagonal
      Gaussian over the initial state [x_0]. The prior is the model's
      {!Initial} prior.
    - {b Controller.} A second pair of encoder RNNs reads the observations
      bidirectionally and produces a feature at every step. A controller RNN
      runs forward in time over the concatenation of that feature with the
      previous generator state; an affine readout of its state gives the
      mean and log-variance of a diagonal Gaussian over the input [u_t].
    - {b Joint posterior.} [q(x_0, u_{0:T-1}) = q(x_0) · Π_t q(u_t | u_{<t}, x_{<t}, o)], an autoregressive Gaussian whose [u_t] factor conditions
      on the generator state reached by the inputs before it — the paper's
      dependence on the previous factor, written through the generator state
      because in this library the generator is the model's dynamics. A
      feedback through a linear factor readout [W_fac g] would only compose
      with the controller's own input weights, so the state is fed directly.

    {b The paper's variants.} [~with_inputs:false] drops the controller and
    its encoders from the posterior (they still exist, unused): the inputs
    are then the point mass at zero and only the initial state is inferred,
    which is the original LFADS model without inferred inputs. [~n_ic],
    [~n_ci] and [~n_ctrl] are the state sizes of the two encoder pairs and
    the controller.

    {b Initialization.} The encoder cells start from learned initial states,
    as the paper's references do, and the input readout is initialized small
    so that the posterior starts close to the model's input prior. *)
module Lfads = struct
  type 'a params =
    { ic_fwd : 'a Dynamics.Mgu.params (** initial-condition encoder, forward *)
    ; ic_bwd : 'a Dynamics.Mgu.params (** initial-condition encoder, backward *)
    ; ci_fwd : 'a Dynamics.Mgu.params (** controller-input encoder, forward *)
    ; ci_bwd : 'a Dynamics.Mgu.params (** controller-input encoder, backward *)
    ; con : 'a Dynamics.Mgu.params (** controller RNN *)
    ; ic_fwd_init : 'a (** [[n_ic]]; learned initial state *)
    ; ic_bwd_init : 'a (** [[n_ic]] *)
    ; ci_fwd_init : 'a (** [[n_ci]] *)
    ; ci_bwd_init : 'a (** [[n_ci]] *)
    ; con_init : 'a (** [[n_ctrl]] *)
    ; ic_readout : 'a (** [[2 n_ic; 2 n]]; the mean and log-variance of [x_0] *)
    ; ic_readout_bias : 'a (** [[2 n]] *)
    ; u_readout : 'a (** [[n_ctrl; 2 m]]; the mean and log-variance of [u_t] *)
    ; u_readout_bias : 'a (** [[2 m]] *)
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type 'a t = 'a params

        let walk = walk_params
      end)

  (* [run_mgu ~phi p ~init inputs] folds the cell over the leading (time)
     axis of [inputs] from the learned [init] state and returns every state,
     of shape [T; n_state]. *)
  let run_mgu ~phi (p : 'w value Dynamics.Mgu.params) ~(init : 'w value) inputs =
    let t_steps = (Nx.shape inputs).(0) in
    let ks = arange (Nx.dtype inputs) 0 t_steps 1 in
    let _, ys =
      Rune.scan
        Nx.Ptree.tensor
        Nx.Ptree.(pair tensor tensor)
        Nx.Ptree.tensor
        ~f:(fun x (u, k) ->
          let x' = Dynamics.Mgu.dyn ~phi p ~k ~x ~u in
          x', x')
        ~init
        (inputs, ks)
    in
    ys

  (* [encode_ic ~phi p data] is the concatenation of the two
     initial-condition encoders' final states, of shape [[2 n_ic]]. The
     backward encoder reads the reversed data. *)
  let encode_ic ~phi (p : 'w value params) data =
    let t_steps = (Nx.shape data).(0) in
    let data_rev = flip ~axes:[ 0 ] data in
    let ic_fwd = run_mgu ~phi p.ic_fwd ~init:p.ic_fwd_init data in
    let ic_bwd = run_mgu ~phi p.ic_bwd ~init:p.ic_bwd_init data_rev in
    concatenate ~axis:0 [ get [ t_steps - 1 ] ic_fwd; get [ t_steps - 1 ] ic_bwd ]

  (* [encode_con ~phi p data] holds, at row [t], the concatenation of the
     controller-input encoders' states at time [t], of shape [[T; 2 n_ci]].
     The backward encoder's outputs are flipped back to time order. *)
  let encode_con ~phi (p : 'w value params) data =
    let data_rev = flip ~axes:[ 0 ] data in
    let ci_fwd = run_mgu ~phi p.ci_fwd ~init:p.ci_fwd_init data in
    let ci_bwd = flip ~axes:[ 0 ] (run_mgu ~phi p.ci_bwd ~init:p.ci_bwd_init data_rev) in
    concatenate ~axis:1 [ ci_fwd; ci_bwd ]

  (* [ic_posterior p e_ic] is the mean and log-variance of [x_0], of shape
     [[n]] each. *)
  let ic_posterior (p : 'w value params) e_ic =
    split2 ~axis:0 Infix.((e_ic *@ p.ic_readout) + p.ic_readout_bias)

  (* [rollout ~phi gen gp p ~e_con ~x0 ~eps] runs the controller and the
     generator forward in time over a batch of [n_samples] trajectories:
     [x0] is [[n_samples; n]], [eps] is [[n_steps; n_samples; m]] and holds
     the standard normal draw of every step. It returns the sampled inputs
     [[n_steps; n_samples; m]] and the per-step means and log-variances of
     their Gaussian conditionals, both of the same shape. *)
  let rollout
        (type w d l p e)
        ~phi
        (gen : (w, d, l, p, e) generative)
        (gp : (w, d, l, p, e) gparams)
        (p : w value params)
        ~e_con
        ~x0
        ~eps
    =
    let n_samples = (Nx.shape x0).(0) in
    let n_ctrl = Array.get (Nx.shape p.con_init) 0 in
    let t_steps = gen.n_steps in
    let ks = arange gen.dtype 0 t_steps 1 in
    let c0 = broadcast_to [| n_samples; n_ctrl |] p.con_init in
    let step (g, c) ((e_con_t, eps_t), k) =
      let e = broadcast_to [| n_samples; (Nx.shape e_con_t).(0) |] e_con_t in
      let inp = concatenate ~axis:1 [ e; g ] in
      let c' = Dynamics.Mgu.dyn ~phi p.con ~k ~x:c ~u:inp in
      let mu, logvar = split2 ~axis:1 Infix.((c' *@ p.u_readout) + p.u_readout_bias) in
      let u = Infix.(mu + (eps_t * exp (logvar *$ 0.5))) in
      let g' = gen.dynamics.Dynamics.dyn gp.dynamics ~k ~x:g ~u in
      (g', c'), (u, (mu, logvar))
    in
    let _, (us, (mus, logvars)) =
      Rune.scan
        Nx.Ptree.(pair tensor tensor)
        Nx.Ptree.(pair (pair tensor tensor) tensor)
        Nx.Ptree.(pair tensor (pair tensor tensor))
        ~f:step
        ~init:(x0, c0)
        ((e_con, eps), ks)
    in
    us, mus, logvars

  (* [replay ~phi gen gp p ~e_con ~x0 ~u] recomputes the controller
     conditionals along a given trajectory and returns [log q(u | x0)], of
     shape [[n_samples]]. The controller reads the generator state reached
     by the given inputs, so the replay is exact — the chain rule needs the
     conditionals at the values they conditioned on, nothing else. *)
  let replay
        (type w d l p e)
        ~phi
        (gen : (w, d, l, p, e) generative)
        (gp : (w, d, l, p, e) gparams)
        (p : w value params)
        ~e_con
        ~x0
        ~u
    =
    let n_samples = (Nx.shape x0).(0) in
    let n_ctrl = Array.get (Nx.shape p.con_init) 0 in
    let t_steps = gen.n_steps in
    let ks = arange gen.dtype 0 t_steps 1 in
    let c0 = broadcast_to [| n_samples; n_ctrl |] p.con_init in
    let step (g, c) ((e_con_t, u_t), k) =
      let e = broadcast_to [| n_samples; (Nx.shape e_con_t).(0) |] e_con_t in
      let inp = concatenate ~axis:1 [ e; g ] in
      let c' = Dynamics.Mgu.dyn ~phi p.con ~k ~x:c ~u:inp in
      let mu, logvar = split2 ~axis:1 Infix.((c' *@ p.u_readout) + p.u_readout_bias) in
      let lp = gaussian_logp ~x:u_t ~mu ~logvar in
      let g' = gen.dynamics.Dynamics.dyn gp.dynamics ~k ~x:g ~u:u_t in
      (g', c'), lp
    in
    let _, lps =
      Rune.scan
        Nx.Ptree.(pair tensor tensor)
        Nx.Ptree.(pair (pair tensor tensor) tensor)
        Nx.Ptree.tensor
        ~f:step
        ~init:(x0, c0)
        ((e_con, u), ks)
    in
    sum ~axes:[ 0 ] lps

  let infer
        (type w d l p e)
        ~phi
        ~with_inputs
        (gen : (w, d, l, p, e) generative)
        (gp : (w, d, l, p, e) gparams)
        ~params
        ?(iters = 0)
        ~n_samples
        ~key
        data
    =
    let _ = iters in
    check_data ~n_steps:gen.n_steps ~n_output:gen.n_output data;
    if n_samples <= 0 then invalid_arg "Recognition.Lfads: n_samples must be positive";
    let e_ic = encode_ic ~phi params data in
    let ic_mean, ic_logvar = ic_posterior params e_ic in
    let ic_mean = reshape [| 1; gen.n |] ic_mean
    and ic_logvar = reshape [| 1; gen.n |] ic_logvar in
    let keys = Nx.Rng.split ~n:2 key in
    let eps_ic = Nx.Rng.normal keys.(0) gen.dtype [| n_samples; gen.n |] in
    let x0 = Infix.(ic_mean + (eps_ic * exp (ic_logvar *$ 0.5))) in
    (* The densities are evaluated with the Gaussian means and log-variances
       detached, so that the gradient of the bound reaches the encoder and
       controller only through the sampled latents: the path-derivative
       (sticking-the-landing) estimator of the KL gradient. The values are
       the exact log densities. *)
    let ic_logp =
      gaussian_logp ~x:x0 ~mu:(Rune.detach ic_mean) ~logvar:(Rune.detach ic_logvar)
    in
    if not with_inputs
    then
      { x0; u = zeros gen.dtype [| gen.n_steps; n_samples; gen.m |]; log_prob = ic_logp }
    else (
      let e_con = encode_con ~phi params data in
      let eps = Nx.Rng.normal keys.(1) gen.dtype [| gen.n_steps; n_samples; gen.m |] in
      let u, mus, logvars = rollout ~phi gen gp params ~e_con ~x0 ~eps in
      let u_logp =
        sum
          ~axes:[ 0 ]
          (gaussian_logp ~x:u ~mu:(Rune.detach mus) ~logvar:(Rune.detach logvars))
      in
      { x0; u; log_prob = Infix.(ic_logp + u_logp) })

  let mean
        (type w d l p e)
        ~phi
        ~with_inputs
        (gen : (w, d, l, p, e) generative)
        (gp : (w, d, l, p, e) gparams)
        ~params
        ?(iters = 0)
        data
    =
    let _ = iters in
    check_data ~n_steps:gen.n_steps ~n_output:gen.n_output data;
    let e_ic = encode_ic ~phi params data in
    let ic_mean, _ = ic_posterior params e_ic in
    let x0 = reshape [| 1; gen.n |] ic_mean in
    if not with_inputs
    then { x0; u = zeros gen.dtype [| gen.n_steps; 1; gen.m |] }
    else (
      let e_con = encode_con ~phi params data in
      let eps = zeros gen.dtype [| gen.n_steps; 1; gen.m |] in
      let u, _, _ = rollout ~phi gen gp params ~e_con ~x0 ~eps in
      { x0; u })

  let log_prob
        (type w d l p e)
        ~phi
        ~with_inputs
        (gen : (w, d, l, p, e) generative)
        (gp : (w, d, l, p, e) gparams)
        ~params
        ?(iters = 0)
        data
        (samples : w samples)
    =
    let _ = iters in
    check_data ~n_steps:gen.n_steps ~n_output:gen.n_output data;
    let e_ic = encode_ic ~phi params data in
    let ic_mean, ic_logvar = ic_posterior params e_ic in
    let ic_logp =
      gaussian_logp
        ~x:samples.x0
        ~mu:(reshape [| 1; gen.n |] ic_mean)
        ~logvar:(reshape [| 1; gen.n |] ic_logvar)
    in
    if not with_inputs
    then ic_logp
    else (
      let e_con = encode_con ~phi params data in
      Infix.(ic_logp + replay ~phi gen gp params ~e_con ~x0:samples.x0 ~u:samples.u))

  (** [make ?n_ic ?n_ci ?n_ctrl ?with_inputs ?phi ?key ~n ~m ~n_steps ~n_output dtype] is the LFADS recognition. [n_ic] (default [64]) and
      [n_ci] (default [n_ic]) are the encoder state sizes, [n_ctrl] (default
      [32]) the controller's. [with_inputs] (default [true]) selects whether
      the controller infers inputs at all. [phi] (default {!Nx.tanh}) is the
      activation of every {!Dynamics.Mgu} cell. [key] seeds the initialization
      and defaults to the current random scope. *)
  let make
        ?(n_ic = 64)
        ?n_ci
        ?(n_ctrl = 32)
        ?(with_inputs = true)
        ?(phi = tanh)
        ?key
        ~n
        ~m
        ~n_steps
        ~n_output
        dtype
    =
    let n_ci = Option.value n_ci ~default:n_ic in
    if n_ic <= 0 then invalid_arg "Recognition.Lfads.make: n_ic must be positive";
    if n_ci <= 0 then invalid_arg "Recognition.Lfads.make: n_ci must be positive";
    if n_ctrl <= 0 then invalid_arg "Recognition.Lfads.make: n_ctrl must be positive";
    let keys = Option.map key ~f:(fun key -> Nx.Rng.split ~n:7 key) in
    let sub i = Option.map keys ~f:(fun keys -> keys.(i)) in
    let cell key ~n ~m = (Dynamics.Mgu.make ?key ~n ~m ~phi dtype).Dynamics.params in
    let params =
      { ic_fwd = cell (sub 0) ~n:n_ic ~m:n_output
      ; ic_bwd = cell (sub 1) ~n:n_ic ~m:n_output
      ; ci_fwd = cell (sub 2) ~n:n_ci ~m:n_output
      ; ci_bwd = cell (sub 3) ~n:n_ci ~m:n_output
      ; con = cell (sub 4) ~n:n_ctrl ~m:((2 * n_ci) + n)
      ; ic_fwd_init = zeros dtype [| n_ic |]
      ; ic_bwd_init = zeros dtype [| n_ic |]
      ; ci_fwd_init = zeros dtype [| n_ci |]
      ; ci_bwd_init = zeros dtype [| n_ci |]
      ; con_init = zeros dtype [| n_ctrl |]
      ; ic_readout =
          scaled_randn
            ?key:(sub 5)
            ~sigma:(1.0 /. Float.sqrt (Float.of_int (2 * n_ic)))
            dtype
            [| 2 * n_ic; 2 * n |]
      ; ic_readout_bias = zeros dtype [| 2 * n |]
      ; u_readout =
          scaled_randn
            ?key:(sub 6)
            ~sigma:(1e-2 /. Float.sqrt (Float.of_int n_ctrl))
            dtype
            [| n_ctrl; 2 * m |]
      ; u_readout_bias = zeros dtype [| 2 * m |]
      }
    in
    { dims = { n; m; n_steps; n_output }
    ; infers_initial = true
    ; name = (if with_inputs then "lfads" else "lfads_initial_condition")
    ; structure = structure ()
    ; params
    ; infer = infer ~phi ~with_inputs
    ; mean = mean ~phi ~with_inputs
    ; log_prob = log_prob ~phi ~with_inputs
    }
end
