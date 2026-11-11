(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** MPI training for {!Ilqr_vae}: one mini-batch of trials over several ranks.

    A training step estimates the bound on a mini-batch, differentiates it, and
    updates the parameters. Every trial's bound is independent of every other
    trial's, so the estimate splits over the cores of a cluster:

    {v
      rank 0          rank 1          rank 2
      trials 0-1      trials 2-3      trials 4
          |               |               |
      loss, grad      loss, grad      loss, grad
          \_______________|_______________/
                          |
                 sum over the ranks      (MPI reduction)
                          |
             mean over the mini-batch
                          |
                 update (root only)
                          |
            broadcast of the parameters  (MPI)
                          |
      rank 0          rank 1          rank 2    same parameters
    v}

    The parameters and the step's key are the same on every rank. {!make}
    compiles one single-trial objective per process, {!step} runs one
    distributed step, and {!run} loops over steps.

    {b The objective, and why the split does not matter.} The step's key is
    split into one subkey per trial, exactly as a batched
    {!Ilqr_vae.Elbo.loss_fixed} splits it, and each rank estimates the bound of
    its own trials at their subkeys. The ranks' gradients are summed and
    divided by the number of trials in the mini-batch, so the result is the
    gradient of the batch's mean bound however the batch was split: a run with
    four ranks and a run with one compute the same numbers, up to
    floating-point summation order. What changes with the rank count is the
    wall-clock time, not the estimate.

    {b What runs where.} Every rank holds the whole mini-batch and reads its
    share — {!partition} is the split, contiguous blocks, with the first ranks
    taking one more trial when the batch does not divide evenly. Each rank
    differentiates its own trials. The losses and gradients are summed onto
    rank 0, which divides by the number of trials and calls the caller's
    [update] function; the new parameters are then broadcast, so every rank
    returns from a step holding the same value. An optimizer that keeps state —
    Adam's moment estimates, say — therefore keeps it on the root only, and a
    non-root process never reads its own copy.

    {b Where it runs.} The [mpi] library initializes MPI when the program
    starts, so every program that links this library is an MPI program, run
    with [mpirun] or [srun]; run without a launcher, it is a world of one rank,
    and the driver reduces to a single-process training step.

    {[
      (* train.exe, launched with: mpirun -np 4 train.exe *)
      open Ilqr_vae
      open Ilqr_vae_mpi

      let () =
        let model = … in
        let params = Model.init ~at:Nx.Host.on ~key:(Nx.Rng.key 0) model in
        let driver = make model ~iters:4 ~n_samples:16 () in
        (* Plain SGD: p - lr * g on every leaf. The cast makes the step apply
           to leaves of every dtype, which is what Rune.Ptree.map2 asks of it. *)
        let sgd lr params grads =
          Rune.Ptree.map2
            (Model.structure model)
            (fun _ p g ->
               Nx.Infix.(p - (Nx.cast (Nx.dtype p) (Nx.scalar Nx.float64 lr) * g)))
            params
            grads
        in
        let _, losses =
          run
            driver
            params
            ~steps:200
            ~key:(Nx.Rng.key 1)
            ~data:(fun i -> batches.(i mod Array.length batches))
            ~report:(fun i loss -> Printf.printf "step %d  loss %.4f\n%!" i loss)
            (sgd 1e-3)
        in
        if is_root driver then Printf.printf "final loss %.4f\n%!" losses.(199)
    ]} *)

(** {1:driver The driver} *)

(** A driver: the model's single-trial objective compiled for the calling
    process, and the communicator the ranks share. It holds no parameters and
    no data, and its compiled function takes a trial, not a mini-batch, so one
    driver serves every split of a batch and every batch of the model's shape. *)
type ('w, 'd, 'l, 'p, 'e, 'r, 's) t

(** [make model ~iters ~n_samples ()] compiles the single-trial objective on
    the calling process: the negative ELBO of one trial, with [iters] iLQR
    iterations and [n_samples] recognition draws, differentiated with respect
    to every parameter of the model. [comm] is the communicator the ranks
    share, [Mpi.comm_world] by default.

    Every rank must call [make] with the same model and the same [iters] and
    [n_samples]; each process compiles its own copy and draws nothing, so the
    parameters are still {!Ilqr_vae.Model.init}'s to make. Compilation happens
    here, once per process, not at every step.

    [iters] and [n_samples] are constants of the compiled program, so changing
    them means building another driver; the [beta] weight, the data, the key
    and the parameters are arguments of the step and can change freely.

    Raises [Invalid_argument] if [iters] or [n_samples] is not positive. *)
val make
  :  ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.t
  -> ?comm:Mpi.communicator
  -> iters:int
  -> n_samples:int
  -> unit
  -> ('w, 'd, 'l, 'p, 'e, 'r, 's) t

(** [size driver] is the number of ranks in the driver's communicator. *)
val size : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> int

(** [rank driver] is the calling process's rank, in [0 … size driver − 1]. *)
val rank : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> int

(** [is_root driver] is [true] on rank 0, the process that takes the
    optimizer step. *)
val is_root : ('w, 'd, 'l, 'p, 'e, 'r, 's) t -> bool

(** {1:split The trial split} *)

(** A half-open range of global trial indices,
    [[first] … [first + count − 1]]. *)
type range =
  { first : int
  ; count : int
  }

(** [partition ~ranks ~trials ~rank] is the share of [trials] trials that rank
    [rank] owns when the mini-batch is spread over [ranks] ranks: one
    contiguous block per rank, in rank order, so that trial indices grow with
    the rank. When [trials] does not divide by [ranks], the first
    [trials mod ranks] ranks take one more trial than the others; a rank owns
    nothing when there are more ranks than trials. The ranges of the ranks
    cover [0 … trials − 1] exactly once, which is what makes the distributed
    estimate the batch's own.

    This is also the function a data pipeline uses to read only the trials its
    rank needs, rather than the whole batch.

    Raises [Invalid_argument] if [ranks] is not positive, if [rank] is outside
    [0 … ranks − 1], or if [trials] is negative. *)
val partition : ranks:int -> trials:int -> rank:int -> range

(** {1:training Training} *)

(** [step driver params ~key ?beta ~data update] is one distributed training
    step: every rank estimates and differentiates the bound of its own trials,
    the ranks' gradients are averaged over the mini-batch, the root updates the
    parameters, and the new parameters are broadcast. Returns the parameters
    every rank now holds and the mini-batch's mean negative ELBO — a scalar,
    lower for a better fit. Both are the same on every rank.

    [data] is the mini-batch, [[batch…; n_steps; n_output]], with the model's
    horizon and observation dimension and one or more leading trial axes whose
    product is the number of trials. It must be the same tensor on every rank,
    and every rank slices its share with {!partition}. [key] is the step's key;
    it is split into one subkey per trial, so trial [i] is estimated at the same
    subkey in every run. [beta] is the KL weight of {!Ilqr_vae.Elbo.loss_fixed}
    (a scalar tensor, a one of the model's dtype by default).

    [update params grads] is called on the root only, with the parameters and
    the mini-batch's mean gradient, and returns the new parameters. It must keep
    the model's structure, dtypes and shapes; it is free to keep state between
    calls, which is then root state, since it is not called anywhere else. The
    parameters this function returns are what every rank holds after the step,
    so a non-root rank that started the step with other parameters ends it with
    the root's. The exchange carries the float leaves; a leaf that cannot carry
    a gradient — an integer counter, a key — is not sent and keeps the value
    each rank had.

    Raises [Invalid_argument] if [data] is not a mini-batch of the model's
    shape, if the parameters do not pack to the same number of values on every
    rank, or if [update] changes their structure. *)
val step
  :  ('w, 'd, 'l, 'p, 'e, 'r, 's) t
  -> ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params
  -> key:'s Nx.Rng.key
  -> ?beta:('w, 's) Ilqr_vae.value
  -> data:('w, 's) Ilqr_vae.value
  -> (('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params
      -> ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params
      -> ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params)
  -> ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params * float

(** [run driver params ~steps ~key ~data update ?beta ?report] is [steps]
    calls to {!step}, step [i] with the key [Nx.Rng.fold_in key i], the
    mini-batch [data i] and, when [beta] is given, the weight [beta i]. Without
    [beta] every step uses the ELBO's default weight of one, so a KL warm-up
    passes [beta] and a schedule's {!Ilqr_vae.Schedule.at}.

    [data] and [beta] are called on every rank and must give the same value on
    every rank. [report] is called on the root after each step with the step
    index and the mean loss, which is where a training log belongs. Returns the
    parameters after the last step and one loss per step; the losses are
    identical on every rank.

    Raises [Invalid_argument] if [steps] is negative. *)
val run
  :  ('w, 'd, 'l, 'p, 'e, 'r, 's) t
  -> ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params
  -> steps:int
  -> key:'s Nx.Rng.key
  -> ?beta:(int -> ('w, 's) Ilqr_vae.value)
  -> data:(int -> ('w, 's) Ilqr_vae.value)
  -> ?report:(int -> float -> unit)
  -> (('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params
      -> ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params
      -> ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params)
  -> ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params * float array
