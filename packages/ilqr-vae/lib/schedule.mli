(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Schedules for the KL weight [β].

    The ELBO is [log p(o | z) − β · KL(q ‖ p)]; β = 1 is the variational
    bound. A recognition trained from scratch — LFADS-style ones in
    particular — can collapse onto the prior before it has learned to read
    the data, and the likelihood term alone is then too weak to pull it back.
    Starting β at 0 and raising it to 1 over the first few thousand steps is
    the KL warm-up of Bowman et al. (2016) and of the LFADS reference
    implementation. iLQR-VAE needs it less, because its encoder already
    solves a control problem whose cost contains the encoder prior, but the
    weight is available either way.

    A schedule is host-side data. {!at} turns its value at a step into the
    scalar tensor the ELBO multiplies the KL by; the value can be computed at
    every training step without recompiling, since it is a tensor the
    function takes rather than a constant it closes over.

    {[
      let schedule = Schedule.linear_warmup ~steps:2000 dtype in
      let step =
        Rune.jit
          Nx.Ptree.(structure @-> tensor @-> Nx.Rng.ptree @-> returns …)
          (fun params beta key ->
             Elbo.loss_fixed model params ~beta ~iters:4 ~key ~n_samples:16 trials)
      in
      let loss = step params (Schedule.at schedule i) key in
    ]} *)

open Types

(** A schedule over steps. The type parameter is the dtype of the scalar
    tensor {!at} builds. *)
type 'w t

(** [custom f dtype] is the schedule [f] over steps. *)
val custom : (int -> float) -> (float, 'w) Nx.dtype -> 'w t

(** [constant beta dtype] is [beta] at every step. *)
val constant : float -> (float, 'w) Nx.dtype -> 'w t

(** [linear_warmup ?start ?stop ?start_step ~steps dtype] is [start] up to
    [start_step], then linear to [stop] over the next [steps] steps, then
    [stop]. [start] defaults to [0.], [stop] to [1.], [start_step] to [0].

    Raises [Invalid_argument] if [steps] is not positive or [start_step] is
    negative. *)
val linear_warmup
  :  ?start:float
  -> ?stop:float
  -> ?start_step:int
  -> steps:int
  -> (float, 'w) Nx.dtype
  -> 'w t

(** [value s step] is the weight at [step], as a host float, for logging. *)
val value : 'w t -> int -> float

(** [at s step] is the weight at [step] as a scalar tensor, of the schedule's
    dtype. Pass it to the ELBO as [~beta]. *)
val at : 'w t -> int -> 'w value
