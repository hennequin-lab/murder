(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Types

(* The ELBO is [log p(o | z) − β · KL(q ‖ p)] ({!Elbo}). β = 1 is the
   variational bound, but a recognition trained from scratch can collapse
   onto the prior before it has learned to read the data, and the likelihood
   term alone is then too weak to pull it back. Starting β at 0 and raising
   it to 1 over the first few thousand steps — the KL warm-up of Bowman et
   al. (2016) and the LFADS reference implementation — keeps the posterior
   open while the decoder learns. iLQR-VAE needs it less, because its
   encoder already solves a control problem whose cost contains the encoder
   prior, but the weight is available either way.

   A schedule is host-side data: [value s step] is a float, [at s step] the
   scalar tensor the ELBO multiplies the KL by. Under {!Rune.jit}, pass that
   tensor as an argument of the compiled step — a β captured from the
   closure is a constant of the program and stops following the schedule
   after the first trace. *)

type 'w t =
  { dtype : (float, 'w) Nx.dtype
  ; value : int -> float
  }

(** [custom f dtype] is the schedule [f] over steps. *)
let custom value dtype = { dtype; value }

(** [constant beta dtype] is [beta] at every step. *)
let constant beta dtype = { dtype; value = (fun _ -> beta) }

(** [linear_warmup ?start ?stop ?start_step ~steps dtype] is [start] up to
    [start_step], then linear to [stop] over the next [steps] steps, then
    [stop]. [start] defaults to [0.], [stop] to [1.], [start_step] to [0].
    Raises [Invalid_argument] if [steps] is not positive or [start_step] is
    negative. *)
let linear_warmup ?(start = 0.0) ?(stop = 1.0) ?(start_step = 0) ~steps dtype =
  if steps <= 0 then invalid_arg "Schedule.linear_warmup: steps must be positive";
  if start_step < 0
  then invalid_arg "Schedule.linear_warmup: start_step must be non-negative";
  let value step =
    if step <= start_step
    then start
    else if step >= start_step + steps
    then stop
    else (
      let fraction = Float.of_int (step - start_step) /. Float.of_int steps in
      start +. ((stop -. start) *. fraction))
  in
  { dtype; value }

let value s step = s.value step
let at s step = Nx.scalar s.dtype (s.value step)
