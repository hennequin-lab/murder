(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Backtracking line search along the iLQR update direction. *)

(** [schedule ~n] is the decreasing trial-step sequence
    [1; 1/2; 1/4; ...] with [n] entries. *)
let schedule ?(tau = 0.5) ~n () =
  let rec loop acc k alpha =
    if k >= n then List.rev acc else loop (alpha :: acc) (k + 1) (alpha *. tau)
  in
  loop [] 0 1.0

(** [accepts ~f0 ~f ~df ~c1] is the Armijo condition: the trial cost [f]
    decreases enough relative to the modelled decrease [df] (both [f0 - f] and
    [-df] are positive when the step is good). *)
let accepts ~f0 ~f ~df ~c1 = Float.(f < f0 -. (c1 *. df))
