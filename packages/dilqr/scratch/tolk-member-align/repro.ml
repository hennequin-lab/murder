(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The smallest dilqr program I found that tolk's spec rejects:
   packages/dilqr/scratch/tolk-member-align/README.md is the report.

   An unbatched float64 pendulum, solved by the compiled adaptive solve. The
   horizon is the knob: 1 trips nothing, 2 and up raise

   Invalid_argument("UOp verification failed at 31 on Ops.CALL ... 'member'")

   from Tolk.Spec.type_verify, for the reason the README gives: a host member's
   body keeps the parameter alignment it was built with (16, the default) while
   the call passes a float64 view at an 8-byte offset.

   With verification off the same program compiles and runs, and its numbers are
   right: SPEC=0 is the control. *)

open Base
open Nx

module P = struct
  module Params = struct
    type ('a, _) t =
      { gravity : 'a
      ; mass : 'a
      ; length : 'a
      }
    [@@deriving ptree]
  end

  let n = 2
  let dt = 0.05
  let target = 3.14159265358979

  let dynamics { Params.gravity; mass; length } ~k:_ ~x ~u =
    let open Infix in
    let theta = slice [ R (0, 1) ] x
    and omega = slice [ R (1, 2) ] x in
    let ml2 = mass * (length * length) in
    let grav_torque = gravity * (mass * length) * sin theta in
    let acc = (u - (grav_torque + (omega *$ 0.1))) / ml2 in
    concatenate ~axis:0 [ theta + (omega *$ dt); omega + (acc *$ dt) ]

  let running_loss _ ~k:_ ~x ~u =
    let open Infix in
    let theta = slice [ R (0, 1) ] x
    and omega = slice [ R (1, 2) ] x in
    let dtheta = theta -$ target in
    sum ((((dtheta * dtheta) + (omega * omega)) *$ 0.5) + (u * u *$ 0.01))

  let final_loss _ ~x =
    let open Infix in
    let theta = slice [ R (0, 1) ] x
    and omega = slice [ R (1, 2) ] x in
    let dtheta = theta -$ target in
    sum ((dtheta * dtheta) + (omega * omega)) *$ 5.0

  let problem horizon
    : (Nx.float64_elt, (Nx.host Nx.float64_t, Nx.host) Params.t, Nx.host) Dilqr.Problem.t
    =
    { dtype = float64
    ; n
    ; m = 1
    ; horizon
    ; free_initial = false
    ; verbose = false
    ; params = Nx.Ptree.instantiate (module Params)
    ; dynamics
    ; running_loss
    ; final_loss
    }
end

let () =
  let horizon = Int.of_string (Sys.get_argv ()).(1) in
  let theta : (Nx.host Nx.float64_t, Nx.host) P.Params.t =
    { gravity = Nx.scalar float64 9.81
    ; mass = Nx.scalar float64 1.0
    ; length = Nx.scalar float64 1.0
    }
  in
  let x0 = Nx.create float64 [| 2 |] [| P.target -. 0.3; 0.0 |] in
  let us0 = Array.init horizon ~f:(fun _ -> Nx.create float64 [| 1 |] [| 0.0 |]) in
  let out =
    (Dilqr.Solver.compile (P.problem horizon) ~beam:0 ~parallel:1)
      ~max_iter:20
      theta
      x0
      us0
  in
  Stdio.printf "horizon %d: cost %.6g\n%!" horizon (Nx.item [] out.cost)
