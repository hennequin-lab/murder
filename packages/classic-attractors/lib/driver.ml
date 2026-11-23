(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The shared integrator: one RK4 step per [dt], the states stacked on a new
   leading axis, then sliced and shuffled as the interface says. *)

open Base
open Nx
open Jera

module type Param = sig
  val name : string
  val dim : int
  val bounds : (float * float) array
  val init : float array
  val field : (float, 't, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
end

module Make (P : Param) : Attractor.S = struct
  let dim = P.dim
  let field = P.field

  let fail what fmt =
    Printf.ksprintf (fun s -> invalid_arg (P.name ^ "." ^ what ^ ": " ^ s)) fmt

  let check ~what ~dt ~n_steps ~n ~discard_first =
    if (not (Float.is_finite dt)) || Float.(dt <= 0.)
    then fail what "dt is %g, expected a finite positive step" dt;
    if n_steps < 1 then fail what "n_steps is %d, expected at least 1" n_steps;
    if n < 1 then fail what "the run count is %d, expected at least 1" n;
    if discard_first < 0
    then fail what "discard_first is %d, expected a non-negative count" discard_first

  (* [march ~dt ~n_steps y0] is the [n_steps] states of [y0]'s flow, one RK4 step
     of [dt] apart, stacked on a new leading axis: [[n_steps; n; dim]]. Times
     are float64 whatever the state's dtype; [~at] holds one time per state and
     every interval takes one step, so nothing is interpolated. *)
  let march ~dt ~n_steps y0 =
    let at = linspace float64 0. (Float.of_int (n_steps - 1) *. dt) n_steps in
    Ode.march Ptree.tensor Ode.rk4 ~steps:1 P.field ~at y0

  (* [n] initial conditions drawn uniformly in the bounds. *)
  let draw_ic ~key dtype n =
    let at i = P.bounds.(i) in
    let lo = init dtype [| 1; dim |] (fun i -> fst (at i.(1))) in
    let hi = init dtype [| 1; dim |] (fun i -> snd (at i.(1))) in
    let u = Rng.uniform ~key dtype [| n; dim |] in
    Infix.((u * (hi - lo)) + lo)

  let simulate ?(discard_first = 0) ?(key = Rng.key 0) dtype ~dt ~n_steps n =
    check ~what:"simulate" ~dt ~n_steps ~n ~discard_first;
    let total = discard_first + n_steps in
    let states = march ~dt ~n_steps:total (draw_ic ~key dtype n) in
    slice [ R (discard_first, total); A; A ] states

  let snippets ?(discard_first = 0) ?(key = Rng.key 0) dtype ~dt ~n_steps n =
    check ~what:"snippets" ~dt ~n_steps ~n ~discard_first;
    let total = discard_first + (n * n_steps) in
    let y0 = create dtype [| 1; dim |] P.init in
    let states = march ~dt ~n_steps:total y0 in
    let chunks =
      slice [ R (discard_first, total); A; A ] states |> reshape [| n; n_steps; dim |]
    in
    let order = Rng.permutation ~key n in
    take ~axis:0 order chunks |> transpose ~axes:[ 1; 0; 2 ]
end
