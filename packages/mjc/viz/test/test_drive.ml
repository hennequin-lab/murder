(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The scripted control: a sine every actuator follows, which is what makes a
   mirrored pair of tendons open and close together. *)

open Base
open Windtrap

let close a b = Float.( <= ) (Float.abs (a -. b)) 1e-12

let test_sine () =
  let t = { Mjc_viz.Drive.amplitude = 2.; period = 4. } in
  let at time = (Mjc_viz.Drive.ctrl t ~nu:1 ~time).(0) in
  List.iter
    [ 0., 0.; 1., 2.; 2., 0.; 3., -2. ]
    ~f:(fun (time, want) ->
      equal ~msg:(Printf.sprintf "t=%g" time) bool true (close want (at time)))

let test_shared () =
  let u = Mjc_viz.Drive.ctrl Mjc_viz.Drive.default ~nu:3 ~time:0.37 in
  equal ~msg:"count" int 3 (Array.length u);
  equal ~msg:"same value" bool true (Array.for_all u ~f:(fun x -> close u.(0) x))

let () =
  Stdlib.exit
  @@ run
       "mjc viz: drive"
       [ test "the four points of a sine" test_sine
       ; test "every actuator follows" test_shared
       ]
