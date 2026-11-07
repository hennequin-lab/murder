(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Does the supported quadruped rock in place, with friction on the floor?

   The drive is the one the viewer sends: the knees only (pattern 0,1,0,1,...),
   amplitude 0.2 rad, period 0.6 s, and a quarter turn of phase per actuator so
   the front and back legs move against each other. With the feet on a frictional
   floor that rocks the body without walking it; the same amplitude on the hips,
   or a larger one, is a gait and travels most of a metre.

   dune exec packages/mjc/bench/wig.exe *)

open Base
module Printf = Stdlib.Printf

let pattern = [| 0.; 1.; 0.; 1.; 0.; 1.; 0.; 1. |]

let () =
  let m =
    match Mjc.Mjcb.read "test/fixtures/quad.mjcb" with
    | Ok m -> m
    | Error e -> failwith e
  in
  let d = Mjc.Forward.forward m (Mjc.Data.create m) in
  let nu = (Nx.shape d.Mjc.Data.ctrl).(0) in
  let x0 = (Nx.to_array (Nx.slice [ I 0 ] d.Mjc.Data.qpos)).(0) in
  let d = ref d in
  let xs = ref []
  and zs = ref []
  and touching = ref 0 in
  for _ = 1 to 900 do
    let t = Nx.item [] !d.Mjc.Data.time in
    let ctrl =
      Nx.init Nx.float64 [| nu |] (fun i ->
        let i = i.(0) in
        let weight = pattern.(Int.rem i (Array.length pattern)) in
        let phase = 1.5707963 *. Float.of_int i in
        0.2 *. weight *. Float.sin ((2. *. Float.pi *. t /. 0.6) +. phase))
    in
    d := Mjc.Step.step m { !d with Mjc.Data.ctrl };
    xs := (Nx.to_array (Nx.slice [ I 0 ] !d.Mjc.Data.qpos)).(0) :: !xs;
    zs := (Nx.to_array (Nx.slice [ I 2 ] !d.Mjc.Data.qpos)).(0) :: !zs;
    if Stdlib.Int32.to_int (Nx.item [] !d.Mjc.Data.ncon) > 0 then Int.incr touching
  done;
  let xs = Array.of_list (List.rev !xs)
  and zs = Array.of_list (List.rev !zs) in
  let range a =
    Array.fold a ~init:(-1e9) ~f:Float.max -. Array.fold a ~init:1e9 ~f:Float.min
  in
  Printf.printf
    "900 steps with friction: drift %+.4f m, x-range %.4f, z-range %.4f, in contact %d \
     of 900\n"
    (xs.(Array.length xs - 1) -. x0)
    (range xs)
    (range zs)
    !touching
