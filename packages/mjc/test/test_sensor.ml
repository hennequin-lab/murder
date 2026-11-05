(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The sensor block against MuJoCo, on a seventeen-body chain: the
   state-projection family, the frame family at sites and at both body frames,
   the actuator and tendon readings, and the subtree reductions.

   A quaternion is compared up to sign: [q] and [-q] are the same orientation,
   and the engine picks the representative by which component is largest, so a
   frame within rounding of that boundary can come out negated. *)

open Base
open Windtrap
open Support

let name = "sensors"
let m = model name
let f = fixture name
let sub xs ~pos ~len = Array.sub xs ~pos ~len

let check_state t s =
  let expected = row (tensor f "sensordata") s in
  let d = data_with_qpos m (row (tensor f "qpos") s) in
  let d = { d with Mjc.Data.qvel = vec1 (row (tensor f "qvel") s) } in
  let d = { d with Mjc.Data.ctrl = vec1 (row (tensor f "ctrl") s) } in
  let d = Mjc.Forward.forward m d in
  let actual = Nx.to_array d.Mjc.Data.sensordata in
  Array.iteri t.Mjc.Topo.sensor_type ~f:(fun i kind ->
    let pos = t.Mjc.Topo.sensor_adr.(i)
    and len = t.Mjc.Topo.sensor_dim.(i) in
    let label = Printf.sprintf "%s state %d sensor %d" name s i in
    let e = sub expected ~pos ~len
    and a = sub actual ~pos ~len in
    match kind with
    | Mjc.Topo.Sens_framequat ->
      (* the same orientation: the engine's branch can hand back the negated
         representative, and a negation is the same rotation *)
      let dot = Array.foldi a ~init:0. ~f:(fun k acc x -> acc +. (e.(k) *. x)) in
      if Float.( < ) dot 0.
      then parity ~tol:1e-12 ~name:(label ^ " negated") e (Array.map a ~f:Float.neg)
      else parity ~tol:1e-12 ~name:label e a
    | _ -> parity ~tol:1e-12 ~name:label e a)

let () =
  Stdlib.exit
  @@ run
       "mjc sensor"
       [ test "sensordata" (fun () ->
           let t = m.Mjc.Model.topo in
           List.iter (List.range 0 f.nstates) ~f:(fun s -> check_state t s))
       ]
