(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The sensor block against MuJoCo, on a seventeen-body chain: the
   state-projection family, the frame family at sites and bodies, the actuator
   and tendon readings, and the subtree reductions, which is why the chain is
   deep enough for a subtree to mean something. *)

open Base
open Windtrap
open Support

let name = "sensors"
let m = model name
let f = fixture name

let () =
  Stdlib.exit
  @@ run
       "mjc sensor"
       [ test "sensordata" (fun () ->
           let total = Array.length (row (tensor f "sensordata") 0) in
           List.iter (List.range 0 f.nstates) ~f:(fun s ->
             let d = data_with_qpos m (row (tensor f "qpos") s) in
             let d = { d with Mjc.Data.qvel = vec1 (row (tensor f "qvel") s) } in
             let d =
               match tensor f "ctrl" with
               | c -> { d with Mjc.Data.ctrl = vec1 (row c s) }
             in
             (* A forward pass, so the tendon lengths and actuator forces the
                sensors read are the ones the engine computed. *)
             let d = Mjc.Forward.forward m d in
             parity
               ~tol:1e-12
               ~name:(Printf.sprintf "%s state %d sensordata" name s)
               (Array.sub (row (tensor f "sensordata") s) ~pos:0 ~len:total)
               (Nx.to_array d.Mjc.Data.sensordata)))
       ]
