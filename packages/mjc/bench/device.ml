(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Does a whole mjc step run on the GPU, and does it agree with the host?

   This is the end the device-polymorphism refactor was for: take a fixture's
   model and data, place both on the iGPU, compile `Step.step_with` for that
   device, and compare a hundred steps against the same program on the host.
   The structure (`Step.config`) comes from the host model, because structure is
   host data and is the same on every device; the parameters and the state are
   the compiled program's inputs.

   dune exec packages/mjc/bench/device.exe -- hopper *)

open Base
module Printf = Stdlib.Printf
module Step = Mjc.Step
module Data = Mjc.Data
module Model = Mjc.Model

let load name =
  let path = Stdlib.Filename.concat "test/fixtures" (name ^ ".mjcb") in
  match Mjc.Mjcb.read path with
  | Ok m -> m
  | Error e -> failwith e

(* One program, at whatever device its data lives on. A functor over the device
   set, so the compiled program can name the set's brand; nothing branded
   escapes: the result is a host array. *)
module Run (Set : Nx.Devices) = struct
  let place = Set.on

  let run (c : Step.config) m0 d0 steps =
    (* Parameters and state both live at the device; the config is host data. *)
    let m = Model.place place m0 in
    let d = Data.place place d0 in
    let step =
      Rune.jit
        (Nx.Ptree.( @-> ) (Data.ptree ()) (Nx.Ptree.returns (Data.ptree ())))
        (fun d -> Step.step_with c m d)
    in
    let last = ref d in
    for _ = 1 to steps do
      last := step !last
    done;
    (* One copy back, outside the loop. *)
    (* One copy back, outside the loop: the two leaves this compares. *)
    let out = Data.place Nx.Host.on !last in
    Nx.reshape [| -1 |] (Nx.concatenate ~axis:0 [ out.Data.qpos; out.qvel ])
end

let worst a b =
  let a = Nx.to_array a
  and b = Nx.to_array b in
  let n = Stdlib.min (Array.length a) (Array.length b) in
  let acc = ref 0.0 in
  for i = 0 to n - 1 do
    acc := Float.max !acc (Float.abs (a.(i) -. b.(i)))
  done;
  !acc

let () =
  let argv = Sys.get_argv () in
  let arg i default = if Array.length argv > i then argv.(i) else default in
  let name = arg 1 "hopper" in
  let steps = Stdlib.int_of_string (arg 2 "100") in
  let m = load name in
  let d0 = Data.create m in
  let c = Step.config m in
  let host =
    let module R = Run (Nx.Host) in
    R.run c m d0 steps
  in
  Printf.printf "%s: %d steps on the host\n%!" name steps;
  match Rig_amd_amdgpu.count () with
  | Error e -> Printf.printf "  no GPU: %s\n%!" e
  | Ok 0 -> Printf.printf "  no GPU to compare\n%!"
  | Ok _ ->
    (match Rig.open_ (Rig_amd_amdgpu.gpu 0) with
     | Error e -> Printf.printf "  gpu: %s\n%!" e
     | Ok dev ->
       Printf.printf "  %s (%s)\n%!" (Rig.name dev) (Rig.arch dev);
       let go () =
         let module D = (val Nx.devices [ dev ] : Nx.Devices) in
         let module R = Run (D) in
         R.run c m d0 steps
       in
       (match go () with
        | gpu ->
          Printf.printf
            "  gpu vs host after %d steps: worst %.3g\n%!"
            steps
            (worst gpu host)
        | exception e -> Printf.printf "  gpu raised: %s\n%!" (Exn.to_string e)))
