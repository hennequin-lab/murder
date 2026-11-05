(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Where does a contact gradient become a NaN?

   One stage at a time: each row is the reverse mode of a scalar read from that
   stage's output, with respect to the position, on a *fresh* data record (a
   value a differentiation read is dead afterwards, so reusing one is how a
   diagnostic lies to you). The first row that is not finite names the stage.

   Run: dune exec packages/mjc/bench/spike_contact_grad.exe -- boxes2 *)

open Base
module Printf = Stdlib.Printf
module Nxi32 = Stdlib.Int32

let f64 = Nx.float64

let load name =
  let candidate d =
    let path = d ^ "/" ^ name ^ ".mjcb" in
    Option.some_if (Stdlib.Sys.file_exists path) path
  in
  let path =
    match
      List.find_map [ "test/fixtures"; "packages/mjc/test/fixtures" ] ~f:candidate
    with
    | Some p -> p
    | None -> failwith (Printf.sprintf "no fixture %s" name)
  in
  match Mjc.Mjcb.read path with
  | Ok m -> m
  | Error e -> failwith e

let () =
  let name = if Array.length Stdlib.Sys.argv > 1 then Stdlib.Sys.argv.(1) else "boxes2" in
  let m = load name in
  let d0 = Mjc.Data.create m in
  Printf.printf "%s: nq %d\n%!" name (Array.fold (Nx.shape d0.qpos) ~init:1 ~f:( * ));
  let open Mjc in
  let stage label get =
    let f qpos =
      let d = { d0 with qpos } in
      get (Forward.forward m d)
    in
    let out = f d0.qpos in
    let _, pullback = Rune.vjp Nx.Ptree.tensor Nx.Ptree.tensor f d0.qpos in
    let g = pullback (Nx.ones_like out) in
    let a = Nx.to_array g in
    let nan = Array.count a ~f:Float.is_nan in
    let mx = Array.fold a ~init:0.0 ~f:(fun acc v -> Float.max acc (Float.abs v)) in
    Printf.printf
      "  %-16s %3d NaN of %3d, max |dq| %.4g\n%!"
      label
      nan
      (Array.length a)
      mx
  in
  stage "xpos" (fun d -> d.xpos);
  stage "xquat" (fun d -> d.xquat);
  stage "xipos" (fun d -> d.xipos);
  stage "cdof" (fun d -> d.cdof);
  stage "qfrc_bias" (fun d -> d.qfrc_bias);
  stage "mass_matrix" (fun d -> d.mass_matrix);
  stage "contact_dist" (fun d -> d.contact_dist);
  stage "contact_pos" (fun d -> d.contact_pos);
  stage "contact_frame" (fun d -> d.contact_frame);
  stage "efc_pos" (fun d -> d.efc_pos);
  stage "efc_J" (fun d -> d.efc_J);
  stage "efc_aref" (fun d -> d.efc_aref);
  stage "efc_force" (fun d -> d.efc_force);
  stage "qacc" (fun d -> d.qacc);
  ()
