(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* One state's contact list and row types, for comparing with MuJoCo's when a
   trajectory disagrees. `cdump <model> <state>` reads the long recording. *)

open Base

let () =
  let name = Stdlib.Sys.argv.(1) in
  let k = Stdlib.int_of_string Stdlib.Sys.argv.(2) in
  let m =
    match Mjc.Mjcb.read (Printf.sprintf "test/fixtures/%s.mjcb" name) with
    | Ok m -> m
    | Error e -> failwith e
  in
  let g4 = G4lib.fixture (Printf.sprintf "test/fixtures/%s.g4.mjbf" name) in
  let qpos, qvel, _qacc, act, ctrl, _nefc, _ncon, _ = g4 in
  let d = Mjc.Data.create m in
  let d =
    { d with
      Mjc.Data.qpos = Nx.slice [ I k; A ] qpos
    ; qvel = Nx.slice [ I k; A ] qvel
    ; act = Nx.slice [ I k; A ] act
    ; ctrl = Nx.slice [ I k; A ] ctrl
    }
  in
  let d = Mjc.Forward.forward m d in
  let ncon = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.ncon) in
  Stdlib.Printf.printf
    "ncon %d nefc %d\n"
    ncon
    (Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.nefc));
  let arr x = Nx.to_array x in
  let show label x =
    Stdlib.Printf.printf
      "%s %s\n"
      label
      (String.concat
         ~sep:" "
         (Array.to_list (Array.map (arr x) ~f:(fun v -> Printf.sprintf "%.9g" v))))
  in
  show "qacc       " (Nx.slice [ R (0, 6) ] d.Mjc.Data.qacc);
  let smooth =
    Mjc.Mass.solve (Mjc.Mass.factor d.Mjc.Data.mass_matrix) d.Mjc.Data.qfrc_smooth
  in
  show "qacc_smooth" (Nx.slice [ R (0, 6) ] smooth);
  show "qfrc_smooth" (Nx.slice [ R (0, 6) ] d.Mjc.Data.qfrc_smooth);
  for c = 0 to ncon - 1 do
    let g a = Stdlib.Int32.to_int (Nx.item [ c; a ] d.Mjc.Data.contact_geom) in
    Stdlib.Printf.printf
      "  %d-%d dist %.9g dim %d friction %s\n"
      (g 0)
      (g 1)
      (Nx.item [ c ] d.Mjc.Data.contact_dist)
      (Stdlib.Int32.to_int (Nx.item [ c ] d.Mjc.Data.contact_dim))
      (Float.to_string (Nx.item [ c; 0 ] d.Mjc.Data.contact_friction))
  done;
  let nefc = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.nefc) in
  let counts = Array.create ~len:8 0 in
  for i = 0 to nefc - 1 do
    let t = Stdlib.Int32.to_int (Nx.item [ i ] d.Mjc.Data.efc_type) in
    counts.(t) <- counts.(t) + 1
  done;
  Array.iteri counts ~f:(fun t n ->
    if n > 0 then Stdlib.Printf.printf "  type %d: %d rows\n" t n)
