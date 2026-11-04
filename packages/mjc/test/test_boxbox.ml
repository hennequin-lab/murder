(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Box-box manifolds against the C engine, pair by pair.

   [Collision_box.manifold] is a host port of MuJoCo's [mjc_BoxBox], not yet
   wired into the tensor narrow phase, so this test drives it directly. For
   every recorded state of [boxes2] it takes each box pair the fixture says made
   contacts, recomputes the manifold from the poses our kinematics puts the
   geoms at, and compares count, distances, points and normals. Then it does the
   other direction: every box pair the fixture did *not* report must produce no
   manifold, or the kernel invents contacts. *)

open Base
module Printf = Stdlib.Printf
module Nxi32 = Stdlib.Int32

let int_of_t x = Nxi32.to_int (Nx.item [] x)
let take n a = Array.sub a ~pos:0 ~len:n
let failures = ref 0
let compared = ref 0
let refuted = ref 0

let check ~msg ok =
  if not ok
  then (
    Int.incr failures;
    Printf.printf "FAIL %s\n%!" msg)

let close ~msg a b = check ~msg (Float.( <= ) (Float.abs (a -. b)) 1e-12)
let slice3 a i = Array.sub a ~pos:(3 * i) ~len:3
let slice9 a i = Array.sub a ~pos:(9 * i) ~len:9
let pair_key g1 g2 = Printf.sprintf "%d %d" g1 g2

(* One pair the fixture reported: our manifold must equal it, contact for
   contact, in the same order. *)
let compare_pair ~state ~g1 ~g2 ~got ~dist ~pos ~nrm ~ks =
  Int.incr compared;
  let label tag = Printf.sprintf "state %d pair %d-%d %s" state g1 g2 tag in
  check ~msg:(label "count") (Int.equal (Array.length got) (List.length ks));
  if Int.equal (Array.length got) (List.length ks)
  then
    List.iteri ks ~f:(fun n k ->
      let c = got.(n) in
      close ~msg:(label "dist") c.Mjc.Collision_box.dist dist.(k);
      for i = 0 to 2 do
        close ~msg:(label "pos") c.pos.(i) pos.((3 * k) + i);
        close ~msg:(label "normal") c.normal.(i) nrm.((3 * k) + i)
      done)

let () =
  let name = "boxes2" in
  let m = Support.model name in
  let f = Support.fixture name in
  let qpos = Support.tensor f "qpos"
  and ncon_t = Support.tensor f "ncon"
  and cdist = Support.tensor f "contact_dist"
  and cpos = Support.tensor f "contact_pos"
  and cnormal = Support.tensor f "contact_normal"
  and cgeom = Support.tensor f "contact_geom"
  and cmargin = Support.tensor f "contact_includemargin" in
  let size = Nx.to_array m.Mjc.Model.params.Mjc.Model.Params.geom_size in
  let ngeom = m.Mjc.Model.topo.ngeom in
  List.iter
    (List.range 0 (f.nstates - 1))
    ~f:(fun state ->
      let d =
        Mjc.Kinematics.forward m (Support.data_with_qpos m (Support.row qpos state))
      in
      let ncon = Int.of_float (Support.row ncon_t state).(0) in
      let pos = Nx.to_array d.Mjc.Data.geom_xpos
      and mat = Nx.to_array d.Mjc.Data.geom_xmat in
      let geom = take (ncon * 2) (Support.row cgeom state)
      and dist = take ncon (Support.row cdist state)
      and cpt = take (ncon * 3) (Support.row cpos state)
      and nrm = take (ncon * 3) (Support.row cnormal state)
      and margin = take ncon (Support.row cmargin state) in
      (* the fixture's contacts, grouped by pair, in recorded order *)
      let pairs = Hashtbl.create (module String) in
      for k = 0 to ncon - 1 do
        let g1 = Int.of_float geom.(2 * k)
        and g2 = Int.of_float geom.((2 * k) + 1) in
        let key = pair_key g1 g2 in
        let prev = Option.value (Hashtbl.find pairs key) ~default:[] in
        Hashtbl.set pairs ~key ~data:(k :: prev)
      done;
      Hashtbl.iteri pairs ~f:(fun ~key ~data ->
        let ks = List.rev data in
        let g1, g2 =
          match String.split key ~on:' ' with
          | [ a; b ] -> Int.of_string a, Int.of_string b
          | _ -> Stdlib.failwith "bad pair key"
        in
        let got =
          Mjc.Collision_box.manifold
            ~pos1:(slice3 pos g1)
            ~mat1:(slice9 mat g1)
            ~size1:(slice3 size g1)
            ~pos2:(slice3 pos g2)
            ~mat2:(slice9 mat g2)
            ~size2:(slice3 size g2)
            ~margin:margin.(List.hd_exn ks)
        in
        compare_pair ~state ~g1 ~g2 ~got ~dist ~pos:cpt ~nrm ~ks);
      (* the other direction: no manifold where the fixture reports none *)
      for g1 = 0 to ngeom - 1 do
        for g2 = g1 + 1 to ngeom - 1 do
          let recorded = Hashtbl.mem pairs (pair_key g1 g2) in
          let both_boxes =
            Poly.equal m.Mjc.Model.topo.geom_type.(g1) Mjc.Topo.Box
            && Poly.equal m.Mjc.Model.topo.geom_type.(g2) Mjc.Topo.Box
          in
          if (not recorded) && both_boxes
          then (
            Int.incr refuted;
            let got =
              Mjc.Collision_box.manifold
                ~pos1:(slice3 pos g1)
                ~mat1:(slice9 mat g1)
                ~size1:(slice3 size g1)
                ~pos2:(slice3 pos g2)
                ~mat2:(slice9 mat g2)
                ~size2:(slice3 size g2)
                ~margin:0.
            in
            check
              ~msg:(Printf.sprintf "state %d pair %d-%d must be empty" state g1 g2)
              (Int.equal (Array.length got) 0))
        done
      done);
  if !failures > 0
  then (
    Printf.printf "%d checks failed\n%!" !failures;
    Stdlib.exit 1)
  else (
    Printf.printf
      "box-box: %d pairs match the C engine at 1e-12, %d pairs correctly make no contact\n\
       %!"
      !compared
      !refuted;
    Stdlib.exit 0)
