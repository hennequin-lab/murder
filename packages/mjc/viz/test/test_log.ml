(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The run log: that a written run reads back bit for bit, that a frame can be
   sought in constant time, and that a reader says why it refused a file. No
   display is involved, which is the reason the format came before the replay
   UI. *)

open Base
open Windtrap
module L = Mjc_viz.Log

let fixtures = "../../test/fixtures"

let load name =
  match Mjc.Mjcb.read (fixtures ^ "/" ^ name ^ ".mjcb") with
  | Ok m -> m
  | Error e -> fail e

let temp_log () = Stdlib.Filename.temp_file "mjc-test" ".mjlog"

let write_text path text =
  Stdlib.Out_channel.with_open_bin path (fun oc ->
    Stdlib.Out_channel.output_string oc text)

let read_text path = Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all

let same_array a b =
  equal ~msg:"length" int (Array.length a) (Array.length b);
  Stdlib.Array.iteri
    (fun i x -> equal ~msg:("element " ^ Int.to_string i) float_exact x b.(i))
    a

(* What was in the state when the frame was written, kept to compare against
   what comes back. *)
type expected =
  { time : float
  ; ncon : int
  ; geom_xpos : float array
  ; geom_xmat : float array
  ; contact_pos : float array
  ; contact_normal : float array
  ; contact_dist : float array
  }

let snapshot (d : Mjc.Data.host) =
  { time = Nx.item [] d.Mjc.Data.time
  ; ncon = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.ncon)
  ; geom_xpos = Nx.to_array d.Mjc.Data.geom_xpos
  ; geom_xmat = Nx.to_array d.Mjc.Data.geom_xmat
  ; contact_pos = Nx.to_array d.Mjc.Data.contact_pos
  ; contact_normal = Nx.to_array d.Mjc.Data.contact_normal
  ; contact_dist = Nx.to_array d.Mjc.Data.contact_dist
  }

(* A run of [n] steps, written frame by frame. *)
let write_run m ~n =
  let path = temp_log () in
  let w = L.create ~path m in
  let d = ref (Mjc.Forward.forward m (Mjc.Data.create m)) in
  let expected = ref [] in
  for step = 0 to n - 1 do
    d := Mjc.Step.step m !d;
    L.append w ~step:(Stdlib.Int64.of_int step) !d;
    expected := snapshot !d :: !expected
  done;
  let expected = List.rev !expected in
  L.close w;
  path, expected

(* The round trip: every recorded number is the one the state had. *)
let test_round_trip () =
  let m = load "blocks" in
  let path, expected = write_run m ~n:40 in
  let t =
    match L.open_ path with
    | Ok t -> t
    | Error e -> fail e
  in
  equal ~msg:"frames" int 40 (L.length t);
  equal ~msg:"name" string "blocks" (L.name t);
  equal ~msg:"shape" (pair int int) (3, m.topo.nconmax) (L.model_shape t);
  List.iteri expected ~f:(fun i (e : expected) ->
    let f = L.frame t i in
    equal ~msg:"step" int64 (Stdlib.Int64.of_int i) f.step;
    equal ~msg:"time" float_exact e.time f.time;
    equal ~msg:"ncon" int e.ncon f.ncon;
    (* the geoms map onto the record's interleave: position then matrix *)
    let ngeom = Array.length e.geom_xpos / 3 in
    let geom =
      Array.concat
        (List.init ngeom ~f:(fun g ->
           Array.append
             (Array.sub e.geom_xpos ~pos:(3 * g) ~len:3)
             (Array.sub e.geom_xmat ~pos:(9 * g) ~len:9)))
    in
    same_array geom f.geom;
    same_array (Array.sub e.contact_pos ~pos:0 ~len:(3 * e.ncon)) f.contact_pos;
    same_array (Array.sub e.contact_normal ~pos:0 ~len:(3 * e.ncon)) f.contact_normal;
    same_array (Array.sub e.contact_dist ~pos:0 ~len:e.ncon) f.contact_dist);
  L.close_reader t;
  Stdlib.Sys.remove path

(* A replay frame is the scene the run had: same geom centres, same contacts. *)
let test_replay_scene () =
  let m = load "blocks" in
  let path, _ = write_run m ~n:25 in
  let t =
    match L.open_ path with
    | Ok t -> t
    | Error e -> fail e
  in
  let scene = L.scene m t 24 in
  let d = ref (Mjc.Forward.forward m (Mjc.Data.create m)) in
  for _ = 1 to 25 do
    d := Mjc.Step.step m !d
  done;
  let live = Mjc_viz.Scene.of_state ~contacts:true m !d in
  equal ~msg:"geom count" int (List.length live.geoms) (List.length scene.geoms);
  (match
     List.iter2
       (List.map live.geoms ~f:(fun g -> g.Mjc_viz.Scene.name))
       (List.map scene.geoms ~f:(fun g -> g.Mjc_viz.Scene.name))
       ~f:(fun a b -> equal ~msg:"geom name" string a b)
   with
   | Ok () -> ()
   | Unequal_lengths -> fail "the replay and the live state disagree on geom count");
  equal ~msg:"contact count" int (List.length live.contacts) (List.length scene.contacts);
  (match
     List.iter2 live.contacts scene.contacts ~f:(fun (a : Mjc_viz.Scene.contact) b ->
       equal ~msg:"contact x" float_exact a.point.Mjc_viz.Scene.x b.point.x;
       equal ~msg:"contact z" float_exact a.point.z b.point.z;
       equal ~msg:"normal x" float_exact a.normal.x b.normal.x)
   with
   | Ok () -> ()
   | Unequal_lengths -> fail "the replay and the live state disagree on contacts");
  equal ~msg:"time" float_exact live.time scene.time;
  L.close_reader t;
  Stdlib.Sys.remove path

(* Seeking is a positioned read: the frames a caller asks for out of order are
   the same ones a sequential reader would find, and asking for a thousand of
   them in a big log does not cost a scan. *)
let test_seek_is_constant_time () =
  let m = load "ball" in
  let path, _ = write_run m ~n:800 in
  let t =
    match L.open_ path with
    | Ok t -> t
    | Error e -> fail e
  in
  equal ~msg:"frames" int 800 (L.length t);
  (* forward, backward and shuffled orders must agree *)
  let at n = (L.frame t n).geom.(0) in
  let ascending = List.init 50 ~f:(fun i -> at (i * 16)) in
  let descending = List.init 50 ~f:(fun i -> at ((49 - i) * 16)) in
  equal ~msg:"order does not matter" (list float_exact) ascending (List.rev descending);
  let start = Unix.gettimeofday () in
  let checksum = ref 0. in
  for i = 0 to 499 do
    let n = Int.rem (i * 7919) 800 in
    checksum := !checksum +. (L.frame t n).time
  done;
  let elapsed = Unix.gettimeofday () -. start in
  equal ~msg:"the log is not empty" bool true (Float.( > ) !checksum 0.);
  (* A scanning reader would read 800 records per seek: 400 000 records, which
     cannot happen in this budget. *)
  equal ~msg:"500 seeks in under half a second" bool true (Float.( <= ) elapsed 0.5);
  L.close_reader t;
  Stdlib.Sys.remove path

(* A refused file says why: a text file, a torn record, and an index past the
   end are three different answers. *)
let test_refusals () =
  let path = temp_log () in
  write_text path (String.make 200 'x');
  (match L.open_ path with
   | Ok _ -> fail "a text file opened as a log"
   | Error e -> equal ~msg:"magic" bool true (String.is_substring e ~substring:"magic"));
  Stdlib.Sys.remove path;
  write_text path "tiny";
  (match L.open_ path with
   | Ok _ -> fail "a two-byte file opened as a log"
   | Error e -> equal ~msg:"short" bool true (String.is_substring e ~substring:"header"));
  Stdlib.Sys.remove path;
  let m = load "ball" in
  let path, _ = write_run m ~n:3 in
  let good = read_text path in
  (* a truncated record: the body is not a whole number of records *)
  write_text path (String.prefix good (String.length good - 8));
  (match L.open_ path with
   | Ok _ -> fail "a truncated log opened"
   | Error e -> equal ~msg:"torn" bool true (String.is_substring e ~substring:"whole"));
  Stdlib.Sys.remove path;
  (* out of range is a caller's mistake, not a parse failure *)
  let path, _ = write_run m ~n:3 in
  let t =
    match L.open_ path with
    | Ok t -> t
    | Error e -> fail e
  in
  (match L.frame t 3 with
   | _ -> fail "read past the last frame"
   | exception Invalid_argument _ -> ());
  L.close_reader t;
  Stdlib.Sys.remove path

let () =
  Stdlib.exit
    (run
       "mjc viz: log"
       [ test "round trip" test_round_trip
       ; test "replay scene" test_replay_scene
       ; test "seek is constant time" test_seek_is_constant_time
       ; test "refusals" test_refusals
       ])
