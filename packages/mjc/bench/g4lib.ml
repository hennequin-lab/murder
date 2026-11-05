(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reading the long recording, for the benches that measure G4.

   Two questions, which are not the same and are often conflated:

   - One step from a reference state: set the state from the recording, step
     once, compare with the recording's next state. This is accuracy per step,
     with no error accumulated, and the gate's criterion is 1e-8 relative on at
     least 99% of a thousand steps.
   - A free run: step without ever reading the recording back, and see how long
     the trajectory's shadow lasts. Every engine diverges eventually; the number
     worth printing is where.

   The fixture is written by `tools/record_fixture.py --long`. *)

open Base

let fixture path =
  let b =
    match Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all with
    | s -> s
    | exception e -> failwith (Printf.sprintf "%s: %s" path (Stdlib.Printexc.to_string e))
  in
  let at = ref 0 in
  let u16 () =
    let v = Stdlib.Bytes.get_uint16_le (Stdlib.Bytes.of_string b) !at in
    at := !at + 2;
    v
  in
  let i32 () =
    let v =
      Stdlib.Int32.to_int (Stdlib.Bytes.get_int32_le (Stdlib.Bytes.of_string b) !at)
    in
    at := !at + 4;
    v
  in
  let i64_bits () =
    let v = Stdlib.Bytes.get_int64_le (Stdlib.Bytes.of_string b) !at in
    at := !at + 8;
    v
  in
  let string () =
    let len = i32 () in
    let v = Stdlib.String.sub b !at len in
    at := !at + len;
    v
  in
  let skip n = at := !at + n in
  (* magic, a u16 version and a u16 flags word, then the model name *)
  skip 4;
  ignore (u16 () : int);
  ignore (u16 () : int);
  ignore (string () : string);
  let nstates = i32 () in
  let nfields = i32 () in
  let fields = Hashtbl.create (module String) in
  for _ = 1 to nfields do
    let name = string () in
    let rank = i32 () in
    let shape = Array.init rank ~f:(fun _ -> i32 ()) in
    let numel = Array.fold shape ~init:1 ~f:( * ) in
    let data = Array.init numel ~f:(fun _ -> Stdlib.Int64.float_of_bits (i64_bits ())) in
    Hashtbl.set fields ~key:name ~data:(shape, data)
  done;
  let get name =
    match Hashtbl.find fields name with
    | Some x -> x
    | None -> failwith (Printf.sprintf "%s has no %S" path name)
  in
  let tensor name =
    let shape, data = get name in
    Nx.create Nx.float64 shape data
  in
  let width name =
    let shape, _ = get name in
    Array.fold (Array.sub shape ~pos:1 ~len:(Array.length shape - 1)) ~init:1 ~f:( * )
  in
  ( tensor "traj_qpos"
  , tensor "traj_qvel"
  , tensor "traj_qacc"
  , tensor "traj_act"
  , tensor "traj_ctrl"
  , tensor "traj_nefc"
  , tensor "traj_ncon"
  , (nstates - 1, width "traj_qpos") )

(* The same relative error the tests use, and the same denominator: the larger
   of the two values, so a quantity that should be zero and is not fails
   instead of passing under an absolute tolerance. *)
let rel_error expected actual =
  let e = Nx.to_array expected
  and a = Nx.to_array actual in
  Array.fold2_exn e a ~init:0.0 ~f:(fun acc x y ->
    if Float.equal x y
    then acc
    else (
      let den = Float.max (Float.abs x) (Float.abs y) in
      Float.max acc (Float.abs (x -. y) /. den)))
