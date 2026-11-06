(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shared helpers for the mjc test suite: the fixture reader, the parity
   comparison and the tensor assertions. *)

open Base
open Nx
open Windtrap
open Mjc
module Stdbytes = Stdlib.Bytes
module Stdint32 = Stdlib.Int32
module Stdint64 = Stdlib.Int64

let f64 = Nx.float64
let fixtures = "fixtures"

(* {1 Reading a fixture} *)

type fixture =
  { name : string
  ; nstates : int
  ; tensors : (string * (int array * float array)) list
  }

exception Bad of string

let read_fixture path : fixture =
  let src = Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all in
  let pos = ref 0 in
  let need n = if !pos + n > String.length src then raise (Bad "truncated") in
  let raw n =
    need n;
    let s = Stdlib.String.sub src !pos n in
    pos := !pos + n;
    s
  in
  let i32 () =
    need 4;
    let v = Stdbytes.get_int32_le (Stdbytes.unsafe_of_string src) !pos in
    pos := !pos + 4;
    Stdint32.to_int v
  in
  let f64v () =
    need 8;
    let v = Stdbytes.get_int64_le (Stdbytes.unsafe_of_string src) !pos in
    pos := !pos + 8;
    Stdint64.float_of_bits v
  in
  let string () =
    let n = i32 () in
    raw n
  in
  if not (String.equal (raw 4) "MJBF") then raise (Bad "not a fixture (bad magic)");
  ignore (i32 () : int);
  let name = string () in
  let nstates = i32 () in
  let nfields = i32 () in
  let tensors =
    List.init nfields ~f:(fun _ ->
      let field = string () in
      let rank = i32 () in
      let shape = Array.init rank ~f:(fun _ -> i32 ()) in
      let numel = Array.fold shape ~init:1 ~f:( * ) in
      let data = Array.init numel ~f:(fun _ -> f64v ()) in
      field, (shape, data))
  in
  if not (String.equal (raw 4) "MJFE") then raise (Bad "missing trailer");
  { name; nstates; tensors }

let tensor f name =
  match List.Assoc.find f.tensors ~equal:String.equal name with
  | Some x -> x
  | None -> raise (Bad (Printf.sprintf "fixture %s has no %S" f.name name))

(* Row [s] of a tensor with a leading state axis, flattened. *)
let row (shape, data) s =
  let width =
    Array.fold (Array.sub shape ~pos:1 ~len:(Array.length shape - 1)) ~init:1 ~f:( * )
  in
  Array.sub data ~pos:(s * width) ~len:width

(* {1 Comparing} *)

(* The relative error of two equally sized arrays, and where it is largest.
   The denominator is the larger of the two values, so a value that should be
   zero but is not fails rather than passing under an absolute tolerance. *)
let max_rel_error expected actual =
  if Array.length expected <> Array.length actual
  then
    raise
      (Bad
         (Printf.sprintf
            "expected %d values, got %d"
            (Array.length expected)
            (Array.length actual)));
  let worst = ref 0.0
  and at = ref (-1) in
  for i = 0 to Array.length expected - 1 do
    let e = expected.(i)
    and a = actual.(i) in
    (* Mixed criterion: |e - a| <= tol * (1 + max |e| |a|). A component that
       should be zero is held to the same absolute bound as one of size one,
       which a pure relative error cannot express. *)
    let scale = 1.0 +. Float.max (Float.abs e) (Float.abs a) in
    let r = Float.abs (e -. a) /. scale in
    if Float.(r > !worst)
    then (
      worst := r;
      at := i)
  done;
  !worst, !at

let parity ?(tol = 1e-12) ~name (expected : float array) (actual : float array) : unit =
  let worst, at = max_rel_error expected actual in
  let msg =
    if at < 0
    then Printf.sprintf "%s: no values" name
    else
      Printf.sprintf
        "%s: worst relative error %.3g at %d (expected %.17g, got %.17g)"
        name
        worst
        at
        expected.(at)
        actual.(at)
  in
  equal ~msg float_exact 0.0 (Float.max 0.0 (worst -. tol))

let equal_row ?(tol = 1e-12) ~name (shape, data) s x =
  parity ~tol ~name (row (shape, data) s) (to_array x)

(* {1 Building}

   A model is always read from the corpus; the tests never hand-build one. *)

let model name =
  match Mjcb.read (fixtures ^ "/" ^ name ^ ".mjcb") with
  | Ok m -> m
  | Error e -> raise (Bad e)

let fixture name = read_fixture (fixtures ^ "/" ^ name ^ ".mjbf")

let data_with_qpos m qpos =
  let d = Data.create m in
  { d with qpos = Nx.create f64 [| Array.length qpos |] qpos }

(* {1 Generators} *)

let unit_quat_gen =
  Gen.map
    (fun v ->
       let n = Float.sqrt (Array.fold v ~init:0.0 ~f:(fun acc x -> acc +. (x *. x))) in
       Array.map v ~f:(fun x -> x /. n))
    (Gen.array ~size:(Gen.constant 4) (Gen.float_range (-1.) 1.))

let vec3_gen = Gen.array ~size:(Gen.constant 3) (Gen.float_range (-2.) 2.)
let quat_and_vec_gen = Gen.pair unit_quat_gen vec3_gen
let quat_of_array a = Nx.create f64 [| 1; 4 |] a
let vec_of_array a = Nx.create f64 [| 1; 3 |] a

(* {1 Batching}

   [batch_data ds] stacks every leaf of the data values along a new leading
   axis, which is the shape [Rune.vmap] maps over. The description carries no
   shapes, so folding [map2] over the list builds the batched structure one
   state at a time. *)

let vec1 a = Nx.create f64 [| Array.length a |] a

let batch_data ds =
  match ds with
  | [] -> invalid_arg "batch_data"
  | [ d ] -> d
  | d0 :: d1 :: rest ->
    let both =
      Nx.Ptree.map2 (Data.ptree ()) (fun _ a b -> Nx.stack ~axis:0 [ a; b ]) d0 d1
    in
    List.fold rest ~init:both ~f:(fun acc x ->
      Nx.Ptree.map2
        (Data.ptree ())
        (fun _ a b -> Nx.concatenate ~axis:0 [ a; Nx.unsqueeze ~axes:[ 0 ] b ])
        acc
        x)

let fixture_traj name = read_fixture (fixtures ^ "/" ^ name ^ ".traj.mjbf")
