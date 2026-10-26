(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shared helpers for the geom test suite. *)

open Windtrap

let f32 = Nx.float32
let f64 = Nx.float64
let scalar x = Nx.scalar f32 x
let scalar64 x = Nx.scalar f64 x
let vec2 x y = Nx.create f32 [| 2 |] [| x; y |]
let vec3 x y z = Nx.create f32 [| 3 |] [| x; y; z |]
let vec4 x y z w = Nx.create f32 [| 4 |] [| x; y; z; w |]
let vec6 a b c d e f = Nx.create f32 [| 6 |] [| a; b; c; d; e; f |]
let item indices x = Nx.item indices x

(* The tolerance follows float32 arithmetic through a handful of products. *)
let rel = 2e-4
let abs = 2e-4

let close_scalar ?(rel = rel) ?(abs = abs) a b =
  a = b
  || Float.abs (a -. b) <= abs
  || Float.abs (a -. b) <= rel *. Float.max (Float.abs a) (Float.abs b)

let close ?(rel = rel) ?(abs = abs) a b =
  let sa = Nx.shape a
  and sb = Nx.shape b in
  Array.length sa = Array.length sb
  && Array.for_all2 Int.equal sa sb
  &&
  let a = Nx.to_array (Nx.cast f64 a)
  and b = Nx.to_array (Nx.cast f64 b) in
  Array.length a = Array.length b && Array.for_all2 (close_scalar ~rel ~abs) a b

let tensor () = Testable.make ~pp:Nx.pp ~equal:close

let tensor_exact () =
  Testable.make ~pp:Nx.pp ~equal:(fun a b -> close ~rel:0. ~abs:0. a b)

let equal_tensor ?msg ?(rel = rel) ?(abs = abs) expected actual =
  equal ?msg (Testable.make ~pp:Nx.pp ~equal:(close ~rel ~abs)) expected actual

let equal_floats ?msg ?(eps = 1e-4) expected actual =
  equal ?msg (array (float eps)) expected (Nx.to_array actual)

let equal_bools ?msg expected actual =
  equal ?msg (array bool) expected (Nx.to_array actual)

let equal_scalar ?msg ?(eps = 1e-4) expected actual =
  equal ?msg (float eps) expected (item [] actual)

let equal_bool ?msg expected actual = equal ?msg bool expected (item [] actual)

let gen_mv size of_array =
  Gen.map of_array (Gen.array ~size:(Gen.constant size) (Gen.float_range (-2.) 2.))

let nonzero_gen gen pred = Gen.such_that pred (Gen.map (fun x -> x) gen)
let string_of_pp pp v = Format.asprintf "%a" pp v
