(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = int

let of_int i =
  if i < 0 then invalid_arg "Blade.of_int: negative index";
  i

let to_int b = b

let of_basis i =
  if i < 1 then invalid_arg "Blade.of_basis: index must be >= 1";
  1 lsl (i - 1)

let contains b i =
  if i < 1 then invalid_arg "Blade.contains: index must be >= 1";
  b land (1 lsl (i - 1)) <> 0

let is_scalar b = b = 0
let rec popcount x = if x = 0 then 0 else (x land 1) + popcount (x lsr 1)
let grade = popcount
let equal (a : t) b = a = b
let compare (a : t) b = Stdlib.compare a b
let hash (b : t) = Hashtbl.hash b

let all n =
  if n < 0 then invalid_arg "Blade.all: negative dimension";
  Array.init (1 lsl n) (fun i -> i)

let reverse_sign b =
  let k = grade b in
  if k * (k - 1) / 2 land 1 = 0 then 1 else -1

let grade_involution_sign b = if grade b land 1 = 0 then 1 else -1

let clifford_conjugation_sign b =
  let k = grade b in
  if k * (k + 1) / 2 land 1 = 0 then 1 else -1

let mul_sign a b =
  let count = ref 0 in
  let x = ref a in
  while !x <> 0 do
    let ia = ref 0 in
    let xa = ref !x in
    while !xa land 1 = 0 do
      incr ia;
      xa := !xa lsr 1
    done;
    let y = ref b in
    let ib = ref 0 in
    while !y <> 0 do
      if !y land 1 <> 0 && !ib < !ia then incr count;
      incr ib;
      y := !y lsr 1
    done;
    x := !x land (!x - 1)
  done;
  if !count land 1 = 0 then 1 else -1

let complement ~n b =
  if n < 0 then invalid_arg "Blade.complement: negative dimension";
  ((1 lsl n) - 1) lxor b

let complement_sign ~n b = mul_sign b (complement ~n b)

let to_string b =
  if b = 0
  then "1"
  else (
    let buf = Buffer.create 8 in
    Buffer.add_char buf 'e';
    let i = ref 1 in
    let x = ref b in
    while !x <> 0 do
      if !x land 1 <> 0 then Buffer.add_string buf (string_of_int !i);
      incr i;
      x := !x lsr 1
    done;
    Buffer.contents buf)

let pp ppf b = Format.pp_print_string ppf (to_string b)
