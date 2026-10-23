(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = int array

let check sigma =
  Array.iter
    (fun s ->
       if s <> -1 && s <> 0 && s <> 1
       then invalid_arg "Signature.make: entries must be -1, 0 or 1")
    sigma

let make sigma =
  check sigma;
  Array.copy sigma

let euclidean n =
  if n < 0 then invalid_arg "Signature.euclidean: negative dimension";
  Array.make n 1

let projective n =
  if n < 0 then invalid_arg "Signature.projective: negative dimension";
  Array.init (n + 1) (fun i -> if i < n then 1 else 0)

let conformal n =
  if n < 0 then invalid_arg "Signature.conformal: negative dimension";
  Array.init (n + 2) (fun i -> if i < n + 1 then 1 else -1)

let conformal3 = conformal 3
let to_array sigma = Array.copy sigma
let dim sigma = Array.length sigma
let size sigma = 1 lsl Array.length sigma
let count p sigma = Array.fold_left (fun acc s -> if p s then acc + 1 else acc) 0 sigma
let positives sigma = count (fun s -> s = 1) sigma
let negatives sigma = count (fun s -> s = -1) sigma
let nulls sigma = count (fun s -> s = 0) sigma
let is_degenerate sigma = nulls sigma > 0

let pseudoscalar_square sigma =
  let n = dim sigma in
  if is_degenerate sigma
  then 0
  else (
    let q = negatives sigma in
    let sign_reorder = if n * (n - 1) / 2 land 1 = 0 then 1 else -1 in
    let sign_metric = if q land 1 = 0 then 1 else -1 in
    sign_reorder * sign_metric)

let equal (a : t) b = a = b

let to_string sigma =
  Printf.sprintf "(%d,%d,%d)" (positives sigma) (negatives sigma) (nulls sigma)

let pp ppf sigma = Format.pp_print_string ppf (to_string sigma)
