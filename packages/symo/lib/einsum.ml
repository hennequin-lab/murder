(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [pattern eq] is the einsum equation [eq], written in one-character labels
   as the compiler builds it, as an [Nx.Pattern.t]: each character is a name,
   and the names the result lacks are summed. ["zaa,a->z"] is
   ["z a a, a -> z | a"]. *)

open Base

let pattern eq =
  let i = String.substr_index_exn eq ~pattern:"->" in
  let lhs = String.prefix eq i in
  let rhs = String.drop_prefix eq (i + 2) in
  let names labels = String.concat ~sep:" " (List.map labels ~f:String.of_char) in
  let summed =
    String.to_list lhs
    |> List.filter ~f:(fun c -> Char.is_alpha c && not (String.mem rhs c))
    |> List.dedup_and_sort ~compare:Char.compare
  in
  let operands = List.map (String.split lhs ~on:',') ~f:String.to_list in
  let sums = if List.is_empty summed then "" else " | " ^ names summed in
  Nx.Pattern.v
    (String.concat ~sep:", " (List.map operands ~f:names)
     ^ " -> "
     ^ names (String.to_list rhs)
     ^ sums)
