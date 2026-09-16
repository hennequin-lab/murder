(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Symmetry specifications.

   The user describes each parameter tensor's symmetry by one spec per axis:
   [Id] for an axis left untouched by the symmetry group ("free": it survives
   in the factors), [Perm i] for an axis acted on by the group [i] (axes tied
   to the same [i] are transformed by the same group element), and [Absent]
   for a side with no axis, used to pad the side of a compilation. *)

open Base

type spec =
  | Absent
  | Id
  | Perm of int

type t = spec list Sides.t

let equal_spec (a : spec) (b : spec) = Poly.equal a b
let compare_spec (a : spec) (b : spec) = Poly.compare a b

let label_of =
  let label_of_side side =
    List.map side ~f:(function
      | Absent -> ""
      | Id -> "I"
      | Perm i -> "P" ^ Int.to_string Int.(i + 1))
    |> String.concat ~sep:"x"
  in
  fun (symm : t) ->
    String.concat ~sep:"_" [ label_of_side symm.left; label_of_side symm.right ]

(* [surrogate_dims ~surrogate_dim specs dims] is [dims] with every permuted
   axis replaced by [min surrogate_dim dim], while [Id] axes keep their
   dimension. The surrogate group must stay small but *non-trivial*: with a
   permuted axis of size 1 the surrogate group is trivial, the distinct basis
   components coincide, and the estimated curvature is identically zero. The
   paper uses a surrogate dimension of 2 for permutation groups. *)
let surrogate_dims ~surrogate_dim specs dims =
  List.map2_exn specs dims ~f:(fun spec dim ->
    match spec with
    | Id -> dim
    | Perm _ -> Int.min surrogate_dim dim
    | Absent ->
      invalid_arg "Symmetry.surrogate_dims: Absent axis in a parameter symmetry spec")
