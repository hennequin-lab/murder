(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Axis indices of a commutation equation.

   Every index names an axis of one of the two sides of a commutation
   equation: [Left i] is the [i]-th axis of the left side, [Right i] the
   [i]-th axis of the right side. Indices are compared and collected in sets
   (a term ties copies of the same axis together), and compiled to einsum
   characters in {!Compiler}. *)

open Base

module T = struct
  type t =
    | Left of int
    | Right of int

  let compare (a : t) (b : t) = Poly.compare a b
  let equal (a : t) (b : t) = Poly.equal a b

  let sexp_of_t = function
    | Left i -> Sexp.List [ Sexp.Atom "Left"; Sexp.Atom (Int.to_string i) ]
    | Right i -> Sexp.List [ Sexp.Atom "Right"; Sexp.Atom (Int.to_string i) ]
end

include T
include Comparator.Make (T)

exception Out_of_range of t

let transpose = function
  | Left id -> Right id
  | Right id -> Left id

(* Einsum characters are allocated per side so that a character identifies an
   axis unambiguously: left axes [0..12] map to [a..m], right axes [0..11] to
   [n..y], and [z] is reserved for the compiler's batch axis. *)
let to_char = function
  | Left i when i >= 0 && i < 13 -> Char.of_int_exn (Char.to_int 'a' + i)
  | Right i when i >= 0 && i < 12 -> Char.of_int_exn (Char.to_int 'n' + i)
  | id -> raise (Out_of_range id)

let string_of_ids ids = String.of_char_list (List.map ids ~f:to_char)

let is_left = function
  | Left _ -> true
  | Right _ -> false

let is_right = function
  | Left _ -> false
  | Right _ -> true

(* [dim_of ~dims id] is the size of axis [id] in [dims]. *)
let dim_of ~(dims : int list Sides.t) = function
  | Left i -> List.nth_exn dims.left i
  | Right i -> List.nth_exn dims.right i

(* [axis_of ~dims id] is the position of [id] in the flattened
   [dims.left @ dims.right] axis order. *)
let axis_of ~(dims : int list Sides.t) =
  let n_left = List.length dims.left in
  function
  | Left i -> i
  | Right i -> i + n_left

let all_indices ~(dims : int list Sides.t) =
  List.mapi dims.left ~f:(fun i _ -> Left i)
  @ List.mapi dims.right ~f:(fun i _ -> Right i)

let all_axes ~(dims : int list Sides.t) =
  List.range 0 Int.(List.length dims.left + List.length dims.right)

let filter_left =
  List.filter_map ~f:(function
    | Left i -> Some i
    | Right _ -> None)

let filter_right =
  List.filter_map ~f:(function
    | Left _ -> None
    | Right i -> Some i)
