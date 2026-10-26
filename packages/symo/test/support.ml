(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Brute-force references for the test suites: enumerate the whole index space
   of a [dims] specification, evaluate the raw indicator tensor of a term, and
   sum the symbolic inner products the compiler computes directly. *)

open Base

let dims ~left ~right = { Symo.Sides.left; right }

(* Every index tuple of the index space, in [Index.all_indices] order. *)
let all_tuples (dims : int list Symo.Sides.t) =
  let axes = Symo.Index.all_indices ~dims in
  let rec go acc = function
    | [] -> [ List.rev acc ]
    | i :: rest ->
      let d = Symo.Index.dim_of ~dims i in
      List.concat_map (List.range 0 d) ~f:(fun v -> go ((i, v) :: acc) rest)
  in
  go [] axes

(* The raw indicator tensor of a term: [1] when every tie holds, [0]
   otherwise. No normalization. *)
let raw_term_value (term : Symo.Term.t) tuple =
  let get i = List.Assoc.find tuple i ~equal:Symo.Index.equal in
  List.for_all term.ties ~f:(function
    | [] -> true
    | x :: rest -> List.for_all rest ~f:(fun y -> Option.equal Int.equal (get x) (get y)))

(* [<term_1, term_2>] as a sum over the index space. Free axes are factor
   coordinates, not contracted indices: the symbolic inner product ignores
   them, so the full-space sum is divided by the number of free assignments.
   Ties never involve free axes, so that count is the same for every term. *)
let inner_product_brute ~dims t1 t2 =
  let n1 = Symo.Term.normalization ~dims t1 in
  let n2 = Symo.Term.normalization ~dims t2 in
  let free_count =
    List.fold t1.free ~init:1 ~f:(fun acc i -> acc * Symo.Index.dim_of ~dims i)
  in
  let total =
    List.fold (all_tuples dims) ~init:0. ~f:(fun acc tuple ->
      if raw_term_value t1 tuple && raw_term_value t2 tuple
      then acc +. (n1 *. n2)
      else acc)
  in
  total /. Float.of_int free_count

let flat_index dims coords =
  List.fold2_exn dims coords ~init:0 ~f:(fun acc d v -> (acc * d) + v)

(* The coefficient tensor of a term by direct summation: for every assignment
   of the non-free axes, the normalized data value is accumulated at the free
   coordinates. *)
let coefficient_brute ~dims (term : Symo.Term.t) data =
  let norm = Symo.Term.normalization ~dims term in
  let free_dims = List.map term.free ~f:(Symo.Index.dim_of ~dims) in
  let cells = List.fold free_dims ~init:1 ~f:Int.( * ) in
  let acc = Array.create ~len:cells 0.0 in
  List.iter (all_tuples dims) ~f:(fun tuple ->
    if raw_term_value term tuple
    then (
      let free_coords =
        List.map term.free ~f:(fun i ->
          List.Assoc.find_exn tuple i ~equal:Symo.Index.equal)
      in
      let idx = flat_index free_dims free_coords in
      acc.(idx) <- acc.(idx) +. (norm *. Nx.item (List.map tuple ~f:snd) data)));
  Nx.create Nx.float32 (Array.of_list free_dims) acc
