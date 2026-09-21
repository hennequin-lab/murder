(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Terms of the commutation-equation basis.

   Invariance forces an orbit average to be a linear superposition of sparse
   basis tensors; a term describes one such tensor by the index ties it
   imposes (groups of indices forced equal, i.e. products of Kronecker deltas)
   and the free axes left to carry its factor. Indices name axes of the left
   or right side of the equation (see {!Index}); the same axis may occur
   several times within a term (once per side of the orbit average). *)

open Base

type t =
  { ties : Index.t list list
  ; free : Index.t list
  }
[@@deriving compare, equal, sexp]

let map t ~f =
  { ties = List.map t.ties ~f:(fun group -> List.map group ~f)
  ; free = List.map t.free ~f
  }

(* Canonical order: ties sorted by first index, indices sorted within a tie. *)
let sort t =
  let ties =
    List.map t.ties ~f:(List.sort ~compare:Index.compare)
    |> List.sort ~compare:(fun a b ->
      match a, b with
      | a :: _, b :: _ -> Index.compare a b
      | _ -> assert false)
  in
  let free = List.sort t.free ~compare:Index.compare in
  { ties; free }

(* Equality of terms, insensitive to the order of ties, of indices within a
   tie, and of free axes. *)
let equal a b = equal (sort a) (sort b)
let transpose t = map t ~f:Index.transpose |> sort
let is_symmetric t = equal t (transpose t)

let to_string { ties; free } =
  let ties = List.map ties ~f:(fun ids -> "delta_" ^ Index.string_of_ids ids) in
  (match free with
   | [] -> ties
   | ids -> ("A_" ^ Index.string_of_ids ids) :: ties)
  |> String.concat ~sep:" "

let indices_involved t =
  List.fold (List.concat t.ties @ t.free) ~init:(Set.empty (module Index)) ~f:Set.add

(* [normalization ~dims t] is [1 / sqrt D], where [D] is the product of the
   dimensions of the axes the term does not involve. The convention makes the
   basis elements orthonormal under the normalized inner product below. *)
let normalization ~(dims : int list Sides.t) t =
  let involved = indices_involved t in
  let dim_of = Index.dim_of ~dims in
  let total =
    Index.all_indices ~dims
    |> List.filter ~f:(fun i -> not (Set.mem involved i))
    |> List.fold ~init:1 ~f:(fun acc i -> Int.(acc * dim_of i))
  in
  Float.(1.0 / sqrt (of_int total))

(* Symbolic inner product of two basis terms: the delta constraints of both
   terms are merged into one partition of the involved indices (each merged
   class contributes the dimension of the axis it forces equal), axes that
   appear in neither term contribute their dimension as a free sum, and the
   result is scaled by both normalizations. *)
let inner_product ~(dims : int list Sides.t) term1 term2 =
  assert (List.equal Index.equal term1.free term2.free);
  let dim_of = Index.dim_of ~dims in
  let normalizer = Float.(normalization ~dims term1 * normalization ~dims term2) in
  let constraint_groups =
    let rec merge_constraints constraints =
      let rec merge_one changed acc = function
        | [] -> changed, acc
        | group :: rest ->
          let overlapping, non_overlapping =
            List.partition_tf acc ~f:(fun existing_group ->
              List.exists group ~f:(fun idx ->
                List.mem existing_group idx ~equal:Index.equal))
          in
          (match overlapping with
           | [] -> merge_one changed (group :: acc) rest
           | _ ->
             let merged =
               List.fold overlapping ~init:group ~f:(fun acc existing ->
                 List.dedup_and_sort ~compare:Index.compare (acc @ existing))
             in
             merge_one true (merged :: non_overlapping) rest)
      in
      let changed, new_constraints = merge_one false [] constraints in
      if changed then merge_constraints new_constraints else new_constraints
    in
    merge_constraints (term1.ties @ term2.ties)
    |> List.map ~f:(List.sort ~compare:Index.compare)
  in
  let contrib_from_ones =
    let to_exclude =
      List.fold
        (term1.free @ List.concat constraint_groups)
        ~init:(Set.empty (module Index))
        ~f:Set.add
    in
    Index.all_indices ~dims
    |> List.filter ~f:(fun i -> not (Set.mem to_exclude i))
    |> List.fold ~init:1 ~f:(fun acc i -> Int.(acc * dim_of i))
  in
  List.fold
    constraint_groups
    ~init:Float.(normalizer * of_int contrib_from_ones)
    ~f:(fun acc group ->
      match group with
      | [] -> acc
      | idx :: _ -> Float.(acc * of_int (dim_of idx)))

(* ------------------------------------------------------------------------
   Tensor side
   ------------------------------------------------------------------------ *)

(* [coefficient ~dims term data] projects [data] onto the basis tensor
   described by [term]: every tie is a repeated index in the einsum equation
   (a repeated index names the diagonal), the axes that carry no factor are
   summed, and the free axes are left as the result's shape. [data] is either
   a dense tensor of shape [dims.left @ dims.right] or the outer product of a
   left-shaped and a right-shaped tensor. The result is scaled by the term's
   normalization. *)
let coefficient ~(dims : int list Sides.t) term =
  let dim_of = Index.dim_of ~dims in
  let normalizer = normalization ~dims term in
  let ids =
    Sides.
      { left = List.mapi dims.left ~f:(fun i _ -> Index.Left i)
      ; right = List.mapi dims.right ~f:(fun i _ -> Index.Right i)
      }
  in
  let in_eq =
    Sides.map ids ~f:(List.map ~f:Index.to_char) |> Sides.map ~f:Bytes.of_char_list
  in
  (* at the end only free axes remain *)
  let out_eq = List.map term.free ~f:Index.to_char |> String.of_char_list in
  (* merge every member of a tie group into the label of its representative *)
  let sub_in c =
    List.iter ~f:(function
      | Index.Left id -> Bytes.set in_eq.left id c
      | Index.Right id -> Bytes.set in_eq.right id c)
  in
  List.iter term.ties ~f:(function
    | Index.Left rep :: rest -> sub_in (Bytes.get in_eq.left rep) rest
    | Index.Right rep :: rest -> sub_in (Bytes.get in_eq.right rep) rest
    | [] -> assert false);
  let in_eq = Sides.map in_eq ~f:Bytes.to_string in
  let full_equation = in_eq.left ^ in_eq.right ^ "->" ^ out_eq in
  (* [Nx.einsum] rejects an empty operand, so a one-sided term uses the
     single-operand form of the equation; the missing side is then a scalar
     (it has no axes) and multiplies the result. *)
  let split_equation, sides =
    match in_eq.left, in_eq.right with
    | left, "" -> left ^ "->" ^ out_eq, `Left
    | "", right -> right ^ "->" ^ out_eq, `Right
    | left, right -> left ^ "," ^ right ^ "->" ^ out_eq, `Both
  in
  let free_shape = Array.of_list (List.map term.free ~f:dim_of) in
  let expected_shape = Array.of_list (dims.left @ dims.right) in
  fun data ->
    let result =
      match data with
      | `Full x -> Nx.einsum full_equation [| Nx.reshape expected_shape x |]
      | `Outer_product (x_left, x_right) ->
        let x_left = Nx.reshape (Array.of_list dims.left) x_left in
        let x_right = Nx.reshape (Array.of_list dims.right) x_right in
        (match sides with
         | `Left -> Nx.einsum split_equation [| x_left |] |> Nx.mul x_right
         | `Right -> Nx.einsum split_equation [| x_right |] |> Nx.mul x_left
         | `Both -> Nx.einsum split_equation [| x_left; x_right |])
    in
    let result = Nx.contiguous result in
    Nx.Infix.(result *$ normalizer) |> Nx.reshape free_shape
