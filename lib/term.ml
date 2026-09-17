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

let raw_equal (a : t) (b : t) = Poly.equal a b
let compare (a : t) (b : t) = Poly.compare a b

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
let equal a b = raw_equal (sort a) (sort b)
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
   described by [term]: every tie is contracted against a delta operand, the
   axes that carry no factor are summed, and the free axes are left as the
   result's shape. [data] is either a dense tensor of shape
   [dims.left @ dims.right], or the outer product of a left-shaped and a
   right-shaped tensor. The result is scaled by the term's normalization. *)
let coefficient ~(dims : int list Sides.t) term data =
  let dim_of = Index.dim_of ~dims in
  let normalizer = normalization ~dims term in
  let label = Index.to_char in
  let free_labels = List.map term.free ~f:label in
  let delta group =
    Delta.tensor ~order:(List.length group) (dim_of (List.hd_exn group))
  in
  match data with
  | `Full x ->
    let data = Nx.reshape (Array.of_list (dims.left @ dims.right)) x in
    let labels = List.map (Index.all_indices ~dims) ~f:label in
    let t, labels =
      List.fold term.ties ~init:(data, labels) ~f:(fun (t, labels) group ->
        let tie_labels = List.map group ~f:label in
        let out =
          List.filter labels ~f:(fun c -> not (List.mem tie_labels c ~equal:Char.equal))
        in
        Contract.binary ~labels_a:labels ~labels_b:tie_labels ~out t (delta group), out)
    in
    Nx.mul_s (Contract.permute_sum ~labels ~output:free_labels t) normalizer
  | `Outer_product (x_left, x_right) ->
    (* Never form [x_left ⊗ x_right]: that intermediate has the size of the
       dense tensor, [prod dims.left * prod dims.right] (quartic in the
       hidden dimension for a square weight pair), and it was what made the
       second-order estimation OOM. Contract each tie into the operand that
       carries its indices instead; a tie with left indices keeps its right
       labels on the left operand, so the two operands meet only once, in
       the final contraction, whose intermediate is never larger than the
       coefficient itself. *)
    let left_labels = List.mapi dims.left ~f:(fun i _ -> label (Index.Left i)) in
    let right_labels = List.mapi dims.right ~f:(fun i _ -> label (Index.Right i)) in
    let t_left, left_labels, t_right, right_labels =
      List.fold
        term.ties
        ~init:(x_left, left_labels, x_right, right_labels)
        ~f:(fun (t_left, left_labels, t_right, right_labels) group ->
          let tie_labels = List.map group ~f:label in
          let kept labels =
            List.filter labels ~f:(fun c -> List.mem tie_labels c ~equal:Char.equal)
          in
          let dropped labels =
            List.filter labels ~f:(fun c -> not (List.mem tie_labels c ~equal:Char.equal))
          in
          if List.exists group ~f:Index.is_left
          then (
            (* The tie's right labels move to the left operand, where the
               final contraction pairs them with the right operand's. *)
            let out = dropped left_labels @ kept right_labels in
            ( Contract.binary
                ~labels_a:left_labels
                ~labels_b:tie_labels
                ~out
                t_left
                (delta group)
            , out
            , t_right
            , right_labels ))
          else (
            let out = dropped right_labels in
            ( t_left
            , left_labels
            , Contract.binary
                ~labels_a:right_labels
                ~labels_b:tie_labels
                ~out
                t_right
                (delta group)
            , out )))
    in
    let t =
      match left_labels, right_labels with
      | [], [] -> Nx.mul t_left t_right
      | _, [] ->
        Nx.mul
          (Contract.permute_sum ~labels:left_labels ~output:free_labels t_left)
          t_right
      | [], _ ->
        Nx.mul
          t_left
          (Contract.permute_sum ~labels:right_labels ~output:free_labels t_right)
      | _, _ ->
        Contract.binary
          ~labels_a:left_labels
          ~labels_b:right_labels
          ~out:free_labels
          t_left
          t_right
    in
    Nx.mul_s t normalizer
