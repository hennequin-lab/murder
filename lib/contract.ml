(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Pairwise einsum contraction.

   Multi-operand [Nx.einsum] chooses a contraction path whose intermediate can
   be a transposed view; a later contraction then reshapes that view, which
   fails on non-contiguous strides. The compiler therefore contracts one
   operand at a time, materializing each intermediate, with axis labels carried
   explicitly as characters (one per axis, unique within an operand). *)

open Base

let check_unique ~what labels =
  let rec go = function
    | a :: (b :: _ as rest) ->
      if Char.equal a b
      then invalid_arg (Printf.sprintf "Contract: duplicate label '%c' in %s" a what)
      else go rest
    | _ -> ()
  in
  go (List.sort labels ~compare:Char.compare)

(* [binary ~labels_a ~labels_b ~out a b] contracts [a] and [b], keeping the
   labels [out] (in that order) and summing the rest. Operands are
   materialized first: Nx's multi-operand contraction can reshape a
   non-contiguous intermediate, and a broadcast view (an outer product, a
   broadcast factor) then trips the [Invalid_argument] on incompatible strides
   (see PLAN.md §10). *)
let binary ~labels_a ~labels_b ~out a b =
  check_unique ~what:"left operand" labels_a;
  check_unique ~what:"right operand" labels_b;
  let equation =
    String.concat ~sep:"," [ String.of_char_list labels_a; String.of_char_list labels_b ]
    ^ "->"
    ^ String.of_char_list out
  in
  Nx.contiguous (Nx.einsum equation [| Nx.contiguous a; Nx.contiguous b |])

(* [permute_sum ~labels ~output t] moves the axes labelled [output] (in that
   order) to the front and sums the remaining axes. *)
let permute_sum ~labels ~output t =
  let index c =
    fst (Option.value_exn (List.findi labels ~f:(fun _ c' -> Char.equal c c')))
  in
  let out_index = List.map output ~f:index in
  let rest_index =
    List.filter_map
      (List.mapi labels ~f:(fun i c -> i, c))
      ~f:(fun (i, _) -> if List.mem out_index i ~equal:Int.equal then None else Some i)
  in
  let t = Nx.transpose ~axes:(out_index @ rest_index) t |> Nx.contiguous in
  let n = List.length output in
  let out_shape = Array.sub (Nx.shape t) ~pos:0 ~len:n in
  let out_size = Array.fold out_shape ~init:1 ~f:Int.( * ) in
  let rest = Nx.numel t / out_size in
  Nx.reshape [| out_size; rest |] t |> Nx.sum ~axes:[ 1 ] |> Nx.reshape out_shape
