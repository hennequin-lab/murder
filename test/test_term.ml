(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The pure front end: index bookkeeping, symmetry labels, term
   canonicalization, and the symbolic normalization and inner product checked
   against brute-force sums over the full index space. *)

open Base
open Windtrap
open Symo

let dims ~left ~right = { Sides.left; right }
let left i = Index.Left i
let right i = Index.Right i
let term ~ties ~free = { Term.ties; free }

let test_to_char () =
  is_true ~msg:"left starts at a" (Char.equal 'a' (Index.to_char (left 0)));
  is_true ~msg:"left ends at m" (Char.equal 'm' (Index.to_char (left 12)));
  is_true ~msg:"right starts at n" (Char.equal 'n' (Index.to_char (right 0)));
  is_true ~msg:"right ends at y" (Char.equal 'y' (Index.to_char (right 11)))

let test_to_char_out_of_range () =
  raises (Index.Out_of_range (left 13)) (fun () -> ignore (Index.to_char (left 13)))

let test_axis_helpers () =
  let d = dims ~left:[ 2; 3 ] ~right:[ 4 ] in
  equal int 3 (Index.dim_of ~dims:d (left 1));
  equal int 4 (Index.dim_of ~dims:d (right 0));
  equal int 2 (Index.axis_of ~dims:d (right 0));
  is_true (Index.equal (left 1) (Index.transpose (right 1)));
  is_true (Index.equal (right 1) (Index.transpose (left 1)))

let test_symmetry_label () =
  let symm = { Sides.left = [ Symmetry.Id; Perm 0 ]; right = [ Symmetry.Perm 0 ] } in
  equal string "IxP1_P1" (Symmetry.label_of symm)

let test_surrogate_dims () =
  equal
    (list int)
    [ 3; 2; 2 ]
    (Symmetry.surrogate_dims ~surrogate_dim:2 [ Symmetry.Id; Perm 0; Symmetry.Id ] [ 3; 5; 2 ])

let sample = term ~ties:[ [ left 0; left 1 ]; [ right 0; right 0 ] ] ~free:[ left 2 ]

let test_term_canonicalization () =
  let reordered =
    term ~ties:[ [ right 0; right 0 ]; [ left 1; left 0 ] ] ~free:[ left 2 ]
  in
  is_true
    ~msg:"reordering ties and indices does not change a term"
    (Term.equal sample reordered);
  is_true ~msg:"sort is idempotent" (Term.equal sample (Term.sort (Term.sort sample)));
  is_true ~msg:"sort preserves the term" (Term.equal sample (Term.sort sample))

let test_term_transpose () =
  is_true
    ~msg:"transpose is an involution"
    (Term.equal sample (Term.transpose (Term.transpose sample)));
  let symmetric = term ~ties:[ [ left 0; right 0 ] ] ~free:[] in
  is_true ~msg:"a self-transpose term is symmetric" (Term.is_symmetric symmetric);
  is_false ~msg:"a generic term is not symmetric" (Term.is_symmetric sample)

let test_normalization () =
  (* [left 0] is tied with itself, [right 0] is free, so only [left 1]
     (dimension 3) is uninvolved. *)
  let t = term ~ties:[ [ left 0; left 0 ] ] ~free:[ right 0 ] in
  let d = dims ~left:[ 2; 3 ] ~right:[ 2 ] in
  equal (float 1e-12) (1.0 /. Float.sqrt 3.0) (Term.normalization ~dims:d t)

let test_inner_product () =
  let d = dims ~left:[ 2; 3 ] ~right:[ 2 ] in
  let t1 = term ~ties:[ [ left 0; left 0 ] ] ~free:[ right 0 ] in
  let t2 = term ~ties:[ [ left 0; left 1 ] ] ~free:[ right 0 ] in
  let t3 = term ~ties:[] ~free:[ right 0 ] in
  List.iter
    [ t1, t1; t1, t2; t2, t3; t3, t3 ]
    ~f:(fun (a, b) ->
      equal
        (float 1e-9)
        (Support.inner_product_brute ~dims:d a b)
        (Term.inner_product ~dims:d a b));
  equal (float 1e-9) (Term.inner_product ~dims:d t1 t3) (Term.inner_product ~dims:d t3 t1)

let test_component_inner_product () =
  let d = dims ~left:[ 2; 3 ] ~right:[ 2 ] in
  let t1 = term ~ties:[ [ left 0; left 0 ] ] ~free:[ right 0 ] in
  let t2 = term ~ties:[] ~free:[ right 0 ] in
  let sum = Component.Sum [ t1; t2 ] in
  equal
    (float 1e-9)
    (Term.inner_product ~dims:d t1 t1
     +. Term.inner_product ~dims:d t1 t2
     +. Term.inner_product ~dims:d t2 t1
     +. Term.inner_product ~dims:d t2 t2)
    (Component.inner_product ~dims:d sum sum)

let tests =
  [ test "index characters" test_to_char
  ; test "characters run out" test_to_char_out_of_range
  ; test "axis helpers" test_axis_helpers
  ; test "symmetry labels" test_symmetry_label
  ; test "surrogate dimensions keep Id axes and shrink permuted ones" test_surrogate_dims
  ; test "term canonicalization" test_term_canonicalization
  ; test "term transpose" test_term_transpose
  ; test "normalization" test_normalization
  ; test "inner product" test_inner_product
  ; test "component inner product" test_component_inner_product
  ]

let () = run "symo front end" tests
