(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Geom.Signature: constructors, counts, the pseudoscalar square and
   validation. *)

open Windtrap
module S = Geom.Signature

let test_make () =
  let arr = [| 1; -1; 0 |] in
  let s = S.make arr in
  arr.(0) <- -1;
  equal ~msg:"make copies its argument" (array int) [| 1; -1; 0 |] (S.to_array s);
  raises_match
    ~msg:"entries must be -1, 0 or 1"
    (Exn.invalid_arg ~substring:"entries")
    (fun () -> ignore (S.make [| 1; 2 |] : S.t));
  raises_match
    ~msg:"negative two is not an entry"
    (Exn.invalid_arg ~substring:"entries")
    (fun () -> ignore (S.make [| -2 |] : S.t))

let test_constructors () =
  equal ~msg:"euclidean 3" (array int) [| 1; 1; 1 |] (S.to_array (S.euclidean 3));
  equal ~msg:"projective 3" (array int) [| 1; 1; 1; 0 |] (S.to_array (S.projective 3));
  equal ~msg:"conformal 3" (array int) [| 1; 1; 1; 1; -1 |] (S.to_array (S.conformal 3));
  equal
    ~msg:"conformal3 is conformal 3"
    (array int)
    (S.to_array (S.conformal 3))
    (S.to_array S.conformal3);
  equal ~msg:"euclidean 0 is empty" (array int) [||] (S.to_array (S.euclidean 0));
  equal
    ~msg:"projective 0 is one null direction"
    (array int)
    [| 0 |]
    (S.to_array (S.projective 0));
  equal
    ~msg:"conformal 0 is a hyperbolic pair"
    (array int)
    [| 1; -1 |]
    (S.to_array (S.conformal 0));
  raises_match
    ~msg:"euclidean rejects a negative dimension"
    (Exn.invalid_arg ~substring:"negative")
    (fun () -> ignore (S.euclidean (-1) : S.t));
  raises_match
    ~msg:"projective rejects a negative dimension"
    (Exn.invalid_arg ~substring:"negative")
    (fun () -> ignore (S.projective (-1) : S.t));
  raises_match
    ~msg:"conformal rejects a negative dimension"
    (Exn.invalid_arg ~substring:"negative")
    (fun () -> ignore (S.conformal (-1) : S.t))

let test_properties () =
  let e3 = S.euclidean 3 in
  equal ~msg:"dim" int 3 (S.dim e3);
  equal ~msg:"size" int 8 (S.size e3);
  equal ~msg:"positives" int 3 (S.positives e3);
  equal ~msg:"negatives" int 0 (S.negatives e3);
  equal ~msg:"nulls" int 0 (S.nulls e3);
  equal ~msg:"not degenerate" bool false (S.is_degenerate e3);
  let p3 = S.projective 3 in
  equal ~msg:"projective dim" int 4 (S.dim p3);
  equal ~msg:"projective size" int 16 (S.size p3);
  equal ~msg:"projective positives" int 3 (S.positives p3);
  equal ~msg:"projective nulls" int 1 (S.nulls p3);
  equal ~msg:"projective is degenerate" bool true (S.is_degenerate p3);
  let c3 = S.conformal3 in
  equal ~msg:"conformal dim" int 5 (S.dim c3);
  equal ~msg:"conformal size" int 32 (S.size c3);
  equal ~msg:"conformal positives" int 4 (S.positives c3);
  equal ~msg:"conformal negatives" int 1 (S.negatives c3);
  equal ~msg:"conformal not degenerate" bool false (S.is_degenerate c3);
  equal ~msg:"empty signature size" int 1 (S.size (S.euclidean 0));
  equal
    ~msg:"to_array returns the diagonal"
    (array int)
    [| 1; -1 |]
    (S.to_array (S.make [| 1; -1 |]))

let test_pseudoscalar_square () =
  let check ~msg n expected =
    equal ~msg int expected (S.pseudoscalar_square (S.euclidean n))
  in
  check ~msg:"Cl(0)" 0 1;
  check ~msg:"Cl(1)" 1 1;
  check ~msg:"Cl(2)" 2 (-1);
  check ~msg:"Cl(3)" 3 (-1);
  check ~msg:"Cl(4)" 4 1;
  equal ~msg:"Cl(4,1)" int (-1) (S.pseudoscalar_square S.conformal3);
  equal ~msg:"Cl(3,0,1)" int 0 (S.pseudoscalar_square (S.projective 3));
  equal ~msg:"Cl(5,1)" int 1 (S.pseudoscalar_square (S.conformal 4));
  equal ~msg:"Cl(4,2)" int (-1) (S.pseudoscalar_square (S.make [| 1; 1; 1; 1; -1; -1 |]))

let test_equality_printing () =
  equal ~msg:"equal signatures" bool true (S.equal (S.euclidean 3) (S.euclidean 3));
  equal ~msg:"different signatures" bool false (S.equal (S.euclidean 3) (S.projective 3));
  equal ~msg:"to_string euclidean" string "(3,0,0)" (S.to_string (S.euclidean 3));
  equal ~msg:"to_string projective" string "(3,0,1)" (S.to_string (S.projective 3));
  equal ~msg:"to_string conformal" string "(4,1,0)" (S.to_string S.conformal3);
  equal ~msg:"pp" string "(2,1,0)" (Format.asprintf "%a" S.pp (S.make [| 1; 1; -1 |]))

let () =
  exit
    (run
       "geom signature"
       [ group
           "constructors"
           [ test "make validates and copies" test_make
           ; test "the standard constructors" test_constructors
           ]
       ; group
           "properties"
           [ test "counts, dimension and degeneracy" test_properties
           ; test "the pseudoscalar square" test_pseudoscalar_square
           ; test "equality and printing" test_equality_printing
           ]
       ])
