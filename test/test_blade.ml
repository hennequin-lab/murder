(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Geom.Blade: index arithmetic, grades, signs, products and
   complements. Every law is stated against a reference written with integer
   bit operations and popcount. *)

open Windtrap
module B = Geom.Blade

let blade = Testable.contramap B.to_int int

let popcount x =
  let rec loop x acc = if x = 0 then acc else loop (x lsr 1) (acc + (x land 1)) in
  loop x 0

let low_bits n = (1 lsl n) - 1

(* Constructors and accessors *)

let test_constructors () =
  equal ~msg:"of_int/to_int round trip" int 0 (B.to_int (B.of_int 0));
  equal ~msg:"a dense bitmask" int 42 (B.to_int (B.of_int 42));
  raises_match
    ~msg:"negative index is rejected"
    (Exn.invalid_arg ~substring:"negative")
    (fun () -> ignore (B.of_int (-1) : B.t));
  equal ~msg:"e_1 is bit 0" int 1 (B.to_int (B.of_basis 1));
  equal ~msg:"e_4 is bit 3" int 8 (B.to_int (B.of_basis 4));
  raises_match
    ~msg:"basis indices start at one"
    (Exn.invalid_arg ~substring:"index")
    (fun () -> ignore (B.of_basis 0 : B.t));
  raises_match
    ~msg:"negative basis index is rejected"
    (Exn.invalid_arg ~substring:"index")
    (fun () -> ignore (B.of_basis (-2) : B.t))

let test_contains () =
  let b = B.of_int 0b1011 in
  equal ~msg:"e_1 in e124" bool true (B.contains b 1);
  equal ~msg:"e_2 in e124" bool true (B.contains b 2);
  equal ~msg:"e_3 not in e124" bool false (B.contains b 3);
  equal ~msg:"e_4 in e124" bool true (B.contains b 4);
  raises_match ~msg:"indices start at one" (Exn.invalid_arg ~substring:"index") (fun () ->
    ignore (B.contains b 0));
  equal ~msg:"the scalar contains nothing" bool false (B.contains (B.of_int 0) 1);
  equal ~msg:"is_scalar on the empty product" bool true (B.is_scalar (B.of_int 0));
  equal ~msg:"is_scalar off a vector" bool false (B.is_scalar (B.of_int 1))

let test_equality_hash_order () =
  let a = B.of_int 0b101 in
  let b = B.of_int 0b101 in
  let c = B.of_int 0b110 in
  equal ~msg:"equal blades" bool true (B.equal a b);
  equal ~msg:"different blades" bool false (B.equal a c);
  equal ~msg:"equal hashes" int (B.hash a) (B.hash b);
  equal ~msg:"compare equal" int 0 (B.compare a b);
  less ~msg:"compare orders by bitmask" int ~than:(B.compare c a) (B.compare a c)

let test_all () =
  equal ~msg:"all 0 is the scalar" (array blade) [| B.of_int 0 |] (B.all 0);
  equal
    ~msg:"all 3 is every bitmask in order"
    (array int)
    [| 0; 1; 2; 3; 4; 5; 6; 7 |]
    (Array.map B.to_int (B.all 3));
  equal ~msg:"all 4 has 16 entries" int 16 (Array.length (B.all 4));
  raises_match
    ~msg:"negative dimension is rejected"
    (Exn.invalid_arg ~substring:"negative")
    (fun () -> ignore (B.all (-1)))

let test_grade () =
  equal ~msg:"scalar grade" int 0 (B.grade (B.of_int 0));
  equal ~msg:"vector grade" int 1 (B.grade (B.of_int 0b100));
  equal ~msg:"bivector grade" int 2 (B.grade (B.of_int 0b101));
  equal ~msg:"trivector grade" int 3 (B.grade (B.of_int 0b111));
  for b = 0 to 63 do
    equal ~msg:(Printf.sprintf "popcount of %d" b) int (popcount b) (B.grade (B.of_int b))
  done

let sign_name s = if s = 1 then "1" else if s = -1 then "-1" else string_of_int s

let test_signs () =
  for b = 0 to 63 do
    let blade = B.of_int b in
    let k = B.grade blade in
    let reverse = if k * (k - 1) / 2 land 1 = 0 then 1 else -1 in
    let grade_involution = if k land 1 = 0 then 1 else -1 in
    let clifford = if k * (k + 1) / 2 land 1 = 0 then 1 else -1 in
    equal
      ~msg:(Printf.sprintf "reverse sign of %d" b)
      string
      (sign_name reverse)
      (sign_name (B.reverse_sign blade));
    equal
      ~msg:(Printf.sprintf "grade involution of %d" b)
      string
      (sign_name grade_involution)
      (sign_name (B.grade_involution_sign blade));
    equal
      ~msg:(Printf.sprintf "Clifford conjugate of %d" b)
      string
      (sign_name clifford)
      (sign_name (B.clifford_conjugation_sign blade));
    equal
      ~msg:(Printf.sprintf "signs square to one for %d" b)
      int
      1
      (B.reverse_sign blade * B.reverse_sign blade);
    equal
      ~msg:(Printf.sprintf "conjugation composes for %d" b)
      int
      (B.reverse_sign blade * B.grade_involution_sign blade)
      (B.clifford_conjugation_sign blade)
  done

let test_mul_sign_examples () =
  equal ~msg:"e1 e2" int 1 (B.mul_sign (B.of_int 0b1) (B.of_int 0b10));
  equal ~msg:"e2 e1" int (-1) (B.mul_sign (B.of_int 0b10) (B.of_int 0b1));
  equal ~msg:"e1 e1" int 1 (B.mul_sign (B.of_int 0b1) (B.of_int 0b1));
  equal ~msg:"e12 e3" int 1 (B.mul_sign (B.of_int 0b11) (B.of_int 0b100));
  equal ~msg:"e3 e12" int 1 (B.mul_sign (B.of_int 0b100) (B.of_int 0b11));
  equal ~msg:"e2 e13" int (-1) (B.mul_sign (B.of_int 0b10) (B.of_int 0b101));
  equal ~msg:"the scalar" int 1 (B.mul_sign (B.of_int 0) (B.of_int 0b1011))

let test_mul_sign_law () =
  for a = 0 to 31 do
    for b = 0 to 31 do
      let s = B.mul_sign (B.of_int a) (B.of_int b) in
      equal
        ~msg:(Printf.sprintf "mul_sign %d %d is a sign" a b)
        bool
        true
        (s = 1 || s = -1);
      if a land b = 0
      then (
        let swapped = B.mul_sign (B.of_int b) (B.of_int a) in
        let expected =
          if B.grade (B.of_int a) * B.grade (B.of_int b) land 1 = 0 then 1 else -1
        in
        equal
          ~msg:(Printf.sprintf "mul_sign symmetry %d %d" a b)
          int
          (s * swapped)
          expected)
    done
  done

let test_complement () =
  for n = 0 to 5 do
    let full = low_bits n in
    for b = 0 to full do
      let blade = B.of_int b in
      let c = B.complement ~n blade in
      equal
        ~msg:(Printf.sprintf "complement of %d in %d grades" b n)
        int
        (n - B.grade blade)
        (B.grade c);
      equal
        ~msg:(Printf.sprintf "complement of %d in %d is disjoint" b n)
        int
        0
        (b land B.to_int c);
      equal
        ~msg:(Printf.sprintf "complement of %d in %d covers" b n)
        int
        full
        (b lor B.to_int c);
      equal
        ~msg:(Printf.sprintf "complement is an involution at %d, %d" b n)
        int
        b
        (B.to_int (B.complement ~n c));
      let s = B.complement_sign ~n blade in
      equal
        ~msg:(Printf.sprintf "complement sign at %d, %d" b n)
        bool
        true
        (s = 1 || s = -1);
      equal
        ~msg:(Printf.sprintf "complement sign is the product sign at %d, %d" b n)
        int
        s
        (B.mul_sign blade c)
    done
  done;
  raises_match
    ~msg:"negative dimension is rejected"
    (Exn.invalid_arg ~substring:"negative")
    (fun () -> ignore (B.complement ~n:(-1) (B.of_int 0)))

let test_printing () =
  equal ~msg:"scalar" string "1" (B.to_string (B.of_int 0));
  equal ~msg:"vector" string "e1" (B.to_string (B.of_basis 1));
  equal ~msg:"bivector" string "e13" (B.to_string (B.of_int 0b101));
  equal ~msg:"four terms" string "e234" (B.to_string (B.of_int 0b1110));
  equal
    ~msg:"pp uses the same name"
    string
    "e124"
    (Format.asprintf "%a" B.pp (B.of_int 0b1011))

(* Properties *)

let test_grade_prop =
  prop
    "grade is the number of basis vectors"
    Gen.(int_range 0 4095)
    (fun b -> equal int (popcount b) (B.grade (B.of_int b)))

let test_signs_prop =
  prop
    "every sign squares to one"
    Gen.(int_range 0 4095)
    (fun b ->
       let b = B.of_int b in
       equal int 1 (B.reverse_sign b * B.reverse_sign b);
       equal int 1 (B.grade_involution_sign b * B.grade_involution_sign b);
       equal int 1 (B.clifford_conjugation_sign b * B.clifford_conjugation_sign b);
       equal
         int
         (B.reverse_sign b * B.grade_involution_sign b)
         (B.clifford_conjugation_sign b))

let test_complement_prop =
  prop
    "complement is an involution and a signed product"
    Gen.(pair (int_range 0 5) (int_range 0 31))
    (fun (n, b) ->
       let b = b land low_bits n in
       let blade = B.of_int b in
       let c = B.complement ~n blade in
       equal int b (B.to_int (B.complement ~n c));
       equal int (B.complement_sign ~n blade) (B.mul_sign blade c))

let test_mul_sign_prop =
  prop
    "mul_sign is antisymmetric on disjoint blades"
    Gen.(pair (int_range 0 15) (int_range 0 15))
    (fun (a, b0) ->
       let b = b0 lsl 4 in
       let a = B.of_int a
       and b = B.of_int b in
       let s = B.mul_sign a b
       and t = B.mul_sign b a in
       let expected = if B.grade a * B.grade b land 1 = 0 then 1 else -1 in
       equal int expected (s * t))

let () =
  exit
    (run
       "geom blade"
       [ group
           "constructors"
           [ test "of_int, to_int and basis access" test_constructors
           ; test "contains and is_scalar" test_contains
           ; test "equality, hash and order" test_equality_hash_order
           ; test "all enumerates the bitmasks" test_all
           ; test "grade counts the basis vectors" test_grade
           ]
       ; group
           "signs"
           [ test "involutions follow the grade formulas" test_signs
           ; test "reordering signs on named products" test_mul_sign_examples
           ; test "every product sign is a sign" test_mul_sign_law
           ]
       ; group
           "complement"
           [ test "the complement is a signed index complement" test_complement ]
       ; group "printing" [ test "blades print with increasing indices" test_printing ]
       ; group
           "properties"
           [ test_grade_prop; test_signs_prop; test_complement_prop; test_mul_sign_prop ]
       ])
