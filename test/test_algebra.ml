(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Geom.Algebra: the generic dense Clifford algebra. Most laws are
   stated for the Euclidean algebra Cl(3,0,0), with a float64 algebra for
   duals and analytic checks and a degenerate algebra for the documented
   failures. *)

open Windtrap
open Support
module B = Geom.Blade

module A =
  Geom.Algebra.Make
    (struct
      let signature = Geom.Signature.euclidean 3
    end)
    (struct
      type kind = Nx.float32_elt

      let dtype = Nx.float32
    end)

module Q =
  Geom.Algebra.Make
    (struct
      let signature = Geom.Signature.euclidean 2
    end)
    (struct
      type kind = Nx.float64_elt

      let dtype = Nx.float64
    end)

module P =
  Geom.Algebra.Make
    (struct
      let signature = Geom.Signature.projective 2
    end)
    (struct
      type kind = Nx.float32_elt

      let dtype = Nx.float32
    end)

let e i = A.e i
let b i = A.basis (B.of_int i)
let zero = A.zero
let one = A.one
let e1 = e 1
let e2 = e 2
let e3 = e 3
let e12 = b 0b11
let e13 = b 0b101
let e23 = b 0b110
let e123 = b 0b111
let neg a = Nx.mul_s a (-1.)
let add = Nx.add
let arr n f = Array.init n f
let coeff a i = A.coeff a (B.of_int i)

(* An independent reference product, written from the bitmask rule. *)
let ref_metric i j =
  let sigma = Geom.Signature.to_array A.signature in
  let rec loop bit acc =
    if bit >= Array.length sigma
    then acc
    else if i land j land (1 lsl bit) <> 0
    then loop (bit + 1) (acc * sigma.(bit))
    else loop (bit + 1) acc
  in
  loop 0 1

let ref_gp i j =
  let m = ref_metric i j in
  if m = 0
  then zero
  else (
    let s = B.mul_sign (B.of_int i) (B.of_int j) in
    A.of_array (arr A.size (fun k -> if k = i lxor j then float_of_int (s * m) else 0.)))

(* Construction and shapes *)

let test_construction () =
  equal ~msg:"n" int 3 A.n;
  equal ~msg:"size" int 8 A.size;
  equal ~msg:"dtype" string "float32" (Format.asprintf "%a" Nx.pp_dtype A.dtype);
  equal_tensor
    ~msg:"scalar"
    (A.of_array [| 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |])
    (A.scalar 1.);
  equal_tensor ~msg:"one" (A.scalar 1.) A.one;
  equal_tensor ~msg:"zero" (A.scalar 0.) A.zero;
  equal_tensor ~msg:"basis" (A.of_array [| 0.; 1.; 0.; 0.; 0.; 0.; 0.; 0. |]) e1;
  equal_tensor ~msg:"e" e1 (b 1);
  equal_tensor ~msg:"pseudoscalar" e123 A.pseudoscalar;
  equal
    ~msg:"of_array/to_array round trip"
    (array (float 1e-6))
    [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
    (A.to_array (A.of_array [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]));
  equal
    ~msg:"of_nx keeps the value"
    (array (float 1e-6))
    [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
    (A.to_array
       (A.of_nx (Nx.create A.dtype [| 8 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |])));
  equal_tensor ~msg:"to_nx is the identity" e12 (A.to_nx e12)

let test_shape () =
  let batch = Nx.create A.dtype [| 2; 8 |] (arr 16 (fun i -> float_of_int i)) in
  equal ~msg:"shape" (array int) [| 2; 8 |] (A.shape batch);
  equal ~msg:"batch shape" (array int) [| 2 |] (A.batch_shape batch);
  equal ~msg:"unbatched shape" (array int) [| 8 |] (A.shape e1);
  equal ~msg:"unbatched batch shape" (array int) [||] (A.batch_shape e1);
  equal
    ~msg:"coeff on a batch"
    (array (float 1e-6))
    [| 1.; 9. |]
    (Nx.to_array (A.coeff batch (B.of_int 1)));
  equal
    ~msg:"coeff shape is the batch shape"
    (array int)
    [| 2 |]
    (Nx.shape (A.coeff batch (B.of_int 1)))

let test_coefficients () =
  let a = A.of_array [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
  for i = 0 to 7 do
    equal_scalar
      ~msg:(Printf.sprintf "coefficient %d" i)
      (float_of_int (i + 1))
      (coeff a i)
  done;
  equal_tensor
    ~msg:"with_coeff replaces one coefficient"
    (A.of_array [| 1.; 9.; 3.; 4.; 5.; 6.; 7.; 8. |])
    (A.with_coeff a (B.of_int 1) (Nx.scalar A.dtype 9.));
  let batch =
    Nx.create A.dtype [| 2; 8 |] (Array.concat [ A.to_array a; A.to_array (neg a) ])
  in
  let replaced =
    A.with_coeff batch (B.of_int 2) (Nx.create A.dtype [| 2 |] [| 10.; 11. |])
  in
  equal
    ~msg:"with_coeff on a batch"
    (array (float 1e-5))
    [| 1.; 2.; 10.; 4.; 5.; 6.; 7.; 8.; -1.; -2.; 11.; -4.; -5.; -6.; -7.; -8. |]
    (Nx.to_array replaced);
  equal
    ~msg:"with_coeff can broadcast a scalar"
    (array (float 1e-5))
    [| 1.; 9.; 3.; 4.; 5.; 6.; 7.; 8.; 1.; 9.; 3.; 4.; 5.; 6.; 7.; 8. |]
    (let batch =
       Nx.create
         A.dtype
         [| 2; 8 |]
         [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
     in
     A.with_coeff batch (B.of_int 1) (Nx.scalar A.dtype 9.) |> Nx.to_array)

let test_grade () =
  let a = A.of_array [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
  equal
    ~msg:"grade 0"
    (array (float 1e-6))
    [| 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
    (A.to_array (A.grade 0 a));
  equal
    ~msg:"grade 1"
    (array (float 1e-6))
    [| 0.; 2.; 3.; 0.; 5.; 0.; 0.; 0. |]
    (A.to_array (A.grade 1 a));
  equal
    ~msg:"grade 2"
    (array (float 1e-6))
    [| 0.; 0.; 0.; 4.; 0.; 6.; 7.; 0. |]
    (A.to_array (A.grade 2 a));
  equal
    ~msg:"grade 3"
    (array (float 1e-6))
    [| 0.; 0.; 0.; 0.; 0.; 0.; 0.; 8. |]
    (A.to_array (A.grade 3 a));
  equal_tensor ~msg:"grade above n is zero" zero (A.grade 4 a);
  equal_tensor ~msg:"negative grade is zero" zero (A.grade (-1) a);
  equal
    ~msg:"even"
    (array (float 1e-6))
    [| 1.; 0.; 0.; 4.; 0.; 6.; 7.; 0. |]
    (A.to_array (A.even a));
  equal
    ~msg:"odd"
    (array (float 1e-6))
    [| 0.; 2.; 3.; 0.; 5.; 0.; 0.; 8. |]
    (A.to_array (A.odd a));
  equal_tensor ~msg:"even plus odd" a (add (A.even a) (A.odd a))

(* Products *)

let test_basis_products () =
  for i = 0 to A.size - 1 do
    for j = 0 to A.size - 1 do
      equal_tensor
        ~msg:
          (Printf.sprintf
             "gp of %s and %s"
             (B.to_string (B.of_int i))
             (B.to_string (B.of_int j)))
        (ref_gp i j)
        (A.gp (b i) (b j))
    done
  done

let test_products () =
  equal_tensor ~msg:"e1 e1 = 1" one (A.gp e1 e1);
  equal_tensor ~msg:"e2 e2 = 1" one (A.gp e2 e2);
  equal_tensor ~msg:"e3 e3 = 1" one (A.gp e3 e3);
  equal_tensor ~msg:"e1 e2 = e12" e12 (A.gp e1 e2);
  equal_tensor ~msg:"e2 e1 = -e12" (neg e12) (A.gp e2 e1);
  equal_tensor ~msg:"e1 ^ e2 = e12" e12 (A.op e1 e2);
  equal_tensor ~msg:"e2 ^ e1 = -e12" (neg e12) (A.op e2 e1);
  equal_tensor ~msg:"e1 ^ e1 = 0" zero (A.op e1 e1);
  equal_tensor ~msg:"e1 . e12 = e2" e2 (A.inner e1 e12);
  equal_tensor ~msg:"e12 . e1 = 0 under left contraction" zero (A.inner e12 e1);
  equal_tensor ~msg:"e1 . e1 = 1" one (A.inner e1 e1);
  equal_scalar ~msg:"the scalar product of e1 and e1" 1. (A.scalar_prod e1 e1);
  equal_scalar ~msg:"the scalar product of e12 with itself" 1. (A.scalar_prod e12 e12);
  equal_tensor
    ~msg:"the commutator of orthogonal bivectors"
    (neg e23)
    (A.commutator e12 e13);
  equal_tensor ~msg:"the commutator of a bivector with itself" zero (A.commutator e12 e12)

let test_product_algebra () =
  let x = A.of_array [| 1.; 0.5; -1.; 2.; 0.25; -0.5; 1.5; -2. |] in
  let y = A.of_array [| -0.5; 1.; 0.75; -1.5; 2.; 0.5; -0.25; 1. |] in
  let z = A.of_array [| 2.; -1.; 0.5; 1.; -0.75; 1.5; 0.25; -0.5 |] in
  equal_tensor ~msg:"gp is associative" (A.gp (A.gp x y) z) (A.gp x (A.gp y z));
  equal_tensor ~msg:"op is associative" (A.op (A.op x y) z) (A.op x (A.op y z));
  equal_tensor
    ~msg:"gp distributes over addition on the left"
    (A.gp x (add y z))
    (add (A.gp x y) (A.gp x z));
  equal_tensor
    ~msg:"gp distributes over addition on the right"
    (A.gp (add x y) z)
    (add (A.gp x z) (A.gp y z));
  equal_tensor ~msg:"one is neutral on the left" x (A.gp one x);
  equal_tensor ~msg:"one is neutral on the right" x (A.gp x one);
  equal_tensor ~msg:"zero absorbs on the left" zero (A.gp zero x);
  equal_tensor ~msg:"zero absorbs on the right" zero (A.gp x zero);
  equal_tensor
    ~msg:"op is alternating on vectors"
    zero
    (A.op
       (A.of_array [| 0.; 1.; 2.; 0.; 3.; 0.; 0.; 0. |])
       (A.of_array [| 0.; 1.; 2.; 0.; 3.; 0.; 0.; 0. |]))

(* Involutions *)

let test_involutions () =
  equal_tensor ~msg:"reverse of a vector" e1 (A.reverse e1);
  equal_tensor ~msg:"reverse of a bivector" (neg e12) (A.reverse e12);
  equal_tensor ~msg:"reverse of the pseudoscalar" (neg e123) (A.reverse e123);
  equal_tensor ~msg:"grade involution of a vector" (neg e1) (A.grade_involution e1);
  equal_tensor ~msg:"grade involution of a bivector" e12 (A.grade_involution e12);
  equal_tensor ~msg:"grade involution of a trivector" (neg e123) (A.grade_involution e123);
  equal_tensor ~msg:"Clifford conjugate of a vector" (neg e1) (A.conjugate e1);
  equal_tensor ~msg:"Clifford conjugate of a bivector" (neg e12) (A.conjugate e12);
  equal_tensor ~msg:"Clifford conjugate of a trivector" e123 (A.conjugate e123);
  let x = A.of_array [| 1.; 0.5; -1.; 2.; 0.25; -0.5; 1.5; -2. |] in
  let y = A.of_array [| -0.5; 1.; 0.75; -1.5; 2.; 0.5; -0.25; 1. |] in
  equal_tensor ~msg:"reverse is an involution" x (A.reverse (A.reverse x));
  equal_tensor
    ~msg:"grade involution is an involution"
    x
    (A.grade_involution (A.grade_involution x));
  equal_tensor
    ~msg:"Clifford conjugation is an involution"
    x
    (A.conjugate (A.conjugate x));
  equal_tensor
    ~msg:"reverse is an anti-automorphism"
    (A.reverse (A.gp x y))
    (A.gp (A.reverse y) (A.reverse x))

(* Duals and complements *)

let test_dual () =
  equal_tensor ~msg:"dual of e1 is -e23" (neg e23) (A.dual e1);
  equal_tensor ~msg:"dual of e2 is e13" e13 (A.dual e2);
  equal_tensor ~msg:"dual of e3 is -e12" (neg e12) (A.dual e3);
  equal_tensor ~msg:"dual of the scalar is minus the pseudoscalar" (neg e123) (A.dual one);
  equal_tensor ~msg:"dual of the pseudoscalar is the scalar" one (A.dual e123);
  equal_tensor ~msg:"undual of the dual" e1 (A.undual (A.dual e1));
  equal_tensor ~msg:"dual of the undual" e12 (A.dual (A.undual e12));
  let q1 = Q.e 1
  and q2 = Q.e 2 in
  equal_tensor ~msg:"dual in Cl(2) sends e1 to -e2" (Nx.mul_s q2 (-1.)) (Q.dual q1);
  equal_tensor ~msg:"dual in Cl(2) sends e2 to e1" q1 (Q.dual q2);
  let x = A.of_array [| 1.; 0.5; -1.; 2.; 0.25; -0.5; 1.5; -2. |] in
  equal_tensor ~msg:"undual after dual is the identity" x (A.undual (A.dual x));
  raises_match
    ~msg:"dual is undefined in a degenerate algebra"
    (Exn.invalid_arg ~substring:"degenerate")
    (fun () -> ignore (P.dual (P.e 1)));
  raises_match
    ~msg:"undual is undefined in a degenerate algebra"
    (Exn.invalid_arg ~substring:"degenerate")
    (fun () -> ignore (P.undual (P.e 1)))

let test_complement () =
  for i = 0 to A.size - 1 do
    let expected =
      let c = B.complement ~n:A.n (B.of_int i) in
      let s = B.complement_sign ~n:A.n (B.of_int i) in
      A.of_array (arr A.size (fun k -> if k = B.to_int c then float_of_int s else 0.))
    in
    equal_tensor ~msg:(Printf.sprintf "complement of %d" i) expected (A.complement (b i))
  done;
  let x = A.of_array [| 1.; 0.5; -1.; 2.; 0.25; -0.5; 1.5; -2. |] in
  equal_tensor ~msg:"complement is an involution" x (A.complement (A.complement x));
  let y = A.of_array [| -0.5; 1.; 0.75; -1.5; 2.; 0.5; -0.25; 1. |] in
  equal_tensor
    ~msg:"regressive is the complemented outer product"
    (A.regressive x y)
    (A.complement (A.op (A.complement x) (A.complement y)));
  equal_tensor ~msg:"regressive of e1 and e2 in Cl(3)" zero (A.regressive e1 e2)

(* Norms and inverses *)

let test_norms () =
  equal_scalar ~msg:"norm2 of a vector" 1. (A.norm2 e1);
  equal_scalar ~msg:"norm2 of a bivector" 1. (A.norm2 e12);
  equal_scalar ~msg:"norm2 of the pseudoscalar" 1. (A.norm2 e123);
  equal_scalar ~msg:"norm2 of a scalar" 9. (A.norm2 (A.scalar (-3.)));
  equal_scalar ~msg:"norm of a scalar" 3. (A.norm (A.scalar (-3.)));
  equal_scalar ~msg:"norm of e12" 1. (A.norm e12);
  equal_tensor
    ~msg:"normalize scales to unit norm"
    e12
    (A.normalize (A.gp e12 (A.scalar 3.)));
  equal_tensor ~msg:"inverse of a vector" e1 (A.inverse e1);
  equal_tensor ~msg:"inverse of a bivector" (neg e12) (A.inverse e12);
  equal_tensor ~msg:"inverse of a scalar" (A.scalar 0.5) (A.inverse (A.scalar 2.));
  for i = 0 to A.size - 1 do
    equal_tensor
      ~msg:(Printf.sprintf "a blade is invertible: %d" i)
      one
      (A.gp (b i) (A.inverse (b i)))
  done;
  let x = A.of_array [| 1.; 0.5; -1.; 2.; 0.25; -0.5; 1.5; -2. |] in
  equal_scalar
    ~msg:"norm2 is the scalar part of reverse times a"
    (A.coeff (A.gp (A.reverse x) x) (B.of_int 0) |> Nx.item [])
    (A.norm2 x)

(* Versor actions and adjoints *)

let test_sandwich () =
  equal_tensor ~msg:"a vector reflects another across itself" (neg e2) (A.sandwich e1 e2);
  equal_tensor ~msg:"a vector preserves itself" e1 (A.sandwich e1 e1);
  equal_tensor ~msg:"a bivector rotates by pi in its plane" (neg e1) (A.sandwich e12 e1);
  equal_tensor ~msg:"a bivector leaves its plane alone" e12 (A.sandwich e12 e12);
  equal_tensor ~msg:"the pseudoscalar fixes vectors" e1 (A.sandwich e123 e1);
  let x = A.of_array [| 1.; 0.5; -1.; 2.; 0.25; -0.5; 1.5; -2. |] in
  equal_tensor ~msg:"sandwich by one is the identity" x (A.sandwich one x);
  equal_tensor
    ~msg:"sandwich by a vector preserves the norm of a vector"
    x
    (A.sandwich e1 (A.sandwich e1 x))

let test_adjoint () =
  let x = A.of_array [| 1.; 0.5; -1.; 2.; 0.25; -0.5; 1.5; -2. |] in
  let check name v =
    let adj = A.adjoint v in
    equal ~msg:(name ^ " shape") (array int) [| A.size; A.size |] (Nx.shape adj);
    let applied =
      Nx.matmul adj (Nx.reshape [| A.size; 1 |] x) |> Nx.reshape [| A.size |]
    in
    equal_tensor ~msg:(name ^ " acts like the sandwich") (A.sandwich v x) applied
  in
  check "adjoint of e1" e1;
  check "adjoint of e12" e12;
  check "adjoint of e123" e123;
  let v = A.gp e1 e12 in
  check "adjoint of a product" v;
  let adj_batch = A.adjoint (A.of_nx (Nx.stack ~axis:0 [ e1; e12 ])) in
  equal
    ~msg:"batched adjoint shape"
    (array int)
    [| 2; A.size; A.size |]
    (Nx.shape adj_batch)

(* Batching and broadcasting *)

let take0 i a = Nx.take ~axis:0 ~indices:(Nx.scalar Nx.int64 (Int64.of_int i)) a

let test_batching () =
  let x = A.of_array [| 1.; 0.5; -1.; 2.; 0.25; -0.5; 1.5; -2. |] in
  let y = A.of_array [| -0.5; 1.; 0.75; -1.5; 2.; 0.5; -0.25; 1. |] in
  let z = A.of_array [| 2.; -1.; 0.5; 1.; -0.75; 1.5; 0.25; -0.5 |] in
  let batch = A.of_nx (Nx.stack ~axis:0 [ x; y ]) in
  let prod = A.gp batch z in
  equal ~msg:"batch shape is kept" (array int) [| 2; A.size |] (A.shape prod);
  equal_tensor ~msg:"first row" (A.gp x z) (take0 0 prod);
  equal_tensor ~msg:"second row" (A.gp y z) (take0 1 prod);
  let left = A.of_nx (Nx.stack ~axis:0 [ x; y ]) in
  let right = A.of_nx (Nx.stack ~axis:0 [ z; y ]) in
  let batched = A.gp left right in
  equal_tensor ~msg:"both operands batched: first" (A.gp x z) (take0 0 batched);
  equal_tensor ~msg:"both operands batched: second" (A.gp y y) (take0 1 batched);
  let grid_l = A.of_nx (Nx.reshape [| 2; 1; A.size |] (A.to_nx batch)) in
  let grid_r = A.of_nx (Nx.reshape [| 1; 3; A.size |] (Nx.stack ~axis:0 [ z; y; x ])) in
  let grid = A.gp grid_l grid_r in
  equal ~msg:"broadcast batch dims" (array int) [| 2; 3; A.size |] (A.shape grid);
  equal_tensor ~msg:"broadcast cell" (A.gp x y) (take0 1 (take0 0 grid))

(* Printing *)

let test_printing () =
  equal ~msg:"zero prints as 0" string "0" (Format.asprintf "%a" A.pp zero);
  equal
    ~msg:"a sum separates its terms"
    string
    "1 + 1*e1"
    (Format.asprintf "%a" A.pp (add one e1));
  let batch = A.of_nx (Nx.stack ~axis:0 [ e1; e2 ]) in
  equal
    ~msg:"a batch prints its shape"
    string
    "<multivector 2x8>"
    (Format.asprintf "%a" A.pp batch);
  (* A first batch axis equal to [size] must not be mistaken for the blade
     axis. *)
  let square = A.of_nx (Nx.zeros A.dtype [| A.size; A.size |]) in
  equal
    ~msg:"a square batch prints its shape"
    string
    "<multivector 8x8>"
    (Format.asprintf "%a" A.pp square)

(* Documented failures *)

let test_errors () =
  raises_match
    ~msg:"basis checks the blade index"
    (Exn.invalid_arg ~substring:"out of range")
    (fun () -> ignore (A.basis (B.of_int A.size)));
  raises_match
    ~msg:"e checks the basis index"
    (Exn.invalid_arg ~substring:"out of range")
    (fun () -> ignore (A.e 0));
  raises_match
    ~msg:"e checks the upper bound"
    (Exn.invalid_arg ~substring:"out of range")
    (fun () -> ignore (A.e 4));
  raises_match
    ~msg:"of_array checks the length"
    (Exn.invalid_arg ~substring:"coefficients")
    (fun () -> ignore (A.of_array [| 1.; 2. |]));
  raises_match
    ~msg:"to_array refuses batched values"
    (Exn.invalid_arg ~substring:"unbatched")
    (fun () -> ignore (A.to_array (A.of_nx (Nx.zeros A.dtype [| 2; A.size |]))));
  raises_match
    ~msg:"of_nx checks the blade axis"
    (Exn.invalid_arg ~substring:"blade axis")
    (fun () -> ignore (A.of_nx (Nx.zeros A.dtype [| 7 |])));
  raises_match
    ~msg:"batch_shape needs a blade axis"
    (Exn.invalid_arg ~substring:"blade axis")
    (fun () -> ignore (A.batch_shape (Nx.scalar A.dtype 1.)));
  raises_match ~msg:"with_coeff checks the coefficient shape" Exn.invalid_arg (fun () ->
    ignore
      (A.with_coeff
         (A.of_nx (Nx.zeros A.dtype [| 2; A.size |]))
         (B.of_int 1)
         (Nx.zeros A.dtype [| 3 |])))

(* Properties *)

let test_properties =
  let a = Support.gen_mv A.size A.of_array in
  group
    "properties"
    [ prop
        "gp is associative"
        Gen.(triple a a a)
        (fun (x, y, z) -> Law.associative (Support.tensor ()) A.gp (x, y, z))
    ; prop
        "gp distributes over addition"
        Gen.(triple a a a)
        (fun (x, y, z) ->
           Law.distributive (Support.tensor ()) A.gp ~over:Nx.add (x, y, z))
    ; prop "reverse is an involution" a (fun x ->
        Law.involutive (Support.tensor ()) A.reverse x)
    ; prop "grade involution is an involution" a (fun x ->
        Law.involutive (Support.tensor ()) A.grade_involution x)
    ; prop "Clifford conjugation is an involution" a (fun x ->
        Law.involutive (Support.tensor ()) A.conjugate x)
    ; prop
        "reverse reverses a product"
        Gen.(pair a a)
        (fun (x, y) ->
           Support.equal_tensor (A.reverse (A.gp x y)) (A.gp (A.reverse y) (A.reverse x)))
    ; prop "the complement is an involution" a (fun x ->
        Law.involutive (Support.tensor ()) A.complement x)
    ; prop "the dual and the undual are inverse" a (fun x ->
        Law.round_trip (Support.tensor ()) (Support.tensor ()) A.dual A.undual x)
    ; prop
        "a blade times its inverse is one"
        Gen.(int_range 0 (A.size - 1))
        (fun i -> Support.equal_tensor one (A.gp (b i) (A.inverse (b i))))
    ; prop
        "a sandwich preserves the geometric product"
        Gen.(pair a a)
        (fun (x, y) ->
           Support.equal_tensor
             (A.sandwich e12 (A.gp x y))
             (A.gp (A.sandwich e12 x) (A.sandwich e12 y)))
    ]

let () =
  exit
    (run
       "geom algebra"
       [ group
           "construction"
           [ test "constructors and conversions" test_construction
           ; test "shapes and batches" test_shape
           ; test "coefficients" test_coefficients
           ; test "grades" test_grade
           ]
       ; group
           "products"
           [ test "every basis product" test_basis_products
           ; test "named products" test_products
           ; test "product algebra" test_product_algebra
           ]
       ; group "involutions" [ test "the three involutions" test_involutions ]
       ; group
           "duals"
           [ test "Hodge duals" test_dual
           ; test "complements and the regressive product" test_complement
           ]
       ; group "norms and inverses" [ test "norms and inverses" test_norms ]
       ; group
           "versors"
           [ test "sandwich actions" test_sandwich
           ; test "the adjoint is the sandwich matrix" test_adjoint
           ]
       ; group "batching" [ test "broadcasting batch axes" test_batching ]
       ; group "printing" [ test "debug printing" test_printing ]
       ; group "errors" [ test "documented failures" test_errors ]
       ; test_properties
       ])
