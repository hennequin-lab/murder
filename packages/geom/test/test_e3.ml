(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Geom.E3: the Euclidean Cl(3,0,0) model, its vectors and rotors. *)

open Windtrap
open Support
module E = Geom.E3
module B = Geom.Blade

let x_axis = vec3 1. 0. 0.
let y_axis = vec3 0. 1. 0.
let z_axis = vec3 0. 0. 1.
let angle a = scalar a
let rotor ~axis a = E.rotor ~axis ~angle:(angle a)
let vector_of a = E.to_vector a |> Nx.to_array

(* Construction *)

let test_basis_and_vectors () =
  equal ~msg:"n" int 3 E.n;
  equal ~msg:"size" int 8 E.size;
  equal_tensor ~msg:"e1" (E.basis (B.of_int 0b001)) E.e1;
  equal_tensor ~msg:"e2" (E.basis (B.of_int 0b010)) E.e2;
  equal_tensor ~msg:"e3" (E.basis (B.of_int 0b100)) E.e3;
  equal_tensor
    ~msg:"vector3 agrees with vector"
    (E.vector (vec3 1. (-2.) 3.))
    (E.vector3 1. (-2.) 3.);
  equal
    ~msg:"to_vector round trip"
    (array (float 1e-6))
    [| 1.; -2.; 3. |]
    (vector_of (E.vector3 1. (-2.) 3.));
  equal
    ~msg:"a batch of vectors"
    (array int)
    [| 2; 8 |]
    (E.shape (E.vector (Nx.create f32 [| 2; 3 |] [| 1.; 2.; 3.; -1.; 0.; 2. |])));
  equal
    ~msg:"to_vector keeps the batch shape"
    (array int)
    [| 2; 3 |]
    (Nx.shape
       (E.to_vector (E.vector (Nx.create f32 [| 2; 3 |] [| 1.; 2.; 3.; -1.; 0.; 2. |]))));
  equal
    ~msg:"to_vector ignores higher grades"
    (array (float 1e-6))
    [| 0.; 0.; 0. |]
    (vector_of (E.gp E.e1 E.e2))

let test_vector_products () =
  equal_tensor ~msg:"e1 e1 = 1" E.one (E.gp E.e1 E.e1);
  equal_tensor ~msg:"e1 e2 = e12" (E.gp E.e1 E.e2) (E.op E.e1 E.e2);
  equal_tensor ~msg:"e2 e1 = -e12" (E.gp E.e2 E.e1) (Nx.mul_s (E.op E.e1 E.e2) (-1.));
  equal_tensor ~msg:"the dot product of orthogonal vectors" E.zero (E.inner E.e1 E.e2);
  equal_scalar
    ~msg:"the dot product of parallel vectors"
    14.
    (E.coeff (E.inner (E.vector3 1. 2. 3.) (E.vector3 1. 2. 3.)) (B.of_int 0));
  equal_tensor ~msg:"the pseudoscalar" (E.gp (E.gp E.e1 E.e2) E.e3) E.pseudoscalar;
  equal_scalar
    ~msg:"the pseudoscalar squares to -1"
    (-1.)
    (E.coeff (E.gp E.pseudoscalar E.pseudoscalar) (B.of_int 0))

(* Rotors *)

let test_rotor_basics () =
  let r = rotor ~axis:z_axis 0. in
  equal_tensor ~msg:"a zero angle is the identity" E.one r;
  let r90 = rotor ~axis:z_axis (Float.pi /. 2.) in
  equal_tensor ~msg:"rotate e1 to e2" E.e2 (E.apply_rotor r90 E.e1);
  equal_tensor ~msg:"rotate e2 to -e1" (Nx.mul_s E.e1 (-1.)) (E.apply_rotor r90 E.e2);
  equal_tensor ~msg:"rotate e3 to e3" E.e3 (E.apply_rotor r90 E.e3);
  equal_scalar ~msg:"a rotor has unit norm" 1. (E.norm2 r90);
  equal_tensor ~msg:"a rotor times its reverse is one" E.one (E.gp r90 (E.reverse r90));
  let r180 = rotor ~axis:z_axis Float.pi in
  equal_tensor
    ~msg:"a half turn reflects the plane"
    (Nx.mul_s E.e1 (-1.))
    (E.apply_rotor r180 E.e1);
  let r360 = rotor ~axis:z_axis (2. *. Float.pi) in
  equal_tensor ~msg:"a full turn is -1 as a rotor" (Nx.mul_s E.one (-1.)) r360;
  equal_tensor ~msg:"a full turn fixes a vector" E.e1 (E.apply_rotor r360 E.e1);
  let scaled = rotor ~axis:(vec3 0. 0. 2.) (Float.pi /. 2.) in
  equal_tensor ~msg:"the axis is normalized" r90 scaled

let test_rotor_composition () =
  let r1 = rotor ~axis:z_axis 0.3 in
  let r2 = rotor ~axis:z_axis 0.5 in
  equal_tensor
    ~msg:"rotors about one axis add angles"
    (E.gp r1 r2)
    (rotor ~axis:z_axis 0.8);
  let x = vec3 1. (-2.) 0.5 |> E.vector in
  equal_tensor
    ~msg:"a product acts in the right order"
    (E.apply_rotor (E.gp r1 r2) x)
    (E.apply_rotor r1 (E.apply_rotor r2 x));
  let r3 = rotor ~axis:x_axis 0.4 in
  equal_tensor
    ~msg:"different axes compose in order"
    (E.apply_rotor (E.gp r1 r3) x)
    (E.apply_rotor r1 (E.apply_rotor r3 x));
  let r = E.gp r1 r3 in
  equal_tensor
    ~msg:"the inverse rotor undoes the action"
    x
    (E.apply_rotor (E.reverse r) (E.apply_rotor r x))

let test_rotor_batch () =
  let axes = Nx.create f32 [| 2; 3 |] [| 0.; 0.; 1.; 1.; 0.; 0. |] in
  let angles = Nx.create f32 [| 2 |] [| Float.pi /. 2.; Float.pi /. 2. |] in
  let r = E.rotor ~axis:axes ~angle:angles in
  equal ~msg:"batched rotor shape" (array int) [| 2; 8 |] (E.shape r);
  let x = Nx.create f32 [| 2; 3 |] [| 1.; 0.; 0.; 0.; 0.; 1. |] |> E.vector in
  let y = E.apply_rotor r x in
  equal ~msg:"batched action shape" (array int) [| 2; 3 |] (Nx.shape (E.to_vector y));
  equal_floats ~msg:"first batch element" [| 0.; 1.; 0.; 0.; -1.; 0. |] (E.to_vector y);
  let single = rotor ~axis:z_axis (Float.pi /. 2.) in
  let y = E.apply_rotor single x in
  equal
    ~msg:"a single rotor broadcasts over a batch"
    (array int)
    [| 2; 3 |]
    (Nx.shape (E.to_vector y));
  equal_floats ~msg:"broadcast result" [| 0.; 1.; 0.; 0.; 0.; 1. |] (E.to_vector y)

(* Properties *)

let axis_gen =
  Gen.map
    (fun (x, y) -> vec3 x y 1.)
    Gen.(pair (float_range (-2.) 2.) (float_range (-2.) 2.))

let angle_gen = Gen.float_range (-.Float.pi) Float.pi

let vector_gen =
  Gen.map
    (fun a -> E.vector (Nx.create f32 [| 3 |] a))
    Gen.(array ~size:(constant 3) (float_range (-2.) 2.))

let rotor_gen = Gen.map (fun (axis, a) -> rotor ~axis a) (Gen.pair axis_gen angle_gen)

let test_properties =
  group
    "properties"
    [ prop
        "a rotor preserves the norm of a vector"
        Gen.(pair rotor_gen vector_gen)
        (fun (r, x) ->
           equal_scalar
             ~msg:"norm2 before and after"
             (Nx.item [] (E.norm2 x))
             (E.norm2 (E.apply_rotor r x)))
    ; prop
        "a rotor is linear"
        Gen.(pair rotor_gen (pair vector_gen vector_gen))
        (fun (r, (x, y)) ->
           equal_tensor
             (E.apply_rotor r (Nx.add x y))
             (Nx.add (E.apply_rotor r x) (E.apply_rotor r y)))
    ; prop
        "a rotor preserves the inner product"
        Gen.(pair rotor_gen (pair vector_gen vector_gen))
        (fun (r, (x, y)) ->
           let dot a b = Nx.item [] (E.coeff (E.inner a b) (B.of_int 0)) in
           equal (float 1e-4) (dot x y) (dot (E.apply_rotor r x) (E.apply_rotor r y)))
    ; prop
        "a rotor and its inverse cancel"
        Gen.(pair rotor_gen vector_gen)
        (fun (r, x) -> equal_tensor x (E.apply_rotor (E.reverse r) (E.apply_rotor r x)))
    ; prop "a full turn fixes every vector" vector_gen (fun x ->
        equal_tensor x (E.apply_rotor (rotor ~axis:z_axis (2. *. Float.pi)) x))
    ]

let () =
  exit
    (run
       "geom e3"
       [ group
           "vectors"
           [ test "basis and vector conversions" test_basis_and_vectors
           ; test "the Euclidean products" test_vector_products
           ]
       ; group
           "rotors"
           [ test "rotations about a coordinate axis" test_rotor_basics
           ; test "rotors compose" test_rotor_composition
           ; test "a batch of rotors" test_rotor_batch
           ]
       ; test_properties
       ])
