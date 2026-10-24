(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Geom.Motor: the packed dual quaternion of SE(3). The coefficient
   order is [1, e12, e13, e23, e1 einf, e2 einf, e3 einf, e123 einf]. *)

open Windtrap
open Support
module M = Geom.Motor

let z_axis = vec3 0. 0. 1.
let x_axis = vec3 1. 0. 0.
let q w x y z = Nx.create f32 [| 4 |] [| w; x; y; z |]
let rot ~axis a = M.rotate ~axis ~angle:(scalar a)
let translation v = M.translate v
let check_motor = equal_tensor

(* A motor and its negative are the same rigid motion. *)
let motor_proj =
  Testable.make ~pp:M.pp ~equal:(fun a b ->
    Support.close a b || Support.close a (Nx.mul_s b (-1.)))

let check_motor_proj ?msg expected actual = equal ?msg motor_proj expected actual

(* Rodrigues' formula, used to build matrices independently of [Motor.rotate]
   so that [of_rotation] is checked against a second construction. *)
let rotation_matrix ~axis ~angle =
  let x = Nx.item [ 0 ] axis
  and y = Nx.item [ 1 ] axis
  and z = Nx.item [ 2 ] axis in
  let n = sqrt ((x *. x) +. (y *. y) +. (z *. z)) in
  let x = x /. n
  and y = y /. n
  and z = z /. n in
  let c = cos angle
  and s = sin angle
  and t = 1. -. cos angle in
  Nx.create
    f32
    [| 3; 3 |]
    [| (t *. x *. x) +. c
     ; (t *. x *. y) -. (s *. z)
     ; (t *. x *. z) +. (s *. y)
     ; (t *. x *. y) +. (s *. z)
     ; (t *. y *. y) +. c
     ; (t *. y *. z) -. (s *. x)
     ; (t *. x *. z) -. (s *. y)
     ; (t *. y *. z) +. (s *. x)
     ; (t *. z *. z) +. c
    |]

(* Construction and representation *)

let test_representation () =
  equal ~msg:"size" int 8 M.size;
  equal ~msg:"dtype" string "float32" (Format.asprintf "%a" Nx.pp_dtype M.dtype);
  equal
    ~msg:"one"
    (array (float 1e-6))
    [| 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
    (M.to_array M.one);
  equal
    ~msg:"zero"
    (array (float 1e-6))
    [| 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
    (M.to_array M.zero);
  let a = [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
  equal
    ~msg:"of_array and to_array round trip"
    (array (float 1e-6))
    a
    (M.to_array (M.of_array a));
  raises_match
    ~msg:"of_array checks the length"
    (Exn.invalid_arg ~substring:"8")
    (fun () -> ignore (M.of_array [| 1.; 2. |]));
  raises_match
    ~msg:"to_array refuses a batch"
    (Exn.invalid_arg ~substring:"unbatched")
    (fun () -> ignore (M.to_array (M.of_nx (Nx.zeros M.dtype [| 2; 8 |]))));
  raises_match
    ~msg:"of_nx checks the motor axis"
    (Exn.invalid_arg ~substring:"8")
    (fun () -> ignore (M.of_nx (Nx.zeros M.dtype [| 7 |])));
  equal ~msg:"shape" (array int) [| 8 |] (M.shape M.one);
  equal ~msg:"batch shape" (array int) [||] (M.batch_shape M.one);
  let batch = M.translate (Nx.create f32 [| 2; 3 |] [| 1.; 2.; 3.; -1.; 0.; 1. |]) in
  equal ~msg:"a batched translation" (array int) [| 2; 8 |] (M.shape batch);
  equal ~msg:"its batch shape" (array int) [| 2 |] (M.batch_shape batch)

let test_motion_constructors () =
  equal
    ~msg:"an identity translation"
    (array (float 1e-6))
    [| 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
    (M.to_array (M.translate3 0. 0. 0.));
  equal
    ~msg:"translate packs half the vector with a minus sign"
    (array (float 1e-6))
    [| 1.; 0.; 0.; 0.; -0.5; -1.; -1.5; 0. |]
    (M.to_array (M.translate3 1. 2. 3.));
  equal
    ~msg:"a zero rotation is one"
    (array (float 1e-6))
    [| 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
    (M.to_array (rot ~axis:z_axis 0.));
  equal
    ~msg:"a quarter turn about z"
    (array (float 1e-6))
    [| cos (Float.pi /. 4.); -.sin (Float.pi /. 4.); 0.; 0.; 0.; 0.; 0.; 0. |]
    (M.to_array (rot ~axis:z_axis (Float.pi /. 2.)));
  equal
    ~msg:"a turn about x"
    (array (float 1e-6))
    [| cos (Float.pi /. 4.); 0.; 0.; -.sin (Float.pi /. 4.); 0.; 0.; 0.; 0. |]
    (M.to_array (rot ~axis:x_axis (Float.pi /. 2.)));
  equal
    ~msg:"the axis is normalized"
    (array (float 1e-6))
    (M.to_array (rot ~axis:z_axis 0.7))
    (M.to_array (rot ~axis:(vec3 0. 0. 5.) 0.7))

let test_quaternions () =
  equal
    ~msg:"the identity quaternion"
    (array (float 1e-6))
    [| 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
    (M.to_array (M.of_quaternion (q 1. 0. 0. 0.)));
  equal
    ~msg:"a half turn about x"
    (array (float 1e-6))
    [| 0.; 0.; 0.; -1.; 0.; 0.; 0.; 0. |]
    (M.to_array (M.of_quaternion (q 0. 1. 0. 0.)));
  equal
    ~msg:"a quarter turn about z from a quaternion"
    (array (float 1e-6))
    (M.to_array (rot ~axis:z_axis (Float.pi /. 2.)))
    (M.to_array (M.of_quaternion (q (cos (Float.pi /. 4.)) 0. 0. (sin (Float.pi /. 4.)))));
  equal
    ~msg:"quaternions are normalized"
    (array (float 1e-6))
    (M.to_array M.one)
    (M.to_array (M.of_quaternion (q 2. 0. 0. 0.)));
  let r = rot ~axis:(vec3 1. 2. 3.) 0.7 in
  let n = 1. /. sqrt 14. in
  equal_floats
    ~msg:"to_quaternion of a rotation"
    [| cos 0.35; sin 0.35 *. n; 2. *. sin 0.35 *. n; 3. *. sin 0.35 *. n |]
    (M.to_quaternion r);
  check_motor
    ~msg:"of_quaternion inverts to_quaternion"
    r
    (M.of_quaternion (M.to_quaternion r));
  equal
    ~msg:"to_quaternion drops the translation"
    (array (float 1e-6))
    (Nx.to_array (M.to_quaternion r))
    (Nx.to_array (M.to_quaternion (M.mul (translation (vec3 1. 2. 3.)) r)))

let test_rotation_matrices () =
  let identity = Nx.create f32 [| 3; 3 |] [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |] in
  check_motor ~msg:"the identity matrix" M.one (M.of_rotation identity);
  let rz90 = Nx.create f32 [| 3; 3 |] [| 0.; -1.; 0.; 1.; 0.; 0.; 0.; 0.; 1. |] in
  check_motor
    ~msg:"a quarter turn about z"
    (rot ~axis:z_axis (Float.pi /. 2.))
    (M.of_rotation rz90);
  let rx90 = Nx.create f32 [| 3; 3 |] [| 1.; 0.; 0.; 0.; 0.; -1.; 0.; 1.; 0. |] in
  check_motor
    ~msg:"a quarter turn about x"
    (rot ~axis:x_axis (Float.pi /. 2.))
    (M.of_rotation rx90);
  raises_match
    ~msg:"of_rotation checks the matrix shape"
    (Exn.invalid_arg ~substring:"3x3")
    (fun () -> ignore (M.of_rotation (Nx.zeros M.dtype [| 2; 3 |])));
  let batch = Nx.stack ~axis:0 [ identity; rz90 ] in
  let m = M.of_rotation batch in
  equal ~msg:"a batch of rotation matrices" (array int) [| 2; 8 |] (M.shape m);
  check_motor
    ~msg:"the first matrix"
    M.one
    (M.of_nx (Nx.take ~axis:0 ~indices:(Nx.scalar Nx.int64 0L) m));
  check_motor
    ~msg:"the second matrix"
    (rot ~axis:z_axis (Float.pi /. 2.))
    (M.of_nx (Nx.take ~axis:0 ~indices:(Nx.scalar Nx.int64 1L) m));
  (* Half turns have trace -1, so [of_rotation] takes one of its three
     diagonal branches instead of the trace branch. *)
  let rx180 = Nx.create f32 [| 3; 3 |] [| 1.; 0.; 0.; 0.; -1.; 0.; 0.; 0.; -1. |] in
  check_motor_proj
    ~msg:"a half turn about x from a matrix"
    (rot ~axis:x_axis Float.pi)
    (M.of_rotation rx180);
  let ry180 = Nx.create f32 [| 3; 3 |] [| -1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; -1. |] in
  check_motor_proj
    ~msg:"a half turn about y from a matrix"
    (rot ~axis:(vec3 0. 1. 0.) Float.pi)
    (M.of_rotation ry180);
  let rz180 = Nx.create f32 [| 3; 3 |] [| -1.; 0.; 0.; 0.; -1.; 0.; 0.; 0.; 1. |] in
  check_motor_proj
    ~msg:"a half turn about z from a matrix"
    (rot ~axis:z_axis Float.pi)
    (M.of_rotation rz180);
  (* Generic rotations beyond pi/2 have a negative trace and non-zero
     off-diagonal terms, so they exercise the diagonal branches and their
     signs. *)
  List.iter
    (fun (axis, angle) ->
       check_motor_proj
         ~msg:(Printf.sprintf "a generic rotation matrix for angle %g" angle)
         (rot ~axis angle)
         (M.of_rotation (rotation_matrix ~axis ~angle)))
    [ vec3 1. 0.3 0.2, 2.0
    ; vec3 0.3 1. 0.2, 2.0
    ; vec3 0.2 0.3 1., 2.0
    ; vec3 0.8 0.2 1., 2.5 (* m00 > m11 but m22 is the largest *)
    ; vec3 1. (-2.) 0.5, 2.5
    ; vec3 1. 2. 3., 1.0
    ]

(* Algebra of motors *)

let test_algebra () =
  let m = M.mul (translation (vec3 0.5 (-0.2) 0.3)) (rot ~axis:z_axis 0.7) in
  check_motor ~msg:"one is neutral on the left" m (M.mul M.one m);
  check_motor ~msg:"one is neutral on the right" m (M.mul m M.one);
  check_motor
    ~msg:"mul is associative"
    (M.mul (M.mul m (translation (vec3 1. 2. 3.))) (rot ~axis:x_axis 0.4))
    (M.mul m (M.mul (translation (vec3 1. 2. 3.)) (rot ~axis:x_axis 0.4)));
  check_motor
    ~msg:"the inverse composes to one on the right"
    M.one
    (M.mul m (M.inverse m));
  check_motor ~msg:"the inverse composes to one on the left" M.one (M.mul (M.inverse m) m);
  check_motor
    ~msg:"normalize gives a unit motor"
    M.one
    (M.mul (M.normalize m) (M.inverse (M.normalize m)));
  check_motor ~msg:"compose [] is one" M.one (M.compose []);
  check_motor
    ~msg:"compose applies the list in order"
    (M.compose [ m; translation (vec3 1. 2. 3.); rot ~axis:x_axis 0.4 ])
    (M.mul (M.mul m (translation (vec3 1. 2. 3.))) (rot ~axis:x_axis 0.4))

let test_translation_algebra () =
  let a = vec3 1. 2. 3.
  and b = vec3 (-0.5) 0.25 4. in
  check_motor
    ~msg:"translations add"
    (translation (Nx.add a b))
    (M.mul (translation a) (translation b));
  check_motor
    ~msg:"the inverse translation"
    (translation (Nx.mul_s a (-1.)))
    (M.inverse (translation a))

let test_rotation_algebra () =
  let r1 = rot ~axis:z_axis 0.3
  and r2 = rot ~axis:z_axis 0.5 in
  check_motor
    ~msg:"rotations about one axis add angles"
    (rot ~axis:z_axis 0.8)
    (M.mul r1 r2);
  let rx = rot ~axis:x_axis 0.4 in
  check_motor ~msg:"a product is not generally commutative" (M.mul r1 rx) (M.mul r1 rx);
  check_motor
    ~msg:"a half turn squares to minus one"
    (Nx.mul_s M.one (-1.))
    (M.mul (rot ~axis:z_axis Float.pi) (rot ~axis:z_axis Float.pi))

(* Exponential and logarithm *)

let test_exp () =
  check_motor ~msg:"exp of zero is one" M.one (M.exp (vec6 0. 0. 0. 0. 0. 0.));
  check_motor
    ~msg:"exp of a translation twist"
    (M.translate3 1. 2. 3.)
    (M.exp (vec6 0. 0. 0. 1. 2. 3.));
  check_motor
    ~msg:"exp of a rotation twist"
    (rot ~axis:z_axis (Float.pi /. 2.))
    (M.exp (vec6 0. 0. (Float.pi /. 2.) 0. 0. 0.));
  check_motor
    ~msg:"exp of a small rotation"
    (rot ~axis:x_axis 1e-4)
    (M.exp (vec6 1e-4 0. 0. 0. 0. 0.));
  raises_match
    ~msg:"exp checks the twist axis"
    (Exn.invalid_arg ~substring:"6")
    (fun () -> ignore (M.exp (Nx.create f32 [| 5 |] [| 0.; 0.; 0.; 0.; 0. |])));
  let twists =
    [ vec6 0. 0. 0. 0. 0. 1.
    ; vec6 0. 0. 0.5 0. 0. 0.
    ; vec6 0.2 (-0.1) 0.5 0.3 0.1 (-0.2)
    ; vec6 0.0 0.0 0.0 0.0 0.0 0.0
    ; vec6 1e-7 (-2e-7) 3e-7 1e-6 (-2e-6) 3e-6
    ]
  in
  List.iter
    (fun tw ->
       let name = Format.asprintf "%a" Nx.pp tw in
       equal_floats
         ~msg:("log (exp twist) for " ^ name)
         ~eps:2e-5
         (Nx.to_array tw)
         (M.log (M.exp tw)))
    twists

let test_log () =
  equal_floats ~msg:"log of one is zero" [| 0.; 0.; 0.; 0.; 0.; 0. |] (M.log M.one);
  let m =
    M.mul (M.translate3 0.5 (-0.2) 0.3) (M.rotate ~axis:z_axis ~angle:(scalar 0.7))
  in
  check_motor ~msg:"exp (log m) is m" m (M.exp (M.log m));
  let angles = [ 0.1; 0.5; 1.0; 2.0; 3.0; -0.5; -2.0 ] in
  List.iter
    (fun a ->
       let m =
         M.mul
           (M.translate3 0.5 (-0.2) 0.3)
           (M.rotate ~axis:(vec3 1. 2. 3.) ~angle:(scalar a))
       in
       check_motor ~msg:(Printf.sprintf "exp (log m) for angle %g" a) m (M.exp (M.log m)))
    angles;
  let tw = vec6 0.2 (-0.1) 0.5 0.3 0.1 (-0.2) in
  equal_floats
    ~msg:"log of a pure translation"
    [| 0.; 0.; 0.; 1.; 2.; 3. |]
    (M.log (M.translate3 1. 2. 3.));
  equal_floats ~msg:"log of a screw" (Nx.to_array tw) (M.log (M.exp tw));
  (* For a rotation angle beyond pi the rotor scalar part is negative. The
     translation must be extracted from the same rotor representative, not
     from the sign-normalised quaternion. *)
  List.iter
    (fun a ->
       let m =
         M.mul (M.translate3 0.5 (-0.2) 0.3) (M.rotate ~axis:z_axis ~angle:(scalar a))
       in
       check_motor_proj
         ~msg:(Printf.sprintf "exp (log m) for angle %g" a)
         m
         (M.exp (M.log m)))
    [ 4.0; 5.0; -4.0 ]

(* Interpolation *)

let test_slerp () =
  let m0 = M.translate3 1. 0. 0.
  and m1 = M.rotate ~axis:z_axis ~angle:(scalar 1.) in
  check_motor ~msg:"slerp at zero" m0 (M.slerp m0 m1 (scalar 0.));
  check_motor_proj ~msg:"slerp at one" m1 (M.slerp m0 m1 (scalar 1.));
  check_motor ~msg:"slerp with equal endpoints" m0 (M.slerp m0 m0 (scalar 0.37));
  let a = vec3 1. 2. 3.
  and b = vec3 (-1.) 0. 2. in
  check_motor
    ~msg:"slerp of translations is linear"
    (translation (Nx.add (Nx.mul_s a 0.75) (Nx.mul_s b 0.25)))
    (M.slerp (translation a) (translation b) (scalar 0.25));
  let r0 = rot ~axis:z_axis 0.3
  and r1 = rot ~axis:z_axis 0.7 in
  check_motor
    ~msg:"slerp of rotations interpolates the angle"
    (rot ~axis:z_axis 0.5)
    (M.slerp r0 r1 (scalar 0.5));
  let b0 = M.mul (translation a) r0
  and b1 = M.mul (translation b) r1 in
  let mid = M.slerp b0 b1 (scalar 0.5) in
  check_motor
    ~msg:"the midpoint is equidistant in the group"
    mid
    (let half = M.exp (Nx.mul_s (M.log (M.mul (M.inverse b0) b1)) 0.5) in
     M.mul b0 half);
  let alpha = Nx.create f32 [| 2 |] [| 0.; 1. |] in
  let batch = M.slerp m0 m1 alpha in
  equal ~msg:"slerp accepts a batch of parameters" (array int) [| 2; 8 |] (M.shape batch)

(* Adjoint *)

let test_adjoint () =
  let m0 = M.translate3 1. 2. 3.
  and m1 = M.mul (M.translate3 0.5 (-0.2) 0.3) (rot ~axis:z_axis 0.7) in
  let tw = vec6 0.1 (-0.2) 0.3 0.2 0.1 (-0.1) in
  let conjugates name m =
    let left = M.log (M.mul (M.mul m (M.exp tw)) (M.inverse m)) in
    let right =
      Nx.matmul (M.adjoint m) (Nx.reshape [| 6; 1 |] tw) |> Nx.reshape [| 6 |]
    in
    equal ~msg:name (array (float 2e-4)) (Nx.to_array left) (Nx.to_array right)
  in
  conjugates "the adjoint of a translation" m0;
  conjugates "the adjoint of a rotation" m1;
  equal ~msg:"the adjoint shape" (array int) [| 6; 6 |] (Nx.shape (M.adjoint m1));
  let product = M.mul m0 m1 in
  equal_tensor
    ~msg:"the adjoint is a homomorphism"
    (M.adjoint product)
    (Nx.matmul (M.adjoint m0) (M.adjoint m1));
  let batch = M.adjoint (M.of_nx (Nx.stack ~axis:0 [ m0; m1 ])) in
  equal ~msg:"a batch of adjoints" (array int) [| 2; 6; 6 |] (Nx.shape batch)

(* Printing *)

let test_printing () =
  equal
    ~msg:"one prints its coefficients"
    string
    "[1; 0; 0; 0; 0; 0; 0; 0]"
    (Format.asprintf "%a" M.pp M.one);
  equal
    ~msg:"a batch prints its shape"
    string
    "<motor 2x8>"
    (Format.asprintf "%a" M.pp (M.of_nx (Nx.stack ~axis:0 [ M.one; M.one ])));
  equal
    ~msg:"a square batch prints its shape"
    string
    "<motor 8x8>"
    (Format.asprintf "%a" M.pp (M.of_nx (Nx.zeros M.dtype [| 8; 8 |])))

(* Properties *)

let motor_gen =
  Gen.map
    (fun tw -> M.exp (Nx.create f32 [| 6 |] tw))
    Gen.(array ~size:(constant 6) (float_range (-1.5) 1.5))

let test_properties =
  group
    "properties"
    [ prop
        "mul is associative"
        Gen.(triple motor_gen motor_gen motor_gen)
        (fun (a, b, c) -> Law.associative (Support.tensor ()) M.mul (a, b, c))
    ; prop "one is neutral" motor_gen (fun a ->
        Law.neutral (Support.tensor ()) M.mul M.one a)
    ; prop "inverse is the group inverse" motor_gen (fun a ->
        Law.invertible (Support.tensor ()) M.mul M.one M.inverse a)
    ; prop
        "exp and log are inverse"
        Gen.(array ~size:(constant 6) (float_range (-1.) 1.))
        (fun tw ->
           let tw = Nx.create f32 [| 6 |] tw in
           equal (array (float 2e-4)) (Nx.to_array tw) (Nx.to_array (M.log (M.exp tw))))
    ; prop
        "translations commute"
        Gen.(
          pair
            (array ~size:(constant 3) (float_range (-2.) 2.))
            (array ~size:(constant 3) (float_range (-2.) 2.)))
        (fun (a, b) ->
           let a = Nx.create f32 [| 3 |] a
           and b = Nx.create f32 [| 3 |] b in
           check_motor (M.translate (Nx.add a b)) (M.mul (M.translate a) (M.translate b)))
    ; prop
        "slerp starts at its first motor"
        Gen.(pair motor_gen motor_gen)
        (fun (a, b) -> check_motor a (M.slerp a b (scalar 0.)))
    ; prop
        "slerp ends at its second motor"
        Gen.(pair motor_gen motor_gen)
        (fun (a, b) -> check_motor_proj b (M.slerp a b (scalar 1.)))
    ; prop
        "the adjoint is a homomorphism"
        Gen.(pair motor_gen motor_gen)
        (fun (a, b) ->
           equal_tensor (M.adjoint (M.mul a b)) (Nx.matmul (M.adjoint a) (M.adjoint b)))
    ; prop
        "of_rotation agrees with rotate"
        Gen.(
          pair
            (map
               (fun (x, y) -> vec3 x y 1.)
               (pair (float_range (-2.) 2.) (float_range (-2.) 2.)))
            (float_range 0.2 2.8))
        (fun (axis, angle) ->
           check_motor_proj
             (rot ~axis angle)
             (M.of_rotation (rotation_matrix ~axis ~angle)))
    ]

let () =
  exit
    (run
       "geom motor"
       [ group
           "representation"
           [ test "packed coefficients and conversions" test_representation
           ; test "translations and rotations" test_motion_constructors
           ; test "quaternion constructors" test_quaternions
           ; test "rotation matrices" test_rotation_matrices
           ]
       ; group
           "algebra"
           [ test "composition, inverse and normalize" test_algebra
           ; test "translation algebra" test_translation_algebra
           ; test "rotation algebra" test_rotation_algebra
           ]
       ; group "exponential" [ test "exp" test_exp; test "log" test_log ]
       ; group "interpolation" [ test "slerp" test_slerp ]
       ; group "adjoint" [ test "the adjoint of a motion" test_adjoint ]
       ; group "printing" [ test "debug printing" test_printing ]
       ; test_properties
       ])
