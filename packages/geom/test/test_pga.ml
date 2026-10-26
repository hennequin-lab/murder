(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Geom.Pga: Cl(3,0,1), the plane-based projective model. *)

open Windtrap
open Support
module P = Geom.Pga
module M = Geom.Motor
module B = Geom.Blade

let point = P.point3
let plane = P.plane3
let coords p = P.point_coords p
let absf = Float.abs

let check_point ?(eps = 1e-4) ~msg p (x, y, z) =
  equal_floats ~msg ~eps [| x; y; z |] (coords p)

(* Points and planes *)

let test_points () =
  check_point ~msg:"point coordinates round trip" (point 1. (-2.) 3.) (1., -2., 3.);
  equal_scalar ~msg:"a point has unit weight" 1. (P.weight (point 1. 2. 3.));
  equal_scalar
    ~msg:"a rescaled point has the scaled weight"
    3.
    (P.weight (Nx.mul_s (point 1. 2. 3.) 3.));
  check_point
    ~msg:"normalize_point keeps the coordinates"
    (P.normalize_point (Nx.mul_s (point 1. 2. 3.) 3.))
    (1., 2., 3.);
  equal_scalar
    ~msg:"normalize_point gives unit weight"
    1.
    (P.weight (P.normalize_point (Nx.mul_s (point 1. 2. 3.) 3.)));
  equal_bool ~msg:"a finite point is not ideal" false (P.is_ideal (point 1. 2. 3.));
  let d = vec3 1. 2. 3. in
  let ideal = P.ideal_point d in
  equal_scalar ~msg:"an ideal point has zero weight" 0. (P.weight ideal);
  equal_bool ~msg:"an ideal point is ideal" true (P.is_ideal ideal);
  equal
    ~msg:"point accepts a batch"
    (array int)
    [| 2; 16 |]
    (P.shape (P.point (Nx.create f32 [| 2; 3 |] [| 1.; 2.; 3.; -1.; 0.; 0.5 |])));
  equal
    ~msg:"point_coords keeps the batch shape"
    (array int)
    [| 2; 3 |]
    (Nx.shape
       (coords (P.point (Nx.create f32 [| 2; 3 |] [| 1.; 2.; 3.; -1.; 0.; 0.5 |]))))

let test_planes () =
  let pi = plane 0. 0. 1. 1. in
  equal_floats ~msg:"plane normal" [| 0.; 0.; 1. |] (P.plane_normal pi);
  equal_floats ~msg:"plane offset" [| 1. |] (P.plane_offset pi);
  equal_bool ~msg:"a point on the plane" true (P.point_on_plane (point 0. 0. 1.) pi);
  equal_bool ~msg:"a point off the plane" false (P.point_on_plane (point 0. 0. 0.) pi);
  equal_bool ~msg:"a point on the other side" false (P.point_on_plane (point 0. 0. 2.) pi);
  let scaled = plane 0. 0. 2. 2. in
  let normalized = P.normalize_plane scaled in
  equal_floats
    ~msg:"normalize_plane gives a unit normal"
    [| 0.; 0.; 1. |]
    (P.plane_normal normalized);
  equal_floats
    ~msg:"normalize_plane keeps the offset"
    [| 1. |]
    (P.plane_offset normalized);
  equal_bool
    ~msg:"a plane with a non-unit normal still tests incidence"
    true
    (P.point_on_plane (point 0. 0. 1.) scaled);
  equal
    ~msg:"plane accepts a batch"
    (array int)
    [| 2; 16 |]
    (P.shape
       (P.plane
          ~normal:(Nx.create f32 [| 2; 3 |] [| 1.; 0.; 0.; 0.; 1.; 0. |])
          ~offset:(Nx.create f32 [| 2 |] [| 1.; 2. |])))

(* Lines *)

let test_lines () =
  let p = point 1. 2. 3.
  and q = point 1. 2. 4. in
  let l = P.line_through p q in
  let d = P.line_direction l |> Nx.to_array in
  let m = P.line_moment l |> Nx.to_array in
  equal_tensor ~msg:"a line is a bivector" l (P.grade 2 l);
  equal
    ~msg:"the line has a unit direction"
    bool
    true
    (absf (sqrt ((d.(0) *. d.(0)) +. (d.(1) *. d.(1)) +. (d.(2) *. d.(2))) -. 1.) < 1e-4);
  equal
    ~msg:"the direction is along z"
    bool
    true
    (absf d.(0) < 1e-4 && absf d.(1) < 1e-4 && absf (absf d.(2) -. 1.) < 1e-4);
  equal
    ~msg:"the moment is r cross the direction"
    bool
    true
    (absf (m.(0) -. (2. *. d.(2))) < 1e-4
     && absf (m.(1) -. (-1. *. d.(2))) < 1e-4
     && absf m.(2) < 1e-4);
  equal_bool ~msg:"an endpoint is on the line" true (P.point_on_line p l);
  equal_bool ~msg:"the other endpoint is on the line" true (P.point_on_line q l);
  equal_bool ~msg:"the midpoint is on the line" true (P.point_on_line (point 1. 2. 3.5) l);
  equal_bool ~msg:"a point off the line" false (P.point_on_line (point 0. 2. 3.) l);
  let l0 = P.line_through (point 0. 0. 0.) (point 1. 0. 0.) in
  let d0 = P.line_direction l0 |> Nx.to_array in
  equal_floats
    ~msg:"the x-axis has no moment"
    ~eps:1e-4
    [| 0.; 0.; 0. |]
    (P.line_moment l0);
  equal
    ~msg:"the x-axis direction"
    bool
    true
    (absf (absf d0.(0) -. 1.) < 1e-4 && absf d0.(1) < 1e-4 && absf d0.(2) < 1e-4)

let test_line_incidence () =
  let l = P.line_of_planes (plane 1. 0. 0. 0.) (plane 0. 1. 0. 0.) in
  equal_bool ~msg:"the z-axis lies in x=0" true (P.line_in_plane l (plane 1. 0. 0. 0.));
  equal_bool ~msg:"the z-axis lies in y=0" true (P.line_in_plane l (plane 0. 1. 0. 0.));
  equal_bool
    ~msg:"the z-axis does not lie in z=1"
    false
    (P.line_in_plane l (plane 0. 0. 1. 1.));
  equal_bool
    ~msg:"a point on the z-axis is on the line"
    true
    (P.point_on_line (point 0. 0. 2.5) l);
  equal_bool
    ~msg:"a point off the z-axis is not on the line"
    false
    (P.point_on_line (point 1. 0. 2.5) l)

let test_meet_and_join () =
  let p = point 1. 2. 3.
  and q = point (-1.) 0. 2. in
  equal_tensor
    ~msg:"the join of two points is their line"
    (P.line_through p q)
    (P.join p q);
  let l = P.line_through p q in
  let r = point 0. 1. 1. in
  let pi = P.join r l in
  equal_bool ~msg:"the joined plane contains r" true (P.point_on_plane r pi);
  equal_bool ~msg:"the joined plane contains p" true (P.point_on_plane p pi);
  equal_bool ~msg:"the joined plane contains q" true (P.point_on_plane q pi);
  let zline = P.line_through (point 0. 0. 0.) (point 0. 0. 1.) in
  let hit = P.meet (plane 0. 0. 1. 0.5) zline in
  check_point ~msg:"a plane and a line meet in a point" hit (0., 0., 0.5);
  let l2 = P.meet (plane 1. 0. 0. 0.) (plane 0. 1. 0. 0.) in
  equal_bool
    ~msg:"the meet of two planes is their line"
    true
    (P.point_on_line (point 0. 0. 7.) l2);
  let parallel = P.line_through (point 0. 0. 0.) (point 1. 0. 0.) in
  let ideal = P.meet (plane 0. 0. 1. 1.) parallel in
  equal_bool ~msg:"parallel objects meet at an ideal point" true (P.is_ideal ideal);
  equal_tensor
    ~msg:"meet is the outer product"
    (P.op (plane 1. 0. 0. 0.) (plane 0. 1. 0. 0.))
    (P.meet (plane 1. 0. 0. 0.) (plane 0. 1. 0. 0.))

(* Motors *)

let test_motors () =
  let p = point 1. 2. 3. in
  let t = M.translate3 0. 0. 1. in
  equal
    ~msg:"of_motor embeds into the algebra"
    (array int)
    [| 16 |]
    (P.shape (P.of_motor t));
  check_point ~msg:"a translation moves a point" (P.apply_motor t p) (1., 2., 4.);
  let r = M.rotate ~axis:(vec3 0. 0. 1.) ~angle:(scalar (Float.pi /. 2.)) in
  check_point
    ~msg:"a rotation turns a point"
    (P.apply_motor r (point 1. 0. 0.))
    (0., 1., 0.);
  let m0 = M.translate3 1. (-2.) 0.5
  and m1 = M.rotate ~axis:(vec3 1. 1. 0.) ~angle:(scalar 0.8) in
  equal_tensor
    ~msg:"the action composes"
    (P.apply_motor (M.mul m0 m1) p)
    (P.apply_motor m0 (P.apply_motor m1 p));
  let pi = plane 1. 0. 0. 1. in
  let moved_p = P.apply_motor m0 p
  and moved_pi = P.apply_motor m0 pi in
  equal_bool ~msg:"a motor preserves incidence" true (P.point_on_plane moved_p moved_pi);
  equal_bool ~msg:"the original incidence" true (P.point_on_plane p pi);
  let batch =
    P.apply_motor
      (M.translate (Nx.create f32 [| 2; 3 |] [| 0.; 0.; 1.; 1.; 0.; 0. |]))
      (P.point (Nx.create f32 [| 2; 3 |] [| 0.; 0.; 0.; 1.; 0.; 0. |]))
  in
  equal
    ~msg:"a batch of points through a batch of motors"
    (array int)
    [| 2; 16 |]
    (P.shape batch);
  equal ~msg:"the batch coordinates" (array int) [| 2; 3 |] (Nx.shape (coords batch))

let test_inherited () =
  raises_match
    ~msg:"the Hodge dual is undefined in a degenerate algebra"
    (Exn.invalid_arg ~substring:"degenerate")
    (fun () -> ignore (P.dual (P.e 1)));
  equal_tensor
    ~msg:"the complement is defined"
    (P.complement (P.e 1))
    (P.complement (P.e 1))

(* Properties *)

let vec_gen =
  Gen.map
    (fun v -> Nx.create f32 [| 3 |] v)
    Gen.(array ~size:(constant 3) (float_range (-2.) 2.))

let point_gen = Gen.map P.point vec_gen

let axis_gen =
  Gen.map
    (fun (x, y) -> Nx.create f32 [| 3 |] [| x; y; 1. |])
    Gen.(pair (float_range (-1.) 1.) (float_range (-1.) 1.))

let plane_gen =
  Gen.map
    (fun (n, d) ->
       P.plane ~normal:(Nx.create f32 [| 3 |] [| n.(0); n.(1); 1. |]) ~offset:(scalar d))
    Gen.(pair (array ~size:(constant 3) (float_range (-1.) 1.)) (float_range (-2.) 2.))

let motor_gen =
  Gen.map
    (fun (t, axis, a) ->
       M.mul (M.translate (Nx.create f32 [| 3 |] t)) (M.rotate ~axis ~angle:(scalar a)))
    Gen.(
      triple
        (array ~size:(constant 3) (float_range (-1.) 1.))
        axis_gen
        (float_range (-2.5) 2.5))

let test_properties =
  group
    "properties"
    [ prop "point_coords inverts point" vec_gen (fun v ->
        equal_floats (Nx.to_array v) (coords (P.point v)))
    ; prop "both endpoints lie on their line" vec_gen (fun v ->
        let p = P.point v in
        let q = P.point (Nx.add v (vec3 1. 0.5 0.25)) in
        let l = P.line_through p q in
        equal_bool true (P.point_on_line p l);
        equal_bool true (P.point_on_line q l);
        let d = P.line_direction l in
        equal_scalar ~eps:1e-4 1. (Nx.sqrt (Nx.sum ~axes:[ 0 ] (Nx.mul d d))))
    ; prop
        "a motor preserves incidence"
        Gen.(pair motor_gen (pair point_gen plane_gen))
        (fun (m, (p, pi)) ->
           equal
             (array bool)
             (Nx.to_array (P.point_on_plane p pi))
             (Nx.to_array (P.point_on_plane (P.apply_motor m p) (P.apply_motor m pi))))
    ; prop
        "the motor action is a homomorphism"
        Gen.(pair motor_gen (pair motor_gen point_gen))
        (fun (m0, (m1, p)) ->
           equal_tensor
             (P.apply_motor (M.mul m0 m1) p)
             (P.apply_motor m0 (P.apply_motor m1 p)))
    ]

let () =
  exit
    (run
       "geom pga"
       [ group
           "objects"
           [ test "points" test_points
           ; test "planes" test_planes
           ; test "lines" test_lines
           ; test "line incidence" test_line_incidence
           ]
       ; group "incidence" [ test "meet and join" test_meet_and_join ]
       ; group "motors" [ test "the packed motor bridge" test_motors ]
       ; group
           "inherited"
           [ test "the generic algebra in a degenerate signature" test_inherited ]
       ; test_properties
       ])
