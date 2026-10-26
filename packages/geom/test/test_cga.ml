(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Geom.Cga: the conformal model of R^3, Cl(4,1). *)

open Windtrap
open Support
module C = Geom.Cga
module M = Geom.Motor
module B = Geom.Blade

let p3 = C.point3
let v3 = vec3
let sc = scalar
let coords p = C.coords p
let absf = Float.abs

let check_point ?(eps = 2e-4) ~msg p (x, y, z) =
  equal_floats ~msg ~eps [| x; y; z |] (coords p)

let on ?msg x o = equal_bool ?msg true (C.incident x o)
let off ?msg x o = equal_bool ?msg false (C.incident x o)

(* The null basis *)

let test_null_basis () =
  equal_scalar ~msg:"e1 squares to 1" 1. (C.coeff (C.gp C.e1 C.e1) (B.of_int 0));
  equal_scalar ~msg:"e4 squares to 1" 1. (C.coeff (C.gp C.e4 C.e4) (B.of_int 0));
  equal_scalar ~msg:"e5 squares to -1" (-1.) (C.coeff (C.gp C.e5 C.e5) (B.of_int 0));
  equal_tensor ~msg:"e0 is (e5 - e4)/2" (Nx.mul_s (Nx.sub C.e5 C.e4) 0.5) C.e0;
  equal_tensor ~msg:"einf is e4 + e5" (Nx.add C.e4 C.e5) C.einf;
  equal_scalar ~msg:"e0 is null" 0. (C.norm2 C.e0);
  equal_scalar ~msg:"einf is null" 0. (C.norm2 C.einf);
  equal_scalar ~msg:"e0 dot einf is -1" (-1.) (C.scalar_prod C.e0 C.einf);
  equal_scalar ~msg:"einf dot e0 is -1" (-1.) (C.scalar_prod C.einf C.e0);
  equal_scalar
    ~msg:"e0 is orthogonal to the Euclidean directions"
    0.
    (C.scalar_prod C.e0 C.e1)

(* Points *)

let test_points () =
  let p = p3 1. 2. 3. in
  check_point ~msg:"coordinate round trip" p (1., 2., 3.);
  equal_scalar ~msg:"a point is null" 0. (C.norm2 p);
  equal_scalar ~msg:"einf dot a point is -1" (-1.) (C.scalar_prod C.einf p);
  equal_bool ~msg:"is_point on a point" true (C.is_point p);
  equal_bool
    ~msg:"is_point off a sphere"
    false
    (C.is_point (C.sphere ~center:(v3 0. 0. 0.) ~radius:(sc 1.)));
  equal_bool
    ~msg:"is_point off a plane"
    false
    (C.is_point (C.plane ~normal:(v3 0. 0. 1.) ~offset:(sc 0.)));
  check_point
    ~msg:"normalize_point keeps the coordinates"
    (C.normalize_point (Nx.mul_s p 3.))
    (1., 2., 3.);
  equal_bool
    ~msg:"a normalized point is a point"
    true
    (C.is_point (C.normalize_point (Nx.mul_s p 3.)));
  equal_scalar ~msg:"distance from the origin" (sqrt 14.) (C.distance p (p3 0. 0. 0.));
  equal_scalar ~msg:"squared distance" 25. (C.distance2 (p3 0. 0. 0.) (p3 3. 4. 0.));
  equal_scalar ~msg:"distance is symmetric" 5. (C.distance (p3 3. 4. 0.) (p3 0. 0. 0.));
  let batch = C.point (Nx.create f32 [| 2; 3 |] [| 1.; 2.; 3.; -1.; 0.; 0.5 |]) in
  equal ~msg:"a batch of points" (array int) [| 2; 32 |] (C.shape batch);
  equal
    ~msg:"its coordinates keep the batch shape"
    (array int)
    [| 2; 3 |]
    (Nx.shape (coords batch));
  equal_floats
    ~msg:"the batched coordinates"
    [| 1.; 2.; 3.; -1.; 0.; 0.5 |]
    (coords batch);
  equal
    ~msg:"a batched is_point"
    (array bool)
    [| true; true |]
    (Nx.to_array (C.is_point batch))

(* Spheres *)

let test_spheres () =
  let s = C.sphere ~center:(v3 1. 0. 0.) ~radius:(sc 2.) in
  equal_scalar ~msg:"the squared radius" 4. (C.sphere_radius2 s);
  check_point ~msg:"the center" (C.sphere_center s) (1., 0., 0.);
  on ~msg:"a point on the sphere" (p3 3. 0. 0.) s;
  on ~msg:"the opposite point on the sphere" (p3 (-1.) 0. 0.) s;
  off ~msg:"the center is not on the sphere" (p3 1. 0. 0.) s;
  off ~msg:"a point inside" (p3 2. 0. 0.) s;
  equal_bool ~msg:"a sphere is round" true (C.is_round s);
  equal_bool ~msg:"a sphere is not flat" false (C.is_flat s);
  equal_scalar ~msg:"einf dot a sphere is -1" (-1.) (C.scalar_prod C.einf s);
  let s' = C.sphere_of_point ~center:(p3 1. 0. 0.) ~radius:(sc 2.) in
  equal_tensor ~msg:"sphere_of_point agrees with sphere" s s';
  let scaled = Nx.mul_s s 3. in
  equal_scalar ~msg:"the radius is scale invariant" 4. (C.sphere_radius2 scaled);
  check_point ~msg:"the center is scale invariant" (C.sphere_center scaled) (1., 0., 0.);
  equal_scalar
    ~msg:"normalize_sphere fixes the einf coefficient"
    (-1.)
    (C.scalar_prod C.einf (C.normalize_sphere scaled));
  let imaginary = C.sphere ~center:(v3 0. 0. 0.) ~radius:(sc (-2.)) in
  equal_scalar ~msg:"an imaginary sphere reports r^2" 4. (C.sphere_radius2 imaginary);
  let batch =
    C.sphere
      ~center:(Nx.create f32 [| 2; 3 |] [| 1.; 0.; 0.; 0.; 2.; 0. |])
      ~radius:(Nx.create f32 [| 2 |] [| 2.; 3. |])
  in
  equal ~msg:"a batch of spheres" (array int) [| 2; 32 |] (C.shape batch);
  equal_floats ~msg:"their squared radii" [| 4.; 9. |] (C.sphere_radius2 batch);
  equal
    ~msg:"their centers keep the batch shape"
    (array int)
    [| 2; 3 |]
    (Nx.shape (coords (C.sphere_center batch)))

(* Planes *)

let test_planes () =
  let pi = C.plane ~normal:(v3 0. 0. 1.) ~offset:(sc 1.) in
  equal_floats ~msg:"the normal" [| 0.; 0.; 1. |] (C.plane_normal pi);
  equal_floats ~msg:"the offset" [| 1. |] (C.plane_offset pi);
  on ~msg:"a point on the plane" (p3 0. 0. 1.) pi;
  off ~msg:"a point below the plane" (p3 0. 0. 0.) pi;
  equal_bool ~msg:"a plane is flat" true (C.is_flat pi);
  equal_bool ~msg:"a plane is not round" false (C.is_round pi);
  let scaled = C.plane ~normal:(v3 0. 0. 2.) ~offset:(sc 2.) in
  let normalized = C.normalize_plane scaled in
  equal_floats
    ~msg:"normalize_plane gives a unit normal"
    [| 0.; 0.; 1. |]
    (C.plane_normal normalized);
  equal_floats
    ~msg:"normalize_plane gives a signed offset"
    [| 1. |]
    (C.plane_offset normalized);
  on ~msg:"the scaled plane still contains the point" (p3 0. 0. 1.) scaled;
  let batch =
    C.plane
      ~normal:(Nx.create f32 [| 2; 3 |] [| 1.; 0.; 0.; 0.; 1.; 0. |])
      ~offset:(Nx.create f32 [| 2 |] [| 1.; 2. |])
  in
  equal ~msg:"a batch of planes" (array int) [| 2; 32 |] (C.shape batch);
  equal
    ~msg:"their normals keep the batch shape"
    (array int)
    [| 2; 3 |]
    (Nx.shape (C.plane_normal batch));
  equal_floats
    ~msg:"the batched normals"
    [| 1.; 0.; 0.; 0.; 1.; 0. |]
    (C.plane_normal batch);
  equal_floats ~msg:"the batched offsets" [| 1.; 2. |] (C.plane_offset batch)

(* Point pairs, lines and circles *)

let test_point_pairs () =
  let a = p3 1. 0. 0.
  and b = p3 0. 1. 0. in
  let pair = C.point_pair a b in
  (* A point pair is a direct 2-blade, so a null point is in the pair exactly
     when its wedge with the pair vanishes. *)
  equal_scalar ~msg:"the first point of a pair" 0. (C.norm (C.op a pair));
  equal_scalar ~msg:"the second point of a pair" 0. (C.norm (C.op b pair));
  greater
    ~msg:"a point off the pair"
    (float 1e-4)
    ~than:0.
    (Nx.item [] (C.norm (C.op (p3 1. 1. 0.) pair)));
  equal_bool ~msg:"a point pair is round" true (C.is_round pair);
  equal_scalar ~msg:"a degenerate pair vanishes" 0. (C.norm2 (C.point_pair a a))

let test_lines () =
  let a = p3 1. 2. 3.
  and b = p3 1. 2. 4. in
  let l = C.line a b in
  equal_bool ~msg:"a line is flat" true (C.is_flat l);
  equal_bool ~msg:"a line is not round" false (C.is_round l);
  let d = C.line_direction l |> Nx.to_array in
  let m = C.line_moment l |> Nx.to_array in
  equal
    ~msg:"the line direction is along z"
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
  on ~msg:"an endpoint is on the line" a l;
  on ~msg:"the other endpoint is on the line" b l;
  on ~msg:"the midpoint is on the line" (p3 1. 2. 3.5) l;
  off ~msg:"a point off the line" (p3 0. 2. 3.) l;
  let l2 =
    C.line_of_planes
      (C.plane ~normal:(v3 1. 0. 0.) ~offset:(sc 0.))
      (C.plane ~normal:(v3 0. 1. 0.) ~offset:(sc 0.))
  in
  on ~msg:"the intersection line contains a point" (p3 0. 0. 5.) l2;
  off ~msg:"a point off the intersection line" (p3 1. 0. 5.) l2

let test_circles () =
  let a = p3 1. 0. 0.
  and b = p3 0. 1. 0.
  and c = p3 (-1.) 0. 0. in
  let circle = C.circle a b c in
  equal_bool ~msg:"a circle is round" true (C.is_round circle);
  equal_bool ~msg:"a circle is not flat" false (C.is_flat circle);
  on ~msg:"the first point is on the circle" a circle;
  on ~msg:"the second point is on the circle" b circle;
  on ~msg:"the third point is on the circle" c circle;
  off ~msg:"a point off the circle" (p3 0. 0. 1.) circle;
  check_point ~msg:"the circle center" (C.circle_center circle) (0., 0., 0.);
  equal_scalar ~msg:"the circle radius" 1. (C.circle_radius2 circle);
  let n = C.circle_normal circle |> Nx.to_array in
  equal
    ~msg:"the carrier normal is along z"
    bool
    true
    (absf n.(0) < 1e-4 && absf n.(1) < 1e-4 && absf (absf n.(2) -. 1.) < 1e-4);
  let shifted = C.circle (p3 3. 2. 1.) (p3 1. 4. 1.) (p3 (-1.) 2. 1.) in
  check_point ~msg:"a shifted circle center" (C.circle_center shifted) (1., 2., 1.);
  equal_scalar ~msg:"a shifted circle radius" 4. (C.circle_radius2 shifted);
  let s1 = C.sphere ~center:(v3 0. 0. 0.) ~radius:(sc 1.)
  and s2 = C.sphere ~center:(v3 1. 0. 0.) ~radius:(sc 1.) in
  let meeting = C.circle_of_spheres s1 s2 in
  check_point
    ~msg:"two spheres meet in a circle at the midpoint"
    (C.circle_center meeting)
    (0.5, 0., 0.);
  equal_scalar ~msg:"the meeting circle radius" 0.75 (C.circle_radius2 meeting);
  on ~msg:"a point on the meeting circle" (p3 0.5 (sqrt 0.75) 0.) meeting;
  off ~msg:"a point off the meeting circle" (p3 0. 0. 0.) meeting;
  on ~msg:"the first sphere point is on the circle" (p3 0.5 (sqrt 0.75) 0.) meeting;
  on ~msg:"the second sphere point is on the circle" (p3 0.5 0. (sqrt 0.75)) meeting

let test_constructed_flats () =
  let a = p3 0. 0. 1.
  and b = p3 1. 0. 1.
  and c = p3 0. 1. 1. in
  let pi = C.plane_through a b c in
  on ~msg:"plane_through contains a" a pi;
  on ~msg:"plane_through contains b" b pi;
  on ~msg:"plane_through contains c" c pi;
  off ~msg:"a point off the constructed plane" (p3 0. 0. 0.) pi;
  let n = C.plane_normal (C.normalize_plane pi) |> Nx.to_array in
  equal
    ~msg:"the constructed plane normal"
    bool
    true
    (absf n.(0) < 1e-4 && absf n.(1) < 1e-4 && absf (absf n.(2) -. 1.) < 1e-4);
  let p1 = p3 1. 0. 0.
  and p2 = p3 (-1.) 0. 0.
  and p3_ = p3 0. 1. 0.
  and p4 = p3 0. 0. 1. in
  let s = C.sphere_through p1 p2 p3_ p4 in
  check_point ~msg:"sphere_through center" (C.sphere_center s) (0., 0., 0.);
  equal_scalar ~msg:"sphere_through radius" 1. (C.sphere_radius2 s);
  on ~msg:"sphere_through contains the first point" p1 s;
  on ~msg:"sphere_through contains the fourth point" p4 s

(* Meets *)

let test_meets () =
  let px = C.plane ~normal:(v3 1. 0. 0.) ~offset:(sc 0.)
  and py = C.plane ~normal:(v3 0. 1. 0.) ~offset:(sc 0.) in
  let line = C.meet px py in
  on ~msg:"the meet of two planes is a line" (p3 0. 0. 1.) line;
  off ~msg:"a point off the line" (p3 1. 0. 0.) line;
  let s = C.sphere ~center:(v3 0. 0. 0.) ~radius:(sc 1.) in
  let circle = C.meet (C.plane ~normal:(v3 0. 0. 1.) ~offset:(sc 0.)) s in
  on ~msg:"the plane cuts the sphere in a unit circle" (p3 1. 0. 0.) circle;
  off ~msg:"the sphere center is not on the circle" (p3 0. 0. 0.) circle;
  let s2 = C.sphere ~center:(v3 1. 0. 0.) ~radius:(sc 1.) in
  let circle2 = C.meet s s2 in
  on ~msg:"the spheres meet in a circle" (p3 0.5 (sqrt 0.75) 0.) circle2;
  off ~msg:"a point off the sphere meet" (p3 0. 0. 0.) circle2

(* Conformal generators *)

let test_conformal () =
  let d = C.dilator (sc (log 2.)) in
  check_point ~msg:"dilation scales a point" (C.sandwich d (p3 1. 0. 0.)) (2., 0., 0.);
  let s = C.sphere ~center:(v3 1. 0. 0.) ~radius:(sc 0.5) in
  let sd = C.sandwich d s in
  check_point ~msg:"dilation moves a sphere center" (C.sphere_center sd) (2., 0., 0.);
  equal_scalar ~msg:"dilation scales a sphere radius" 1. (C.sphere_radius2 sd);
  on ~msg:"dilation preserves incidence" (C.sandwich d (p3 1.5 0. 0.)) sd;
  let tv = C.transversion (v3 0.5 0. 0.) in
  check_point
    ~msg:"transversion maps a point"
    (C.sandwich tv (p3 1. 0. 0.))
    (1.5 /. 2.25, 0., 0.);
  check_point
    ~msg:"a zero transversion is the identity"
    (C.sandwich (C.transversion (v3 0. 0. 0.)) (p3 1. 2. 3.))
    (1., 2., 3.);
  equal_tensor ~msg:"a zero dilator is the identity" C.one (C.dilator (sc 0.));
  let batch_d = C.dilator (Nx.create f32 [| 2 |] [| 0.; log 2. |]) in
  equal ~msg:"a batch of dilators" (array int) [| 2; 32 |] (C.shape batch_d);
  let batch_tv =
    C.transversion (Nx.create f32 [| 2; 3 |] [| 0.; 0.; 0.; 0.5; 0.; 0. |])
  in
  equal ~msg:"a batch of transversions" (array int) [| 2; 32 |] (C.shape batch_tv)

(* Motors *)

let test_motors () =
  let p = p3 1. 2. 3. in
  let t = M.translate3 0. 0. 1. in
  equal
    ~msg:"of_motor embeds into the algebra"
    (array int)
    [| 32 |]
    (C.shape (C.of_motor t));
  check_point ~msg:"a translation moves a point" (C.apply_motor t p) (1., 2., 4.);
  equal_scalar ~msg:"the translated point is null" 0. (C.norm2 (C.apply_motor t p));
  let r = M.rotate ~axis:(v3 0. 0. 1.) ~angle:(sc (Float.pi /. 2.)) in
  check_point ~msg:"a rotation turns a point" (C.apply_motor r (p3 1. 0. 0.)) (0., 1., 0.);
  let m0 = M.translate3 1. (-2.) 0.5
  and m1 = M.rotate ~axis:(v3 1. 1. 0.) ~angle:(sc 0.8) in
  equal_tensor
    ~msg:"the action composes"
    (C.apply_motor (M.mul m0 m1) p)
    (C.apply_motor m0 (C.apply_motor m1 p));
  let s = C.sphere ~center:(v3 1. 0. 0.) ~radius:(sc 2.) in
  let moved = C.apply_motor m0 s in
  check_point ~msg:"a motor moves a sphere center" (C.sphere_center moved) (2., -2., 0.5);
  equal_scalar ~msg:"a motor preserves the sphere radius" 4. (C.sphere_radius2 moved);
  let batch =
    C.apply_motor
      (M.translate (Nx.create f32 [| 2; 3 |] [| 0.; 0.; 1.; 1.; 0.; 0. |]))
      (C.point (Nx.create f32 [| 2; 3 |] [| 0.; 0.; 0.; 1.; 0.; 0. |]))
  in
  equal
    ~msg:"a batch of points through a batch of motors"
    (array int)
    [| 2; 32 |]
    (C.shape batch);
  equal ~msg:"the batch coordinates" (array int) [| 2; 3 |] (Nx.shape (coords batch))

(* Inherited algebra *)

let test_inherited () =
  equal_tensor
    ~msg:"the Hodge dual is round-trip in Cl(4,1)"
    C.e1
    (C.undual (C.dual C.e1));
  raises_match
    ~msg:"a bad blade index is rejected"
    (Exn.invalid_arg ~substring:"out of range")
    (fun () -> ignore (C.basis (B.of_int 32)))

(* Properties *)

let vec_gen =
  Gen.map
    (fun v -> Nx.create f32 [| 3 |] v)
    Gen.(array ~size:(constant 3) (float_range (-2.) 2.))

let point_gen = Gen.map C.point vec_gen

let axis_gen =
  Gen.map
    (fun (x, y) -> Nx.create f32 [| 3 |] [| x; y; 1. |])
    Gen.(pair (float_range (-1.) 1.) (float_range (-1.) 1.))

let sphere_gen =
  Gen.map
    (fun (c, r) ->
       C.sphere ~center:(Nx.create f32 [| 3 |] c) ~radius:(sc (Float.abs r +. 0.1)))
    Gen.(pair (array ~size:(constant 3) (float_range (-2.) 2.)) (float_range (-2.) 2.))

let motor_gen =
  Gen.map
    (fun (t, axis, a) ->
       M.mul (M.translate (Nx.create f32 [| 3 |] t)) (M.rotate ~axis ~angle:(sc a)))
    Gen.(
      triple
        (array ~size:(constant 3) (float_range (-1.) 1.))
        axis_gen
        (float_range (-2.5) 2.5))

let test_properties =
  group
    "properties"
    [ prop "coords inverts point" vec_gen (fun v ->
        equal_floats (Nx.to_array v) (coords (C.point v)))
    ; prop "a point is null" vec_gen (fun v ->
        equal_scalar ~eps:1e-3 0. (C.norm2 (C.point v)))
    ; prop "a sphere contains its surface points" sphere_gen (fun s ->
        let center = C.coords (C.sphere_center s) in
        let r = sqrt (Nx.item [] (C.sphere_radius2 s)) in
        let p = C.point (Nx.add center (v3 r 0. 0.)) in
        on p s)
    ; prop
        "distance is symmetric"
        Gen.(pair point_gen point_gen)
        (fun (a, b) ->
           equal (float 1e-4) (Nx.item [] (C.distance a b)) (Nx.item [] (C.distance b a)))
    ; prop
        "a motor preserves incidence"
        Gen.(pair motor_gen (pair point_gen sphere_gen))
        (fun (m, (p, s)) ->
           equal
             (array bool)
             (Nx.to_array (C.incident p s))
             (Nx.to_array (C.incident (C.apply_motor m p) (C.apply_motor m s))))
    ; prop
        "the motor action is a homomorphism"
        Gen.(pair motor_gen (pair motor_gen point_gen))
        (fun (m0, (m1, p)) ->
           equal_tensor
             (C.apply_motor (M.mul m0 m1) p)
             (C.apply_motor m0 (C.apply_motor m1 p)))
    ]

let () =
  exit
    (run
       "geom cga"
       [ group "basis" [ test "the null basis" test_null_basis ]
       ; group "points" [ test "coordinates, nullness and distance" test_points ]
       ; group
           "round objects"
           [ test "spheres" test_spheres
           ; test "point pairs" test_point_pairs
           ; xfail
               ~reason:
                 "Cga.incident uses X dot O; a point pair is a direct blade, so X dot P \
                  is nonzero at its endpoints"
               (test "incident on a point pair" (fun () ->
                  on (p3 1. 0. 0.) (C.point_pair (p3 1. 0. 0.) (p3 0. 1. 0.))))
           ; test "circles" test_circles
           ]
       ; group
           "flat objects"
           [ test "planes" test_planes
           ; test "lines" test_lines
           ; test "constructed flats" test_constructed_flats
           ]
       ; group "incidence" [ test "meets" test_meets ]
       ; group "conformal" [ test "dilations and transversions" test_conformal ]
       ; group "motors" [ test "the packed motor bridge" test_motors ]
       ; group "inherited" [ test "the generic algebra in Cl(4,1)" test_inherited ]
       ; test_properties
       ])
