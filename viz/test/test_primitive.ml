(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Geom_viz.Primitive: the neutral primitives, the PGA adapters and
   rigid transforms. *)

open Windtrap
open Geom
module P = Geom_viz.Primitive

let vec3 x y z = Nx.create Nx.float32 [| 3 |] [| x; y; z |]
let scalar x = Nx.scalar Nx.float32 x

module M = Geom.Motor

let p3 = P.p3
let equal_bool ?msg expected actual = equal ?msg bool expected actual

let equal_float ?(eps = 5e-3) ?msg expected actual =
  equal ?msg (float eps) expected actual

let equal_point ?(eps = 5e-3) ?msg (x, y, z) (p : P.p3) =
  equal_float ~eps ~msg:"x" x p.x;
  equal_float ~eps ~msg:"y" y p.y;
  equal_float ~eps ~msg:"z" z p.z

let point_coords p = Nx.to_array (Pga.point_coords p)
let point_on_line p l = Nx.item [] (Pga.point_on_line p l)

let plane_normal_and_offset = function
  | P.Plane pl -> pl.P.normal, pl.P.offset
  | _ -> failwith "expected a plane"

let segment_endpoints = function
  | P.Segment (a, b) -> a, b
  | _ -> failwith "expected a segment"

let test_vectors () =
  equal_point ~msg:"normalizing zero is zero" (0., 0., 0.) (P.normalize (p3 0. 0. 0.));
  equal_float
    ~msg:"a normalized vector has unit length"
    1.
    (P.norm (P.normalize (p3 3. 4. 0.)));
  equal_point
    ~msg:"normalize keeps the direction"
    (0.6, 0.8, 0.)
    (P.normalize (p3 3. 4. 0.));
  equal_float ~msg:"the dot product" 11. (P.dot (p3 1. 2. 3.) (p3 4. (-1.) 3.));
  equal_point ~msg:"the cross product" (0., 0., 1.) (P.cross (p3 1. 0. 0.) (p3 0. 1. 0.))

let test_plane_patch () =
  let check ~normal ~offset =
    let patch = P.plane_patch ~normal ~offset in
    equal ~msg:"four patch corners" int 4 (List.length patch);
    List.iter
      (fun (q : P.p3) ->
         equal_float
           ~msg:"the corner lies on the plane"
           offset
           (P.dot (P.normalize normal) q))
      patch;
    let rec sides = function
      | [] | [ _ ] -> []
      | a :: (b :: _ as rest) -> P.norm (P.sub b a) :: sides rest
    in
    List.iter
      (fun side -> equal_float ~msg:"the patch is a square of side four" 4. side)
      (sides patch)
  in
  (* The two orientation branches of the in-plane basis. *)
  check ~normal:(p3 0. 0. 1.) ~offset:1.;
  check ~normal:(p3 1. 0. 0.) ~offset:(-2.)

let test_cga_adapters () =
  (match P.of_cga_point (Cga.point3 1. 2. 3.) with
   | P.Point q -> equal_point ~msg:"a CGA point" (1., 2., 3.) q
   | _ -> failwith "expected a point");
  let sphere = Cga.sphere ~center:(vec3 1. 2. 3.) ~radius:(scalar 0.5) in
  (match P.of_cga_sphere sphere with
   | P.Sphere s ->
     equal_point ~msg:"a CGA sphere centre" (1., 2., 3.) s.P.center;
     equal_float ~msg:"a CGA sphere radius" 0.5 s.P.radius
   | _ -> failwith "expected a sphere");
  let plane = Cga.plane ~normal:(vec3 0. 1. 0.) ~offset:(scalar 2.) in
  let normal, offset = plane_normal_and_offset (P.of_cga_plane plane) in
  equal_point ~msg:"a CGA plane normal" (0., 1., 0.) normal;
  equal_float ~msg:"a CGA plane offset" 2. offset;
  let line = Cga.line (Cga.point3 0. 0. 0.) (Cga.point3 1. 0. 0.) in
  let a, b = segment_endpoints (P.of_cga_line line) in
  equal_float ~msg:"a CGA line segment half-length" 5. (P.norm (P.sub b a));
  let mid = P.scale (P.add a b) 0.5 in
  equal_point ~msg:"the CGA line segment is centred on the origin" (0., 0., 0.) mid;
  let circle =
    Cga.circle (Cga.point3 1. 0. 0.) (Cga.point3 0. 1. 0.) (Cga.point3 (-1.) 0. 0.)
  in
  match P.of_cga_circle circle with
  | P.Circle c ->
    equal_point ~msg:"a CGA circle centre" (0., 0., 0.) c.P.center;
    equal_float ~msg:"a CGA circle radius" 1. c.P.radius;
    equal_float ~msg:"a CGA circle normal is unit" 1. (P.norm c.P.normal)
  | _ -> failwith "expected a circle"

let test_pga_point () =
  let p = P.of_pga_point (Pga.point3 1. (-2.) 3.) in
  (match p with
   | P.Point q -> equal_point ~msg:"the point coordinates" (1., -2., 3.) q
   | _ -> failwith "expected a point");
  (* A rescaled representative is the same projective point. *)
  let scaled = Nx.mul_s (Pga.point3 1. (-2.) 3.) 4. in
  match P.of_pga_point scaled with
  | P.Point q -> equal_point ~msg:"a rescaled point" (1., -2., 3.) q
  | _ -> failwith "expected a point"

let test_cga_plane_normalization () =
  (* 2 z = 4 is the plane z = 2. The adapter must divide out the normal. *)
  let pi =
    Cga.plane
      ~normal:(Nx.create Nx.float32 [| 3 |] [| 0.; 0.; 2. |])
      ~offset:(Nx.scalar Nx.float32 4.)
  in
  let normal, offset = plane_normal_and_offset (P.of_cga_plane pi) in
  equal_point ~msg:"the normalized CGA plane normal" (0., 0., 1.) normal;
  equal_float ~msg:"the normalized CGA plane offset" 2. offset

let test_pga_plane () =
  (* A non-unit normal: Pga.normalize_plane must remove the scale. *)
  let pi = Pga.plane3 0. 0. 2. 4. in
  let primitive = P.of_pga_plane pi in
  let normal, offset = plane_normal_and_offset primitive in
  equal_point ~msg:"the normalized plane normal" (0., 0., 1.) normal;
  equal_float ~msg:"the normalized plane offset" 2. offset;
  match primitive with
  | P.Plane pl ->
    List.iter
      (fun q -> equal_float ~msg:"the patch lies on the plane" 2. (P.dot normal q))
      pl.P.patch
  | _ -> assert false

let test_pga_line () =
  let p = Pga.point3 1. 2. 3.
  and q = Pga.point3 1. 2. 4. in
  let l = Pga.line_through p q in
  let a, b = segment_endpoints (P.of_pga_line ~extent:1.5 l) in
  equal_float ~msg:"the segment half-length" 3. (P.norm (P.sub b a));
  equal_bool
    ~msg:"the first endpoint is on the line"
    true
    (point_on_line (Pga.point3 a.x a.y a.z) l);
  equal_bool
    ~msg:"the second endpoint is on the line"
    true
    (point_on_line (Pga.point3 b.x b.y b.z) l);
  (* The closest point to the origin of this line is (1, 2, 0). *)
  let mid = P.scale (P.add a b) 0.5 in
  equal_point ~msg:"the segment is centred on the closest point" (1., 2., 0.) mid;
  let a, b = segment_endpoints (P.of_pga_line l) in
  equal_float ~msg:"the default extent" 5. (P.norm (P.sub b a))

let test_transform_point_and_plane () =
  let m = M.translate3 0. 0. 2. in
  (match P.transform m (P.Point (p3 1. 2. 3.)) with
   | P.Point q -> equal_point ~msg:"a translated point" (1., 2., 5.) q
   | _ -> failwith "expected a point");
  (* A plane through the origin moves its offset, not only its patch. This is
     the regression that a normal-only transform gets wrong. *)
  let patch = [ p3 (-1.) (-1.) 0.; p3 1. (-1.) 0.; p3 1. 1. 0.; p3 (-1.) 1. 0. ] in
  let plane = P.Plane { normal = p3 0. 0. 1.; offset = 0.; patch } in
  let normal, offset = plane_normal_and_offset (P.transform m plane) in
  equal_point ~msg:"a translated plane keeps its normal" (0., 0., 1.) normal;
  equal_float ~msg:"a translated plane moves its offset" 2. offset;
  (match P.transform m plane with
   | P.Plane pl ->
     List.iter
       (fun (q : P.p3) ->
          equal_float ~msg:"the transformed patch lies on the plane" 2. q.z)
       pl.P.patch
   | _ -> assert false);
  (* A rotation changes the normal; the offset is measured against the new
     normal and the translation. *)
  let m =
    M.mul
      (M.translate3 0. 0. 2.)
      (M.rotate
         ~axis:(Nx.create Nx.float32 [| 3 |] [| 1.; 0.; 0. |])
         ~angle:(Nx.scalar Nx.float32 (Float.pi /. 2.)))
  in
  let normal, offset =
    plane_normal_and_offset
      (P.transform m (P.Plane { normal = p3 0. 0. 1.; offset = 0.5; patch = [] }))
  in
  equal_point ~msg:"a rotated plane carries its normal" (0., -1., 0.) normal;
  equal_float ~msg:"a rotated plane keeps its distance" 0.5 offset

let test_transform_shapes () =
  let m =
    M.mul
      (M.translate3 1. 2. 3.)
      (M.rotate
         ~axis:(Nx.create Nx.float32 [| 3 |] [| 0.; 0.; 1. |])
         ~angle:(Nx.scalar Nx.float32 (Float.pi /. 2.)))
  in
  let expected0 = 1., 2., 3.
  and expected1 = 1., 3., 3. in
  (match P.transform m (P.Segment (p3 0. 0. 0., p3 1. 0. 0.)) with
   | P.Segment (a, b) ->
     equal_point ~msg:"a transformed segment: first endpoint" expected0 a;
     equal_point ~msg:"a transformed segment: second endpoint" expected1 b
   | _ -> failwith "expected a segment");
  (match P.transform m (P.Polyline [ p3 0. 0. 0.; p3 1. 0. 0. ]) with
   | P.Polyline [ a; b ] ->
     equal_point ~msg:"a transformed polyline: first point" expected0 a;
     equal_point ~msg:"a transformed polyline: second point" expected1 b
   | _ -> failwith "expected a two-point polyline");
  (match P.transform m (P.Polygon [ p3 0. 0. 0.; p3 1. 0. 0. ]) with
   | P.Polygon [ a; b ] ->
     equal_point ~msg:"a transformed polygon: first point" expected0 a;
     equal_point ~msg:"a transformed polygon: second point" expected1 b
   | _ -> failwith "expected a two-point polygon");
  match
    P.transform
      m
      (P.Frame { origin = p3 0. 0. 0.; x = p3 1. 0. 0.; y = p3 0. 1. 0.; z = p3 0. 0. 1. })
  with
  | P.Frame f ->
    equal_point ~msg:"a transformed frame origin" expected0 f.P.origin;
    equal_point ~msg:"a transformed frame x" expected1 f.P.x;
    equal_point ~msg:"a transformed frame y" (0., 2., 3.) f.P.y;
    equal_point ~msg:"a transformed frame z" (1., 2., 4.) f.P.z
  | _ -> failwith "expected a frame"

let test_transform_matches_pga () =
  let m =
    M.mul
      (M.translate3 0.4 (-1.2) 0.7)
      (M.rotate
         ~axis:(Nx.create Nx.float32 [| 3 |] [| 0.; 0.; 1. |])
         ~angle:(Nx.scalar Nx.float32 0.9))
  in
  let p = p3 0.3 (-0.6) 1.1 in
  let expected =
    Pga.apply_motor m (Pga.point3 p.x p.y p.z) |> Pga.point_coords |> Nx.to_array
  in
  match P.transform m (P.Point p) with
  | P.Point q ->
    equal_point
      ~msg:"the primitive transform agrees with Pga.apply_motor"
      (expected.(0), expected.(1), expected.(2))
      q
  | _ -> failwith "expected a point"

let test_transform_circle_and_sphere () =
  let m = M.translate3 1. 0. 0. in
  (match
     P.transform m (P.Circle { center = p3 0. 0. 0.; normal = p3 0. 0. 1.; radius = 2. })
   with
   | P.Circle c ->
     equal_point ~msg:"the circle centre" (1., 0., 0.) c.P.center;
     equal_float ~msg:"the circle keeps its radius" 2. c.P.radius;
     equal_point ~msg:"the circle keeps its normal" (0., 0., 1.) c.P.normal
   | _ -> failwith "expected a circle");
  match P.transform m (P.Sphere { center = p3 0. 0. 0.; radius = 0.5 }) with
  | P.Sphere s ->
    equal_point ~msg:"the sphere centre" (1., 0., 0.) s.P.center;
    equal_float ~msg:"the sphere keeps its radius" 0.5 s.P.radius
  | _ -> failwith "expected a sphere"

let () =
  exit
    (run
       "geom-viz primitive"
       [ group
           "vectors and patches"
           [ test "vector arithmetic" test_vectors
           ; test "plane patches" test_plane_patch
           ]
       ; group
           "adapters"
           [ test "PGA points" test_pga_point
           ; test "PGA planes" test_pga_plane
           ; test "PGA lines" test_pga_line
           ; test "CGA objects" test_cga_adapters
           ; test "CGA plane normalization" test_cga_plane_normalization
           ]
       ; group
           "rigid transforms"
           [ test "points and planes" test_transform_point_and_plane
           ; test "segments, polylines, polygons and frames" test_transform_shapes
           ; test "agreement with the PGA motor action" test_transform_matches_pga
           ; test "circles and spheres" test_transform_circle_and_sphere
           ]
       ])
