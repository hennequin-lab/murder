(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Geom_viz.Camera: the PGA image plane, view rays and the
   agreement between the meet projection and the closed-form pinhole path. *)

open Windtrap
open Geom
module C = Geom_viz.Camera
module P = Geom_viz.Primitive
module M = Geom.Motor

let p3 = P.p3
let width = 800.
let height = 600.

let camera ?(ortho = false) ?ortho_scale () =
  C.create
    ?ortho_scale
    ~ortho
    ~eye:(p3 4. (-4.) 3.)
    ~target:(p3 0. 0. 0.5)
    ~up:(p3 0. 0. 1.)
    ~fov_y:(Float.pi /. 4.)
    ~width
    ~height
    ()

let equal_bool ?msg expected actual = equal ?msg bool expected actual

let equal_float ?(eps = 5e-3) ?msg expected actual =
  equal ?msg (float eps) expected actual

let equal_floats ?(eps = 5e-3) ?msg expected actual =
  equal ?msg (array (float eps)) expected actual

let equal_triple ?(eps = 5e-3) ?msg (x, y, d) actual =
  match actual with
  | None ->
    failwith
      (match msg with
       | Some m -> m ^ ": got None"
       | None -> "got None")
  | Some (ax, ay, ad) ->
    equal_float ~eps ~msg:"x" x ax;
    equal_float ~eps ~msg:"y" y ay;
    equal_float ~eps ~msg:"depth" d ad

let coords p = Nx.to_array (Pga.point_coords p)
let pga (p : P.p3) = Pga.point3 p.x p.y p.z

let world_of_camera c (q : P.p3) =
  Pga.apply_motor (M.inverse (C.view c)) (Pga.point3 q.x q.y q.z)

(* [Pga.point_on_line] and [Pga.point_on_plane] use an absolute tolerance on
   the incidence blade. The image plane is [focal] units away, where the
   coefficients are large, so these tests measure the geometric residuals
   directly instead. *)

let cross3 a b =
  ( (a.(1) *. b.(2)) -. (a.(2) *. b.(1))
  , (a.(2) *. b.(0)) -. (a.(0) *. b.(2))
  , (a.(0) *. b.(1)) -. (a.(1) *. b.(0)) )

let line_distance (p : P.p3) l =
  let d = Nx.to_array (Pga.line_direction l) in
  let m = Nx.to_array (Pga.line_moment l) in
  let c0, c1, c2 = cross3 [| p.x; p.y; p.z |] d in
  sqrt (((c0 -. m.(0)) ** 2.) +. ((c1 -. m.(1)) ** 2.) +. ((c2 -. m.(2)) ** 2.))

let plane_residual (p : P.p3) pi =
  let pi = Pga.normalize_plane pi in
  let n = Nx.to_array (Pga.plane_normal pi) in
  let d = Nx.item [] (Pga.plane_offset pi) in
  (n.(0) *. p.x) +. (n.(1) *. p.y) +. (n.(2) *. p.z) -. d

let on_line ?(eps = 1e-3) ?msg p l = equal_float ~eps ?msg 0. (line_distance p l)
let on_plane ?(eps = 1e-3) ?msg p pi = equal_float ~eps ?msg 0. (plane_residual p pi)

(* Constructors reject degenerate cameras. *)

let test_validation () =
  let eye = p3 4. (-4.) 3. in
  let target = p3 0. 0. 0.5 in
  let build ?(up = p3 0. 0. 1.) ?(fov_y = Float.pi /. 4.) ?(near = 0.1) ?(far = 1000.) () =
    C.create ~eye ~target ~up ~fov_y ~width ~height ~near ~far ()
  in
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"fov_y" e)
    (fun () -> build ~fov_y:0. ());
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"fov_y" e)
    (fun () -> build ~fov_y:Float.pi ());
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"parallel" e)
    (fun () -> build ~up:(P.normalize (P.sub target eye)) ());
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"coincide" e)
    (fun () ->
       C.create
         ~eye
         ~target:eye
         ~up:(p3 0. 0. 1.)
         ~fov_y:(Float.pi /. 4.)
         ~width
         ~height
         ());
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"near" e)
    (fun () -> build ~near:10. ~far:5. ());
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"width and height" e)
    (fun () ->
       C.create
         ~eye
         ~target
         ~up:(p3 0. 0. 1.)
         ~fov_y:(Float.pi /. 4.)
         ~width:0.
         ~height
         ());
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"ortho_scale" e)
    (fun () ->
       C.create
         ~ortho:true
         ~ortho_scale:0.
         ~eye
         ~target
         ~up:(p3 0. 0. 1.)
         ~fov_y:(Float.pi /. 4.)
         ~width
         ~height
         ());
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"width and height" e)
    (fun () ->
       C.create
         ~eye
         ~target
         ~up:(p3 0. 0. 1.)
         ~fov_y:(Float.pi /. 4.)
         ~width:Float.nan
         ~height
         ());
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"near" e)
    (fun () ->
       C.create
         ~eye
         ~target
         ~up:(p3 0. 0. 1.)
         ~fov_y:(Float.pi /. 4.)
         ~width
         ~height
         ~near:Float.nan
         ());
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"near" e)
    (fun () -> build ~near:0. ());
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"near" e)
    (fun () -> build ~far:Float.infinity ());
  raises_match
    (fun e -> Exn.invalid_arg ~substring:"width and height" e)
    (fun () ->
       C.create
         ~eye
         ~target
         ~up:(p3 0. 0. 1.)
         ~fov_y:(Float.pi /. 4.)
         ~width
         ~height:Float.infinity
         ())

let test_accessors () =
  let c = camera () in
  equal_float ~msg:"width" width (C.width c);
  equal_float ~msg:"height" height (C.height c);
  equal_float ~msg:"near" 0.1 (C.near c);
  equal_float ~msg:"far" 1000. (C.far c);
  equal_float ~msg:"the focal length" (height /. 2. /. tan (Float.pi /. 8.)) (C.focal c);
  equal_bool ~msg:"perspective by default" false (C.is_orthographic c);
  equal_float ~msg:"the default orthographic scale" (height /. 4.) (C.ortho_scale c);
  let r = C.right c
  and u = C.up c
  and f = C.forward c in
  equal_float ~msg:"the right axis is unit" 1. (P.norm r);
  equal_float ~msg:"the up axis is unit" 1. (P.norm u);
  equal_float ~msg:"the forward axis is unit" 1. (P.norm f);
  equal_float ~msg:"right and up are orthogonal" 0. (P.dot r u);
  equal_float ~msg:"right and forward are orthogonal" 0. (P.dot r f);
  equal_float ~msg:"up and forward are orthogonal" 0. (P.dot u f);
  let expected_forward = P.normalize (P.sub (p3 0. 0. 0.5) (p3 4. (-4.) 3.)) in
  equal_float ~msg:"forward points at the target" expected_forward.x f.x;
  equal_float ~msg:"forward points at the target" expected_forward.y f.y;
  equal_float ~msg:"forward points at the target" expected_forward.z f.z;
  equal_floats ~msg:"the eye point" [| 4.; -4.; 3. |] (coords (C.eye_point c));
  let ortho = camera ~ortho:true ~ortho_scale:250. () in
  equal_bool ~msg:"orthographic" true (C.is_orthographic ortho);
  equal_float ~msg:"an explicit orthographic scale" 250. (C.ortho_scale ortho)

(* The camera is a PGA construction: the view motor, the eye point and the
   image plane must agree with the direct look-at quantities. *)

let test_view_motor () =
  let c = camera () in
  let eye_cam = coords (Pga.apply_motor (C.view c) (pga (C.eye c))) in
  equal_float ~msg:"the eye maps to the origin" 0. eye_cam.(0);
  equal_float ~msg:"the eye maps to the origin" 0. eye_cam.(1);
  equal_float ~msg:"the eye maps to the origin" 0. eye_cam.(2);
  let target = p3 0. 0. 0.5 in
  let target_cam = coords (Pga.apply_motor (C.view c) (pga target)) in
  let distance = P.norm (P.sub target (C.eye c)) in
  equal_float ~msg:"the target lies on the optical axis" 0. target_cam.(0);
  equal_float ~msg:"the target lies on the optical axis" 0. target_cam.(1);
  equal_float ~msg:"the target is in front" (-.distance) target_cam.(2);
  (* The camera axes are the rows of the rotation part of the view motor. *)
  let origin = Nx.to_array (Pga.point_coords (world_of_camera c (p3 0. 0. 0.))) in
  equal_float ~msg:"the inverse view maps the origin to the eye" (C.eye c).x origin.(0);
  equal_float ~msg:"the inverse view maps the origin to the eye" (C.eye c).y origin.(1);
  equal_float ~msg:"the inverse view maps the origin to the eye" (C.eye c).z origin.(2)

let test_image_plane () =
  let c = camera () in
  let pi = C.image_plane c in
  let focal = C.focal c in
  let back = P.scale (C.forward c) (-1.) in
  let expected_offset = P.dot back (C.eye c) -. focal in
  let normal = Nx.to_array (Pga.plane_normal pi) in
  equal_float ~msg:"the image plane normal is the camera back axis" back.x normal.(0);
  equal_float ~msg:"the image plane normal is the camera back axis" back.y normal.(1);
  equal_float ~msg:"the image plane normal is the camera back axis" back.z normal.(2);
  equal_float
    ~msg:"the image plane offset"
    expected_offset
    (Nx.item [] (Pga.plane_offset pi));
  (* A point of the image plane, carried to world space, is on the plane. *)
  let q = p3 30. (-20.) (-.focal) in
  let w = world_of_camera c q in
  let w_p3 =
    let q = Nx.to_array (Pga.point_coords w) in
    p3 q.(0) q.(1) q.(2)
  in
  on_plane ~msg:"a camera image point lies on the image plane" w_p3 pi;
  equal_bool
    ~msg:"the eye is not on the image plane"
    true
    (Float.abs (plane_residual (C.eye c) pi) > 1.);
  (* The same plane built from its normal and offset is the same object. *)
  let direct =
    Pga.normalize_plane
      (Pga.plane
         ~normal:(Nx.create Nx.float32 [| 3 |] [| back.x; back.y; back.z |])
         ~offset:(Nx.scalar Nx.float32 expected_offset))
  in
  let a = Nx.to_array (Pga.normalize_plane pi) in
  let b = Nx.to_array direct in
  Array.iter2 (fun x y -> equal_float ~msg:"the plane coefficients" ~eps:1e-4 x y) a b

(* The meet projection and the pinhole evaluation are the same function. *)

let test_projection_agreement () =
  let c = camera () in
  let points =
    [ p3 0. 0. 0.5
    ; p3 1. 0. 1.
    ; p3 (-1.2) 0.7 0.3
    ; p3 0.4 (-0.8) 2.
    ; p3 2. 2. 0.
    ; p3 (-2.) (-2.) 1.5
    ; P.add (C.eye c) (P.scale (C.forward c) (C.focal c))
    ]
  in
  List.iter
    (fun p ->
       let meet = C.project c p in
       let fast = C.project_pinhole c p in
       match meet, fast with
       | None, None -> ()
       | Some m, Some f -> equal_triple m (Some f)
       | Some _, None -> failwith "the meet projected a point the pinhole path rejected"
       | None, Some _ -> failwith "the pinhole path projected a point the meet rejected")
    points

let test_projection_image () =
  let c = camera () in
  let p = p3 0.7 (-0.4) 1.3 in
  match C.project_image c p, C.project c p with
  | Some (xi, yi, d), Some (x, y, d') ->
    equal_float ~msg:"depth is shared" d d';
    equal_float ~msg:"the page x mapping" ((width /. 2.) +. xi) x;
    equal_float ~msg:"the page y mapping" ((height /. 2.) -. yi) y
  | _ -> failwith "a point in front of the camera must project"

let test_culling () =
  let c = camera () in
  let eye = C.eye c in
  let behind = P.add eye (P.scale (C.forward c) (-1.)) in
  let too_close = P.add eye (P.scale (C.forward c) 0.01) in
  equal_bool
    ~msg:"a point behind the eye is culled"
    true
    (Option.is_none (C.project c behind));
  equal_bool
    ~msg:"a point inside the near plane is culled"
    true
    (Option.is_none (C.project c too_close));
  equal_bool
    ~msg:"the pinhole path culls behind the eye"
    true
    (Option.is_none (C.project_pinhole c behind));
  equal_bool
    ~msg:"the image path culls behind the eye"
    true
    (Option.is_none (C.project_image c behind));
  equal_bool
    ~msg:"the pinhole path culls at the near plane"
    true
    (Option.is_none (C.project_pinhole c too_close));
  let near_camera =
    C.create
      ~near:0.1
      ~far:5.
      ~eye:(p3 4. (-4.) 3.)
      ~target:(p3 0. 0. 0.5)
      ~up:(p3 0. 0. 1.)
      ~fov_y:(Float.pi /. 4.)
      ~width
      ~height
      ()
  in
  let far_point = P.add (C.eye c) (P.scale (C.forward c) 20.) in
  equal_bool
    ~msg:"a point beyond the far plane is culled"
    true
    (Option.is_none (C.project near_camera far_point))

(* The ray through a page point is the line used by the meet, and it passes
   through the eye. *)

let test_rays () =
  let c = camera () in
  let focal = C.focal c in
  let x = 610.
  and y = 180. in
  let r = C.ray c ~x ~y in
  on_line ~msg:"the eye is on the ray" (C.eye c) r;
  let xi = x -. (width /. 2.) in
  let yi = (height /. 2.) -. y in
  let image_world = world_of_camera c (p3 xi yi (-.focal)) in
  let image_world_p3 =
    let q = Nx.to_array (Pga.point_coords image_world) in
    p3 q.(0) q.(1) q.(2)
  in
  on_line ~msg:"the image point is on the ray" image_world_p3 r;
  let image_p3 =
    let q = Nx.to_array (Pga.point_coords image_world) in
    p3 q.(0) q.(1) q.(2)
  in
  match C.project c image_p3 with
  | None -> failwith "the image point must project"
  | Some (px, py, _) ->
    equal_float ~msg:"the ray returns to its page x" x px;
    equal_float ~msg:"the ray returns to its page y" y py;
    (* The ray at the principal point is the optical axis. *)
    let center = C.ray c ~x:(width /. 2.) ~y:(height /. 2.) in
    let target = p3 0. 0. 0.5 in
    on_line ~msg:"the optical axis goes through the target" target center;
    let dir = Nx.to_array (Pga.line_direction center) in
    equal_float ~msg:"the ray points forward" (C.forward c).x dir.(0);
    equal_float ~msg:"the ray points forward" (C.forward c).y dir.(1);
    equal_float ~msg:"the ray points forward" (C.forward c).z dir.(2)

let test_orthographic () =
  let c = camera ~ortho:true () in
  let p = p3 1. (-0.5) 0.2 in
  let xc, yc, _zc =
    let q = P.sub p (C.eye c) in
    P.dot (C.right c) q, P.dot (C.up c) q, P.dot (C.forward c) q
  in
  let s = C.ortho_scale c in
  (match C.project_image c p with
   | None -> failwith "an orthographic point in front must project"
   | Some (xi, yi, _) ->
     equal_float ~msg:"the orthographic x scale" (s *. xc) xi;
     equal_float ~msg:"the orthographic y scale" (s *. yc) yi);
  (match C.project c p, C.project_pinhole c p with
   | Some a, Some b -> equal_triple a (Some b)
   | _ -> failwith "both orthographic paths must project");
  (* The ray is parallel to the view axis. *)
  let r = C.ray c ~x:300. ~y:200. in
  let dir = Nx.to_array (Pga.line_direction r) in
  equal_float
    ~msg:"an orthographic ray is parallel to the view axis"
    (C.forward c).x
    dir.(0);
  equal_float
    ~msg:"an orthographic ray is parallel to the view axis"
    (C.forward c).y
    dir.(1);
  equal_float
    ~msg:"an orthographic ray is parallel to the view axis"
    (C.forward c).z
    dir.(2);
  let xc = (300. -. (width /. 2.)) /. s in
  let yc = ((height /. 2.) -. 200.) /. s in
  let image_world = world_of_camera c (p3 xc yc (-1.)) in
  let image_world_p3 =
    let q = Nx.to_array (Pga.point_coords image_world) in
    p3 q.(0) q.(1) q.(2)
  in
  on_line ~msg:"the orthographic ray goes through the pixel" image_world_p3 r

let () =
  exit
    (run
       "geom-viz camera"
       [ group
           "construction"
           [ test "validation" test_validation
           ; test "parameters and axes" test_accessors
           ; test "the view motor" test_view_motor
           ; test "the image plane" test_image_plane
           ]
       ; group
           "projection"
           [ test "the meet agrees with the pinhole path" test_projection_agreement
           ; test "image and page coordinates" test_projection_image
           ; test "near and far culling" test_culling
           ]
       ; group "rays" [ test "view rays" test_rays ]
       ; group "orthographic" [ test "scale and rays" test_orthographic ]
       ])
