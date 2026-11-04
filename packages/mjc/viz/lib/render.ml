(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
module P = Geom_viz.Primitive
module D = Geom_viz.Draw

type camera =
  { eye : Scene.p3 option
  ; target : Scene.p3 option
  ; up : Scene.p3
  ; fov_y : float
  ; width : float
  ; height : float
  }

let default_camera =
  { eye = None
  ; target = None
  ; up = { Scene.x = 0.; y = 0.; z = 1. }
  ; fov_y = Float.pi /. 4.
  ; width = 960.
  ; height = 540.
  }

let with_size ~width ~height c () = { c with width; height }

(* {1 Colours} *)

let colour (c : Scene.colour) = Hugin_gg.Color.v ~alpha:c.a c.r c.g c.b
let fill c = { D.default with fill = Some (colour c); stroke = None; width = 0.6 }
let edge c = { D.default with fill = None; stroke = Some (colour c); width = 1.2 }

(* {1 Decomposition} *)

let point (p : Scene.p3) = P.p3 p.x p.y p.z

let add (a : Scene.p3) (b : Scene.p3) =
  { Scene.x = a.x +. b.x; y = a.y +. b.y; z = a.z +. b.z }

let smul (a : Scene.p3) s = { Scene.x = a.x *. s; y = a.Scene.y *. s; z = a.Scene.z *. s }
let dot (a : Scene.p3) (b : Scene.p3) = (a.x *. b.x) +. (a.y *. b.y) +. (a.z *. b.z)

let sub (a : Scene.p3) (b : Scene.p3) =
  { Scene.x = a.x -. b.x; y = a.y -. b.y; z = a.z -. b.z }

let box_edges (centre : Scene.p3) (ux, uy, uz) (hx, hy, hz) =
  (* The eight corners, then the twelve edges between them. *)
  let corner sx sy sz =
    add centre (add (add (smul ux (sx *. hx)) (smul uy (sy *. hy))) (smul uz (sz *. hz)))
  in
  let c sx sy sz =
    corner (if sx then 1. else -1.) (if sy then 1. else -1.) (if sz then 1. else -1.)
  in
  let v = Array.init 8 ~f:(fun i -> c (i land 1 <> 0) (i land 2 <> 0) (i land 4 <> 0)) in
  let edge = [ 0, 1; 0, 2; 0, 4; 1, 3; 1, 5; 2, 3; 2, 6; 3, 7; 4, 5; 4, 6; 5, 7; 6, 7 ] in
  List.map edge ~f:(fun (i, j) -> P.Segment (point v.(i), point v.(j)))

let circle ~(centre : Scene.p3) ~(normal : Scene.p3) ~radius =
  P.Circle { center = point centre; normal = point normal; radius }

(* A plane is infinite and a page is not: draw the part of it the camera is
   looking at, as a square of the scene's own size centred under the target.
   Without the clip the patch's far corners end up behind the eye and the
   polygon projects to nonsense. *)
let shapes
      ?(plane_extent = 5.)
      ?(plane_centre = { Scene.x = 0.; y = 0.; z = 0. })
      (g : Scene.geom)
  =
  match g.shape with
  | Scene.Sphere { centre; radius } -> [ P.Sphere { center = point centre; radius } ]
  | Scene.Capsule { a; b; radius } ->
    (* A capsule is a segment thickened by a sphere. Drawing the axis and the
       two ends leaves it looking like a wire, so it is sampled: a handful of
       spheres along the axis, which is the same set to within the sampling. *)
    let steps = 7 in
    let at k = add a (smul (sub b a) (Float.of_int k /. Float.of_int (steps - 1))) in
    List.init steps ~f:(fun k -> P.Sphere { center = point (at k); radius })
  | Scene.Box { centre; axes; half } -> box_edges centre axes half
  | Scene.Cylinder { centre; axis; radius; half_height } ->
    let top = add centre (smul axis half_height)
    and bottom = add centre (smul axis (-.half_height)) in
    let side = smul axis half_height in
    [ circle ~centre:top ~normal:axis ~radius
    ; circle ~centre:bottom ~normal:axis ~radius
    ; P.Segment (point (add top (smul side 0.)), point (add bottom (smul side 0.)))
    ]
  | Scene.Ellipsoid { centre; axes = ux, uy, uz; radii = rx, ry, _rz } ->
    let _ = uz in
    [ circle ~centre ~normal:uz ~radius:(Float.max rx ry)
    ; circle ~centre ~normal:uy ~radius:rx
    ; circle ~centre ~normal:ux ~radius:ry
    ]
  | Scene.Plane { normal; offset; half = hx, hy } ->
    (* The patch is the plane's own rectangle: its normal is the third column
       of the geom frame, so two orthogonal directions span it. *)
    let n = normal in
    let u =
      if Float.( < ) (Float.abs n.Scene.x) 0.9
      then { Scene.x = 0.; y = -.n.z; z = n.y }
      else { Scene.x = n.z; y = 0.; z = -.n.x }
    in
    let norm (v : Scene.p3) = Float.sqrt ((v.x *. v.x) +. (v.y *. v.y) +. (v.z *. v.z)) in
    let u = smul u (1. /. Float.max (norm u) 1e-9) in
    let w =
      { Scene.x = (n.y *. u.z) -. (n.z *. u.y)
      ; y = (n.z *. u.x) -. (n.x *. u.z)
      ; z = (n.x *. u.y) -. (n.y *. u.x)
      }
    in
    (* Centre the patch on the target's projection, and keep it no larger
       than a few scene widths. *)
    let at = sub plane_centre (smul n (dot n plane_centre -. offset)) in
    let hx = Float.min hx plane_extent
    and hy = Float.min hy plane_extent in
    let centre = at in
    let corner sx sy = add centre (add (smul u (sx *. hx)) (smul w (sy *. hy))) in
    let patch =
      [ point (corner (-1.) (-1.))
      ; point (corner 1. (-1.))
      ; point (corner 1. 1.)
      ; point (corner (-1.) 1.)
      ]
    in
    [ P.Plane { normal = point n; offset; patch } ]
  | Scene.Unsupported _ -> []

(* {1 Camera from the scene} *)

let bounds (s : Scene.t) =
  let lo = ref { Scene.x = Float.infinity; y = Float.infinity; z = Float.infinity }
  and hi =
    ref { Scene.x = Float.neg_infinity; y = Float.neg_infinity; z = Float.neg_infinity }
  in
  let seen = ref false in
  let eat (p : Scene.p3) =
    seen := true;
    lo
    := { Scene.x = Float.min p.x !lo.x; y = Float.min p.y !lo.y; z = Float.min p.z !lo.z };
    hi
    := { Scene.x = Float.max p.x !hi.x; y = Float.max p.y !hi.y; z = Float.max p.z !hi.z }
  in
  List.iter s.geoms ~f:(fun g ->
    match g.shape with
    | Scene.Sphere { centre; radius } ->
      eat (add centre (smul { Scene.x = 1.; y = 0.; z = 0. } radius));
      eat (add centre (smul { Scene.x = -1.; y = 0.; z = 0. } radius))
    | Scene.Capsule { a; b; _ } ->
      eat a;
      eat b
    | Scene.Box { centre; _ }
    | Scene.Cylinder { centre; _ }
    | Scene.Ellipsoid { centre; _ } -> eat centre
    | Scene.Plane _ | Scene.Unsupported _ -> ());
  if not !seen
  then { Scene.x = -1.; y = -1.; z = -1. }, { Scene.x = 1.; y = 1.; z = 1. }
  else !lo, !hi

let camera_for (c : camera) (s : Scene.t) =
  let lo, hi = bounds s in
  let centre = smul (add lo hi) 0.5 in
  let size = Float.max (hi.Scene.x -. lo.x) (Float.max (hi.y -. lo.y) (hi.z -. lo.z)) in
  let d = Float.max size 0.5 in
  let eye =
    Option.value
      c.eye
      ~default:(add centre { Scene.x = 2.0 *. d; y = -2.8 *. d; z = 1.6 *. d })
  in
  let target = Option.value c.target ~default:centre in
  Geom_viz.Camera.create
    ~eye:(point eye)
    ~target:(point target)
    ~up:(point c.up)
    ~fov_y:c.fov_y
    ~width:c.width
    ~height:c.height
    ()

(* {1 The picture} *)

let entities
      ?(plane_extent = 5.)
      ?(plane_centre = { Scene.x = 0.; y = 0.; z = 0. })
      (s : Scene.t)
  =
  let of_geom (g : Scene.geom) =
    List.map (shapes ~plane_extent ~plane_centre g) ~f:(fun p ->
      let style =
        match p with
        | P.Segment _ | P.Circle _ -> edge g.colour
        | P.Plane _ ->
          (* Translucent: a floor is context, and the painter's algorithm would
             otherwise let it hide whatever stands on it. *)
          { D.default with
            fill = Some (colour { g.colour with a = 0.18 })
          ; stroke = Some (colour { g.colour with a = 0.5 })
          ; width = 0.4
          }
        | _ -> fill g.colour
      in
      D.static
        ~layer:
          (match p with
           | P.Plane _ -> 0
           | _ -> 2)
        ~style
        p)
  in
  let contacts =
    List.concat_map s.contacts ~f:(fun (c : Scene.contact) ->
      let tip = add c.point (smul c.normal 0.12) in
      [ D.static
          ~layer:3
          ~style:(fill { r = 0.85; g = 0.1; b = 0.1; a = 0.9 })
          (P.Sphere { center = point c.point; radius = 0.02 })
      ; D.static
          ~layer:3
          ~style:(edge { r = 0.85; g = 0.1; b = 0.1; a = 0.9 })
          (P.Segment (point c.point, point tip))
      ])
  in
  let forces =
    List.map s.forces ~f:(fun (f : Scene.arrow) ->
      D.static
        ~layer:4
        ~style:(edge { r = 0.1; g = 0.35; b = 0.85; a = 0.95 })
        (P.Segment (point f.from, point (add f.from f.vector))))
  in
  let frames =
    List.map s.bodies ~f:(fun (b : Scene.body) ->
      D.static
        ~layer:5
        ~style:
          { D.default with
            fill = None
          ; stroke = Some (Hugin_gg.Color.v 0.1 0.1 0.1)
          ; width = 0.9
          }
        (P.Frame
           { origin = point b.frame.origin
           ; x = point b.frame.x
           ; y = point b.frame.y
           ; z = point b.frame.z
           }))
  in
  List.concat_map s.geoms ~f:of_geom @ contacts @ forces @ frames

let scene ?(camera = default_camera) (s : Scene.t) =
  let cam = camera_for camera s in
  let lo, hi = bounds s in
  let centre = smul (add lo hi) 0.5 in
  let extent =
    Float.max
      1.0
      (2.5 *. Float.max (hi.Scene.x -. lo.x) (Float.max (hi.y -. lo.y) (hi.z -. lo.z)))
  in
  D.create
    ~background:(Hugin_gg.Color.v 0.97 0.97 0.98)
    cam
    (entities ~plane_extent:extent ~plane_centre:centre s)

let frame ?(density = 1.5) ?camera (s : Scene.t) =
  let c = Option.value camera ~default:default_camera in
  let sc = scene ?camera s in
  Geom_viz.Frame.create
    ~density
    ~width:c.width
    ~height:c.height
    (D.picture_at sc ~time:0.)

let png ?density ?camera s = Geom_viz.Frame.png (frame ?density ?camera s)

let save_png ?density ?camera path s =
  Geom_viz.Frame.save_png (frame ?density ?camera s) path
