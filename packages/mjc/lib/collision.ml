(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
module Int32 = Stdlib.Int32
open Topo

(* {1 Candidate pairs}

   MuJoCo decides which geoms may touch before it computes any geometry: a
   bitmask, a body filter, a bounding-sphere rejection, and a table of the
   primitive pairs it knows how to solve. All of that is host data, read while
   the program is built, so the candidate list is a host array and only the
   geometry below is tensors. *)

let rec weldbody (t : Topo.t) b =
  if Int.equal b 0 || Int.( > ) t.body_jntnum.(b) 0
  then b
  else weldbody t t.body_parentid.(b)

let weldparent (t : Topo.t) b =
  if Int.equal b 0 then 0 else weldbody t t.body_parentid.(b)

let bitmask_ok (t : Topo.t) g1 g2 =
  (not (Int.equal (t.geom_contype.(g1) land t.geom_conaffinity.(g2)) 0))
  || not (Int.equal (t.geom_contype.(g2) land t.geom_conaffinity.(g1)) 0)

let type_index = function
  | Plane -> 0
  | Hfield -> 1
  | Sphere -> 2
  | Capsule -> 3
  | Ellipsoid -> 4
  | Cylinder -> 5
  | Box -> 6
  | Mesh -> 7

(* The pairs whose distance v1 knows how to compute, MuJoCo's lower type
   first. *)
let supported (t : Topo.t) g1 g2 =
  match t.geom_type.(g1), t.geom_type.(g2) with
  | Plane, Sphere
  | Plane, Capsule
  | Sphere, Sphere
  | Sphere, Capsule
  | Plane, Box
  | Sphere, Box
  | Capsule, Capsule
  | Box, Box -> true
  | _ -> false

(* One contact slot a pair can produce: a capsule against a plane yields one
   per endpoint, every other supported pair yields one. *)
type cand =
  { g1 : int
  ; g2 : int
  ; slot : int
  }

let slots (t : Topo.t) g1 g2 =
  match t.geom_type.(g1), t.geom_type.(g2) with
  | Plane, Capsule -> 2
  | Plane, Box -> 8
  | Capsule, Capsule -> 2
  (* a face contact is a clipped polygon: a quad through four half-planes has at
     most eight vertices, and the tensor kernel holds twelve slots *)
  | Box, Box -> Collision_box_tensor.max_contacts
  | _ -> 1

(* MuJoCo keeps at most four contacts per pair of the primitive kinds, taking
   them in the order the distance function emits them; a box pair is the
   exception and fills the whole manifold. *)
let pair_limit (t : Topo.t) g1 g2 =
  match t.geom_type.(g1), t.geom_type.(g2) with
  | Box, Box -> Collision_box_tensor.max_contacts
  | _ -> 4

let candidates (m : Model.host) : cand array =
  let t = m.topo in
  let out = ref [] in
  if t.contacts_enabled
  then
    for a = 0 to t.ngeom - 1 do
      for b = a + 1 to t.ngeom - 1 do
        let g1, g2 =
          if Int.( <= ) (type_index t.geom_type.(a)) (type_index t.geom_type.(b))
          then a, b
          else b, a
        in
        let weld1 = weldbody t t.geom_bodyid.(g1)
        and weld2 = weldbody t t.geom_bodyid.(g2) in
        let filtered =
          Int.equal weld1 weld2
          || (Int.equal weld1 0 && Int.equal weld2 0)
          (* filterparent, MuJoCo's default: a body does not touch its own
             parent, but the world body is exempt. *)
          || ((not (Int.equal weld1 0))
              && (not (Int.equal weld2 0))
              && (Int.equal weld1 (weldparent t t.geom_bodyid.(g2))
                  || Int.equal weld2 (weldparent t t.geom_bodyid.(g1))))
        in
        (* A pair that may touch but whose distance v1 cannot compute is a hole
           in the physics, not an omission to leave silent. *)
        if (not filtered) && bitmask_ok t g1 g2 && not (supported t g1 g2)
        then
          raise
            (Invalid_argument
               (Printf.sprintf
                  "collision between geom types %d and %d is not implemented"
                  (type_index t.geom_type.(g1))
                  (type_index t.geom_type.(g2))));
        if (not filtered) && bitmask_ok t g1 g2 && supported t g1 g2
        then
          for slot = 0 to slots t g1 g2 - 1 do
            out := { g1; g2; slot } :: !out
          done
      done
    done;
  Array.of_list (List.rev !out)

(* {1 Contact geometry}

   MuJoCo's primitive distance functions for the pairs v1 supports. Each
   returns the signed distance along the normal, the contact point and the
   normal, the normal pointing from geom 1 towards geom 2. *)

let geom_zaxis (d : Data.host) g =
  let m = Nx.reshape [| 3; 3 |] (Nx.slice [ I g; A ] d.geom_xmat) in
  Nx.slice [ A; I 2 ] m

let geom_pos (d : Data.host) g = Nx.slice [ I g; A ] d.geom_xpos

(* The two z axes coincide, so no direction is defined: MuJoCo takes their
   cross product and falls back to x when even that is zero. *)
let fallback_normal axis1 axis2 =
  let cross = Nx.cross axis1 axis2 in
  let len = Nx.sqrt (Nx.dot cross cross) in
  let x = Nx.set [ I 0 ] (Nx.scalar Nx.float64 1.0) (Nx.zeros Nx.float64 [| 3 |]) in
  let v = Nx.where (Nx.greater_s len 1e-15) cross x in
  Nx.div v (Nx.maximum_s (Nx.sqrt (Nx.dot v v)) 1e-300)

let sphere_sphere ~pos1 ~size1 ~pos2 ~size2 ~axis1 ~axis2 =
  let dif = Nx.sub pos1 pos2 in
  let cdist = Nx.sqrt (Nx.dot dif dif) in
  let dist = Nx.sub (Nx.sub cdist size1) size2 in
  let towards = Nx.sub pos2 pos1 in
  let len = Nx.sqrt (Nx.dot towards towards) in
  let normal =
    Nx.where
      (Nx.greater_s len 1e-15)
      (Nx.div towards (Nx.maximum_s len 1e-300))
      (fallback_normal axis1 axis2)
  in
  let pos = Nx.add pos1 (Nx.mul normal (Nx.add size1 (Nx.mul_s dist 0.5))) in
  dist, pos, normal

let plane_sphere ~pos1 ~normal ~pos2 ~size =
  let cdist = Nx.dot (Nx.sub pos2 pos1) normal in
  let dist = Nx.sub cdist size in
  let pos = Nx.add pos2 (Nx.mul normal (Nx.neg (Nx.add (Nx.mul_s dist 0.5) size))) in
  dist, pos, normal

(* [mju_makeFrame]: the contact's third row from its normal and whatever
   tangent the pair supplied, orthonormalised. *)
let make_frame ~normal ~tangent =
  let sc = Nx.scalar Nx.float64 in
  let norm v = Nx.maximum_s (Nx.sqrt (Nx.dot v v)) 1e-300 in
  let n = Nx.div normal (norm normal) in
  let degenerate = Nx.less_s (Nx.dot tangent tangent) 0.25 in
  let fallback =
    Nx.where
      (Nx.logical_and
         (Nx.less_s (Nx.get [ 1 ] n) 0.5)
         (Nx.greater_s (Nx.get [ 1 ] n) (-0.5)))
      (Nx.stack ~axis:0 [ sc 0.0; sc 1.0; sc 0.0 ])
      (Nx.stack ~axis:0 [ sc 0.0; sc 0.0; sc 1.0 ])
  in
  let y0 = Nx.where degenerate fallback tangent in
  let y = Nx.sub y0 (Nx.mul n (Nx.dot n y0)) in
  let y = Nx.div y (norm y) in
  let z = Nx.cross n y in
  Nx.stack ~axis:0 [ n; y; z ]

let geometry (m : Model.host) (d : Data.host) (c : cand) =
  let p = m.params in
  let t = m.topo in
  let size g k = Nx.get [ g; k ] p.geom_size in
  match t.geom_type.(c.g1), t.geom_type.(c.g2) with
  | Plane, Sphere ->
    plane_sphere
      ~pos1:(geom_pos d c.g1)
      ~normal:(geom_zaxis d c.g1)
      ~pos2:(geom_pos d c.g2)
      ~size:(size c.g2 0)
  | Plane, Capsule ->
    let axis = geom_zaxis d c.g2 in
    let sign = if Int.equal c.slot 0 then 1.0 else -1.0 in
    let endpoint = Nx.add (geom_pos d c.g2) (Nx.mul axis (Nx.mul_s (size c.g2 1) sign)) in
    plane_sphere
      ~pos1:(geom_pos d c.g1)
      ~normal:(geom_zaxis d c.g1)
      ~pos2:endpoint
      ~size:(size c.g2 0)
  | Sphere, Sphere ->
    sphere_sphere
      ~pos1:(geom_pos d c.g1)
      ~size1:(size c.g1 0)
      ~pos2:(geom_pos d c.g2)
      ~size2:(size c.g2 0)
      ~axis1:(geom_zaxis d c.g1)
      ~axis2:(geom_zaxis d c.g2)
  | Plane, Box ->
    (* One candidate per corner, in MuJoCo's order, with a corner above the
       plane sent far away so that the compaction drops it. *)
    let normal = geom_zaxis d c.g1 in
    let mat2 = Nx.reshape [| 3; 3 |] (Nx.slice [ I c.g2; A ] d.geom_xmat) in
    let sx = size c.g2 0
    and sy = size c.g2 1
    and sz = size c.g2 2 in
    let pick i flag hi lo = if Int.( land ) i flag <> 0 then hi else lo in
    let vec =
      Nx.stack
        ~axis:0
        [ pick c.slot 1 sx (Nx.neg sx)
        ; pick c.slot 2 sy (Nx.neg sy)
        ; pick c.slot 4 sz (Nx.neg sz)
        ]
    in
    let corner =
      Nx.add
        (geom_pos d c.g2)
        (Nx.matmul mat2 (Nx.reshape [| 3; 1 |] vec) |> Nx.reshape [| 3 |])
    in
    let ldist = Nx.dot normal (Nx.sub corner (geom_pos d c.g2)) in
    let centre = Nx.dot (Nx.sub (geom_pos d c.g2) (geom_pos d c.g1)) normal in
    let dist = Nx.add centre ldist in
    let up = Nx.greater_s ldist 0.0 in
    let far = Nx.scalar Nx.float64 1e10 in
    let dist = Nx.where up far dist in
    let pos = Nx.add corner (Nx.mul normal (Nx.neg (Nx.mul_s dist 0.5))) in
    dist, pos, normal
  | Capsule, Capsule ->
    (* [mjraw_CapsuleCapsule]: the closest pair of points on the two axial
       segments, then a sphere-sphere test on them. Nearly parallel segments
       have two candidate pairs and one slot carries each. *)
    let sc x = Nx.scalar Nx.float64 x in
    let axis1 = geom_zaxis d c.g1
    and axis2 = geom_zaxis d c.g2 in
    let r1 = size c.g1 0
    and l1 = size c.g1 1
    and r2 = size c.g2 0
    and l2 = size c.g2 1 in
    let p1 = Nx.sub (geom_pos d c.g1) (Nx.mul axis1 l1)
    and p2 = Nx.add (geom_pos d c.g1) (Nx.mul axis1 l1)
    and q1 = Nx.sub (geom_pos d c.g2) (Nx.mul axis2 l2)
    and q2 = Nx.add (geom_pos d c.g2) (Nx.mul axis2 l2) in
    let ab = Nx.sub p2 p1
    and cd = Nx.sub q2 q1
    and ca = Nx.sub p1 q1 in
    let dot1 = Nx.dot ab ab
    and dot2 = Nx.dot cd cd
    and dot3 = Nx.dot cd ca
    and dot4 = Nx.dot ab ca
    and dot5 = Nx.dot ab cd in
    let clip x = Nx.minimum (Nx.maximum x (sc 0.0)) (sc 1.0) in
    let s0 = Nx.neg (Nx.div dot4 dot1)
    and s1 = Nx.div (Nx.sub dot5 dot4) dot1 in
    let cross = Nx.cross ab cd in
    let det = Nx.dot cross cross in
    let parallel = Nx.less_equal det (Nx.mul (Nx.mul dot1 dot2) (sc 1e-8)) in
    let s = clip (Nx.div (Nx.dot cross (Nx.cross cd ca)) (Nx.maximum_s det 1e-300)) in
    let t = Nx.div (Nx.add (Nx.mul dot5 s) dot3) dot2 in
    (* [clampSeg]: slide to an endpoint when the projection leaves the segment. *)
    let seg s_ t_ =
      let a =
        Nx.where
          (Nx.less_s t_ 0.0)
          (clip s0)
          (Nx.where (Nx.greater_s t_ 1.0) (clip s1) s_)
      in
      let b =
        Nx.where (Nx.less_s t_ 0.0) (sc 0.0) (Nx.where (Nx.greater_s t_ 1.0) (sc 1.0) t_)
      in
      a, b
    in
    let a0, b0 = seg s t in
    let a1, b1 = seg (sc 0.0) (Nx.div dot3 dot2) in
    let a2, b2 = seg (sc 1.0) (Nx.div (Nx.add dot5 dot3) dot2) in
    let w_a, w_b =
      if Int.equal c.slot 0
      then Nx.where parallel a1 a0, Nx.where parallel b1 b0
      else a2, b2
    in
    let point base w dir = Nx.add base (Nx.mul dir w) in
    let dist, pos, normal =
      sphere_sphere
        ~pos1:(point p1 w_a ab)
        ~size1:r1
        ~pos2:(point q1 w_b cd)
        ~size2:r2
        ~axis1
        ~axis2
    in
    let dist = if Int.equal c.slot 0 then dist else Nx.where parallel dist (sc 1e10) in
    dist, pos, normal
  | Sphere, Box ->
    (* [mjraw_SphereBox]: the sphere centre in the box frame, clamped to the
       half-sizes; inside the box the nearest face sets the normal. *)
    let mat2 = Nx.reshape [| 3; 3 |] (Nx.slice [ I c.g2; A ] d.geom_xmat) in
    let local v =
      Nx.reshape [| 3 |] (Nx.matmul (Nx.matrix_transpose mat2) (Nx.reshape [| 3; 1 |] v))
    in
    let world v = Nx.reshape [| 3 |] (Nx.matmul mat2 (Nx.reshape [| 3; 1 |] v)) in
    let radius = size c.g1 0 in
    let half = Nx.stack ~axis:0 [ size c.g2 0; size c.g2 1; size c.g2 2 ] in
    let centre = local (Nx.sub (geom_pos d c.g1) (geom_pos d c.g2)) in
    let clamped = Nx.minimum (Nx.maximum centre (Nx.neg half)) half in
    let towards = Nx.sub clamped centre in
    let away = Nx.sqrt (Nx.dot towards towards) in
    let touch = Nx.maximum_s away 1e-300 in
    let direction = Nx.div towards touch in
    (* Outside: the deepest point of the sphere and the midpoint to the box. *)
    (* The point of the sphere in the direction of the box: the centre moved
       by one radius, not the clamped point. *)
    let deepest = Nx.add centre (Nx.mul direction radius) in
    let outside_pos = Nx.mul_s (Nx.add clamped deepest) 0.5 in
    let outside_normal = world direction in
    let outside_dist = Nx.sub away radius in
    (* Inside: the nearest face, and the sphere centre pushed halfway to it. *)
    let gap_plus = Nx.sub half centre
    and gap_minus = Nx.add half centre in
    let gap = Nx.minimum gap_plus gap_minus in
    (* MuJoCo enumerates the six faces as -x,+x,-y,... and sets the winning
       direction to +1 on the *even* index, so the nearer -face means -1 for
       the face itself and +1 for the axis it lies on. The sign is therefore
       opposite to the face that wins. *)
    let sign =
      Nx.where (Nx.less gap_plus gap_minus) (Nx.neg (Nx.ones_like gap)) (Nx.ones_like gap)
    in
    let g0 = Nx.get [ 0 ] gap
    and g1 = Nx.get [ 1 ] gap
    and g2 = Nx.get [ 2 ] gap in
    let axis i v = Nx.set [ I i ] v (Nx.zeros Nx.float64 [| 3 |]) in
    let nearest =
      Nx.where
        (Nx.logical_and (Nx.less_equal g0 g1) (Nx.less_equal g0 g2))
        (axis 0 (Nx.get [ 0 ] sign))
        (Nx.where
           (Nx.less_equal g1 g2)
           (axis 1 (Nx.get [ 1 ] sign))
           (axis 2 (Nx.get [ 2 ] sign)))
    in
    let closest = Nx.minimum g0 (Nx.minimum g1 g2) in
    let inside_pos =
      Nx.add centre (Nx.mul nearest (Nx.mul_s (Nx.sub radius closest) 0.5))
    in
    let inside_normal = world nearest in
    let inside = Nx.less_equal away (Nx.scalar Nx.float64 1e-15) in
    let pos_local = Nx.where inside inside_pos outside_pos in
    let normal = Nx.where inside inside_normal outside_normal in
    let dist = Nx.where inside (Nx.sub (Nx.neg closest) radius) outside_dist in
    let pos = Nx.add (world pos_local) (geom_pos d c.g2) in
    dist, pos, normal
  | Sphere, Capsule ->
    let axis = geom_zaxis d c.g2 in
    let len = size c.g2 1 in
    let vec = Nx.sub (geom_pos d c.g1) (geom_pos d c.g2) in
    let along = Nx.minimum (Nx.maximum (Nx.dot axis vec) (Nx.neg len)) len in
    let nearest = Nx.add (geom_pos d c.g2) (Nx.mul axis along) in
    sphere_sphere
      ~pos1:(geom_pos d c.g1)
      ~size1:(size c.g1 0)
      ~pos2:nearest
      ~size2:(size c.g2 0)
      ~axis1:(geom_zaxis d c.g1)
      ~axis2:(geom_zaxis d c.g2)
  | _ -> Nx.scalar Nx.float64 0.0, geom_pos d c.g1, geom_pos d c.g1

(* {1 Contact parameters}

   [mj_contactParam]: the higher-priority geom wins outright, otherwise the
   condim is the max, the friction the elementwise max, and the solver
   parameters are mixed with a weight that favours the geom with the larger
   [solmix]. Standard-format references mix; direct-format ones take the min.

   The C engine does this with scalars because it reads a model it owns. A
   traced program may not: a captured parameter is a traced tensor, and only
   tensor operations are allowed on one. So the same arithmetic is written as
   gathers and [where]s, one row per candidate contact. *)

type mixed =
  { dim : Nx.host Nx.int32_t (** [[ncand]] *)
  ; friction : Nx.host Nx.float64_t (** [[ncand;5]] *)
  ; solref : Nx.host Nx.float64_t (** [[ncand;2]] *)
  ; solimp : Nx.host Nx.float64_t (** [[ncand;5]] *)
  ; includemargin : Nx.host Nx.float64_t (** [[ncand]] *)
  ; reach : Nx.host Nx.float64_t
    (** [[ncand]], the bounding radius: the margin plus the two bounds. *)
  }

let mixed_params (m : Model.host) (cands : cand array) : mixed =
  let p = m.params in
  let n = Array.length cands in
  let minval = 1e-15 in
  let sc x = Nx.scalar Nx.float64 x in
  let g side =
    Nx.init Nx.int64 [| n |] (fun i ->
      Int64.of_int (if Int.equal side 0 then cands.(i.(0)).g1 else cands.(i.(0)).g2))
  in
  let g1 = g 0
  and g2 = g 1 in
  let take g x = Nx.take ~axis:0 g x in
  let prio side =
    Nx.init Nx.float64 [| n |] (fun i ->
      let c = cands.(i.(0)) in
      Float.of_int m.topo.geom_priority.(if Int.equal side 0 then c.g1 else c.g2))
  in
  let prio1 = prio 0
  and prio2 = prio 1 in
  let as_row v = Nx.reshape [| n; 1 |] v in
  (* Picking between two parameter rows: the condition is a column so that it
     broadcasts over the parameter's own width. *)
  let pick v1 v2 v12 =
    Nx.where
      (Nx.logical_not (as_row (Nx.less prio1 prio2)))
      (Nx.where (as_row (Nx.greater prio1 prio2)) v1 v12)
      v2
  in
  let margin1 = take g1 p.geom_margin
  and margin2 = take g2 p.geom_margin
  and gap1 = take g1 p.geom_gap
  and gap2 = take g2 p.geom_gap
  and rbound1 = take g1 p.geom_rbound
  and rbound2 = take g2 p.geom_rbound in
  let includemargin = Nx.add (Nx.add margin1 margin2) (Nx.add gap1 gap2) in
  let reach = Nx.add (Nx.add includemargin rbound1) rbound2 in
  (* The mix weight: s1/(s1+s2) when both are meaningful, 0.5 when neither is,
     and the one that is when only one is. *)
  let s1 = take g1 p.geom_solmix
  and s2 = take g2 p.geom_solmix in
  let both =
    Nx.logical_and (Nx.greater_equal_s s1 minval) (Nx.greater_equal_s s2 minval)
  in
  let neither = Nx.logical_and (Nx.less_s s1 minval) (Nx.less_s s2 minval) in
  let ratio = Nx.div s1 (Nx.maximum_s (Nx.add s1 s2) minval) in
  let mix =
    Nx.where
      both
      ratio
      (Nx.where neither (sc 0.5) (Nx.where (Nx.less_s s1 minval) (sc 0.0) (sc 1.0)))
  in
  let one_minus = Nx.sub (sc 1.0) (as_row mix) in
  let solimp =
    pick
      (take g1 p.geom_solimp)
      (take g2 p.geom_solimp)
      (Nx.add
         (Nx.mul (as_row mix) (take g1 p.geom_solimp))
         (Nx.mul one_minus (take g2 p.geom_solimp)))
  in
  let ref1 = take g1 p.geom_solref
  and ref2 = take g2 p.geom_solref in
  let standard =
    Nx.logical_and
      (Nx.greater_s (Nx.slice [ A; I 0 ] ref1) 0.0)
      (Nx.greater_s (Nx.slice [ A; I 0 ] ref2) 0.0)
  in
  let solref_mixed = Nx.add (Nx.mul (as_row mix) ref1) (Nx.mul one_minus ref2) in
  let solref =
    pick ref1 ref2 (Nx.where (as_row standard) solref_mixed (Nx.minimum ref1 ref2))
  in
  let mu = Nx.maximum (take g1 p.geom_friction) (take g2 p.geom_friction) in
  (* MuJoCo's five contact slots from the three a geom carries. *)
  let col k x = Nx.reshape [| n; 1 |] (Nx.slice [ A; I k ] x) in
  let friction =
    Nx.concatenate ~axis:1 [ col 0 mu; col 0 mu; col 1 mu; col 2 mu; col 2 mu ]
  in
  let dim =
    Nx.cast Nx.int32 (Nx.maximum (take g1 p.geom_condim) (take g2 p.geom_condim))
  in
  { dim; friction; solref; solimp; includemargin; reach }

(* {1 Contacts}

   The candidate list is compacted the way the limit rows are: which contacts
   exist is a tensor predicate, the destination index is the number of live
   contacts before it, and a dead candidate is sent past the buffer where a
   scatter writes nothing. *)

(* {1 Box-box}

   The one pair whose manifold is more than one contact, and the one whose
   geometry is tensor code over a batch of pairs rather than one expression over
   the candidate list: {!Collision_box_tensor} wants a batch, so every box pair
   in the list is gathered once, solved once, and read back slot by slot. *)

type box_manifold =
  { row : (int, int) Hashtbl.t (** [g1 * ngeom + g2] to the pair's row in the batch *)
  ; manifold : Collision_box_tensor.manifold
  }

let is_box_pair (t : Topo.t) g1 g2 =
  match t.geom_type.(g1), t.geom_type.(g2) with
  | Box, Box -> true
  | _ -> false

let box_box (m : Model.host) (d : Data.host) (cands : cand array) =
  let t = m.topo in
  let pairs =
    Array.filter cands ~f:(fun c -> is_box_pair t c.g1 c.g2 && Int.equal c.slot 0)
  in
  let n = Array.length pairs in
  if Int.equal n 0
  then None
  else (
    let p = m.params in
    let g1 i = pairs.(i).g1
    and g2 i = pairs.(i).g2 in
    let gather (x : _ Nx.t) f =
      let idx = Nx.init Nx.int64 [| n |] (fun i -> Int64.of_int (f i.(0))) in
      Nx.take ~axis:0 idx x
    in
    let batch =
      Collision_box_tensor.
        { pos1 = gather d.geom_xpos g1
        ; mat1 = gather d.geom_xmat g1
        ; size1 = gather p.geom_size g1
        ; pos2 = gather d.geom_xpos g2
        ; mat2 = gather d.geom_xmat g2
        ; size2 =
            gather p.geom_size g2
            (* the engine's collision functions take the two geom margins, not the
               gaps, which only widen the driver's inclusion test *)
        ; margin = Nx.add (gather p.geom_margin g1) (gather p.geom_margin g2)
        }
    in
    let man = Collision_box_tensor.manifold batch in
    let row = Hashtbl.create (module Int) in
    for i = 0 to n - 1 do
      Hashtbl.set row ~key:((g1 i * t.ngeom) + g2 i) ~data:i
    done;
    Some { row; manifold = man })

(* [box_slot bm row slot] is one contact of a pair's manifold, or an infinite
   distance for a dead slot, which the inclusion test then drops. *)
let box_slot (bm : box_manifold) row slot =
  let m = bm.manifold in
  let dead = Nx.equal_s (Nx.slice [ I row; I slot ] m.Collision_box_tensor.live) 0. in
  ( Nx.where dead (Nx.scalar Nx.float64 1e30) (Nx.slice [ I row; I slot ] m.dist)
  , Nx.where dead (Nx.zeros Nx.float64 [| 3 |]) (Nx.slice [ I row; I slot; A ] m.pos)
  , Nx.slice [ I row; I slot; A ] m.normal )

let contacts (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo in
  let cands = candidates m in
  let ncand = Array.length cands in
  (* A model with no candidate pair touches no parameter at all, which is most
     of them and keeps their step on the smooth path. *)
  if Int.equal ncand 0
  then { d with ncon = Nx.zeros Nx.int32 [||] }
  else (
    let mix = mixed_params m cands in
    let boxes = box_box m d cands in
    let row_of k x = Nx.slice [ I k ] x in
    let tangent (t : Topo.t) d (c : cand) =
      match t.geom_type.(c.g1), t.geom_type.(c.g2) with
      | Plane, Capsule -> geom_zaxis d c.g2
      | _ -> Nx.zeros Nx.float64 [| 3 |]
    in
    let dist = ref (Nx.zeros Nx.float64 [| ncand |])
    and pos = ref (Nx.zeros Nx.float64 [| ncand; 3 |])
    and normal = ref (Nx.zeros Nx.float64 [| ncand; 3 |])
    and frames = ref (Nx.zeros Nx.float64 [| ncand; 9 |])
    and live = ref (Nx.zeros Nx.float64 [| ncand |])
    and jac = ref (Nx.zeros Nx.float64 [| ncand; 3 * t.nv |]) in
    (* MuJoCo keeps at most four contacts per pair, taking them in corner
       order. The candidate loop runs in that order, so a running count per
       pair does the same cut. *)
    let seen = ref (Nx.scalar Nx.float64 0.0)
    and pair = ref (-1, -1) in
    Array.iteri cands ~f:(fun i c ->
      let dist_i, pos_i, normal_i =
        if is_box_pair t c.g1 c.g2
        then (
          let row =
            match boxes with
            | Some bm -> Hashtbl.find_exn bm.row ((c.g1 * t.ngeom) + c.g2)
            | None -> assert false
          in
          box_slot (Option.value_exn boxes) row c.slot)
        else geometry m d c
      in
      let includemargin = row_of i mix.includemargin
      and reach = row_of i mix.reach in
      let plane =
        Poly.( = ) t.geom_type.(c.g1) Plane || Poly.( = ) t.geom_type.(c.g2) Plane
      in
      let bound =
        if plane
        then Nx.scalar Nx.float64 1.0
        else (
          let dif = Nx.sub (geom_pos d c.g1) (geom_pos d c.g2) in
          Nx.where
            (Nx.less_equal (Nx.dot dif dif) (Nx.mul reach reach))
            (Nx.scalar Nx.float64 1.0)
            (Nx.scalar Nx.float64 0.0))
      in
      let near = Nx.less_equal dist_i includemargin in
      if not (Int.equal (fst !pair) c.g1 && Int.equal (snd !pair) c.g2)
      then (
        pair := c.g1, c.g2;
        seen := Nx.scalar Nx.float64 0.0);
      let room =
        Nx.where
          (Nx.less_s !seen (Float.of_int (pair_limit t c.g1 c.g2)))
          (Nx.scalar Nx.float64 1.0)
          (Nx.scalar Nx.float64 0.0)
      in
      let on =
        Nx.mul
          (Nx.mul bound room)
          (Nx.where near (Nx.scalar Nx.float64 1.0) (Nx.scalar Nx.float64 0.0))
      in
      seen := Nx.add !seen on;
      (* The frame's three rows, each dotted with the Jacobian difference:
         geom 2's point Jacobian minus geom 1's. The normal row leads; a
         pyramidal friction pair is built from it and one tangent. *)
      let b1 = t.geom_bodyid.(c.g1)
      and b2 = t.geom_bodyid.(c.g2) in
      let frame = make_frame ~normal:normal_i ~tangent:(tangent t d c) in
      let jacp1, _ = Jacobian.point m d ~body:b1 ~pos:pos_i
      and jacp2, _ = Jacobian.point m d ~body:b2 ~pos:pos_i in
      let rows =
        Nx.matmul (Nx.sub jacp2 jacp1) (Nx.matrix_transpose frame)
        |> Nx.matrix_transpose
        |> fun x -> Nx.reshape [| 3 * t.nv |] x
      in
      jac := Nx.set [ I i ] rows !jac;
      dist := Nx.set [ I i ] dist_i !dist;
      pos := Nx.set [ I i ] pos_i !pos;
      frames := Nx.set [ I i ] (Nx.reshape [| 9 |] frame) !frames;
      normal := Nx.set [ I i ] normal_i !normal;
      live := Nx.set [ I i ] on !live);
    let before = Nx.sub (Nx.cumsum ~axis:0 !live) !live in
    let far = Nx.full_like before (Float.of_int t.nconmax) in
    let idx = Nx.cast Nx.int64 (Nx.where (Nx.greater_s !live 0.5) before far) in
    let put v x = Nx.scatter ~combine:Set ~axis:0 idx v x in
    let put_cols v x =
      let cols = (Nx.shape v).(1) in
      let idx_rows = Nx.broadcast_to [| ncand; cols |] (Nx.reshape [| ncand; 1 |] idx) in
      Nx.scatter ~combine:Set ~axis:0 idx_rows v x
    in
    let geom =
      Nx.init Nx.int32 [| ncand; 2 |] (fun i ->
        let c = cands.(i.(0)) in
        Int32.of_int (if Int.equal i.(1) 0 then c.g1 else c.g2))
    in
    let body =
      Nx.init Nx.int32 [| ncand; 2 |] (fun i ->
        let c = cands.(i.(0)) in
        Int32.of_int
          (if Int.equal i.(1) 0 then t.geom_bodyid.(c.g1) else t.geom_bodyid.(c.g2)))
    in
    let dim = mix.dim
    and friction = mix.friction
    and solref = mix.solref
    and solimp = mix.solimp
    and includemargin = mix.includemargin in
    { d with
      ncon = Nx.cast Nx.int32 (Nx.sum !live)
    ; contact_dist = put !dist d.contact_dist
    ; contact_pos = put_cols !pos d.contact_pos
    ; contact_normal = put_cols !normal d.contact_normal
    ; contact_frame = put_cols !frames d.contact_frame
    ; contact_geom = put_cols geom d.contact_geom
    ; contact_body = put_cols body d.contact_body
    ; contact_dim = put dim d.contact_dim
    ; contact_includemargin = put includemargin d.contact_includemargin
    ; contact_friction = put_cols friction d.contact_friction
    ; contact_solref = put_cols solref d.contact_solref
    ; contact_solimp = put_cols solimp d.contact_solimp
    ; contact_jac = put_cols !jac d.contact_jac
    })
