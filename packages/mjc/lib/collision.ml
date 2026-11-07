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

(* [weldparent t b] is the weld root of [b]'s *weld root*'s parent. The
   distinction is the bug the humanoid found: a body with no joint is welded
   into its parent, so a hand welded into a forearm must be compared with the
   upper arm it now touches directly, and asking for the parent of the original
   body instead compares the forearm with itself and lets the pair through. *)
let weldparent (t : Topo.t) b =
  let w = weldbody t b in
  if Int.equal w 0 then 0 else weldbody t t.body_parentid.(w)

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

(* [excluded t b1 b2] is whether the model says these two bodies never touch.
   The comparison is on welded bodies, because that is what MuJoCo builds its
   signature from: a body fused into its parent is the parent. *)
let excluded t b1 b2 =
  let a, b = if Int.( <= ) b1 b2 then b1, b2 else b2, b1 in
  Array.exists t.exclude_pairs ~f:(fun (x, y) -> Int.equal a x && Int.equal b y)

let candidates (m : 'a Model.model) : cand array =
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
          (* An explicit exclusion is the model saying these two never touch,
             and it is the only thing that stops two limbs of one robot from
             colliding with each other. *)
          excluded t weld1 weld2
          || Int.equal weld1 weld2
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

(* A root of a squared length, with a finite derivative where the length is
   zero. [Nx.sqrt]'s own rule divides by the root, so a branch that is masked
   out because its pair is degenerate still sends an infinity into the reverse
   pass, where [0 * inf] is a NaN: the gradient of every contact model whose
   frames are built from a zero normal, which is every model with a dead slot
   in a manifold. The value is [sqrt]'s, at every length the geometry can
   produce; only the derivative at zero differs, and there it is finite. *)
(* The pair formulae take either a single point ([3]) or a batch of them
   ([k; 3]), and the same code has to work for both: the contact stage computes
   one candidate's answer when it is called from a single step and a whole
   geom-pair group's at once when it is compiled. These four helpers are what
   that costs -- a component, a dot, a stack and a constant, each rank-agnostic.
   Without them a per-candidate trace emitted a node per dof per candidate, and
   a model with a few hundred candidates spent minutes compiling. *)
let rank x = Array.length (Nx.shape x)

(* The [i]th component: 0-d for a point, [k] for a batch. *)
let comp i v = if Int.equal (rank v) 1 then Nx.slice [ I i ] v else Nx.slice [ A; I i ] v

(* The dot product over the last axis: 0-d for a point, [k] for a batch. *)
let dp a b = Nx.sum ~axes:[ -1 ] (Nx.mul a b)

(* Three values stacked into the last axis: [3] from scalars, [k; 3] from [k]. *)
let stack3 a b c = Nx.stack ~axis:(-1) [ a; b; c ]

(* A constant shaped like [v], which for a batch is one per candidate. *)
let const_like v x = Nx.mul_s (Nx.ones_like v) x

(* A vector times a scalar that may be one per candidate. [k; 3] against [k]
   does not broadcast -- the trailing axes have to agree -- so the batch case
   gets a trailing axis of its own; a lone scalar, which is what the
   single-point path passes, must not. *)
let trail s = if Int.equal (rank s) 1 then Nx.unsqueeze ~axes:[ -1 ] s else s
let smul v s = Nx.mul v (trail s)
let sdiv v s = Nx.div v (trail s)
let wsel c a b = Nx.where (trail c) a b
let ssqrt = Rigid.safe_sqrt_exact
let len v = ssqrt (dp v v)

(* The same length, clamped away from zero for a division. *)
let len_clamped v = Nx.maximum_s (len v) 1e-300

(* The two z axes coincide, so no direction is defined: MuJoCo takes their
   cross product and falls back to x when even that is zero. *)
let fallback_normal axis1 axis2 =
  let cross = Nx.cross axis1 axis2 in
  let norm = len cross in
  let x = stack3 (const_like norm 1.0) (const_like norm 0.0) (const_like norm 0.0) in
  let v = wsel (Nx.greater_s norm 1e-15) cross x in
  sdiv v (len_clamped v)

let sphere_sphere ~pos1 ~size1 ~pos2 ~size2 ~axis1 ~axis2 =
  let dif = Nx.sub pos1 pos2 in
  let cdist = len dif in
  let dist = Nx.sub (Nx.sub cdist size1) size2 in
  let towards = Nx.sub pos2 pos1 in
  let norm = len towards in
  let normal =
    wsel
      (Nx.greater_s norm 1e-15)
      (sdiv towards (len_clamped towards))
      (fallback_normal axis1 axis2)
  in
  let pos = Nx.add pos1 (smul normal (Nx.add size1 (Nx.mul_s dist 0.5))) in
  dist, pos, normal

let plane_sphere ~pos1 ~normal ~pos2 ~size =
  let cdist = dp (Nx.sub pos2 pos1) normal in
  let dist = Nx.sub cdist size in
  let pos = Nx.add pos2 (smul normal (Nx.neg (Nx.add (Nx.mul_s dist 0.5) size))) in
  dist, pos, normal

(* [mju_makeFrame]: the contact's third row from its normal and whatever
   tangent the pair supplied, orthonormalised. *)
let make_frame ~normal ~tangent =
  let norm v = len_clamped v in
  let n = sdiv normal (norm normal) in
  let degenerate = Nx.less_s (dp tangent tangent) 0.25 in
  let ny = comp 1 n in
  let sc = const_like ny in
  let fallback =
    wsel
      (Nx.logical_and (Nx.less_s ny 0.5) (Nx.greater_s ny (-0.5)))
      (stack3 (sc 0.0) (sc 1.0) (sc 0.0))
      (stack3 (sc 0.0) (sc 0.0) (sc 1.0))
  in
  let y0 = wsel degenerate fallback tangent in
  let y = Nx.sub y0 (smul n (dp n y0)) in
  let y = sdiv y (norm y) in
  let z = Nx.cross n y in
  (* Rows, not columns: appending the stack axis *before* the last one gives
     [3; 3] for a single frame and [k; 3; 3] for a batch. Stacking on the last
     axis would transpose the single-frame case. *)
  Nx.stack ~axis:(-2) [ n; y; z ]

(* [m v] for a matrix and a vector: [3;3] x [3] and [k;3;3] x [k;3] both. *)
let matvec m v = Nx.reshape (Nx.shape v) (Nx.matmul m (Nx.unsqueeze ~axes:[ -1 ] v))

(* [mᵀ v], the inverse rotation of the above. *)
let matvec_t m v =
  Nx.reshape
    (Nx.shape v)
    (Nx.matmul (Nx.matrix_transpose m) (Nx.unsqueeze ~axes:[ -1 ] v))

let plane_capsule ~pos1 ~normal ~pos2 ~axis2 ~r2 ~l2 ~sign =
  let endpoint = Nx.add pos2 (smul axis2 (Nx.mul l2 sign)) in
  plane_sphere ~pos1 ~normal ~pos2:endpoint ~size:r2

let plane_box ~pos1 ~normal ~pos2 ~mat2 ~half ~signs =
  (* One candidate per corner, in MuJoCo's order, with a corner above the plane
     sent far away so that the compaction drops it. *)
  let corner = Nx.add pos2 (matvec mat2 (Nx.mul signs half)) in
  let ldist = dp normal (Nx.sub corner pos2) in
  let centre = dp (Nx.sub pos2 pos1) normal in
  let dist = Nx.add centre ldist in
  let up = Nx.greater_s ldist 0.0 in
  let far = const_like dist 1e10 in
  let dist = Nx.where up far dist in
  let pos = Nx.add corner (smul normal (Nx.mul_s dist (-0.5))) in
  dist, pos, normal

let sphere_box ~pos1 ~r1 ~pos2 ~mat2 ~half =
  (* [mjraw_SphereBox]: the sphere centre in the box frame, clamped to the
     half-sizes; inside the box the nearest face sets the normal. *)
  let centre = matvec_t mat2 (Nx.sub pos1 pos2) in
  let clamped = Nx.minimum (Nx.maximum centre (Nx.neg half)) half in
  let towards = Nx.sub clamped centre in
  let away = len towards in
  let touch = Nx.maximum_s away 1e-300 in
  let direction = sdiv towards touch in
  let deepest = Nx.add centre (smul direction r1) in
  let outside_pos = Nx.mul_s (Nx.add clamped deepest) 0.5 in
  let outside_normal = matvec mat2 direction in
  let outside_dist = Nx.sub away r1 in
  (* Inside: the nearest face, and the sphere centre pushed halfway to it. *)
  let gap_plus = Nx.sub half centre
  and gap_minus = Nx.add half centre in
  let gap = Nx.minimum gap_plus gap_minus in
  let sign =
    Nx.where (Nx.less gap_plus gap_minus) (Nx.neg (Nx.ones_like gap)) (Nx.ones_like gap)
  in
  let g0 = comp 0 gap
  and g1 = comp 1 gap
  and g2 = comp 2 gap in
  (* MuJoCo enumerates the six faces as -x,+x,-y,… and the sign is opposite to
     the face that wins, which is the [sign] above; the axis is the winner. *)
  let pick0 = Nx.logical_and (Nx.less_equal g0 g1) (Nx.less_equal g0 g2) in
  let pick1 = Nx.logical_and (Nx.logical_not pick0) (Nx.less_equal g1 g2) in
  let pick2 = Nx.logical_not (Nx.logical_or pick0 pick1) in
  let onehot =
    stack3
      (Nx.cast Nx.float64 pick0)
      (Nx.cast Nx.float64 pick1)
      (Nx.cast Nx.float64 pick2)
  in
  let nearest = smul onehot sign in
  let closest = Nx.minimum g0 (Nx.minimum g1 g2) in
  let inside_pos = Nx.add centre (smul nearest (Nx.mul_s (Nx.sub r1 closest) 0.5)) in
  let inside_normal = matvec mat2 nearest in
  let inside = Nx.less_equal away (const_like away 1e-15) in
  let pos_local = wsel inside inside_pos outside_pos in
  let normal = wsel inside inside_normal outside_normal in
  let dist = Nx.where inside (Nx.sub (Nx.neg closest) r1) outside_dist in
  let pos = Nx.add (matvec mat2 pos_local) pos2 in
  dist, pos, normal

let sphere_capsule ~pos1 ~r1 ~pos2 ~axis2 ~r2 ~l2 ~axis1 =
  let vec = Nx.sub pos1 pos2 in
  let along = Nx.minimum (Nx.maximum (dp axis2 vec) (Nx.neg l2)) l2 in
  let nearest = Nx.add pos2 (smul axis2 along) in
  sphere_sphere ~pos1 ~size1:r1 ~pos2:nearest ~size2:r2 ~axis1 ~axis2

let capsule_capsule ~pos1 ~axis1 ~r1 ~l1 ~pos2 ~axis2 ~r2 ~l2 ~slot =
  (* [mjraw_CapsuleCapsule]: the closest pair of points on the two axial
     segments, then a sphere-sphere test on them. Nearly parallel segments have
     two candidate pairs and one slot carries each. *)
  let p1 = Nx.sub pos1 (smul axis1 l1)
  and p2 = Nx.add pos1 (smul axis1 l1)
  and q1 = Nx.sub pos2 (smul axis2 l2)
  and q2 = Nx.add pos2 (smul axis2 l2) in
  let ab = Nx.sub p2 p1
  and cd = Nx.sub q2 q1
  and ca = Nx.sub p1 q1 in
  let dot1 = dp ab ab
  and dot2 = dp cd cd
  and dot3 = dp cd ca
  and dot4 = dp ab ca
  and dot5 = dp ab cd in
  let clip x = Nx.minimum (Nx.maximum x (const_like x 0.0)) (const_like x 1.0) in
  let s0 = Nx.neg (Nx.div dot4 dot1)
  and s1 = Nx.div (Nx.sub dot5 dot4) dot1 in
  let cross = Nx.cross ab cd in
  let det = dp cross cross in
  let parallel = Nx.less_equal det (Nx.mul (Nx.mul dot1 dot2) (const_like det 1e-8)) in
  let s = clip (Nx.div (dp cross (Nx.cross cd ca)) (Nx.maximum_s det 1e-300)) in
  let t = Nx.div (Nx.add (Nx.mul dot5 s) dot3) dot2 in
  (* [clampSeg]: slide to an endpoint when the projection leaves the segment. *)
  let seg s_ t_ =
    let a =
      Nx.where (Nx.less_s t_ 0.0) (clip s0) (Nx.where (Nx.greater_s t_ 1.0) (clip s1) s_)
    in
    let b =
      Nx.where
        (Nx.less_s t_ 0.0)
        (const_like t_ 0.0)
        (Nx.where (Nx.greater_s t_ 1.0) (const_like t_ 1.0) t_)
    in
    a, b
  in
  let a0, b0 = seg s t in
  let a1, b1 = seg (const_like t 0.0) (Nx.div dot3 dot2) in
  let a2, b2 = seg (const_like t 1.0) (Nx.div (Nx.add dot5 dot3) dot2) in
  let first = Nx.equal_s slot 0.0 in
  let w_a = Nx.where first (Nx.where parallel a1 a0) a2
  and w_b = Nx.where first (Nx.where parallel b1 b0) b2 in
  let dist, pos, normal =
    sphere_sphere
      ~pos1:(Nx.add p1 (smul ab w_a))
      ~size1:r1
      ~pos2:(Nx.add q1 (smul cd w_b))
      ~size2:r2
      ~axis1
      ~axis2
  in
  let dist = Nx.where first dist (Nx.where parallel dist (const_like dist 1e10)) in
  dist, pos, normal

(* {1 Contact parameters}

   [mj_contactParam]: the higher-priority geom wins outright, otherwise the
   condim is the max, the friction the elementwise max, and the solver
   parameters are mixed with a weight that favours the geom with the larger
   [solmix]. Standard-format references mix; direct-format ones take the min.

   The C engine does this with scalars because it reads a model it owns. A
   traced program may not: a captured parameter is a traced tensor, and only
   tensor operations are allowed on one. So the same arithmetic is written as
   gathers and [where]s, one row per candidate contact. *)

type 'a mixed =
  { dim : 'a Nx.int32_t (** [[ncand]] *)
  ; friction : 'a Nx.float64_t (** [[ncand;5]] *)
  ; solref : 'a Nx.float64_t (** [[ncand;2]] *)
  ; solimp : 'a Nx.float64_t (** [[ncand;5]] *)
  ; includemargin : 'a Nx.float64_t (** [[ncand]] *)
  ; reach : 'a Nx.float64_t
    (** [[ncand]], the bounding radius: the margin plus the two bounds. *)
  }

let mixed_params ~at (m : 'a Model.model) (cands : cand array) : 'a mixed =
  let p = m.params in
  let n = Array.length cands in
  let minval = 1e-15 in
  let sc x = Nx.scalar_at at Nx.float64 x in
  let g side =
    Nx.place
      at
      (Nx.init Nx.int64 [| n |] (fun i ->
         Int64.of_int (if Int.equal side 0 then cands.(i.(0)).g1 else cands.(i.(0)).g2)))
  in
  let g1 = g 0
  and g2 = g 1 in
  let take g x = Nx.take ~axis:0 g x in
  let prio side =
    Nx.place
      at
      (Nx.init Nx.float64 [| n |] (fun i ->
         let c = cands.(i.(0)) in
         Float.of_int m.topo.geom_priority.(if Int.equal side 0 then c.g1 else c.g2)))
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

type 'a box_manifold =
  { row : (int, int) Hashtbl.t (** [g1 * ngeom + g2] to the pair's row in the batch *)
  ; manifold : 'a Collision_box_tensor.manifold
  }

let is_box_pair (t : Topo.t) g1 g2 =
  match t.geom_type.(g1), t.geom_type.(g2) with
  | Box, Box -> true
  | _ -> false

let box_box (m : 'a Model.model) (d : 'a Data.data) (cands : cand array) =
  let t = m.topo in
  let pairs =
    Array.filter cands ~f:(fun c -> is_box_pair t c.g1 c.g2 && Int.equal c.slot 0)
  in
  let n = Array.length pairs in
  if Int.equal n 0
  then None
  else (
    let at = Nx.placement d.qpos in
    let p = m.params in
    let g1 i = pairs.(i).g1
    and g2 i = pairs.(i).g2 in
    let gather (x : _ Nx.t) f =
      let idx =
        Nx.place at (Nx.init Nx.int64 [| n |] (fun i -> Int64.of_int (f i.(0))))
      in
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

let contacts (m : 'a Model.model) (d : 'a Data.data) : 'a Data.data =
  let at = Nx.placement d.qpos in
  let t = m.topo in
  let cands = candidates m in
  let ncand = Array.length cands in
  (* A model with no candidate pair touches no parameter at all, which is most
     of them and keeps their step on the smooth path. *)
  if Int.equal ncand 0
  then { d with ncon = Nx.zeros_at (Nx.placement d.ncon) Nx.int32 [||] }
  else (
    let mix = mixed_params ~at m cands in
    let boxes = box_box m d cands in
    let masks = Jacobian.body_dof_masks ~at t in
    (* No per-candidate loop is left in this stage. What the candidate list
       says -- which geoms, which body, which slot -- is read here on the host,
       where it is model structure, and the arithmetic runs once per geom-type
       pair over a batch of that pair's candidates. That is the difference
       between a program whose size follows the model and one whose size follows
       the candidate count: humanoid has 273 candidates, and the traced version
       did not compile at all. *)
    let g1s = Array.map cands ~f:(fun (c : cand) -> c.g1)
    and g2s = Array.map cands ~f:(fun (c : cand) -> c.g2)
    and slots = Array.map cands ~f:(fun (c : cand) -> c.slot) in
    let idx_of (ix : int array) =
      Nx.place
        at
        (Nx.init Nx.int64 [| Array.length ix |] (fun i -> Int64.of_int ix.(i.(0))))
    in
    let take ix (x : _ Nx.t) = Nx.take ~axis:0 (idx_of ix) x in
    let scatter ix (v : 'a Nx.float64_t) (base : 'a Nx.float64_t) =
      if Int.equal (Array.length (Nx.shape v)) 1
      then Nx.scatter ~combine:Set ~axis:0 (idx_of ix) v base
      else (
        let k = Array.length ix in
        let cols = (Nx.shape v).(1) in
        let ix_rows = Nx.broadcast_to [| k; cols |] (Nx.reshape [| k; 1 |] (idx_of ix)) in
        Nx.scatter ~combine:Set ~axis:0 ix_rows v base)
    in
    let ones_like x = Nx.mul_s (Nx.ones_like x) 1.0
    and zeros_like x = Nx.mul_s (Nx.ones_like x) 0.0 in
    let gpos = d.geom_xpos
    and gsize = m.params.geom_size
    and gmat = Nx.reshape [| t.ngeom; 3; 3 |] d.geom_xmat in
    let zaxis = Nx.slice [ A; A; I 2 ] gmat in
    let size ix k = Nx.take ~axis:0 (idx_of ix) (Nx.slice [ A; I k ] gsize) in
    let scalars ix f =
      Nx.place at (Nx.init Nx.float64 [| Array.length ix |] (fun i -> f ix.(i.(0))))
    in
    (* A candidate that produces nothing is dropped by distance, so the base
       values only have to be finite and far; they are the ones the scalar
       formulation returned for an unsupported pair. *)
    let dist = ref (Nx.zeros_at at Nx.float64 [| ncand |])
    and pos = ref (take g1s gpos)
    and normal = ref (take g1s gpos)
    and tangent = ref (Nx.zeros_at at Nx.float64 [| ncand; 3 |]) in
    (* One geom-type pair's candidates, all at once. Every formula is the same
       function the single-point path used, so this is batching, not a second
       implementation. *)
    let batch code ix =
      let a1 = Array.map ix ~f:(fun i -> g1s.(i))
      and a2 = Array.map ix ~f:(fun i -> g2s.(i)) in
      let p1 = take a1 gpos
      and p2 = take a2 gpos
      and x1 = take a1 zaxis
      and x2 = take a2 zaxis in
      match code with
      | 0 -> plane_sphere ~pos1:p1 ~normal:x1 ~pos2:p2 ~size:(size a2 0)
      | 1 ->
        plane_capsule
          ~pos1:p1
          ~normal:x1
          ~pos2:p2
          ~axis2:x2
          ~r2:(size a2 0)
          ~l2:(size a2 1)
          ~sign:(scalars ix (fun i -> if Int.equal slots.(i) 0 then 1.0 else -1.0))
      | 2 ->
        sphere_sphere
          ~pos1:p1
          ~size1:(size a1 0)
          ~pos2:p2
          ~size2:(size a2 0)
          ~axis1:x1
          ~axis2:x2
      | 3 ->
        let signs =
          Nx.place
            at
            (Nx.init
               Nx.float64
               [| Array.length ix; 3 |]
               (fun i ->
                  let slot = slots.(ix.(i.(0))) in
                  let bit f = if Int.( land ) slot f <> 0 then 1.0 else -1.0 in
                  bit
                    (match i.(1) with
                     | 0 -> 1
                     | 1 -> 2
                     | _ -> 4)))
        in
        plane_box
          ~pos1:p1
          ~normal:x1
          ~pos2:p2
          ~mat2:(take a2 gmat)
          ~half:(take a2 gsize)
          ~signs
      | 4 ->
        capsule_capsule
          ~pos1:p1
          ~axis1:x1
          ~r1:(size a1 0)
          ~l1:(size a1 1)
          ~pos2:p2
          ~axis2:x2
          ~r2:(size a2 0)
          ~l2:(size a2 1)
          ~slot:(scalars ix (fun i -> Float.of_int slots.(i)))
      | 5 ->
        sphere_box
          ~pos1:p1
          ~r1:(size a1 0)
          ~pos2:p2
          ~mat2:(take a2 gmat)
          ~half:(take a2 gsize)
      | _ ->
        sphere_capsule
          ~pos1:p1
          ~r1:(size a1 0)
          ~pos2:p2
          ~axis2:x2
          ~r2:(size a2 0)
          ~l2:(size a2 1)
          ~axis1:x1
    in
    let code_of i =
      match t.geom_type.(g1s.(i)), t.geom_type.(g2s.(i)) with
      | Plane, Sphere -> Some 0
      | Plane, Capsule -> Some 1
      | Sphere, Sphere -> Some 2
      | Plane, Box -> Some 3
      | Capsule, Capsule -> Some 4
      | Sphere, Box -> Some 5
      | Sphere, Capsule -> Some 6
      | _ -> None
    in
    let groups = Array.create ~len:7 [] in
    let box_ix = ref [] in
    Array.iteri g1s ~f:(fun i _ ->
      match code_of i with
      | Some c -> groups.(c) <- i :: groups.(c)
      | None -> if is_box_pair t g1s.(i) g2s.(i) then box_ix := i :: !box_ix);
    Array.iteri groups ~f:(fun code group ->
      match List.rev group with
      | [] -> ()
      | group ->
        let ix = Array.of_list group in
        let dist_i, pos_i, normal_i = batch code ix in
        dist := scatter ix dist_i !dist;
        pos := scatter ix pos_i !pos;
        normal := scatter ix normal_i !normal;
        if Int.equal code 1
        then
          tangent := scatter ix (take (Array.map ix ~f:(fun i -> g2s.(i))) zaxis) !tangent);
    (* Box pairs come from the manifold rather than a formula: one gather of
       the [row, slot] each candidate names, and a dead slot sent far away. *)
    (match boxes with
     | Some bm when not (List.is_empty !box_ix) ->
       let ix = Array.of_list (List.rev !box_ix) in
       let k = Array.length ix in
       let mf = bm.manifold in
       let maxc = Collision_box_tensor.max_contacts in
       let flat =
         Array.map ix ~f:(fun i ->
           let c = cands.(i) in
           (Hashtbl.find_exn bm.row ((c.g1 * t.ngeom) + c.g2) * maxc) + c.slot)
       in
       let live_f = Nx.reshape [| -1 |] mf.Collision_box_tensor.live
       and dist_f = Nx.reshape [| -1 |] mf.dist
       and pos_f = Nx.reshape [| -1; 3 |] mf.pos
       and normal_f = Nx.reshape [| -1; 3 |] mf.normal in
       let dead = Nx.equal_s (take flat live_f) 0.0 in
       dist
       := scatter
            ix
            (Nx.where dead (Nx.scalar_at at Nx.float64 1e30) (take flat dist_f))
            !dist;
       pos
       := scatter
            ix
            (wsel dead (Nx.zeros_at at Nx.float64 [| k; 3 |]) (take flat pos_f))
            !pos;
       normal := scatter ix (take flat normal_f) !normal
     | _ -> ());
    (* MuJoCo keeps at most four contacts per pair, in corner order. Candidates
       for a pair are contiguous, so the count of eligible candidates before
       this one within its own run -- a prefix count less the count at the run's
       start -- is the same running total, without the loop. *)
    let start = Array.create ~len:ncand 0
    and limit = Array.create ~len:ncand 0.0 in
    let run = ref 0 in
    Array.iteri cands ~f:(fun i (c : cand) ->
      if i > 0 && not (Int.equal c.g1 g1s.(i - 1) && Int.equal c.g2 g2s.(i - 1))
      then run := i;
      start.(i) <- !run;
      limit.(i) <- Float.of_int (pair_limit t c.g1 c.g2));
    let plane =
      Array.map cands ~f:(fun (c : cand) ->
        Poly.( = ) t.geom_type.(c.g1) Plane || Poly.( = ) t.geom_type.(c.g2) Plane)
    in
    let plane_t =
      scalars (Array.init ncand ~f:Fn.id) (fun i -> if plane.(i) then 1.0 else 0.0)
    in
    let dif = Nx.sub (take g2s gpos) (take g1s gpos) in
    let reach = mix.reach in
    let bound =
      Nx.where
        (Nx.greater_s plane_t 0.5)
        (ones_like plane_t)
        (Nx.where
           (Nx.less_equal (Nx.sum ~axes:[ -1 ] (Nx.mul dif dif)) (Nx.mul reach reach))
           (ones_like plane_t)
           (zeros_like plane_t))
    in
    let eligible =
      Nx.mul
        bound
        (Nx.where
           (Nx.less_equal !dist mix.includemargin)
           (ones_like bound)
           (zeros_like bound))
    in
    let cum = Nx.sub (Nx.cumsum ~axis:0 eligible) eligible in
    let rank = Nx.sub cum (take start cum) in
    let room = Nx.less rank (scalars (Array.init ncand ~f:Fn.id) (fun i -> limit.(i))) in
    let live = ref (Nx.where room eligible (zeros_like eligible)) in
    (* The frame's three rows, each dotted with the Jacobian difference: geom
       2's point Jacobian minus geom 1's. The normal row leads; a pyramidal
       friction pair is built from it and one tangent. *)
    let frame = make_frame ~normal:!normal ~tangent:!tangent in
    let bodies_of f = Array.map cands ~f:(fun (c : cand) -> f c) in
    let jacp1, _ =
      Jacobian.points
        ~masks
        m
        d
        ~bodies:(bodies_of (fun c -> t.geom_bodyid.(c.g1)))
        ~positions:!pos
    and jacp2, _ =
      Jacobian.points
        ~masks
        m
        d
        ~bodies:(bodies_of (fun c -> t.geom_bodyid.(c.g2)))
        ~positions:!pos
    in
    let jac =
      Nx.matmul (Nx.sub jacp2 jacp1) (Nx.transpose ~axes:[ 0; 2; 1 ] frame)
      |> fun x ->
      Nx.transpose ~axes:[ 0; 2; 1 ] x |> fun x -> Nx.reshape [| ncand; 3 * t.nv |] x
    in
    let frames = Nx.reshape [| ncand; 9 |] frame in
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
      Nx.place
        at
        (Nx.init Nx.int32 [| ncand; 2 |] (fun i ->
           let c = cands.(i.(0)) in
           Int32.of_int (if Int.equal i.(1) 0 then c.g1 else c.g2)))
    in
    let body =
      Nx.place
        at
        (Nx.init Nx.int32 [| ncand; 2 |] (fun i ->
           let c = cands.(i.(0)) in
           Int32.of_int
             (if Int.equal i.(1) 0 then t.geom_bodyid.(c.g1) else t.geom_bodyid.(c.g2))))
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
    ; contact_frame = put_cols frames d.contact_frame
    ; contact_geom = put_cols geom d.contact_geom
    ; contact_body = put_cols body d.contact_body
    ; contact_dim = put dim d.contact_dim
    ; contact_includemargin = put includemargin d.contact_includemargin
    ; contact_friction = put_cols friction d.contact_friction
    ; contact_solref = put_cols solref d.contact_solref
    ; contact_solimp = put_cols solimp d.contact_solimp
    ; contact_jac = put_cols jac d.contact_jac
    })
