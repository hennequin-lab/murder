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
  | Plane, Sphere | Plane, Capsule | Sphere, Sphere | Sphere, Capsule | Plane, Box -> true
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
  | _ -> 1

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
    let row_of k x = Nx.slice [ I k ] x in
    let tangent (t : Topo.t) d (c : cand) =
      match t.geom_type.(c.g1), t.geom_type.(c.g2) with
      | Plane, Capsule -> geom_zaxis d c.g2
      | _ -> Nx.zeros Nx.float64 [| 3 |]
    in
    let dist = ref (Nx.zeros Nx.float64 [| ncand |])
    and pos = ref (Nx.zeros Nx.float64 [| ncand; 3 |])
    and normal = ref (Nx.zeros Nx.float64 [| ncand; 3 |])
    and live = ref (Nx.zeros Nx.float64 [| ncand |])
    and jac = ref (Nx.zeros Nx.float64 [| ncand; 3 * t.nv |]) in
    (* MuJoCo keeps at most four contacts per pair, taking them in corner
       order. The candidate loop runs in that order, so a running count per
       pair does the same cut. *)
    let seen = ref (Nx.scalar Nx.float64 0.0)
    and pair = ref (-1, -1) in
    Array.iteri cands ~f:(fun i c ->
      let dist_i, pos_i, normal_i = geometry m d c in
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
          (Nx.less_s !seen 4.0)
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
    ; contact_geom = put_cols geom d.contact_geom
    ; contact_body = put_cols body d.contact_body
    ; contact_dim = put dim d.contact_dim
    ; contact_includemargin = put includemargin d.contact_includemargin
    ; contact_friction = put_cols friction d.contact_friction
    ; contact_solref = put_cols solref d.contact_solref
    ; contact_solimp = put_cols solimp d.contact_solimp
    ; contact_jac = put_cols !jac d.contact_jac
    })
