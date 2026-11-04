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
  | Plane, Sphere | Plane, Capsule | Sphere, Sphere | Sphere, Capsule -> true
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
   Every input is a model constant, so this is host arithmetic. *)

type params =
  { dim : int
  ; friction : float array (** [[5]], MuJoCo's slot layout *)
  ; solref : float array (** [[2]] *)
  ; solimp : float array (** [[5]] *)
  ; includemargin : float
  }

(* The contact's five friction slots from the three a geom carries: MuJoCo
   duplicates the sliding value into the torsional slot and the rolling value
   into the last two, so that every condim has an entry per tangential axis. *)
let unpack (mu : float array) = [| mu.(0); mu.(0); mu.(1); mu.(2); mu.(2) |]

let mix_weight s1 s2 =
  let minval = 1e-15 in
  if Float.(s1 >= minval && s2 >= minval)
  then Float.(s1 / (s1 + s2))
  else if Float.(s1 < minval && s2 < minval)
  then 0.5
  else if Float.(s1 < minval)
  then 0.0
  else 1.0

let contact_params (m : Model.host) (c : cand) : params =
  let p = m.params in
  let host x = Nx.to_array x in
  let margin = host p.geom_margin
  and gap = host p.geom_gap
  and condim = host p.geom_condim
  and priority = Array.map m.topo.geom_priority ~f:Float.of_int in
  ignore priority;
  let friction = host p.geom_friction
  and solref = host p.geom_solref
  and solimp = host p.geom_solimp
  and solmix = host p.geom_solmix in
  let g1 = c.g1
  and g2 = c.g2 in
  let take n at = Array.init n ~f:(fun k -> at k) in
  match Int.compare (Int.of_float priority.(g1)) (Int.of_float priority.(g2)) with
  | 0 ->
    let mix = mix_weight solmix.(g1) solmix.(g2) in
    let width = 2 in
    let standard =
      Float.(solref.(Int.(width * g1)) > 0.0 && solref.(Int.(width * g2)) > 0.0)
    in
    let ref k =
      let a = solref.(Int.((width * g1) + k))
      and b = solref.(Int.((width * g2) + k)) in
      if standard then Float.((mix * a) + ((1.0 - mix) * b)) else Float.min a b
    in
    let imp k =
      let a = solimp.(Int.((5 * g1) + k))
      and b = solimp.(Int.((5 * g2) + k)) in
      Float.((mix * a) + ((1.0 - mix) * b))
    in
    let mu k = Float.max friction.(Int.((3 * g1) + k)) friction.(Int.((3 * g2) + k)) in
    let mu = take 3 mu in
    { dim = Int.max (Int.of_float condim.(g1)) (Int.of_float condim.(g2))
    ; friction = unpack mu
    ; solref = take width ref
    ; solimp = take 5 imp
    ; includemargin = Float.(margin.(g1) + margin.(g2) + gap.(g1) + gap.(g2))
    }
  | n ->
    let g = if Int.( > ) n 0 then g1 else g2 in
    { dim = Int.of_float condim.(g)
    ; friction = unpack (take 3 (fun k -> friction.(Int.((3 * g) + k))))
    ; solref = take 2 (fun k -> solref.(Int.((2 * g) + k)))
    ; solimp = take 5 (fun k -> solimp.(Int.((5 * g) + k)))
    ; includemargin = Float.(margin.(g1) + margin.(g2) + gap.(g1) + gap.(g2))
    }

(* {1 Contacts}

   The candidate list is compacted the way the limit rows are: which contacts
   exist is a tensor predicate, the destination index is the number of live
   contacts before it, and a dead candidate is sent past the buffer where a
   scatter writes nothing. *)

let contacts (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  let cands = candidates m in
  let ncand = Array.length cands in
  (* A model with no candidate pair touches no parameter at all, which is most
     of them and keeps their step on the smooth path. *)
  if Int.equal ncand 0
  then { d with ncon = Nx.zeros Nx.int32 [||] }
  else (
    let rbound = Nx.to_array p.geom_rbound in
    let params = Array.map cands ~f:(contact_params m) in
    let dist = ref (Nx.zeros Nx.float64 [| ncand |])
    and pos = ref (Nx.zeros Nx.float64 [| ncand; 3 |])
    and normal = ref (Nx.zeros Nx.float64 [| ncand; 3 |])
    and live = ref (Nx.zeros Nx.float64 [| ncand |])
    and jac = ref (Nx.zeros Nx.float64 [| ncand; t.nv |]) in
    Array.iteri cands ~f:(fun i c ->
      let dist_i, pos_i, normal_i = geometry m d c in
      let prm = params.(i) in
      let plane =
        Poly.( = ) t.geom_type.(c.g1) Plane || Poly.( = ) t.geom_type.(c.g2) Plane
      in
      let bound =
        if plane
        then Nx.scalar Nx.float64 1.0
        else (
          let reach = Float.(prm.includemargin + rbound.(c.g1) + rbound.(c.g2)) in
          let dif = Nx.sub (geom_pos d c.g1) (geom_pos d c.g2) in
          Nx.where
            (Nx.less_equal (Nx.dot dif dif) (Nx.scalar Nx.float64 Float.(reach * reach)))
            (Nx.scalar Nx.float64 1.0)
            (Nx.scalar Nx.float64 0.0))
      in
      let near = Nx.less_equal dist_i (Nx.scalar Nx.float64 prm.includemargin) in
      let on =
        Nx.mul bound (Nx.where near (Nx.scalar Nx.float64 1.0) (Nx.scalar Nx.float64 0.0))
      in
      (* The normal row of a contact: the normal dotted with the difference of
         the two bodies' point Jacobians, geom 2's minus geom 1's. *)
      let b1 = t.geom_bodyid.(c.g1)
      and b2 = t.geom_bodyid.(c.g2) in
      let row_of body =
        let jacp, _ = Jacobian.point m d ~body ~pos:pos_i in
        Nx.reshape [| t.nv |] (Nx.matmul jacp (Nx.reshape [| 3; 1 |] normal_i))
      in
      jac := Nx.set [ I i ] (Nx.sub (row_of b2) (row_of b1)) !jac;
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
    let dim = Nx.init Nx.int32 [| ncand |] (fun i -> Int32.of_int params.(i.(0)).dim) in
    let friction =
      Nx.init Nx.float64 [| ncand; 5 |] (fun i -> params.(i.(0)).friction.(i.(1)))
    in
    let solref =
      Nx.init Nx.float64 [| ncand; 2 |] (fun i -> params.(i.(0)).solref.(i.(1)))
    in
    let solimp =
      Nx.init Nx.float64 [| ncand; 5 |] (fun i -> params.(i.(0)).solimp.(i.(1)))
    in
    let includemargin =
      Nx.init Nx.float64 [| ncand |] (fun i -> params.(i.(0)).includemargin)
    in
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
