(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx

(* The constants are {!Mjc.Collision_box}'s, not this module's: they decide
   which of two nearly-equal axes wins, and a stack notices. *)
let sep_eps = Collision_box.sep_eps
let par_eps = Collision_box.par_eps
let dupe_eps = Collision_box.dupe_eps
let sgn_eps = Collision_box.sgn_eps
let edge_bias = Collision_box.edge_bias
let max_contacts = Collision_box.max_verts

type batch =
  { pos1 : Nx.host Nx.float64_t
  ; mat1 : Nx.host Nx.float64_t
  ; size1 : Nx.host Nx.float64_t
  ; pos2 : Nx.host Nx.float64_t
  ; mat2 : Nx.host Nx.float64_t
  ; size2 : Nx.host Nx.float64_t
  ; margin : Nx.host Nx.float64_t
  }

type sat =
  { code : Nx.host Nx.float64_t
  ; sep : Nx.host Nx.float64_t
  ; separated : Nx.host Nx.float64_t
  }

type manifold =
  { dist : Nx.host Nx.float64_t
  ; pos : Nx.host Nx.float64_t
  ; normal : Nx.host Nx.float64_t
  ; live : Nx.host Nx.float64_t
  }

(* {1 Shape helpers}

   Every tensor is batched over pairs in axis 0, so a [3]-vector is [[n;3]] and
   a matrix [[n;9]] is reshaped to [[n;3;3]] where the maths reads better. *)

let f64 = Nx.float64

let m3 i =
  let r = i - (3 * (i / 3)) in
  if r < 0 then r + 3 else r

let _sc x = Nx.scalar f64 x
let big = 1e30
let n_of (b : batch) = (Nx.shape b.pos1).(0)
let mat3 (m : _ Nx.t) = Nx.reshape [| (Nx.shape m).(0); 3; 3 |] m
let row m i = Nx.slice [ A; I i ] m

(* component [i] of every pair's [3]-vector *)
let comp (v : _ Nx.t) i = Nx.slice [ A; I i ] v

(* column [c] of the last two axes of a [n;3;3] tensor, as [n;3] *)
let col m c = Nx.slice [ A; A; I c ] m
let dot (a : _ Nx.t) (b : _ Nx.t) = Nx.sum ~axes:[ 1 ] (Nx.mul a b)

(* [first_of] is the index of the first element equal to [best] along axis 1 of
   a [n;k] tensor, which is what "first axis wins a tie" means. *)
let first_of (x : _ Nx.t) (best : _ Nx.t) =
  let n = (Nx.shape x).(0)
  and k = (Nx.shape x).(1) in
  let idx = Nx.init f64 [| k |] (fun i -> Float.of_int i.(0)) in
  let idx = Nx.broadcast_to [| n; k |] (Nx.reshape [| 1; k |] idx) in
  let hit = Nx.equal x (Nx.broadcast_to [| n; k |] (Nx.reshape [| n; 1 |] best)) in
  Nx.min ~axes:[ 1 ] (Nx.where hit idx (Nx.full_like idx big))

(* {1 Stage 1: separating axes} *)

type frame =
  { rot : Nx.host Nx.float64_t (** [[n;3;3]]: box 2's axes in box 1's frame. *)
  ; rotabs : Nx.host Nx.float64_t
  ; pos21 : Nx.host Nx.float64_t (** [[n;3]]: box 2 in box 1's frame. *)
  ; pos12 : Nx.host Nx.float64_t
  }

let frame (b : batch) =
  let rot = Nx.matmul (Nx.matrix_transpose (mat3 b.mat1)) (mat3 b.mat2) in
  let rotabs = Nx.abs rot in
  let to_local mat d =
    Nx.reshape
      [| n_of b; 3 |]
      (Nx.matmul (Nx.matrix_transpose (mat3 mat)) (Nx.reshape [| n_of b; 3; 1 |] d))
  in
  { rot
  ; rotabs
  ; pos21 = to_local b.mat1 (Nx.sub b.pos2 b.pos1)
  ; pos12 = to_local b.mat2 (Nx.sub b.pos1 b.pos2)
  }

let septol (b : batch) =
  Nx.add b.margin (Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.add b.size1 b.size2)) sep_eps)

(* The fifteen candidate separations, [n;15]: six face axes then nine edge
   crosses. A parallel edge cross is rounding noise and gets [-inf], so it can
   never win and never separates. *)
let separations (b : batch) (fr : frame) =
  let n = n_of b in
  let face1 i =
    let radius2 = dot (row fr.rotabs i) b.size2 in
    Nx.sub (Nx.sub (Nx.abs (comp fr.pos21 i)) (comp b.size1 i)) radius2
  and face2 j =
    let radius1 = dot (col fr.rotabs j) b.size1 in
    Nx.sub (Nx.sub (Nx.abs (comp fr.pos12 j)) (comp b.size2 j)) radius1
  in
  let edge i j =
    let i1 = m3 (i + 1)
    and i2 = m3 (i + 2) in
    let j1 = m3 (j + 1)
    and j2 = m3 (j + 2) in
    let ax1 = Nx.neg (Nx.slice [ A; I i2; I j ] fr.rot)
    and ax2 = Nx.slice [ A; I i1; I j ] fr.rot in
    let norm2 = Nx.add (Nx.mul ax1 ax1) (Nx.mul ax2 ax2) in
    let safe = Nx.greater_equal_s norm2 par_eps in
    let inv = Nx.div (Nx.ones_like norm2) (Nx.sqrt (Nx.maximum_s norm2 par_eps)) in
    let ax1 = Nx.mul ax1 inv
    and ax2 = Nx.mul ax2 inv in
    let radius1 =
      Nx.add
        (Nx.mul (comp b.size1 i1) (Nx.abs ax1))
        (Nx.mul (comp b.size1 i2) (Nx.abs ax2))
    in
    let a2_1 =
      Nx.add
        (Nx.mul ax1 (Nx.slice [ A; I i1; I j1 ] fr.rot))
        (Nx.mul ax2 (Nx.slice [ A; I i2; I j1 ] fr.rot))
    and a2_2 =
      Nx.add
        (Nx.mul ax1 (Nx.slice [ A; I i1; I j2 ] fr.rot))
        (Nx.mul ax2 (Nx.slice [ A; I i2; I j2 ] fr.rot))
    in
    let radius2 =
      Nx.add
        (Nx.mul (comp b.size2 j1) (Nx.abs a2_1))
        (Nx.mul (comp b.size2 j2) (Nx.abs a2_2))
    in
    let sep =
      Nx.sub
        (Nx.sub
           (Nx.abs
              (Nx.add (Nx.mul ax1 (comp fr.pos21 i1)) (Nx.mul ax2 (comp fr.pos21 i2))))
           radius1)
        radius2
    in
    (* the axis direction, which stage 2 needs when this one wins *)
    let axis = Nx.zeros f64 [| n; 3 |] in
    let axis = Nx.set [ A; I i1 ] ax1 axis in
    let axis = Nx.set [ A; I i2 ] ax2 axis in
    sep, Nx.where safe sep (Nx.full_like sep (-.big)), axis
  in
  let faces = [ face1 0; face1 1; face1 2; face2 0; face2 1; face2 2 ] in
  let edges =
    List.concat_map (List.range 0 3) ~f:(fun i -> List.map (List.range 0 3) ~f:(edge i))
  in
  let edge_seps = List.map edges ~f:(fun (s, _, _) -> s)
  and edge_masked = List.map edges ~f:(fun (_, s, _) -> s)
  and edge_axes = List.map edges ~f:(fun (_, _, a) -> a) in
  ( Nx.stack ~axis:1 faces
  , Nx.stack ~axis:1 edge_seps
  , Nx.stack ~axis:1 edge_masked
  , Nx.stack ~axis:1 edge_axes )

let sat (b : batch) =
  let fr = frame b in
  let tol = septol b in
  let faces, edge_seps, edge_masked, edge_axes = separations b fr in
  let separated =
    Nx.logical_or
      (Nx.greater (Nx.max ~axes:[ 1 ] faces) tol)
      (Nx.greater (Nx.max ~axes:[ 1 ] edge_seps) tol)
  in
  let as_float t =
    Nx.where
      t
      (Nx.ones_like (Nx.max ~axes:[ 1 ] faces))
      (Nx.zeros_like (Nx.max ~axes:[ 1 ] faces))
  in
  let separated_f = as_float separated in
  let sep_face = Nx.max ~axes:[ 1 ] faces in
  let code_face = first_of faces sep_face in
  (* an edge axis must beat the best face axis by the bias, and beat it at all *)
  let biased = Nx.sub edge_masked (Nx.mul_s (Nx.abs edge_masked) edge_bias) in
  let face2 = Nx.reshape [| n_of b; 1 |] sep_face in
  let ok = Nx.logical_and (Nx.greater biased face2) (Nx.greater edge_masked face2) in
  let masked = Nx.where ok edge_masked (Nx.full_like edge_masked (-.big)) in
  let edge_best = Nx.max ~axes:[ 1 ] masked in
  let has_edge = Nx.greater_s edge_best (-.big) in
  let code_edge = Nx.add_s (first_of masked edge_best) 6. in
  let code = Nx.where has_edge code_edge code_face
  and sep = Nx.where has_edge edge_best sep_face in
  (* Face substitution: an edge within eight degrees of the best face axis
     duplicates it, and the face's manifold is better. *)
  let edge_axis =
    (* the winning edge's direction, gathered from the nine *)
    (* a tensor index replaces the axis with its own axes, so a [n] index picks
       one row per pair *)
    let idx =
      Nx.broadcast_to
        [| n_of b; 1; 3 |]
        (Nx.reshape [| n_of b; 1; 1 |] (Nx.cast Nx.int64 (Nx.sub_s code_edge 6.)))
    in
    Nx.reshape [| n_of b; 3 |] (Nx.take_along_axis ~axis:1 idx edge_axes)
  in
  let edge_norm =
    Nx.maximum_s (Nx.sqrt (Nx.sum ~axes:[ 1 ] (Nx.mul edge_axis edge_axis))) 1e-300
  in
  let edge_axis = Nx.div edge_axis (Nx.reshape [| n_of b; 1 |] edge_norm) in
  (* |axis . face normal|: the normal of the winning face, in box 1's frame, is
     a coordinate axis for codes 0..2 and a column of rot for codes 3..5 *)
  let unit_vec i =
    let v = Nx.zeros f64 [| n_of b; 3 |] in
    Nx.set [ A; I i ] (Nx.ones f64 [| n_of b |]) v
  in
  let normals =
    Nx.stack
      ~axis:1
      [ unit_vec 0; unit_vec 1; unit_vec 2; col fr.rot 0; col fr.rot 1; col fr.rot 2 ]
  in
  let face_normal =
    Nx.reshape
      [| n_of b; 3 |]
      (Nx.take_along_axis
         ~axis:1
         (Nx.broadcast_to
            [| n_of b; 1; 3 |]
            (Nx.reshape [| n_of b; 1; 1 |] (Nx.cast Nx.int64 code_face)))
         normals)
  in
  let face_dot = Nx.abs (dot edge_axis face_normal) in
  let substitute =
    Nx.logical_and
      (Nx.greater_s face_dot 0.99)
      (Nx.less
         sep
         (Nx.add
            (Nx.add sep_face (Nx.mul_s (Nx.abs sep_face) 0.05))
            (Nx.scalar f64 1e-15)))
  in
  let substitute = Nx.logical_and substitute has_edge in
  let code = Nx.where substitute code_face code
  and sep = Nx.where substitute sep_face sep in
  { code; sep; separated = separated_f }

(* Temporary: the intermediate quantities, so a failing pair can be looked at
   without guessing. *)
let debug (b : batch) =
  let fr = frame b in
  let tol = septol b in
  let faces, edge_seps, _, _ = separations b fr in
  Nx.max ~axes:[ 1 ] faces, Nx.max ~axes:[ 1 ] edge_seps, tol

(* {1 Stage 2a: the edge-edge contact}

   One contact per pair, from the closest approach of the two supporting edge
   segments. When the separating axis has a near-zero component in a box's
   frame, the supporting corner is ambiguous and the C engine enumerates both
   signs, keeping the closest witness pair; the four corners are computed here
   and the closest is chosen elementwise, which is the same decision. *)

let take_comp (v : _ Nx.t) (idx : _ Nx.t) =
  (* index the last axis of [v] with a [n]-shaped index *)
  Nx.reshape
    [| (Nx.shape v).(0) |]
    (Nx.take_along_axis ~axis:1 (Nx.reshape [| (Nx.shape v).(0); 1 |] idx) v)

let take_col (m : _ Nx.t) (idx : _ Nx.t) =
  let n = (Nx.shape m).(0) in
  Nx.reshape
    [| n; 3 |]
    (Nx.take_along_axis
       ~axis:1
       (Nx.broadcast_to [| n; 1; 3 |] (Nx.reshape [| n; 1; 1 |] idx))
       m)

type triple =
  { dist : Nx.host Nx.float64_t
  ; pos : Nx.host Nx.float64_t
  ; normal : Nx.host Nx.float64_t
  }

(* [edges b fr axes]: for each of the nine edge axes, the contact it makes. *)
let edges (b : batch) (fr : frame) (axes : _ Nx.t) =
  let n = n_of b in
  (* d2: box 2's edge direction for axis (i, j), which is column j of rot *)
  let column j =
    Nx.stack
      ~axis:1
      [ Nx.slice [ A; I 0; I j ] fr.rot
      ; Nx.slice [ A; I 1; I j ] fr.rot
      ; Nx.slice [ A; I 2; I j ] fr.rot
      ]
  in
  let one k (i, j) =
    let i1 = m3 (i + 1)
    and i2 = m3 (i + 2)
    and j1 = m3 (j + 1)
    and j2 = m3 (j + 2) in
    let axis = take_col axes (Nx.full Nx.int64 [| n |] (Int64.of_int k)) in
    (* oriented from box 1 towards box 2 *)
    let towards = Nx.sum ~axes:[ 1 ] (Nx.mul axis fr.pos21) in
    let orient =
      Nx.where (Nx.less_s towards 0.) (Nx.full_like towards (-1.)) (Nx.ones_like towards)
    in
    let axis = Nx.mul axis (Nx.reshape [| n; 1 |] orient) in
    let a2 =
      Nx.reshape
        [| n; 3 |]
        (Nx.matmul (Nx.matrix_transpose fr.rot) (Nx.reshape [| n; 3; 1 |] axis))
    in
    let d2 = column j in
    let bdot = take_comp d2 (Nx.full Nx.int64 [| n |] (Int64.of_int i)) in
    let denom = Nx.sub (Nx.ones_like bdot) (Nx.mul bdot bdot) in
    let x1 = take_comp axis (Nx.full Nx.int64 [| n |] (Int64.of_int i1))
    and x2 = take_comp axis (Nx.full Nx.int64 [| n |] (Int64.of_int i2))
    and y1 = take_comp a2 (Nx.full Nx.int64 [| n |] (Int64.of_int j1))
    and y2 = take_comp a2 (Nx.full Nx.int64 [| n |] (Int64.of_int j2)) in
    (* which component, if any, is ambiguous *)
    let nothing = Nx.full_like x1 (-1.)
    and is_i1 = Nx.full_like x1 (Float.of_int i1)
    and is_i2 = Nx.full_like x1 (Float.of_int i2)
    and is_j1 = Nx.full_like x1 (Float.of_int j1)
    and is_j2 = Nx.full_like x1 (Float.of_int j2) in
    let amb1 =
      Nx.where
        (Nx.less_s (Nx.abs x1) sgn_eps)
        is_i1
        (Nx.where (Nx.less_s (Nx.abs x2) sgn_eps) is_i2 nothing)
    and amb2 =
      Nx.where
        (Nx.less_s (Nx.abs y1) sgn_eps)
        is_j1
        (Nx.where (Nx.less_s (Nx.abs y2) sgn_eps) is_j2 nothing)
    in
    let size1 comp = take_comp b.size1 (Nx.full Nx.int64 [| n |] (Int64.of_int comp))
    and size2 comp = take_comp b.size2 (Nx.full Nx.int64 [| n |] (Int64.of_int comp)) in
    let s1_i1 = size1 i1
    and s1_i2 = size1 i2
    and s2_j1 = size2 j1
    and s2_j2 = size2 j2 in
    let corner1 flip =
      (* support along +axis; [flip] negates a component chosen by amb1 *)
      let c = Nx.zeros f64 [| n; 3 |] in
      let supp v s =
        Nx.mul
          (Nx.where (Nx.greater_equal_s v 0.) (Nx.ones_like v) (Nx.full_like v (-1.)))
          s
      in
      let c = Nx.set [ A; I i1 ] (supp x1 s1_i1) c in
      let c = Nx.set [ A; I i2 ] (supp x2 s1_i2) c in
      if not flip
      then c
      else (
        let at k = Nx.set [ A; I k ] (Nx.neg (Nx.slice [ A; I k ] c)) c in
        (* choose per pair which component the ambiguity flips *)
        let c_i1 = at i1
        and c_i2 = at i2 in
        let cond k = Nx.reshape [| n; 1 |] (Nx.equal_s amb1 (Float.of_int k)) in
        Nx.where (cond i1) c_i1 (Nx.where (cond i2) c_i2 c))
    in
    let corner2 flip =
      let cc = Nx.zeros f64 [| n; 3 |] in
      let supp v s =
        Nx.mul
          (Nx.where (Nx.greater_equal_s v 0.) (Nx.full_like v (-1.)) (Nx.ones_like v))
          s
      in
      let cc = Nx.set [ A; I j1 ] (supp y1 s2_j1) cc in
      let cc = Nx.set [ A; I j2 ] (supp y2 s2_j2) cc in
      if not flip
      then cc
      else (
        let at k = Nx.set [ A; I k ] (Nx.neg (Nx.slice [ A; I k ] cc)) cc in
        let c_j1 = at j1
        and c_j2 = at j2 in
        let cond k = Nx.reshape [| n; 1 |] (Nx.equal_s amb2 (Float.of_int k)) in
        Nx.where (cond j1) c_j1 (Nx.where (cond j2) c_j2 cc))
    in
    let variant flip1 flip2 =
      let c1 = corner1 flip1 in
      let c2 =
        Nx.add
          (Nx.reshape
             [| n; 3 |]
             (Nx.matmul fr.rot (Nx.reshape [| n; 3; 1 |] (corner2 flip2))))
          fr.pos21
      in
      let e = Nx.sub c2 c1 in
      let d1e = take_comp e (Nx.full Nx.int64 [| n |] (Int64.of_int i))
      and d2e = Nx.sum ~axes:[ 1 ] (Nx.mul d2 e) in
      let s =
        Nx.where
          (Nx.less_s denom 1e-14)
          (Nx.zeros_like denom)
          (Nx.div (Nx.sub d1e (Nx.mul bdot d2e)) denom)
      in
      (* the clamps are per component i, whose size is size1[i] *)
      let s_lim = size1 i in
      let s = Nx.minimum (Nx.maximum s (Nx.neg s_lim)) s_lim in
      let t_lim =
        (* clamp into the box 2 edge, which runs along j *)
        let v = Nx.sub (Nx.mul bdot s) d2e in
        let lim = size2 j in
        Nx.minimum (Nx.maximum v (Nx.neg lim)) lim
      in
      let s =
        Nx.minimum (Nx.maximum (Nx.add d1e (Nx.mul bdot t_lim)) (Nx.neg s_lim)) s_lim
      in
      let p1 = Nx.set [ A; I i ] (Nx.add (Nx.slice [ A; I i ] c1) s) c1 in
      let p2 = Nx.add c2 (Nx.mul d2 (Nx.reshape [| n; 1 |] t_lim)) in
      let gap = Nx.sub p2 p1 in
      let gap2 = Nx.sum ~axes:[ 1 ] (Nx.mul gap gap) in
      gap2, p1, p2
    in
    let g00, w1_00, w2_00 = variant false false in
    let g10, w1_10, w2_10 = variant true false in
    let g01, w1_01, w2_01 = variant false true in
    let g11, w1_11, w2_11 = variant true true in
    let pick gx wx wxp gcur wcur wcurp =
      let better = Nx.less gx gcur in
      ( Nx.where better gx gcur
      , Nx.where (Nx.reshape [| n; 1 |] better) wx wcur
      , Nx.where (Nx.reshape [| n; 1 |] better) wxp wcurp )
    in
    let _, w1, w2 = pick g10 w1_10 w2_10 g00 w1_00 w2_00 in
    let _, w1, w2 = pick g01 w1_01 w2_01 (Nx.minimum g10 g00) w1 w2 in
    let _, w1, w2 =
      let g = Nx.minimum (Nx.minimum g10 g00) g01 in
      pick g11 w1_11 w2_11 g w1 w2
    in
    let gap = Nx.sub w2 w1 in
    let dist = Nx.sum ~axes:[ 1 ] (Nx.mul gap axis) in
    let mid = Nx.mul_s (Nx.add w1 w2) 0.5 in
    let pos =
      Nx.add
        (Nx.reshape [| n; 3 |] (Nx.matmul (mat3 b.mat1) (Nx.reshape [| n; 3; 1 |] mid)))
        b.pos1
    in
    let normal =
      Nx.reshape [| n; 3 |] (Nx.matmul (mat3 b.mat1) (Nx.reshape [| n; 3; 1 |] axis))
    in
    ignore (j2, denom);
    dist, pos, normal
  in
  let pairs =
    List.concat_map (List.range 0 3) ~f:(fun i ->
      List.map (List.range 0 3) ~f:(fun j -> i, j))
  in
  let results = List.mapi pairs ~f:(fun k ij -> one k ij) in
  { dist = Nx.stack ~axis:1 (List.map results ~f:(fun (d, _, _) -> d))
  ; pos = Nx.stack ~axis:1 (List.map results ~f:(fun (_, p, _) -> p))
  ; normal = Nx.stack ~axis:1 (List.map results ~f:(fun (_, _, n) -> n))
  }

(* {1 Stage 2b: the face contact}

   The intersection of the reference face and the incident face is a convex
   polygon, so its vertices are exactly: the incident corners that lie inside
   the reference rectangle, the reference corners that lie inside the incident
   rectangle, and the crossings of an incident edge with a reference side plane.
   Those twenty-four candidates are generated here and the survivors are the
   contact points.

   The C engine instead runs a Sutherland-Hodgman clip, which emits the same
   vertices in clip order; taking the set directly costs the order, which is why
   a pair's rows can come out in a different order from the engine's. Row order
   reaches the solver only through the warm start, never through the geometry.

   The face code is a host loop over the six faces and each iteration computes
   that face's manifold for every pair, merged where the pair's code matches.
   Six times the arithmetic buys constant indices: inside an iteration the
   reference axis is a host number, and only the incident axis is data. *)

type cand =
  { p : Nx.host Nx.float64_t array
    (** the three coordinates in the reference face's rearranged frame:
        [(ax, ay, a)], the last one the signed height above the face. *)
  ; ok : Nx.host Nx.float64_t (** [[n]], 1. where this candidate is a vertex. *)
  }

(* [one_case b fr margin ref1 a] is the manifold of the face [a] of box [ref1]'s
   box, for every pair. *)
let one_case (b : batch) (fr : frame) (margin : _ Nx.t) ref1 a =
  let n = n_of b in
  let matref = if ref1 then mat3 b.mat1 else mat3 b.mat2
  and posref = if ref1 then b.pos1 else b.pos2
  and sizeref = if ref1 then b.size1 else b.size2
  and sizeinc = if ref1 then b.size2 else b.size1
  and posoi = if ref1 then fr.pos21 else fr.pos12
  and rinc = if ref1 then fr.rot else Nx.matrix_transpose fr.rot in
  let ax = m3 (a + 1)
  and ay = m3 (a + 2) in
  let axis_of = [| ax; ay; a |] in
  let zero = Nx.zeros f64 [| n |] in
  let one = Nx.add_s zero 1.
  and two = Nx.add_s zero 2. in
  let sgn =
    let pa = comp posoi a in
    Nx.where (Nx.greater_equal_s pa 0.) (Nx.ones_like pa) (Nx.full_like pa (-1.))
  in
  (* the incident face: the incident axis most opposed to the reference
     normal, chosen the way the host chooses it — strictly greater wins *)
  let ra = row rinc a in
  let absra k = Nx.abs (comp ra k) in
  let binc =
    Nx.where
      (Nx.greater (absra 1) (absra 0))
      (Nx.where (Nx.greater (absra 2) (absra 1)) two one)
      (Nx.where (Nx.greater (absra 2) (absra 0)) two zero)
  in
  let binc_i = Nx.cast Nx.int64 binc in
  let tinc =
    Nx.where
      (Nx.greater_s (Nx.mul sgn (take_comp ra binc_i)) 0.)
      (Nx.full_like zero (-1.))
      (Nx.ones_like zero)
  in
  let bu = Nx.where (Nx.equal binc two) zero (Nx.add_s binc 1.)
  and bv = Nx.where (Nx.equal binc two) one (Nx.where (Nx.equal binc one) zero two) in
  let elem c k = take_comp (row rinc c) (Nx.cast Nx.int64 k) in
  let s_binc = take_comp sizeinc binc_i
  and s_bu = take_comp sizeinc (Nx.cast Nx.int64 bu)
  and s_bv = take_comp sizeinc (Nx.cast Nx.int64 bv) in
  (* the incident face's centre, in the reference box's frame and on its
     plane, then the two half-edge vectors along the incident face *)
  let centre =
    Array.init 3 ~f:(fun r ->
      let c = axis_of.(r) in
      Nx.add (comp posoi c) (Nx.mul (Nx.mul tinc s_binc) (elem c binc)))
  and du0 =
    Array.init 3 ~f:(fun r ->
      let c = axis_of.(r) in
      Nx.mul s_bu (elem c bu))
  and dv0 =
    Array.init 3 ~f:(fun r ->
      let c = axis_of.(r) in
      Nx.mul s_bv (elem c bv))
  in
  (* the height above the reference face, so the last coordinate is a signed
     depth: negative inside the reference box *)
  let cx = Array.init 3 ~f:(fun r -> centre.(r)) in
  cx.(2) <- Nx.sub (Nx.mul sgn centre.(2)) (comp sizeref a);
  let du =
    Array.init 3 ~f:(fun r -> if Int.equal r 2 then Nx.mul du0.(r) sgn else du0.(r))
  and dv =
    Array.init 3 ~f:(fun r -> if Int.equal r 2 then Nx.mul dv0.(r) sgn else dv0.(r))
  in
  let signs = [| 1., 1.; -1., 1.; -1., -1.; 1., -1. |] in
  let corner (su, sv) =
    { p =
        Array.init 3 ~f:(fun r ->
          Nx.add cx.(r) (Nx.add (Nx.mul_s du.(r) su) (Nx.mul_s dv.(r) sv)))
    ; ok = one
    }
  in
  let corners = Array.map signs ~f:corner in
  let lim_ax = comp sizeref ax
  and lim_ay = comp sizeref ay
  and lim_a = comp sizeref a in
  let planes = [| 0, 1., lim_ax; 0, -1., lim_ax; 1, 1., lim_ay; 1, -1., lim_ay |] in
  let nxt k = if Int.equal k 3 then 0 else k + 1 in
  let intersect k (coord, s, lim) =
    let p = corners.(k).p
    and q = corners.(nxt k).p in
    let dp = Nx.sub (Nx.mul_s p.(coord) s) lim
    and dq = Nx.sub (Nx.mul_s q.(coord) s) lim in
    let denom = Nx.sub dp dq in
    let t =
      Nx.where (Nx.less_s (Nx.abs denom) 1e-300) (Nx.zeros_like denom) (Nx.div dp denom)
    in
    let cross =
      Nx.where
        (Nx.logical_or
           (Nx.logical_and (Nx.less_s dp 0.) (Nx.greater_s dq 0.))
           (Nx.logical_and (Nx.greater_s dp 0.) (Nx.less_s dq 0.)))
        one
        (Nx.zeros_like one)
    in
    { p = Array.init 3 ~f:(fun r -> Nx.add p.(r) (Nx.mul (Nx.sub q.(r) p.(r)) t))
    ; ok = cross
    }
  in
  (* the reference corners, lifted onto the incident face's plane, kept when
     the incident face covers them *)
  let u =
    Array.init 3 ~f:(fun r ->
      let c = axis_of.(r) in
      elem c bu)
  and v =
    Array.init 3 ~f:(fun r ->
      let c = axis_of.(r) in
      elem c bv)
  in
  let ref_corner (cu, cv) =
    let x = Nx.mul_s lim_ax cu
    and y = Nx.mul_s lim_ay cv in
    let dz0x = Nx.sub x centre.(0)
    and dz0y = Nx.sub y centre.(1) in
    let z =
      Nx.sub
        centre.(2)
        (Nx.div
           (Nx.add (Nx.mul (elem ax binc) dz0x) (Nx.mul (elem ay binc) dz0y))
           (elem a binc))
    in
    let dz = Nx.sub z centre.(2) in
    let pu = Nx.add (Nx.add (Nx.mul u.(0) dz0x) (Nx.mul u.(1) dz0y)) (Nx.mul u.(2) dz)
    and pv = Nx.add (Nx.add (Nx.mul v.(0) dz0x) (Nx.mul v.(1) dz0y)) (Nx.mul v.(2) dz) in
    let slack = 1e-12 in
    let covered =
      Nx.logical_and
        (Nx.less_equal (Nx.abs pu) (Nx.add s_bu (Nx.mul_s (Nx.abs s_bu) slack)))
        (Nx.less_equal (Nx.abs pv) (Nx.add s_bv (Nx.mul_s (Nx.abs s_bv) slack)))
    in
    { p = [| x; y; Nx.sub (Nx.mul sgn z) lim_a |]
    ; ok = Nx.where covered one (Nx.zeros_like one)
    }
  in
  let refs = Array.map signs ~f:ref_corner in
  let cands =
    Array.to_list corners
    @ List.concat_map (List.range 0 4) ~f:(fun k ->
      List.map (Array.to_list planes) ~f:(intersect k))
    @ Array.to_list refs
  in
  let ncand = List.length cands in
  let px = Nx.stack ~axis:1 (List.map cands ~f:(fun c -> c.p.(0)))
  and py = Nx.stack ~axis:1 (List.map cands ~f:(fun c -> c.p.(1)))
  and pz = Nx.stack ~axis:1 (List.map cands ~f:(fun c -> c.p.(2)))
  and ok = Nx.stack ~axis:1 (List.map cands ~f:(fun c -> c.ok)) in
  (* a crossing lies *on* its own side plane, and rounding can put it a few
     ulps outside, which the C never notices because it does not re-test a
     vertex against the plane that made it: guard the boundary *)
  let slack = Nx.mul_s (Nx.add lim_ax lim_ay) 1e-12 in
  let lim_ax2 = Nx.reshape [| n; 1 |] (Nx.add lim_ax slack)
  and lim_ay2 = Nx.reshape [| n; 1 |] (Nx.add lim_ay slack)
  and margin2 = Nx.reshape [| n; 1 |] margin in
  let within v lim =
    Nx.where (Nx.less_equal (Nx.abs v) lim) (Nx.ones_like v) (Nx.zeros_like v)
  in
  let in_x = within px lim_ax2
  and in_y = within py lim_ay2
  (* the depth test is one-sided: a contact is *below* the margin, and the abs
     that the side planes want would drop every penetration *)
  and in_z = Nx.where (Nx.less_equal pz margin2) (Nx.ones_like pz) (Nx.zeros_like pz) in
  let base = Nx.mul ok (Nx.mul in_x (Nx.mul in_y in_z)) in
  (* drop near-duplicates in the plane, keeping the earliest — the host's rule,
     and the reference corners can coincide with a crossing exactly *)
  let dupe2 = Nx.mul_s (Nx.add (Nx.mul lim_ax lim_ax) (Nx.mul lim_ay lim_ay)) dupe_eps in
  let dx = Nx.sub (Nx.reshape [| n; 1; ncand |] px) (Nx.reshape [| n; ncand; 1 |] px)
  and dy = Nx.sub (Nx.reshape [| n; 1; ncand |] py) (Nx.reshape [| n; ncand; 1 |] py) in
  let d2 = Nx.add (Nx.mul dx dx) (Nx.mul dy dy) in
  let earlier =
    Nx.broadcast_to
      [| n; ncand; ncand |]
      (Nx.reshape
         [| 1; ncand; ncand |]
         (Nx.init f64 [| ncand; ncand |] (fun i -> if i.(1) < i.(0) then 1. else 0.)))
  in
  let close =
    Nx.mul
      (Nx.mul earlier (Nx.reshape [| n; ncand; 1 |] base))
      (Nx.where
         (Nx.less d2 (Nx.reshape [| n; 1; 1 |] dupe2))
         (Nx.ones_like d2)
         (Nx.zeros_like d2))
  in
  let dup = Nx.max ~axes:[ 2 ] close in
  let keep =
    Nx.mul
      base
      (Nx.where
         (Nx.greater dup (Nx.zeros_like dup))
         (Nx.zeros_like dup)
         (Nx.ones_like dup))
  in
  let slot = Nx.sub (Nx.cumsum ~axis:1 keep) (Nx.ones_like keep) in
  let count = Nx.sum ~axes:[ 1 ] keep in
  let idx =
    Nx.where
      (Nx.greater keep (Nx.zeros_like keep))
      slot
      (Nx.full_like slot (Float.of_int max_contacts))
  in
  let idx3 =
    Nx.broadcast_to
      [| n; ncand; 3 |]
      (Nx.reshape [| n; ncand; 1 |] (Nx.cast Nx.int64 idx))
  in
  let all = Nx.stack ~axis:2 [ px; py; pz ] in
  let scattered =
    Nx.scatter ~combine:Set ~axis:1 idx3 all (Nx.zeros f64 [| n; max_contacts + 1; 3 |])
  in
  let posc = Nx.slice [ A; R (0, max_contacts); A ] scattered in
  let vx = Nx.slice [ A; A; I 0 ] posc
  and vy = Nx.slice [ A; A; I 1 ] posc
  and vz = Nx.slice [ A; A; I 2 ] posc in
  (* the candidate coordinates are in the rearranged frame [(ax, ay, a)], so
     they go back into the parent box's slots here — for a face axis other
     than 2 the two frames are a cycle apart *)
  let z_face =
    Nx.mul
      (Nx.ones_like vz)
      (Nx.mul
         (Nx.reshape [| n; 1 |] sgn)
         (Nx.add (Nx.reshape [| n; 1 |] lim_a) (Nx.mul_s vz 0.5)))
  in
  let in_parent p = if Int.equal p a then z_face else if Int.equal p ax then vx else vy in
  let pos_face = Nx.stack ~axis:2 [ in_parent 0; in_parent 1; in_parent 2 ] in
  let nsign = if ref1 then sgn else Nx.neg sgn in
  let out_col r = comp (row matref r) a in
  let normal =
    Nx.stack
      ~axis:2
      (List.map (List.range 0 3) ~f:(fun r ->
         Nx.mul
           (Nx.ones_like vz)
           (Nx.mul (Nx.reshape [| n; 1 |] nsign) (Nx.reshape [| n; 1 |] (out_col r)))))
  in
  let pos =
    Nx.add
      (Nx.reshape
         [| n; max_contacts; 3 |]
         (Nx.matmul
            (Nx.reshape [| n; 1; 3; 3 |] matref)
            (Nx.reshape [| n; max_contacts; 3; 1 |] pos_face)))
      (Nx.reshape [| n; 1; 3 |] posref)
  in
  let index =
    Nx.broadcast_to
      [| n; max_contacts |]
      (Nx.reshape
         [| 1; max_contacts |]
         (Nx.init f64 [| max_contacts |] (fun i -> Float.of_int i.(0))))
  in
  let live =
    Nx.where
      (Nx.less index (Nx.reshape [| n; 1 |] count))
      (Nx.ones_like vz)
      (Nx.zeros_like vz)
  in
  { dist = vz; pos; normal; live }

let faces (b : batch) (fr : frame) (code : _ Nx.t) (margin : _ Nx.t) =
  let n = n_of b in
  let zeros =
    { dist = Nx.zeros f64 [| n; max_contacts |]
    ; pos = Nx.zeros f64 [| n; max_contacts; 3 |]
    ; normal = Nx.zeros f64 [| n; max_contacts; 3 |]
    ; live = Nx.zeros f64 [| n; max_contacts |]
    }
  in
  let merged =
    List.fold (List.range 0 6) ~init:zeros ~f:(fun acc c ->
      let ref1 = c < 3 in
      let m = one_case b fr margin ref1 (if ref1 then c else c - 3) in
      let sel =
        Nx.where
          (Nx.equal_s code (Float.of_int c))
          (Nx.ones_like code)
          (Nx.zeros_like code)
      in
      let sel1 = Nx.reshape [| n; 1 |] sel
      and sel3 = Nx.reshape [| n; 1; 1 |] sel in
      { dist = Nx.add acc.dist (Nx.mul m.dist sel1)
      ; pos = Nx.add acc.pos (Nx.mul m.pos sel3)
      ; normal = Nx.add acc.normal (Nx.mul m.normal sel3)
      ; live = Nx.add acc.live (Nx.mul m.live sel1)
      })
  in
  merged

(* [slot0 v rank d] puts [v] — [n] for rank 1, [n;3] for rank 3 — in slot 0 and
   fills the rest of the twelve slots with [d]. *)
let slot0 (v : _ Nx.t) rank d =
  let n = (Nx.shape v).(0) in
  if Int.equal rank 1
  then (
    let v = Nx.reshape [| n; 1 |] v in
    Nx.concatenate ~axis:1 [ v; Nx.full f64 [| n; max_contacts - 1 |] d ])
  else (
    let v = Nx.reshape [| n; 1; 3 |] v in
    Nx.concatenate ~axis:1 [ v; Nx.full f64 [| n; max_contacts - 1; 3 |] d ])

(* {1 Manifold}

   A pair takes one generator or the other: edge codes take the single contact
   of the winning edge cross, face codes take the clipped polygon. The two are
   disjoint, so the edge contact is written into slot 0 and the face manifold
   over all twelve. *)
let manifold (b : batch) =
  let fr = frame b in
  let s = sat b in
  let _, _, _, axes = separations b fr in
  let e = edges b fr axes in
  let f = faces b fr s.code b.margin in
  let n = n_of b in
  let edge_index =
    Nx.reshape [| n; 1 |] (Nx.cast Nx.int64 (Nx.maximum_s (Nx.sub_s s.code 6.) 0.))
  in
  let wide = Nx.reshape [| n; 1; 1 |] edge_index in
  let edge_dist = Nx.reshape [| n |] (Nx.take_along_axis ~axis:1 edge_index e.dist)
  and edge_pos =
    Nx.reshape
      [| n; 3 |]
      (Nx.take_along_axis ~axis:1 (Nx.broadcast_to [| n; 1; 3 |] wide) e.pos)
  and edge_normal =
    Nx.reshape
      [| n; 3 |]
      (Nx.take_along_axis ~axis:1 (Nx.broadcast_to [| n; 1; 3 |] wide) e.normal)
  in
  let mask of_ t = Nx.where t (Nx.ones_like of_) (Nx.zeros_like of_) in
  let is_edge = mask s.code (Nx.greater_equal_s s.code 6.)
  and is_face = mask s.code (Nx.less_s s.code 6.)
  and touching = mask s.code (Nx.less_s s.separated 0.5) in
  let edge = Nx.mul is_edge touching in
  let face = Nx.mul is_face touching in
  (* the dead-slot filler must not survive into a face pair's slots, so the whole
     edge branch — filler included — is masked by the branch, not just its value *)
  { dist =
      Nx.add
        (Nx.mul (Nx.reshape [| n; 1 |] edge) (slot0 edge_dist 1 (-.big)))
        (Nx.mul (Nx.reshape [| n; 1 |] face) f.dist)
  ; pos =
      Nx.add
        (slot0 (Nx.mul (Nx.reshape [| n; 1 |] edge) edge_pos) 3 0.)
        (Nx.mul (Nx.reshape [| n; 1; 1 |] face) f.pos)
  ; normal =
      Nx.add
        (slot0 (Nx.mul (Nx.reshape [| n; 1 |] edge) edge_normal) 3 0.)
        (Nx.mul (Nx.reshape [| n; 1; 1 |] face) f.normal)
  ; live = Nx.add (slot0 edge 1 0.) (Nx.mul (Nx.reshape [| n; 1 |] face) f.live)
  }
