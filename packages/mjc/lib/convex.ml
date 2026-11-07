(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Support functions, one per shape, written against MuJoCo's own so that the
   iteration built on them can be compared with its results. All of them do the
   same three things: rotate the direction into the geom's frame with [matᵀ],
   take the extreme point there, and rotate it back with [mat] and add [pos]. *)

type shape =
  | Point
  | Sphere
  | Line
  | Capsule
  | Ellipsoid
  | Cylinder
  | Box

let rank x = Array.length (Nx.shape x)

(* A component of a point [3] or of a batch [k; 3]. *)
let comp i v = if Int.equal (rank v) 1 then Nx.slice [ I i ] v else Nx.slice [ A; I i ] v

(* A size component: 0-d for a single geom, [k] for a batch. *)
let size_k ~dir k size =
  if Int.equal (rank dir) 1 then Nx.slice [ I k ] size else Nx.slice [ A; I k ] size

let dp a b = Nx.sum ~axes:[ -1 ] (Nx.mul a b)
let stack3 a b c = Nx.stack ~axis:(-1) [ a; b; c ]
let ssqrt = Rigid.safe_sqrt_exact
let const_like v x = Nx.mul_s (Nx.ones_like v) x

(* A per-candidate scalar times a vector: [k] against [k;3] does not broadcast,
   so the batch case grows a trailing axis and a lone scalar must not. *)
(* The trailing axis belongs only where the vector has one more axis than the
   scalar: [k;3] against [k]. A [k] against [k] must not grow one, or the product
   is [k;k]. *)
let align v s = if Int.equal (rank v) (rank s + 1) then Nx.unsqueeze ~axes:[ -1 ] s else s
let smul v s = Nx.mul v (align v s)
let sdiv v s = Nx.div v (align v s)
let wsel c a b = Nx.where (align a c) a b

let transpose mat =
  if Int.equal (rank mat) 2
  then Nx.matrix_transpose mat
  else Nx.transpose ~axes:[ 0; 2; 1 ] mat

(* [matᵀ dir]: the direction in the geom's frame. *)
let to_local mat dir =
  Nx.reshape (Nx.shape dir) (Nx.matmul (transpose mat) (Nx.unsqueeze ~axes:[ -1 ] dir))

(* [mat local + pos]: back to the world. *)
let to_world ~pos ~mat local =
  Nx.add
    pos
    (Nx.reshape (Nx.shape local) (Nx.matmul mat (Nx.unsqueeze ~axes:[ -1 ] local)))

let support shape ~pos ~mat ~size ~dir =
  let local = to_local mat dir in
  let z = comp 2 local in
  let up = Nx.greater_equal z (const_like z 0.0) in
  let along value = Nx.where up value (Nx.neg value) in
  match shape with
  | Point -> pos
  | Sphere ->
    (* MuJoCo does not normalize here: the radius scales the direction as given,
       and a unit direction is what the callers pass. *)
    Nx.add pos (smul dir (size_k ~dir 0 size))
  | Line ->
    let axis = along (size_k ~dir 1 size) in
    to_world ~pos ~mat (stack3 (const_like z 0.0) (const_like z 0.0) axis)
  | Capsule ->
    let body = smul local (size_k ~dir 0 size) in
    let zz = Nx.add (comp 2 body) (along (size_k ~dir 1 size)) in
    to_world ~pos ~mat (stack3 (comp 0 body) (comp 1 body) zz)
  | Ellipsoid ->
    let radii = stack3 (size_k ~dir 0 size) (size_k ~dir 1 size) (size_k ~dir 2 size) in
    let scaled = Nx.mul local radii in
    let norm = ssqrt (dp scaled scaled) in
    let degenerate = Nx.less norm (const_like norm 1e-15) in
    let unit =
      wsel
        degenerate
        (stack3 (const_like norm 1.0) (const_like norm 0.0) (const_like norm 0.0))
        (sdiv scaled (Nx.maximum_s norm 1e-300))
    in
    to_world ~pos ~mat (Nx.mul unit radii)
  | Cylinder ->
    let r = size_k ~dir 0 size in
    let lx = comp 0 local
    and ly = comp 1 local in
    let flat_sqr = Nx.add (Nx.mul lx lx) (Nx.mul ly ly) in
    let flat = Nx.greater flat_sqr (const_like flat_sqr 1e-30) in
    let scale = sdiv r (Nx.maximum_s (ssqrt flat_sqr) 1e-300) in
    let x = Nx.where flat (Nx.mul lx scale) (const_like flat_sqr 0.0)
    and y = Nx.where flat (Nx.mul ly scale) (const_like flat_sqr 0.0) in
    to_world ~pos ~mat (stack3 x y (along (size_k ~dir 1 size)))
  | Box ->
    let pick i = size_k ~dir i size in
    let sx = pick 0
    and sy = pick 1
    and sz = pick 2 in
    let cx = Nx.where (Nx.greater_equal (comp 0 local) (const_like z 0.0)) sx (Nx.neg sx)
    and cy = Nx.where (Nx.greater_equal (comp 1 local) (const_like z 0.0)) sy (Nx.neg sy)
    and cz = Nx.where up sz (Nx.neg sz) in
    to_world ~pos ~mat (stack3 cx cy cz)
