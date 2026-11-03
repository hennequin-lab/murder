(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx

type ('s, 'd) f = (float, 's, 'd) Nx.t

(* Component [k] of the last axis, for a rank-2 tensor. *)
let comp k x = Nx.slice [ A; I k ] x

(* Four 1-d columns to one [n;4]. *)
let of_cols xs = Nx.stack ~axis:1 xs

let quat_mul a b =
  let open Infix in
  let aw = comp 0 a
  and ax = comp 1 a
  and ay = comp 2 a
  and az = comp 3 a in
  let bw = comp 0 b
  and bx = comp 1 b
  and by = comp 2 b
  and bz = comp 3 b in
  of_cols
    [ (aw * bw) - (ax * bx) - (ay * by) - (az * bz)
    ; (aw * bx) + (ax * bw) + (ay * bz) - (az * by)
    ; (aw * by) - (ax * bz) + (ay * bw) + (az * bx)
    ; (aw * bz) + (ax * by) - (ay * bx) + (az * bw)
    ]

let quat_mul_axis q v =
  let z = Nx.slice [ R (0, 1) ] v in
  quat_mul q (Nx.concatenate ~axis:0 [ Nx.mul_s z 0.0; v ])

let quat_deriv q omega =
  (* MuJoCo's [mju_derivQuat] multiplies from the left: [(0, omega) * q]. *)
  let z = Nx.slice [ R (0, 1) ] omega in
  let pure = Nx.concatenate ~axis:0 [ Nx.mul_s z 0.0; omega ] in
  Nx.mul_s (quat_mul pure q) 0.5

let quat_conj q =
  of_cols [ comp 0 q; Nx.neg (comp 1 q); Nx.neg (comp 2 q); Nx.neg (comp 3 q) ]

let norm_along_1 x = Nx.sqrt (Nx.sum ~axes:[ 1 ] (Nx.mul x x))

let safe_norm x =
  let n = norm_along_1 x in
  Nx.where (Nx.equal_s n 0.0) (Nx.ones_like n) n

let quat_normalize q = Nx.div q (Nx.unsqueeze ~axes:[ 1 ] (safe_norm q))
let vec_normalize v = Nx.div v (Nx.unsqueeze ~axes:[ 1 ] (safe_norm v))

let quat_rot_vec q v =
  let open Infix in
  (* [t = w v + u x v], [res = v + 2 u x t], with [u] the vector part. *)
  let u = Nx.slice [ A; R (1, 4) ] q in
  let w = Nx.unsqueeze ~axes:[ 1 ] (comp 0 q) in
  let t = (w * v) + Nx.cross u v in
  v + Nx.mul_s (Nx.cross u t) 2.0

let quat_to_mat q =
  let open Infix in
  let w = comp 0 q
  and x = comp 1 q
  and y = comp 2 q
  and z = comp 3 q in
  let q00 = w * w
  and q01 = w * x
  and q02 = w * y
  and q03 = w * z in
  let q11 = x * x
  and q12 = x * y
  and q13 = x * z in
  let q22 = y * y
  and q23 = y * z in
  let q33 = z * z in
  of_cols
    [ q00 + q11 - q22 - q33
    ; Nx.mul_s (q12 - q03) 2.0
    ; Nx.mul_s (q13 + q02) 2.0
    ; Nx.mul_s (q12 + q03) 2.0
    ; q00 - q11 + q22 - q33
    ; Nx.mul_s (q23 - q01) 2.0
    ; Nx.mul_s (q13 - q02) 2.0
    ; Nx.mul_s (q23 + q01) 2.0
    ; q00 - q11 - q22 + q33
    ]

let quat_of_axis_angle ~axis ~angle =
  let half = Nx.mul_s angle 0.5 in
  let s = Nx.sin half in
  let u = Nx.mul (Nx.unsqueeze ~axes:[ 1 ] s) axis in
  of_cols [ Nx.cos half; comp 0 u; comp 1 u; comp 2 u ]

let quat_of_rotvec r =
  let t2 = Nx.sum ~axes:[ 1 ] (Nx.mul r r) in
  let theta = Nx.sqrt t2 in
  let half = Nx.mul_s theta 0.5 in
  (* sin(half)/theta tends to 1/2; its series is 1/2 - theta^2/48. *)
  let small = Nx.less_s theta 1e-8 in
  let ratio =
    Nx.where
      small
      (Nx.sub (Nx.full_like theta 0.5) (Nx.div_s t2 48.0))
      (Nx.div (Nx.sin half) theta)
  in
  let v = Nx.mul (Nx.unsqueeze ~axes:[ 1 ] ratio) r in
  of_cols [ Nx.cos half; comp 0 v; comp 1 v; comp 2 v ]

let quat_integrate q w ~scale =
  quat_mul (quat_normalize q) (quat_of_rotvec (Nx.mul_s w scale))

let vec_dot a b = Nx.sum ~axes:[ 1 ] (Nx.mul a b)
let vec_cross a b = Nx.cross a b
let vec_norm = norm_along_1

let mat_vec m v =
  let n = Nx.dim 0 m in
  let m3 = Nx.reshape [| n; 3; 3 |] m in
  let v3 = Nx.reshape [| n; 3; 1 |] v in
  Nx.reshape [| n; 3 |] (Nx.matmul m3 v3)

let dot6 a b = Nx.dot a b

let cross_motion vel v =
  let g = Nx.get in
  let open Infix in
  let z = g [ 0 ] vel
  and o = g [ 1 ] vel
  and t = g [ 2 ] vel in
  let th = g [ 3 ] vel
  and fo = g [ 4 ] vel
  and fi = g [ 5 ] vel in
  let v0 = g [ 0 ] v
  and v1 = g [ 1 ] v
  and v2 = g [ 2 ] v in
  let v3 = g [ 3 ] v
  and v4 = g [ 4 ] v
  and v5 = g [ 5 ] v in
  let r0 = Nx.neg (t * v1) + (o * v2) in
  let r1 = (t * v0) - (z * v2) in
  let r2 = Nx.neg (o * v0) + (z * v1) in
  let r3 = Nx.neg (t * v4) + (o * v5) + (Nx.neg (fi * v1) + (fo * v2)) in
  let r4 = (t * v3) - (z * v5) + ((fi * v0) - (th * v2)) in
  let r5 = Nx.neg (o * v3) + (z * v4) + (Nx.neg (fo * v0) + (th * v1)) in
  Nx.stack ~axis:0 [ r0; r1; r2; r3; r4; r5 ]

let cross_force vel f =
  let g = Nx.get in
  let open Infix in
  let z = g [ 0 ] vel
  and o = g [ 1 ] vel
  and t = g [ 2 ] vel in
  let th = g [ 3 ] vel
  and fo = g [ 4 ] vel
  and fi = g [ 5 ] vel in
  let f0 = g [ 0 ] f
  and f1 = g [ 1 ] f
  and f2 = g [ 2 ] f in
  let f3 = g [ 3 ] f
  and f4 = g [ 4 ] f
  and f5 = g [ 5 ] f in
  let r0 = Nx.neg (t * f1) + (o * f2) + (Nx.neg (fi * f4) + (fo * f5)) in
  let r1 = (t * f0) - (z * f2) + ((fi * f3) - (th * f5)) in
  let r2 = Nx.neg (o * f0) + (z * f1) + (Nx.neg (fo * f3) + (th * f4)) in
  let r3 = Nx.neg (t * f4) + (o * f5) in
  let r4 = (t * f3) - (z * f5) in
  let r5 = Nx.neg (o * f3) + (z * f4) in
  Nx.stack ~axis:0 [ r0; r1; r2; r3; r4; r5 ]

let mul_inert_vec i v =
  let ic k = Nx.get [ k ] i in
  let vc k = Nx.get [ k ] v in
  let open Infix in
  let r0 =
    (ic 0 * vc 0) + (ic 3 * vc 1) + (ic 4 * vc 2) - (ic 8 * vc 4) + (ic 7 * vc 5)
  in
  let r1 =
    (ic 3 * vc 0) + (ic 1 * vc 1) + (ic 5 * vc 2) + (ic 8 * vc 3) - (ic 6 * vc 5)
  in
  let r2 =
    (ic 4 * vc 0) + (ic 5 * vc 1) + (ic 2 * vc 2) - (ic 7 * vc 3) + (ic 6 * vc 4)
  in
  let r3 = (ic 8 * vc 1) - (ic 7 * vc 2) + (ic 9 * vc 3) in
  let r4 = (ic 6 * vc 2) - (ic 8 * vc 0) + (ic 9 * vc 4) in
  let r5 = (ic 7 * vc 0) - (ic 6 * vc 1) + (ic 9 * vc 5) in
  Nx.stack ~axis:0 [ r0; r1; r2; r3; r4; r5 ]
