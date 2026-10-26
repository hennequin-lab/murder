(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A =
  Algebra.Make
    (struct
      let signature = Signature.conformal3
    end)
    (struct
      type kind = Nx.float32_elt

      let dtype = Nx.float32
    end)

include A

let dtype = Nx.float32
let dty = Nx.float32
let batch_shape = A.batch_shape
let expand batch x = Nx.reshape (Array.append batch [| 1 |]) x
let comp a i = A.coeff a (Blade.of_int i)
let constant batch x = Nx.broadcast_to batch (Nx.scalar dty x)

let batch_of_shape sh =
  if Array.length sh = 0 then [||] else Array.sub sh 0 (Array.length sh - 1)

let dense batch comps =
  let z = Nx.zeros dty batch in
  let a = Array.make 32 z in
  List.iter (fun (i, v) -> a.(i) <- v) comps;
  Nx.stack ~axis:(-1) (Array.to_list a)

let take3 v =
  let r = Array.length (Nx.shape v) in
  let f i = Nx.take ~axis:(r - 1) ~indices:(Nx.scalar Nx.int64 (Int64.of_int i)) v in
  f 0, f 1, f 2

let e1 = A.e 1
let e2 = A.e 2
let e3 = A.e 3
let e4 = A.e 4
let e5 = A.e 5
let e0 = dense [||] [ 8, constant [||] (-0.5); 16, constant [||] 0.5 ]
let einf = dense [||] [ 8, constant [||] 1.; 16, constant [||] 1. ]

let point v =
  let sh = Nx.shape v in
  let batch = batch_of_shape sh in
  let x, y, z = take3 v in
  let x2 = Nx.add (Nx.add (Nx.mul x x) (Nx.mul y y)) (Nx.mul z z) in
  let c4 = Nx.mul_s (Nx.sub_s x2 1.) 0.5 in
  let c5 = Nx.mul_s (Nx.add_s x2 1.) 0.5 in
  dense batch [ 1, x; 2, y; 4, z; 8, c4; 16, c5 ]

let point3 x y z = point (Nx.create dty [| 3 |] [| x; y; z |])

let coords p =
  let c1 = comp p 1
  and c2 = comp p 2
  and c3 = comp p 4 in
  let c4 = comp p 8
  and c5 = comp p 16 in
  let b = Nx.sub c5 c4 in
  Nx.stack ~axis:(-1) [ Nx.div c1 b; Nx.div c2 b; Nx.div c3 b ]

let is_point p =
  let n2 = A.norm2 p in
  let b = Nx.sub (comp p 16) (comp p 8) in
  Nx.logical_and (Nx.less_s (Nx.abs n2) 1e-5) (Nx.less_s (Nx.abs (Nx.sub_s b 1.)) 1e-5)

let sphere ~center ~radius =
  let batch = batch_of_shape (Nx.shape center) in
  let x, y, z = take3 center in
  let x, y, z, r =
    match Nx.broadcast_arrays [ x; y; z; radius ] with
    | [ x; y; z; r ] -> x, y, z, r
    | _ -> assert false
  in
  let x2 = Nx.add (Nx.add (Nx.mul x x) (Nx.mul y y)) (Nx.mul z z) in
  let r2 = Nx.mul r r in
  let d = Nx.sub x2 r2 in
  let c4 = Nx.sub (Nx.mul_s d 0.5) (constant batch 0.5) in
  let c5 = Nx.add (Nx.mul_s d 0.5) (constant batch 0.5) in
  dense batch [ 1, x; 2, y; 4, z; 8, c4; 16, c5 ]

let sphere_of_point ~center ~radius =
  let batch = A.batch_shape center in
  let r2 = expand batch (Nx.mul radius radius) in
  Nx.sub center (Nx.mul_s (Nx.mul r2 einf) 0.5)

let sphere_radius2 s =
  let s2 = A.coeff (A.gp s s) (Blade.of_int 0) in
  let b = Nx.sub (comp s 16) (comp s 8) in
  Nx.div s2 (Nx.mul b b)

let sphere_center s =
  let b = Nx.sub (comp s 16) (comp s 8) in
  let batch = A.batch_shape s in
  let sn = Nx.div s (expand batch b) in
  let s2 = A.coeff (A.gp sn sn) (Blade.of_int 0) in
  Nx.add sn (Nx.mul (expand batch (Nx.mul_s s2 0.5)) einf)

let plane ~normal ~offset =
  let batch = batch_of_shape (Nx.shape normal) in
  let x, y, z = take3 normal in
  let x, y, z, d =
    match Nx.broadcast_arrays [ x; y; z; offset ] with
    | [ x; y; z; d ] -> x, y, z, d
    | _ -> assert false
  in
  dense batch [ 1, x; 2, y; 4, z; 8, d; 16, d ]

let plane_normal p = Nx.stack ~axis:(-1) [ comp p 1; comp p 2; comp p 4 ]
let plane_offset p = comp p 8

let normalize_plane p =
  let n = plane_normal p in
  let n2 = Nx.sum ~axes:[ Array.length (Nx.shape n) - 1 ] (Nx.mul n n) in
  Nx.div p (expand (A.batch_shape p) (Nx.sqrt n2))

let normalize_point p =
  let b = Nx.sub (comp p 16) (comp p 8) in
  Nx.div p (expand (A.batch_shape p) b)

let normalize_sphere = normalize_point

let is_zero a =
  let r = Array.length (Nx.shape a) in
  let s = Nx.sum ~axes:[ r - 1 ] (Nx.abs a) in
  Nx.less_s s 1e-5

let is_flat a = is_zero (A.inner einf a)
let is_round a = Nx.logical_not (is_flat a)
let point_pair x y = A.op x y
let line x y = A.dual (A.op (A.op x y) einf)
let line_of_planes p1 p2 = A.op p1 p2
let circle x y z = A.dual (A.op (A.op x y) z)
let circle_of_spheres s1 s2 = A.op s1 s2
let plane_through x y z = A.dual (A.op (A.op (A.op x y) z) einf)
let sphere_through x y z w = A.dual (A.op (A.op (A.op x y) z) w)
let incident x o = is_zero (A.inner x o)
let distance2 x y = Nx.mul_s (A.scalar_prod x y) (-2.)
let distance x y = Nx.sqrt (distance2 x y)
let meet a b = A.op a b
let coeff_i a i = A.coeff a (Blade.of_int i)

let line_scale l =
  let b12 = coeff_i l 3
  and b13 = coeff_i l 5
  and b23 = coeff_i l 6 in
  Nx.sqrt (Nx.add (Nx.add (Nx.mul b12 b12) (Nx.mul b13 b13)) (Nx.mul b23 b23))

let line_direction l =
  let b12 = coeff_i l 3
  and b13 = coeff_i l 5
  and b23 = coeff_i l 6 in
  let s = line_scale l in
  Nx.stack ~axis:(-1) [ Nx.div b23 s; Nx.div (Nx.neg b13) s; Nx.div b12 s ]

let line_moment l =
  let s = line_scale l in
  Nx.stack
    ~axis:(-1)
    [ Nx.div (coeff_i l 9) s; Nx.div (coeff_i l 10) s; Nx.div (coeff_i l 12) s ]

let cross3 ax ay az bx by bz =
  let cx = Nx.sub (Nx.mul ay bz) (Nx.mul az by) in
  let cy = Nx.sub (Nx.mul az bx) (Nx.mul ax bz) in
  let cz = Nx.sub (Nx.mul ax by) (Nx.mul ay bx) in
  cx, cy, cz

let circle_frame c =
  (* Recover the null-basis coefficients of the circle blade. In the null basis
     a circle has z = B + u_i e_i e_0 + v_i e_i e_inf + w e_0 e_inf, with u the
     normal of its carrier plane (not necessarily unit), w = -d and
     v = (1/2)(c^2 - r^2) n - d c. *)
  let a1 = coeff_i c 9
  and a2 = coeff_i c 10
  and a3 = coeff_i c 12 in
  let b1 = coeff_i c 17
  and b2 = coeff_i c 18
  and b3 = coeff_i c 20 in
  let u1 = Nx.sub b1 a1
  and u2 = Nx.sub b2 a2
  and u3 = Nx.sub b3 a3 in
  let s = Nx.sqrt (Nx.add (Nx.add (Nx.mul u1 u1) (Nx.mul u2 u2)) (Nx.mul u3 u3)) in
  let n1 = Nx.div u1 s
  and n2 = Nx.div u2 s
  and n3 = Nx.div u3 s in
  let w = coeff_i c 24 in
  let d = Nx.div w s in
  let b12 = Nx.div (coeff_i c 3) s in
  let b13 = Nx.div (coeff_i c 5) s in
  let b23 = Nx.div (coeff_i c 6) s in
  let v1 = Nx.div (Nx.add a1 b1) (Nx.mul_s s 2.) in
  let v2 = Nx.div (Nx.add a2 b2) (Nx.mul_s s 2.) in
  let v3 = Nx.div (Nx.add a3 b3) (Nx.mul_s s 2.) in
  let vpar = Nx.add (Nx.add (Nx.mul v1 n1) (Nx.mul v2 n2)) (Nx.mul v3 n3) in
  let b2 = Nx.add (Nx.add (Nx.mul b12 b12) (Nx.mul b13 b13)) (Nx.mul b23 b23) in
  let r2 = Nx.sub (Nx.sub b2 (Nx.mul d d)) (Nx.mul_s vpar 2.) in
  let wx = b23
  and wy = Nx.neg b13
  and wz = b12 in
  let cx, cy, cz = cross3 wx wy wz n1 n2 n3 in
  let cx = Nx.add (Nx.mul d n1) cx in
  let cy = Nx.add (Nx.mul d n2) cy in
  let cz = Nx.add (Nx.mul d n3) cz in
  n1, n2, n3, cx, cy, cz, r2

let circle_normal c =
  let n1, n2, n3, _, _, _, _ = circle_frame c in
  Nx.stack ~axis:(-1) [ n1; n2; n3 ]

let circle_center c =
  let _, _, _, cx, cy, cz, _ = circle_frame c in
  point (Nx.stack ~axis:(-1) [ cx; cy; cz ])

let circle_radius2 c =
  let _, _, _, _, _, _, r2 = circle_frame c in
  r2

let dilator lambda =
  let batch = Nx.shape lambda in
  let half = Nx.mul_s lambda 0.5 in
  let c = Nx.cosh half
  and s = Nx.sinh half in
  dense batch [ 0, c; 24, Nx.mul_s s (-1.) ]

let transversion a =
  let batch = batch_of_shape (Nx.shape a) in
  let ax, ay, az = take3 a in
  let one = constant batch 1. in
  let hx = Nx.mul_s ax 0.5
  and hy = Nx.mul_s ay 0.5
  and hz = Nx.mul_s az 0.5 in
  dense
    batch
    [ 0, one; 9, hx; 17, Nx.neg hx; 10, hy; 18, Nx.neg hy; 12, hz; 20, Nx.neg hz ]

let motor_embedding =
  let basis idxs =
    let a = Array.make 32 0. in
    Array.iter (fun i -> a.(i) <- 1.) idxs;
    a
  in
  let rows =
    [| basis [| 0 |]
     ; basis [| 3 |]
     ; basis [| 5 |]
     ; basis [| 6 |]
     ; basis [| 9; 17 |]
     ; basis [| 10; 18 |]
     ; basis [| 12; 20 |]
     ; basis [| 15; 23 |]
    |]
  in
  let e = Array.make (8 * 32) 0. in
  for k = 0 to 7 do
    for d = 0 to 31 do
      e.((k * 32) + d) <- rows.(k).(d)
    done
  done;
  Nx.create dty [| 8; 32 |] e

let of_motor m =
  let sh = Nx.shape m in
  let batch = Array.sub sh 0 (Array.length sh - 1) in
  let bsz = Array.fold_left ( * ) 1 batch in
  let m2 = Nx.reshape [| bsz; 8 |] m in
  let out = Nx.matmul m2 motor_embedding in
  Nx.reshape (Array.append batch [| 32 |]) out

let apply_motor m x = A.sandwich (of_motor m) x
