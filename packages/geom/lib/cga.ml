(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx

module A =
  Algebra.Make
    (struct
      let signature = Signature.conformal3
    end)
    (struct
      type kind = Nx.float32_elt

      let dtype = float32
    end)

include A

let dtype = float32
let dty = float32
let batch_shape = A.batch_shape
let expand batch x = reshape (Array.append batch [| 1 |]) x
let comp a i = A.coeff a (Blade.of_int i)
let constant p batch x = broadcast_to batch (Nx.scalar_at p dty x)

let batch_of_shape sh =
  if Array.length sh = 0 then [||] else Array.sub sh 0 (Array.length sh - 1)

let dense batch comps =
  let z = zeros_at (beside (snd (List.hd comps))) dty batch in
  let a = Array.make 32 z in
  List.iter (fun (i, v) -> a.(i) <- v) comps;
  stack ~axis:(-1) (Array.to_list a)

let take3 v =
  let r = Array.length (shape v) in
  let f i = take ~axis:(r - 1) (Nx.scalar_at (beside v) int64 (Int64.of_int i)) v in
  f 0, f 1, f 2

let e1 = A.e 1
let e2 = A.e 2
let e3 = A.e 3
let e4 = A.e 4
let e5 = A.e 5
let e0 = dense [||] [ 8, constant Host.on [||] (-0.5); 16, constant Host.on [||] 0.5 ]
let einf = dense [||] [ 8, constant Host.on [||] 1.; 16, constant Host.on [||] 1. ]

let point v =
  let open Infix in
  let sh = shape v in
  let batch = batch_of_shape sh in
  let x, y, z = take3 v in
  let x2 = (x * x) + (y * y) + (z * z) in
  let c4 = (x2 -$ 1.) *$ 0.5 in
  let c5 = (x2 +$ 1.) *$ 0.5 in
  dense batch [ 1, x; 2, y; 4, z; 8, c4; 16, c5 ]

let point3 x y z = point (create dty [| 3 |] [| x; y; z |])

let coords p =
  let open Infix in
  let c1 = comp p 1
  and c2 = comp p 2
  and c3 = comp p 4 in
  let c4 = comp p 8
  and c5 = comp p 16 in
  let b = c5 - c4 in
  stack ~axis:(-1) [ c1 / b; c2 / b; c3 / b ]

let is_point p =
  let open Infix in
  let n2 = A.norm2 p in
  let b = comp p 16 - comp p 8 in
  logical_and (less_s (Nx.abs n2) 1e-5) (less_s (Nx.abs (b -$ 1.)) 1e-5)

let sphere ~center ~radius =
  let open Infix in
  let batch = batch_of_shape (shape center) in
  let x, y, z = take3 center in
  let x, y, z, r =
    match broadcast_arrays [ x; y; z; radius ] with
    | [ x; y; z; r ] -> x, y, z, r
    | _ -> assert false
  in
  let x2 = (x * x) + (y * y) + (z * z) in
  let r2 = r * r in
  let d = x2 - r2 in
  let c4 = (d *$ 0.5) - constant (beside d) batch 0.5 in
  let c5 = (d *$ 0.5) + constant (beside d) batch 0.5 in
  dense batch [ 1, x; 2, y; 4, z; 8, c4; 16, c5 ]

let sphere_of_point ~center ~radius =
  let open Infix in
  let batch = A.batch_shape center in
  let r2 = expand batch (radius * radius) in
  center - (r2 * place (beside center) einf *$ 0.5)

let sphere_radius2 s =
  let open Infix in
  let s2 = A.coeff (A.gp s s) (Blade.of_int 0) in
  let b = comp s 16 - comp s 8 in
  s2 / (b * b)

let sphere_center s =
  let open Infix in
  let b = comp s 16 - comp s 8 in
  let batch = A.batch_shape s in
  let sn = s / expand batch b in
  let s2 = A.coeff (A.gp sn sn) (Blade.of_int 0) in
  sn + (expand batch (s2 *$ 0.5) * place (beside s) einf)

let plane ~normal ~offset =
  let batch = batch_of_shape (shape normal) in
  let x, y, z = take3 normal in
  let x, y, z, d =
    match broadcast_arrays [ x; y; z; offset ] with
    | [ x; y; z; d ] -> x, y, z, d
    | _ -> assert false
  in
  dense batch [ 1, x; 2, y; 4, z; 8, d; 16, d ]

let plane_normal p = stack ~axis:(-1) [ comp p 1; comp p 2; comp p 4 ]
let plane_offset p = comp p 8

let normalize_plane p =
  let n = plane_normal p in
  let n2 = sum ~axes:[ Array.length (shape n) - 1 ] Infix.(n * n) in
  Infix.(p / expand (A.batch_shape p) (Nx.sqrt n2))

let normalize_point p =
  let open Infix in
  let b = comp p 16 - comp p 8 in
  p / expand (A.batch_shape p) b

let normalize_sphere = normalize_point

let is_zero a =
  let r = Array.length (shape a) in
  let s = sum ~axes:[ r - 1 ] (Nx.abs a) in
  less_s s 1e-5

let is_flat a = is_zero (A.inner (place (beside a) einf) a)
let is_round a = logical_not (is_flat a)
let point_pair x y = A.op x y
let line x y = A.dual (A.op (A.op x y) (place (beside x) einf))
let line_of_planes p1 p2 = A.op p1 p2
let circle x y z = A.dual (A.op (A.op x y) z)
let circle_of_spheres s1 s2 = A.op s1 s2
let plane_through x y z = A.dual (A.op (A.op (A.op x y) z) (place (beside x) einf))
let sphere_through x y z w = A.dual (A.op (A.op (A.op x y) z) w)
let incident x o = is_zero (A.inner x o)
let distance2 x y = Infix.(A.scalar_prod x y *$ -2.)

(* The root reads a placeholder where [distance2] is not positive, so that
   neither its value nor its infinite derivative at zero reaches the result:
   coincident points get distance zero and derivative zero in both modes. *)
let distance x y =
  let d2 = distance2 x y in
  let pos = greater_s d2 0. in
  where pos (Nx.sqrt (where pos d2 (ones_like d2))) (zeros_like d2)

let meet a b = A.op a b
let coeff_i a i = A.coeff a (Blade.of_int i)

let line_scale l =
  let b12 = coeff_i l 3
  and b13 = coeff_i l 5
  and b23 = coeff_i l 6 in
  Nx.sqrt Infix.((b12 * b12) + (b13 * b13) + (b23 * b23))

let line_direction l =
  let open Infix in
  let b12 = coeff_i l 3
  and b13 = coeff_i l 5
  and b23 = coeff_i l 6 in
  let s = line_scale l in
  stack ~axis:(-1) [ b23 / s; neg b13 / s; b12 / s ]

let line_moment l =
  let open Infix in
  let s = line_scale l in
  stack ~axis:(-1) [ coeff_i l 9 / s; coeff_i l 10 / s; coeff_i l 12 / s ]

let cross3 ax ay az bx by bz =
  let open Infix in
  let cx = (ay * bz) - (az * by) in
  let cy = (az * bx) - (ax * bz) in
  let cz = (ax * by) - (ay * bx) in
  cx, cy, cz

let circle_frame c =
  (* Recover the null-basis coefficients of the circle blade. In the null basis
     a circle has z = B + u_i e_i e_0 + v_i e_i e_inf + w e_0 e_inf, with u the
     normal of its carrier plane (not necessarily unit), w = -d and
     v = (1/2)(c^2 - r^2) n - d c. *)
  let open Infix in
  let a1 = coeff_i c 9
  and a2 = coeff_i c 10
  and a3 = coeff_i c 12 in
  let b1 = coeff_i c 17
  and b2 = coeff_i c 18
  and b3 = coeff_i c 20 in
  let u1 = b1 - a1
  and u2 = b2 - a2
  and u3 = b3 - a3 in
  let s = Nx.sqrt ((u1 * u1) + (u2 * u2) + (u3 * u3)) in
  let n1 = u1 / s
  and n2 = u2 / s
  and n3 = u3 / s in
  let w = coeff_i c 24 in
  let d = w / s in
  let b12 = coeff_i c 3 / s in
  let b13 = coeff_i c 5 / s in
  let b23 = coeff_i c 6 / s in
  let v1 = (a1 + b1) / (s *$ 2.) in
  let v2 = (a2 + b2) / (s *$ 2.) in
  let v3 = (a3 + b3) / (s *$ 2.) in
  let vpar = (v1 * n1) + (v2 * n2) + (v3 * n3) in
  let b2 = (b12 * b12) + (b13 * b13) + (b23 * b23) in
  let r2 = b2 - (d * d) - (vpar *$ 2.) in
  let wx = b23
  and wy = -b13
  and wz = b12 in
  let cx, cy, cz = cross3 wx wy wz n1 n2 n3 in
  let cx = (d * n1) + cx in
  let cy = (d * n2) + cy in
  let cz = (d * n3) + cz in
  n1, n2, n3, cx, cy, cz, r2

let circle_normal c =
  let n1, n2, n3, _, _, _, _ = circle_frame c in
  stack ~axis:(-1) [ n1; n2; n3 ]

let circle_center c =
  let _, _, _, cx, cy, cz, _ = circle_frame c in
  point (stack ~axis:(-1) [ cx; cy; cz ])

let circle_radius2 c =
  let _, _, _, _, _, _, r2 = circle_frame c in
  r2

let dilator lambda =
  let open Infix in
  let batch = shape lambda in
  let half = lambda *$ 0.5 in
  let c = Nx.cosh half
  and s = Nx.sinh half in
  dense batch [ 0, c; 24, s *$ -1. ]

let transversion a =
  let open Infix in
  let batch = batch_of_shape (shape a) in
  let ax, ay, az = take3 a in
  let one = constant (beside a) batch 1. in
  let hx = ax *$ 0.5
  and hy = ay *$ 0.5
  and hz = az *$ 0.5 in
  dense batch [ 0, one; 9, hx; 17, -hx; 10, hy; 18, -hy; 12, hz; 20, -hz ]

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
  create dty [| 8; 32 |] e

let of_motor m =
  let sh = shape m in
  let batch = Array.sub sh 0 (Array.length sh - 1) in
  let bsz = Array.fold_left ( * ) 1 batch in
  let m2 = reshape [| bsz; 8 |] m in
  let out = Infix.(m2 *@ place (beside m2) motor_embedding) in
  reshape (Array.append batch [| 32 |]) out

let apply_motor m x = A.sandwich (of_motor m) x
