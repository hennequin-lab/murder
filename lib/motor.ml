(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = Nx.float32_t

let dtype = Nx.float32
let size = 8

module A =
  Algebra.Make
    (struct
      let signature = Signature.conformal3
    end)
    (struct
      type kind = Nx.float32_elt

      let dtype = Nx.float32
    end)

(* The packed basis in the dense Cl(4,1) algebra. The selected dense indices
   form an identity submatrix, so a packed coefficient is read directly from
   the dense product. *)
let dense_basis =
  let make idxs =
    let a = Array.make 32 0. in
    Array.iter (fun i -> a.(i) <- 1.) idxs;
    A.of_array a
  in
  [| make [| 0 |]
   ; make [| 3 |]
   ; make [| 5 |]
   ; make [| 6 |]
   ; make [| 9; 17 |]
   ; make [| 10; 18 |]
   ; make [| 12; 20 |]
   ; make [| 15; 23 |]
  |]

let selected = [| 0; 3; 5; 6; 9; 10; 12; 15 |]

let packed_table =
  let t = Array.make (size * size * size) 0. in
  for i = 0 to size - 1 do
    for j = 0 to size - 1 do
      let r = A.to_array (A.gp dense_basis.(i) dense_basis.(j)) in
      for k = 0 to size - 1 do
        t.((((k * size) + i) * size) + j) <- r.(selected.(k))
      done
    done
  done;
  t

let packed_tbl = Nx.create dtype [| size; size; size |] packed_table

let embedding =
  let e = Array.make (size * 32) 0. in
  for k = 0 to size - 1 do
    let dense = A.to_array dense_basis.(k) in
    for d = 0 to 31 do
      e.((k * 32) + d) <- dense.(d)
    done
  done;
  Nx.create dtype [| size; 32 |] e

let reverse_sign = Nx.create dtype [| size |] [| 1.; -1.; -1.; -1.; -1.; -1.; -1.; 1. |]
let rank a = Array.length (Nx.shape a)

let batch_shape a =
  let s = Nx.shape a in
  if Array.length s = 0 then invalid_arg "Motor.batch_shape: missing motor axis";
  Array.sub s 0 (Array.length s - 1)

let last_axis a =
  let r = rank a in
  if r = 0 then invalid_arg "Motor: missing motor axis";
  r - 1

let take_comp a i =
  Nx.take ~axis:(last_axis a) ~indices:(Nx.scalar Nx.int64 (Int64.of_int i)) a

let expand batch x = Nx.reshape (Array.append batch [| 1 |]) x

let of_array a =
  if Array.length a <> size then invalid_arg "Motor.of_array: expected 8 coefficients";
  Nx.create dtype [| size |] a

let to_array m =
  if batch_shape m <> [||] then invalid_arg "Motor.to_array: expected an unbatched motor";
  Nx.to_array m

let of_nx x =
  let s = Nx.shape x in
  if Array.length s = 0 || s.(Array.length s - 1) <> size
  then invalid_arg "Motor.of_nx: last axis must have length 8";
  x

let to_nx x = x
let shape m = Nx.shape m
let one = of_array [| 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
let zero = of_array [| 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]

let contract a b =
  let a, b =
    match Nx.broadcast_arrays [ a; b ] with
    | [ a; b ] -> a, b
    | _ -> assert false
  in
  let batch = batch_shape a in
  let bsz = Array.fold_left ( * ) 1 batch in
  let a2 = Nx.reshape [| bsz; size |] a in
  let b2 = Nx.reshape [| bsz; size |] b in
  let out = Nx.einsum "bi,bj,kij->bk" [| a2; b2; packed_tbl |] in
  Nx.reshape (Array.append batch [| size |]) out

let mul a b = contract a b
let compose ms = List.fold_left mul one ms
let reverse m = Nx.mul m reverse_sign
let norm2 m = take_comp (mul m (reverse m)) 0
let norm m = Nx.sqrt (Nx.abs (norm2 m))
let normalize m = Nx.div m (expand (batch_shape m) (norm m))
let inverse m = Nx.div (reverse m) (expand (batch_shape m) (norm2 m))
let zero_comp batch = Nx.zeros dtype batch
let one_comp batch = Nx.ones dtype batch
let vector3 x y z = Nx.create dtype [| 3 |] [| x; y; z |]

let translate v =
  let batch = Array.sub (Nx.shape v) 0 (Array.length (Nx.shape v) - 1) in
  let z = zero_comp batch in
  let o = one_comp batch in
  let m = Nx.mul_s v (-0.5) in
  Nx.stack ~axis:(-1) [ o; z; z; z; take_comp m 0; take_comp m 1; take_comp m 2; z ]

let translate3 x y z = translate (vector3 x y z)

let cross a b =
  let ax = take_comp a 0
  and ay = take_comp a 1
  and az = take_comp a 2 in
  let bx = take_comp b 0
  and by = take_comp b 1
  and bz = take_comp b 2 in
  Nx.stack
    ~axis:(-1)
    [ Nx.sub (Nx.mul ay bz) (Nx.mul az by)
    ; Nx.sub (Nx.mul az bx) (Nx.mul ax bz)
    ; Nx.sub (Nx.mul ax by) (Nx.mul ay bx)
    ]

let rotate ~axis ~angle =
  let batch = batch_shape axis in
  let n2 = Nx.sum ~axes:[ Array.length (Nx.shape axis) - 1 ] (Nx.mul axis axis) in
  let n = Nx.sqrt n2 in
  let axis = Nx.div axis (expand batch n) in
  let half = Nx.mul_s angle 0.5 in
  let c = Nx.cos half
  and s = Nx.sin half in
  let nx = take_comp axis 0
  and ny = take_comp axis 1
  and nz = take_comp axis 2 in
  let z = zero_comp batch in
  Nx.stack
    ~axis:(-1)
    [ c
    ; Nx.mul_s (Nx.mul s nz) (-1.)
    ; Nx.mul s ny
    ; Nx.mul_s (Nx.mul s nx) (-1.)
    ; z
    ; z
    ; z
    ; z
    ]

let of_quaternion q =
  let batch = batch_shape q in
  let n2 = Nx.sum ~axes:[ Array.length (Nx.shape q) - 1 ] (Nx.mul q q) in
  let n = Nx.sqrt n2 in
  let q = Nx.div q (expand batch n) in
  let w = take_comp q 0
  and x = take_comp q 1
  and y = take_comp q 2
  and z = take_comp q 3 in
  let zero = zero_comp batch in
  Nx.stack ~axis:(-1) [ w; Nx.mul_s z (-1.); y; Nx.mul_s x (-1.); zero; zero; zero; zero ]

let to_quaternion m =
  let batch = batch_shape m in
  let w = take_comp m 0 in
  let e12 = take_comp m 1
  and e13 = take_comp m 2
  and e23 = take_comp m 3 in
  let q = Nx.stack ~axis:(-1) [ w; Nx.mul_s e23 (-1.); e13; Nx.mul_s e12 (-1.) ] in
  let n2 = Nx.sum ~axes:[ Array.length (Nx.shape q) - 1 ] (Nx.mul q q) in
  let q = Nx.div q (expand batch (Nx.sqrt n2)) in
  let negative = Nx.reshape (Array.append batch [| 1 |]) (Nx.less_s w 0.) in
  Nx.where negative (Nx.neg q) q

let of_rotation r =
  let sh = Nx.shape r in
  let rk = Array.length sh in
  if rk < 2 || sh.(rk - 2) <> 3 || sh.(rk - 1) <> 3
  then invalid_arg "Motor.of_rotation: expected a 3x3 matrix on the last axes";
  let batch = Array.sub sh 0 (rk - 2) in
  let bsz = Array.fold_left ( * ) 1 batch in
  let src = Nx.to_array (Nx.reshape [| bsz; 3; 3 |] r) in
  let out = Array.make (bsz * size) 0. in
  for b = 0 to bsz - 1 do
    let o = b * 9 in
    let m00 = src.(o)
    and m01 = src.(o + 1)
    and m02 = src.(o + 2) in
    let m10 = src.(o + 3)
    and m11 = src.(o + 4)
    and m12 = src.(o + 5) in
    let m20 = src.(o + 6)
    and m21 = src.(o + 7)
    and m22 = src.(o + 8) in
    let tr = m00 +. m11 +. m22 in
    let w, x, y, z =
      if tr > 0.
      then (
        let s = 2. *. sqrt (tr +. 1.) in
        0.25 *. s, (m21 -. m12) /. s, (m02 -. m20) /. s, (m10 -. m01) /. s)
      else if m00 > m11 && m00 > m22
      then (
        let s = 2. *. sqrt (1. +. m00 -. m11 -. m22) in
        (m21 -. m12) /. s, 0.25 *. s, (m01 +. m10) /. s, (m02 +. m20) /. s)
      else if m11 > m22
      then (
        let s = 2. *. sqrt (1. +. m11 -. m00 -. m22) in
        (m02 -. m20) /. s, (m01 +. m10) /. s, 0.25 *. s, (m12 +. m21) /. s)
      else (
        let s = 2. *. sqrt (1. +. m22 -. m00 -. m11) in
        (m10 -. m01) /. s, (m02 +. m20) /. s, (m12 +. m21) /. s, 0.25 *. s)
    in
    (* q = (w, x, y, z) maps to packed [w, -z, y, -x, 0, 0, 0, 0]. *)
    let o8 = b * size in
    out.(o8) <- w;
    out.(o8 + 1) <- -.z;
    out.(o8 + 2) <- y;
    out.(o8 + 3) <- -.x
  done;
  Nx.reshape (Array.append batch [| size |]) (Nx.create dtype [| bsz; size |] out)

let exp twist =
  let sh = Nx.shape twist in
  if sh.(Array.length sh - 1) <> 6
  then invalid_arg "Motor.exp: expected a twist on the last axis of length 6";
  let batch = Array.sub sh 0 (Array.length sh - 1) in
  let omega =
    Nx.stack ~axis:(-1) [ take_comp twist 0; take_comp twist 1; take_comp twist 2 ]
  in
  let v =
    Nx.stack ~axis:(-1) [ take_comp twist 3; take_comp twist 4; take_comp twist 5 ]
  in
  let theta =
    Nx.sqrt (Nx.sum ~axes:[ Array.length (Nx.shape omega) - 1 ] (Nx.mul omega omega))
  in
  let small = Nx.less_s theta 1e-6 in
  let theta_safe = Nx.where small (Nx.ones_like theta) theta in
  let theta2 = Nx.mul theta_safe theta_safe in
  let theta3 = Nx.mul theta2 theta_safe in
  let sin_t = Nx.sin theta
  and cos_t = Nx.cos theta in
  let b0 = Nx.div (Nx.sub (Nx.ones_like cos_t) cos_t) theta2 in
  let c0 = Nx.div (Nx.sub theta sin_t) theta3 in
  let one = Nx.ones_like theta in
  let b = Nx.where small (Nx.mul_s one 0.5) b0 in
  let c = Nx.where small (Nx.mul_s one (1. /. 6.)) c0 in
  let cross1 = cross omega v in
  let cross2 = cross omega cross1 in
  let t =
    Nx.add v (Nx.add (Nx.mul cross1 (expand batch b)) (Nx.mul cross2 (expand batch c)))
  in
  let half = Nx.mul_s theta 0.5 in
  let w = Nx.cos half in
  let s = Nx.sin half in
  let xyz = Nx.mul omega (expand batch (Nx.div s theta_safe)) in
  let q = Nx.stack ~axis:(-1) [ w; take_comp xyz 0; take_comp xyz 1; take_comp xyz 2 ] in
  mul (translate t) (of_quaternion q)

let raw_quaternion m =
  let w = take_comp m 0 in
  let e12 = take_comp m 1
  and e13 = take_comp m 2
  and e23 = take_comp m 3 in
  let q = Nx.stack ~axis:(-1) [ w; Nx.mul_s e23 (-1.); e13; Nx.mul_s e12 (-1.) ] in
  let batch = batch_shape m in
  let n2 = Nx.sum ~axes:[ Array.length (Nx.shape q) - 1 ] (Nx.mul q q) in
  Nx.div q (expand batch (Nx.sqrt n2))

let translation_of_motor m =
  (* Use the rotation representative that belongs to [m] itself. [to_quaternion]
     normalises the sign of the scalar part, which changes the sign of the whole
     rotor and would flip the translation extracted from the dual part. *)
  let q = raw_quaternion m in
  let r = of_quaternion q in
  let t = mul m (inverse r) in
  Nx.stack
    ~axis:(-1)
    [ Nx.mul_s (take_comp t 4) (-2.)
    ; Nx.mul_s (take_comp t 5) (-2.)
    ; Nx.mul_s (take_comp t 6) (-2.)
    ]

let log m =
  let q = to_quaternion m in
  let w = take_comp q 0
  and x = take_comp q 1
  and y = take_comp q 2
  and z = take_comp q 3 in
  let qv = Nx.sqrt (Nx.add (Nx.add (Nx.mul x x) (Nx.mul y y)) (Nx.mul z z)) in
  let small = Nx.less_s qv 1e-6 in
  let qv_safe = Nx.where small (Nx.ones_like qv) qv in
  let theta = Nx.mul_s (Nx.atan2 qv w) 2. in
  let scale = Nx.where small (Nx.mul_s (Nx.ones_like qv) 2.) (Nx.div theta qv_safe) in
  let omega = Nx.stack ~axis:(-1) [ Nx.mul scale x; Nx.mul scale y; Nx.mul scale z ] in
  let t = translation_of_motor m in
  let theta =
    Nx.sqrt (Nx.sum ~axes:[ Array.length (Nx.shape omega) - 1 ] (Nx.mul omega omega))
  in
  let half = Nx.mul_s theta 0.5 in
  let sin_half = Nx.sin half
  and cos_half = Nx.cos half in
  let small = Nx.less_s theta 1e-6 in
  let sin_safe = Nx.where small (Nx.ones_like sin_half) sin_half in
  let theta_safe = Nx.where small (Nx.ones_like theta) theta in
  let cot = Nx.div cos_half sin_safe in
  let d0 =
    Nx.div (Nx.sub (Nx.ones_like theta) (Nx.mul half cot)) (Nx.mul theta_safe theta_safe)
  in
  let d = Nx.where small (Nx.mul_s (Nx.ones_like theta) (1. /. 12.)) d0 in
  let oxt = cross omega t in
  let oxt2 = cross omega oxt in
  let v = Nx.add (Nx.sub t (Nx.mul_s oxt 0.5)) (Nx.mul oxt2 (expand (batch_shape t) d)) in
  Nx.stack
    ~axis:(-1)
    [ take_comp omega 0
    ; take_comp omega 1
    ; take_comp omega 2
    ; take_comp v 0
    ; take_comp v 1
    ; take_comp v 2
    ]

let slerp m0 m1 alpha =
  let rel = mul (inverse m0) m1 in
  let tw = log rel in
  let abatch = Nx.shape alpha in
  mul m0 (exp (Nx.mul tw (Nx.reshape (Array.append abatch [| 1 |]) alpha)))

let rotation_matrix q =
  let w = take_comp q 0
  and x = take_comp q 1
  and y = take_comp q 2
  and z = take_comp q 3 in
  let r00 = Nx.sub_s (Nx.mul_s (Nx.add (Nx.mul y y) (Nx.mul z z)) 2.) 1. in
  let r00 = Nx.mul_s r00 (-1.) in
  let r01 = Nx.mul_s (Nx.sub (Nx.mul x y) (Nx.mul w z)) 2. in
  let r02 = Nx.mul_s (Nx.add (Nx.mul x z) (Nx.mul w y)) 2. in
  let r10 = Nx.mul_s (Nx.add (Nx.mul x y) (Nx.mul w z)) 2. in
  let r11 = Nx.sub_s (Nx.mul_s (Nx.add (Nx.mul x x) (Nx.mul z z)) 2.) 1. in
  let r11 = Nx.mul_s r11 (-1.) in
  let r12 = Nx.mul_s (Nx.sub (Nx.mul y z) (Nx.mul w x)) 2. in
  let r20 = Nx.mul_s (Nx.sub (Nx.mul x z) (Nx.mul w y)) 2. in
  let r21 = Nx.mul_s (Nx.add (Nx.mul y z) (Nx.mul w x)) 2. in
  let r22 = Nx.sub_s (Nx.mul_s (Nx.add (Nx.mul x x) (Nx.mul y y)) 2.) 1. in
  let r22 = Nx.mul_s r22 (-1.) in
  let row a b c = Nx.stack ~axis:(-1) [ a; b; c ] in
  let rows = [ row r00 r01 r02; row r10 r11 r12; row r20 r21 r22 ] in
  Nx.stack ~axis:(Array.length (Nx.shape (List.hd rows)) - 1) rows

let translation m = translation_of_motor m
let rotation m = rotation_matrix (to_quaternion m)

let adjoint m =
  let q = to_quaternion m in
  let r = rotation_matrix q in
  let t = translation_of_motor m in
  let tx = take_comp t 0
  and ty = take_comp t 1
  and tz = take_comp t 2 in
  let zero = Nx.zeros_like tx in
  let skew =
    let row a b c = Nx.stack ~axis:(-1) [ a; b; c ] in
    let rows =
      [ row zero (Nx.neg tz) ty; row tz zero (Nx.neg tx); row (Nx.neg ty) tx zero ]
    in
    Nx.stack ~axis:(Array.length (Nx.shape (List.hd rows)) - 1) rows
  in
  let skew_r = Nx.matmul skew r in
  let z3 = Nx.zeros_like r in
  let top = Nx.concatenate ~axis:(-1) [ r; z3 ] in
  let bottom = Nx.concatenate ~axis:(-1) [ skew_r; r ] in
  Nx.concatenate ~axis:(Array.length (Nx.shape top) - 2) [ top; bottom ]

let pp ppf m =
  let sh = Nx.shape m in
  if Array.length sh = 1 && sh.(0) = size
  then (
    let a = Nx.to_array m in
    Format.fprintf
      ppf
      "@[<hov 1>[%g; %g; %g; %g; %g; %g; %g; %g]@]"
      a.(0)
      a.(1)
      a.(2)
      a.(3)
      a.(4)
      a.(5)
      a.(6)
      a.(7))
  else
    Format.fprintf
      ppf
      "<motor %s>"
      (String.concat "x" (Array.to_list (Array.map string_of_int sh)))
