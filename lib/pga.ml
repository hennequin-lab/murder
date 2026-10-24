(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A =
  Algebra.Make
    (struct
      let signature = Signature.projective 3
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

let batch_of_shape sh =
  if Array.length sh = 0 then [||] else Array.sub sh 0 (Array.length sh - 1)

let dense batch comps =
  let z = Nx.zeros dty batch in
  let a = Array.make 16 z in
  List.iter (fun (i, v) -> a.(i) <- v) comps;
  Nx.stack ~axis:(-1) (Array.to_list a)

let take3 v =
  let r = Array.length (Nx.shape v) in
  let f i = Nx.take ~axis:(r - 1) ~indices:(Nx.scalar Nx.int64 (Int64.of_int i)) v in
  f 0, f 1, f 2

let e1 = A.e 1
let e2 = A.e 2
let e3 = A.e 3
let e0 = A.e 4

let point v =
  let batch = batch_of_shape (Nx.shape v) in
  let x, y, z = take3 v in
  let p1 = Nx.sub e1 (Nx.mul (expand batch x) e0) in
  let p2 = Nx.sub e2 (Nx.mul (expand batch y) e0) in
  let p3 = Nx.sub e3 (Nx.mul (expand batch z) e0) in
  A.op (A.op p1 p2) p3

let point3 x y z = point (Nx.create dty [| 3 |] [| x; y; z |])
let weight p = A.coeff p (Blade.of_int 0b111)

let point_coords p =
  let w = weight p in
  let x = Nx.div (Nx.neg (A.coeff p (Blade.of_int 0b1110))) w in
  let y = Nx.div (A.coeff p (Blade.of_int 0b1101)) w in
  let z = Nx.div (Nx.neg (A.coeff p (Blade.of_int 0b1011))) w in
  Nx.stack ~axis:(-1) [ x; y; z ]

let is_ideal p = Nx.less_s (Nx.abs (weight p)) 1e-6
let normalize_point p = Nx.div p (expand (A.batch_shape p) (weight p))

let ideal_point d =
  let batch = batch_of_shape (Nx.shape d) in
  let dx, dy, dz = take3 d in
  let b = dense batch [ 3, dz; 5, Nx.mul_s dy (-1.); 6, dx ] in
  A.op e0 b

let plane ~normal ~offset =
  let batch = batch_of_shape (Nx.shape normal) in
  let x, y, z = take3 normal in
  let x, y, z, d =
    match Nx.broadcast_arrays [ x; y; z; offset ] with
    | [ x; y; z; d ] -> x, y, z, d
    | _ -> assert false
  in
  dense batch [ 1, x; 2, y; 4, z; 8, Nx.neg d ]

let plane3 x y z d =
  plane ~normal:(Nx.create dty [| 3 |] [| x; y; z |]) ~offset:(Nx.scalar dty d)

let plane_normal p =
  Nx.stack
    ~axis:(-1)
    [ A.coeff p (Blade.of_int 1); A.coeff p (Blade.of_int 2); A.coeff p (Blade.of_int 4) ]

let plane_offset p = Nx.neg (A.coeff p (Blade.of_int 8))

let normalize_plane p =
  let n = plane_normal p in
  let n2 = Nx.sum ~axes:[ Array.length (Nx.shape n) - 1 ] (Nx.mul n n) in
  Nx.div p (expand (A.batch_shape p) (Nx.sqrt n2))

let line_through p q = A.regressive p q
let line_of_planes p1 p2 = A.op p1 p2

let line_scale l =
  let b12 = A.coeff l (Blade.of_int 3) in
  let b13 = A.coeff l (Blade.of_int 5) in
  let b23 = A.coeff l (Blade.of_int 6) in
  Nx.sqrt (Nx.add (Nx.add (Nx.mul b12 b12) (Nx.mul b13 b13)) (Nx.mul b23 b23))

let line_direction l =
  let b12 = A.coeff l (Blade.of_int 3) in
  let b13 = A.coeff l (Blade.of_int 5) in
  let b23 = A.coeff l (Blade.of_int 6) in
  let s = line_scale l in
  Nx.stack ~axis:(-1) [ Nx.div b23 s; Nx.div (Nx.neg b13) s; Nx.div b12 s ]

let line_moment l =
  let s = line_scale l in
  Nx.stack
    ~axis:(-1)
    [ Nx.div (Nx.neg (A.coeff l (Blade.of_int 9))) s
    ; Nx.div (Nx.neg (A.coeff l (Blade.of_int 10))) s
    ; Nx.div (Nx.neg (A.coeff l (Blade.of_int 12))) s
    ]

let is_zero a =
  let r = Array.length (Nx.shape a) in
  let s = Nx.sum ~axes:[ r - 1 ] (Nx.abs a) in
  Nx.less_s s 1e-5

let point_on_plane p pi = is_zero (A.op p pi)
let point_on_line p l = is_zero (A.op (A.complement p) (A.complement l))
let line_in_plane l pi = is_zero (A.op l pi)
let meet a b = A.op a b
let join a b = A.regressive a b

let motor_embedding =
  let basis idxs =
    let a = Array.make 16 0. in
    Array.iter (fun i -> a.(i) <- 1.) idxs;
    a
  in
  let basis_neg idxs = Array.map (fun x -> -.x) (basis idxs) in
  (* The dual part of the packed basis is mapped with the opposite sign. This
     is the algebra automorphism eps -> -eps of the dual quaternions; it aligns
     the packed translator with the PGA point convention. *)
  let rows =
    [| basis [| 0 |]
     ; basis [| 3 |]
     ; basis [| 5 |]
     ; basis [| 6 |]
     ; basis_neg [| 9 |]
     ; basis_neg [| 10 |]
     ; basis_neg [| 12 |]
     ; basis_neg [| 15 |]
    |]
  in
  let e = Array.make (8 * 16) 0. in
  for k = 0 to 7 do
    for d = 0 to 15 do
      e.((k * 16) + d) <- rows.(k).(d)
    done
  done;
  Nx.create dty [| 8; 16 |] e

let of_motor m =
  let sh = Nx.shape m in
  let batch = Array.sub sh 0 (Array.length sh - 1) in
  let bsz = Array.fold_left ( * ) 1 batch in
  let m2 = Nx.reshape [| bsz; 8 |] m in
  let out = Nx.matmul m2 motor_embedding in
  Nx.reshape (Array.append batch [| 16 |]) out

let apply_motor m x = A.sandwich (of_motor m) x
