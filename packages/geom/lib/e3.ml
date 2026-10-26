(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A =
  Algebra.Make
    (struct
      let signature = Signature.euclidean 3
    end)
    (struct
      type kind = Nx.float32_elt

      let dtype = Nx.float32
    end)

include A
module V = Versor.Make (A)

let dtype = Nx.float32
let dty = Nx.float32
let batch_shape = A.batch_shape
let expand batch x = Nx.reshape (Array.append batch [| 1 |]) x

let batch_of_shape sh =
  if Array.length sh = 0 then [||] else Array.sub sh 0 (Array.length sh - 1)

let dense batch comps =
  let z = Nx.zeros dty batch in
  let a = Array.make 8 z in
  List.iter (fun (i, v) -> a.(i) <- v) comps;
  Nx.stack ~axis:(-1) (Array.to_list a)

let take3 v =
  let r = Array.length (Nx.shape v) in
  let f i = Nx.take ~axis:(r - 1) ~indices:(Nx.scalar Nx.int64 (Int64.of_int i)) v in
  f 0, f 1, f 2

let e1 = A.e 1
let e2 = A.e 2
let e3 = A.e 3

let vector v =
  let batch = batch_of_shape (Nx.shape v) in
  let x, y, z = take3 v in
  dense batch [ 1, x; 2, y; 4, z ]

let vector3 x y z = vector (Nx.create dty [| 3 |] [| x; y; z |])

let to_vector a =
  Nx.stack
    ~axis:(-1)
    [ A.coeff a (Blade.of_int 1); A.coeff a (Blade.of_int 2); A.coeff a (Blade.of_int 4) ]

let rotor ~axis ~angle =
  let batch = batch_shape axis in
  let n2 = Nx.sum ~axes:[ Array.length (Nx.shape axis) - 1 ] (Nx.mul axis axis) in
  let n = Nx.sqrt n2 in
  let axis = Nx.div axis (expand batch n) in
  let nx, ny, nz = take3 axis in
  let half = Nx.mul_s angle (-0.5) in
  let bx = Nx.mul half nx
  and by = Nx.mul half ny
  and bz = Nx.mul half nz in
  let b = dense batch [ 3, bz; 5, Nx.mul_s by (-1.); 6, bx ] in
  V.exp_simple b

let apply_rotor r x = A.sandwich r x
