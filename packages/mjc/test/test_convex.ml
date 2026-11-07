(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The support functions, checked against the property that defines them: the
   support point is the farthest point of the shape along the direction, so no
   sampled point of the shape may beat it.

   This is the part of the convex collision path that can be verified without
   implementing the iteration on top of it, which is why it comes first: when GJK
   and MPR arrive, a wrong answer can be blamed on them rather than on the
   geometry they are fed. *)

open Base
open Windtrap
module Convex = Mjc.Convex

let tol = 1e-12

(* Points on the shape, sampled finely enough that a wrong support point cannot
   hide: vertices for the polytopes, a grid over each surface for the curved
   ones. *)
let samples = function
  | Convex.Point -> [ 0., 0., 0. ]
  | Sphere ->
    List.concat_map [ -1.0; -0.5; 0.0; 0.5; 1.0 ] ~f:(fun a ->
      List.concat_map [ -1.0; -0.5; 0.0; 0.5; 1.0 ] ~f:(fun b ->
        List.concat_map [ -1.0; 0.0; 1.0 ] ~f:(fun c ->
          let n = Float.sqrt ((a *. a) +. (b *. b) +. (c *. c)) in
          if Float.( <= ) n 1.0 && Float.( > ) n 1e-9
          then [ a /. n, b /. n, c /. n ]
          else [])))
  | Line -> [ 0., 0., -1.; 0., 0., 1. ]
  | Capsule ->
    List.concat_map [ -1.0; 0.0; 1.0 ] ~f:(fun z ->
      List.concat_map [ 0.0; 1.0; -1.0 ] ~f:(fun y -> [ 0., y, z; 1., 0., z; 0., 0., z ]))
  | Ellipsoid ->
    List.concat_map [ -1.0; 0.0; 1.0 ] ~f:(fun x ->
      List.concat_map [ -1.0; 0.0; 1.0 ] ~f:(fun y ->
        List.concat_map [ -1.0; 0.0; 1.0 ] ~f:(fun z ->
          let n = Float.sqrt ((x *. x) +. (y *. y) +. (z *. z)) in
          if Float.( > ) n 1e-9 then [ x /. n, y /. n, z /. n ] else [])))
  | Cylinder ->
    List.concat_map [ -1.0; 0.0; 1.0 ] ~f:(fun z ->
      List.concat_map [ 0.0; 0.7071; 1.0; -1.0; -0.7071 ] ~f:(fun y ->
        let x = Float.sqrt (Float.max 0.0 (1.0 -. (y *. y))) in
        [ x, y, z; Float.neg x, y, z ]))
  | Box ->
    List.concat_map [ -1.0; 1.0 ] ~f:(fun x ->
      List.concat_map [ -1.0; 1.0 ] ~f:(fun y ->
        List.concat_map [ -1.0; 1.0 ] ~f:(fun z -> [ x, y, z ])))

(* The size triple each shape is defined by: [size] in MuJoCo's convention. *)
let size_of = function
  | Convex.Point -> [| 0.; 0.; 0. |]
  | Sphere -> [| 0.5; 0.; 0. |]
  | Line -> [| 0.; 0.7; 0. |]
  | Capsule -> [| 0.25; 0.7; 0. |]
  | Ellipsoid -> [| 0.8; 0.4; 0.6 |]
  | Cylinder -> [| 0.3; 0.5; 0. |]
  | Box -> [| 0.2; 0.35; 0.1 |]

(* A rotation that is not a multiple of a quarter turn, so a transposed matrix
   or a row/column mix-up cannot pass. *)
let mat =
  let a = 0.4
  and b = -0.7
  and c = 1.1 in
  let ca, sa = Float.cos a, Float.sin a
  and cb, sb = Float.cos b, Float.sin b
  and cc, sc = Float.cos c, Float.sin c in
  Array.concat
    [ [| ca *. cb; (ca *. sb *. sc) -. (sa *. cc); (ca *. sb *. cc) +. (sa *. sc) |]
    ; [| sa *. cb; (sa *. sb *. sc) +. (ca *. cc); (sa *. sb *. cc) -. (ca *. sc) |]
    ; [| -.sb; cb *. sc; cb *. cc |]
    ]

let pos = [| 0.3; -0.2; 1.4 |]
let point_of p = Nx.create Nx.float64 [| 3 |] p

let apply m p =
  Array.init 3 ~f:(fun i ->
    let i3 = i * 3 in
    (m.(i3) *. p.(0)) +. (m.(i3 + 1) *. p.(1)) +. (m.(i3 + 2) *. p.(2)))

let dot a b = (a.(0) *. b.(0)) +. (a.(1) *. b.(1)) +. (a.(2) *. b.(2))

let check shape direction =
  let s =
    Convex.support
      shape
      ~pos:(point_of pos)
      ~mat:(Nx.create Nx.float64 [| 3; 3 |] mat)
      ~size:(point_of (size_of shape))
      ~dir:(point_of direction)
  in
  let support = Nx.to_array s in
  let best = dot support direction in
  List.iter (samples shape) ~f:(fun (lx, ly, lz) ->
    (* The sample is in the shape's own frame. Its sizes are MuJoCo's -- radius
       and half-length for a capsule or a cylinder, not an extent per axis -- so
       scaling it has to know which is which. *)
    let sz = size_of shape in
    let local =
      match shape with
      | Convex.Sphere -> [| lx *. sz.(0); ly *. sz.(0); lz *. sz.(0) |]
      | Line -> [| 0.; 0.; lz *. sz.(1) |]
      | Capsule -> [| lx *. sz.(0); ly *. sz.(0); lz *. sz.(1) |]
      | Cylinder -> [| lx *. sz.(0); ly *. sz.(0); lz *. sz.(1) |]
      | Point -> [| 0.; 0.; 0. |]
      | Ellipsoid | Box -> [| lx *. sz.(0); ly *. sz.(1); lz *. sz.(2) |]
    in
    let world = apply mat local in
    let world = Array.map2_exn world pos ~f:( +. ) in
    let value = dot world direction in
    if Float.( > ) value (best +. tol)
    then
      failwith
        (Printf.sprintf
           "shape %s dir (%g %g %g): a sample reaches %.17g but the support only %.17g"
           (match shape with
            | Convex.Point -> "point"
            | Sphere -> "sphere"
            | Line -> "line"
            | Capsule -> "capsule"
            | Ellipsoid -> "ellipsoid"
            | Cylinder -> "cylinder"
            | Box -> "box")
           direction.(0)
           direction.(1)
           direction.(2)
           value
           best))

(* Unit vectors only, and that is a property of the interface rather than a
   convenience: MuJoCo's sphere support answers [radius * dir + pos] without
   normalizing, so it is the support function for unit directions and libccd is
   what guarantees they are unit. Checking it against a non-unit direction would
   fail on the shape rather than on the code. *)
let unit v =
  let n = Float.sqrt ((v.(0) *. v.(0)) +. (v.(1) *. v.(1)) +. (v.(2) *. v.(2))) in
  Array.map v ~f:(fun x -> x /. n)

let directions =
  List.map
    [ [| 1.; 0.; 0. |]
    ; [| 0.; 1.; 0. |]
    ; [| 0.; 0.; 1. |]
    ; [| -1.; 0.; 0. |]
    ; [| 0.; -1.; 0. |]
    ; [| 0.; 0.; -1. |]
    ; [| 0.3; 0.5; -0.8 |]
    ; [| -0.6; 0.2; 0.75 |]
    ; [| 0.577; 0.577; 0.577 |]
    ]
    ~f:unit

let shapes = [ Convex.Point; Sphere; Line; Capsule; Ellipsoid; Cylinder; Box ]

let test_support () =
  List.iter shapes ~f:(fun shape ->
    List.iter directions ~f:(fun direction -> check shape direction))

(* The batch is the same arithmetic over a leading axis, and the contact stage
   will feed it one: every lane must answer what the single-point call does. *)
let test_batch () =
  let k = List.length directions in
  let point_batch name f =
    Nx.create Nx.float64 [| k; 3 |] (Array.concat (List.map directions ~f))
    |> fun x ->
    ignore name;
    x
  in
  let dirs = point_batch "dir" Fn.id in
  let pos_batch =
    Nx.create Nx.float64 [| k; 3 |] (Array.concat (List.init k ~f:(fun _ -> pos)))
  in
  let mat_batch =
    Nx.create Nx.float64 [| k; 3; 3 |] (Array.concat (List.init k ~f:(fun _ -> mat)))
  in
  List.iter shapes ~f:(fun shape ->
    let sizes = Array.concat (List.init k ~f:(fun _ -> size_of shape)) in
    let size_batch = Nx.create Nx.float64 [| k; 3 |] sizes in
    let batched =
      Convex.support shape ~pos:pos_batch ~mat:mat_batch ~size:size_batch ~dir:dirs
      |> Nx.to_array
    in
    List.iteri directions ~f:(fun i direction ->
      let single =
        Convex.support
          shape
          ~pos:(point_of pos)
          ~mat:(Nx.create Nx.float64 [| 3; 3 |] mat)
          ~size:(point_of (size_of shape))
          ~dir:(point_of direction)
        |> Nx.to_array
      in
      Array.iteri single ~f:(fun j want ->
        let got = batched.((i * 3) + j) in
        if Float.( > ) (Float.abs (got -. want)) (Float.max 1.0 (Float.abs want) *. 1e-14)
        then
          failwith
            (Printf.sprintf
               "lane %d component %d: batched %.17g, single %.17g"
               i
               j
               got
               want))))

let () =
  Stdlib.exit
    (run
       "mjc convex"
       [ test "the support point is the farthest point" test_support
       ; test "a batch answers what a single point does" test_batch
       ])
