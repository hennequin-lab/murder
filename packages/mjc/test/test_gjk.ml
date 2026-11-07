(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GJK against shapes whose answer is known in closed form.

   The contact test is a yes/no question about convex geometry, and four of the
   engine's primitive pairs answer it analytically -- sphere-sphere,
   sphere-capsule, capsule-capsule and box-box. Those are the oracle: for random
   poses, GJK over {!Convex.support} must say "touching" exactly when the closed
   form says the distance is negative. A convex path that cannot reproduce its own
   primitives is not ready for the pairs that have no primitive at all.

   The box-box case is axis-aligned so that the closed form is the separating axis
   test over three axes, which is exact for that case and does not need a
   manifold. *)

open Base
module Printf = Stdlib.Printf
open Windtrap
module Convex = Mjc.Convex
module Gjk = Mjc.Gjk

let identity = [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]

(* A rotation about a random axis by a random angle, as a row-major 3x3. *)
let rotation rng =
  let axis =
    let v = Array.init 3 ~f:(fun _ -> Random.State.float rng 2. -. 1.) in
    let n = Float.sqrt ((v.(0) *. v.(0)) +. (v.(1) *. v.(1)) +. (v.(2) *. v.(2))) in
    if Float.( <= ) n 1e-6 then [| 1.; 0.; 0. |] else Array.map v ~f:(fun x -> x /. n)
  in
  let angle = Random.State.float rng (2. *. Float.pi) in
  let c = Float.cos angle
  and s = Float.sin angle in
  let x = axis.(0)
  and y = axis.(1)
  and z = axis.(2) in
  [| c +. (x *. x *. (1. -. c))
   ; (x *. y *. (1. -. c)) -. (z *. s)
   ; (x *. z *. (1. -. c)) +. (y *. s)
   ; (y *. x *. (1. -. c)) +. (z *. s)
   ; c +. (y *. y *. (1. -. c))
   ; (y *. z *. (1. -. c)) -. (x *. s)
   ; (z *. x *. (1. -. c)) -. (y *. s)
   ; (z *. y *. (1. -. c)) +. (x *. s)
   ; c +. (z *. z *. (1. -. c))
  |]

let near rng scale =
  Array.init 3 ~f:(fun _ -> Random.State.float rng (2. *. scale) -. scale)

let geom shape ~pos ~mat ~size = Gjk.of_geom shape ~pos ~mat ~size ~margin:0.

let sphere rng =
  let pos = near rng 1.0 in
  let size = [| Random.State.float rng 0.4 +. 0.05; 0.; 0. |] in
  geom Convex.Sphere ~pos ~mat:identity ~size, (pos, size)

let capsule rng =
  let pos = near rng 1.0 in
  let mat = rotation rng in
  let size =
    [| Random.State.float rng 0.2 +. 0.05; Random.State.float rng 0.4 +. 0.05; 0. |]
  in
  geom Convex.Capsule ~pos ~mat ~size, (pos, mat, size)

(* Closed forms, in the same terms the engine's primitive functions use. *)

let sq x = x *. x

let dist3 a b =
  Float.sqrt (sq (a.(0) -. b.(0)) +. sq (a.(1) -. b.(1)) +. sq (a.(2) -. b.(2)))

let axis mat = [| mat.(2); mat.(5); mat.(8) |]
let dot3 a b = (a.(0) *. b.(0)) +. (a.(1) *. b.(1)) +. (a.(2) *. b.(2))
let sub3 a b = [| a.(0) -. b.(0); a.(1) -. b.(1); a.(2) -. b.(2) |]

(* The closest points of two segments: the standard clamped pair. *)
let segment_distance (p1, a1, l1) (p2, a2, l2) =
  let s1 =
    [| p1.(0) -. (a1.(0) *. l1); p1.(1) -. (a1.(1) *. l1); p1.(2) -. (a1.(2) *. l1) |]
  and s2 =
    [| p1.(0) +. (a1.(0) *. l1); p1.(1) +. (a1.(1) *. l1); p1.(2) +. (a1.(2) *. l1) |]
  and t1 =
    [| p2.(0) -. (a2.(0) *. l2); p2.(1) -. (a2.(1) *. l2); p2.(2) -. (a2.(2) *. l2) |]
  and t2 =
    [| p2.(0) +. (a2.(0) *. l2); p2.(1) +. (a2.(1) *. l2); p2.(2) +. (a2.(2) *. l2) |]
  in
  let best = ref 1e9 in
  let step = 400 in
  for i = 0 to step do
    for j = 0 to step do
      let u = Float.of_int i /. Float.of_int step
      and v = Float.of_int j /. Float.of_int step in
      let a =
        [| s1.(0) +. (u *. (s2.(0) -. s1.(0)))
         ; s1.(1) +. (u *. (s2.(1) -. s1.(1)))
         ; s1.(2) +. (u *. (s2.(2) -. s1.(2)))
        |]
      in
      let b =
        [| t1.(0) +. (v *. (t2.(0) -. t1.(0)))
         ; t1.(1) +. (v *. (t2.(1) -. t1.(1)))
         ; t1.(2) +. (v *. (t2.(2) -. t1.(2)))
        |]
      in
      best := Float.min !best (dist3 a b)
    done
  done;
  !best

let test_sphere_sphere () =
  let rng = Random.State.make [| 7 |] in
  let mismatches = ref 0 in
  for _ = 1 to 400 do
    let (ga, (pa, sa)), (gb, (pb, sb)) = sphere rng, sphere rng in
    let analytic = dist3 pa pb -. sa.(0) -. sb.(0) in
    let gjk = Gjk.intersect ga gb in
    if Bool.( <> ) gjk (Float.( < ) analytic 0.)
    then (
      Int.incr mismatches;
      if Int.( <= ) !mismatches 2
      then (
        Printf.printf
          "  sphere-sphere: analytic dist %.4f, gjk says %b (r %.3f %.3f, |c| %.3f)\n"
          analytic
          gjk
          sa.(0)
          sb.(0)
          (dist3 pa pb);
        ignore (Gjk.intersect ~trace:true ga gb)))
  done;
  if Int.( > ) !mismatches 0
  then failwith (Printf.sprintf "%d of 400 sphere pairs disagree" !mismatches)

let test_sphere_capsule () =
  let rng = Random.State.make [| 11 |] in
  let mismatches = ref 0 in
  for _ = 1 to 300 do
    let gs, (ps, ss) = sphere rng
    and gc, (pc, mc, sc) = capsule rng in
    let ac = axis mc in
    let along = dot3 ac (sub3 ps pc) in
    let along = Float.max (-.sc.(1)) (Float.min sc.(1) along) in
    let nearest =
      [| pc.(0) +. (ac.(0) *. along)
       ; pc.(1) +. (ac.(1) *. along)
       ; pc.(2) +. (ac.(2) *. along)
      |]
    in
    let analytic = dist3 ps nearest -. ss.(0) -. sc.(0) in
    let gjk = Gjk.intersect gs gc in
    if Bool.( <> ) gjk (Float.( < ) analytic 0.)
    then (
      Int.incr mismatches;
      if Int.( <= ) !mismatches 3
      then
        Printf.printf "  sphere-capsule: analytic dist %.4f, gjk says %b\n" analytic gjk)
  done;
  if Int.( > ) !mismatches 0
  then failwith (Printf.sprintf "%d of 300 sphere-capsule pairs disagree" !mismatches)

let test_capsule_capsule () =
  let rng = Random.State.make [| 13 |] in
  let mismatches = ref 0 in
  for _ = 1 to 120 do
    let ga, (pa, ma, sa) = capsule rng
    and gb, (pb, mb, sb) = capsule rng in
    let analytic =
      segment_distance (pa, axis ma, sa.(1)) (pb, axis mb, sb.(1)) -. sa.(0) -. sb.(0)
    in
    let gjk = Gjk.intersect ga gb in
    if Bool.( <> ) gjk (Float.( < ) analytic 0.)
    then (
      Int.incr mismatches;
      if Int.( <= ) !mismatches 3
      then
        Printf.printf "  capsule-capsule: analytic dist %.4f, gjk says %b\n" analytic gjk)
  done;
  if Int.( > ) !mismatches 0
  then failwith (Printf.sprintf "%d of 120 capsule pairs disagree" !mismatches)

(* Axis-aligned boxes: overlap on all three axes. *)
let test_box_box () =
  let rng = Random.State.make [| 17 |] in
  let mismatches = ref 0 in
  for _ = 1 to 300 do
    let ha = Array.init 3 ~f:(fun _ -> Random.State.float rng 0.3 +. 0.05)
    and hb = Array.init 3 ~f:(fun _ -> Random.State.float rng 0.3 +. 0.05) in
    let pa = near rng 0.6
    and pb = near rng 0.6 in
    let overlap =
      Array.for_all (Array.init 3 ~f:Fn.id) ~f:(fun i ->
        Float.( < ) (Float.abs (pa.(i) -. pb.(i))) (ha.(i) +. hb.(i)))
    in
    let ga = geom Convex.Box ~pos:pa ~mat:identity ~size:ha
    and gb = geom Convex.Box ~pos:pb ~mat:identity ~size:hb in
    let gjk = Gjk.intersect ga gb in
    if Bool.( <> ) gjk overlap
    then (
      Int.incr mismatches;
      if Int.( <= ) !mismatches 3
      then Printf.printf "  box-box: sat says %b, gjk says %b\n" overlap gjk)
  done;
  if Int.( > ) !mismatches 0
  then failwith (Printf.sprintf "%d of 300 box pairs disagree" !mismatches)

(* A margin widens a shape: each object contributes half of its own margin, so a
   caller passing the *pair* margin to both -- which is what MuJoCo does -- gets
   the pair margin as the contact threshold. *)
let test_margin () =
  let sphere_at x = [| x; 0.; 0. |] in
  let a =
    Gjk.of_geom
      Convex.Sphere
      ~pos:(sphere_at 0.)
      ~mat:identity
      ~size:[| 0.3; 0.; 0. |]
      ~margin:0.4
  and b =
    Gjk.of_geom
      Convex.Sphere
      ~pos:(sphere_at 0.75)
      ~mat:identity
      ~size:[| 0.3; 0.; 0. |]
      ~margin:0.4
  in
  (* 0.15 apart, 0.4 of margin shared: touching. *)
  if not (Gjk.intersect a b)
  then failwith "a 0.15 gap with 0.4 of shared margin should touch";
  let c =
    Gjk.of_geom
      Convex.Sphere
      ~pos:(sphere_at 1.2)
      ~mat:identity
      ~size:[| 0.3; 0.; 0. |]
      ~margin:0.4
  and d =
    Gjk.of_geom
      Convex.Sphere
      ~pos:(sphere_at 0.)
      ~mat:identity
      ~size:[| 0.3; 0.; 0. |]
      ~margin:0.4
  in
  (* 0.6 apart, wider than the margin: not touching. *)
  if Gjk.intersect c d
  then failwith "a 0.6 gap with 0.4 of shared margin should not touch";
  (* And with no margin, the same 0.15 gap is a miss. *)
  let e =
    Gjk.of_geom
      Convex.Sphere
      ~pos:(sphere_at 0.)
      ~mat:identity
      ~size:[| 0.3; 0.; 0. |]
      ~margin:0.
  and f =
    Gjk.of_geom
      Convex.Sphere
      ~pos:(sphere_at 0.75)
      ~mat:identity
      ~size:[| 0.3; 0.; 0. |]
      ~margin:0.
  in
  if Gjk.intersect e f then failwith "a 0.15 gap with no margin should not touch"

(* The distance subalgorithm against a brute force: the origin's distance to the
   convex hull, by dense barycentric sampling. If this disagrees, GJK cycles. *)
let test_closest () =
  let rng = Random.State.make [| 23 |] in
  let brute ps =
    let ps = Array.of_list ps in
    let n = Array.length ps in
    let best = ref 1e9 in
    let steps = 60 in
    (* Enumerate barycentric weights on a simplex grid and keep the best. *)
    let rec go i acc remaining =
      if Int.equal i (n - 1)
      then (
        let w = Array.append acc [| remaining |] in
        let p = ref [| 0.; 0.; 0. |] in
        for j = 0 to n - 1 do
          for k = 0 to 2 do
            p
            := Array.mapi !p ~f:(fun kk v ->
                 if Int.equal kk k then v +. (w.(j) *. ps.(j).(k)) else v)
          done
        done;
        let p = !p in
        best := Float.min !best (Float.sqrt (dot3 p p)))
      else
        for s = 0 to steps do
          let x = remaining *. Float.of_int s /. Float.of_int steps in
          go (i + 1) (Array.append acc [| x |]) (remaining -. x)
        done
    in
    go 0 [||] 1.;
    !best
  in
  let worst = ref 0. in
  for _ = 1 to 300 do
    let n = 1 + Random.State.int rng 4 in
    let ps = List.init n ~f:(fun _ -> near rng 1.0) in
    let got = Gjk.closest_point ps in
    let g = Float.sqrt (dot3 got got) in
    let b = brute ps in
    worst := Float.max !worst (g -. b);
    if Float.( > ) (g -. b) 0.05 && Float.( < ) !worst 1e8
    then (
      worst := 1e9;
      Printf.printf
        "  n=%d got %.4f (%.3f %.3f %.3f) brute %.4f\n"
        n
        g
        got.(0)
        got.(1)
        got.(2)
        b;
      List.iter ps ~f:(fun p ->
        Printf.printf "    p (%.3f %.3f %.3f)\n" p.(0) p.(1) p.(2)))
  done;
  if Float.( > ) !worst 0.02
  then failwith (Printf.sprintf "closest point beats brute force by up to %.4f" !worst)

let () =
  Stdlib.exit
    (run
       "mjc gjk"
       [ test "sphere-sphere agrees with the closed form" test_sphere_sphere
       ; test "sphere-capsule agrees with the closed form" test_sphere_capsule
       ; test "capsule-capsule agrees with the closed form" test_capsule_capsule
       ; test "box-box agrees with the separating axis test" test_box_box
       ; test "a margin widens a shape" test_margin
       ; test "the distance subalgorithm matches a brute force" test_closest
       ])
