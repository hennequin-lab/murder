(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base

(* GJK over the support functions, deciding contact only.

   The simplex is a list of Minkowski-difference vertices. Each iteration asks
   the pair for the farthest point along the direction of the current closest
   point, and reduces the simplex to the face of it that is closest to the
   origin; if the origin is inside a tetrahedron the shapes touch, and if the new
   support point lies beyond the origin's plane the shapes cannot touch at all.
   That is the classical algorithm, and MuJoCo's is a variant of it with the
   Frank-Wolfe duality gap as its stopping rule. *)

type obj =
  { support : float array -> float array
  ; margin : float
  }

let sub a b = [| a.(0) -. b.(0); a.(1) -. b.(1); a.(2) -. b.(2) |]
let add a b = [| a.(0) +. b.(0); a.(1) +. b.(1); a.(2) +. b.(2) |]
let scale a s = [| a.(0) *. s; a.(1) *. s; a.(2) *. s |]
let dot a b = (a.(0) *. b.(0)) +. (a.(1) *. b.(1)) +. (a.(2) *. b.(2))

let cross a b =
  [| (a.(1) *. b.(2)) -. (a.(2) *. b.(1))
   ; (a.(2) *. b.(0)) -. (a.(0) *. b.(2))
   ; (a.(0) *. b.(1)) -. (a.(1) *. b.(0))
  |]

let norm a = Float.sqrt (dot a a)

let unit a =
  let n = norm a in
  if Float.( <= ) n 1e-300 then [| 1.; 0.; 0. |] else scale a (1. /. n)

(* A vertex of the simplex: the Minkowski point [support_a d - support_b (-d)] and
   the two witness points it came from. The witnesses are what turns a depth from
   EPA into a pair of contact points. *)
type vertex = { m : float array }

let support (a : obj) (b : obj) dir =
  let dir = unit dir in
  let opposite = scale dir (-1.) in
  (* Each support point is pushed out by half its own margin, which is what makes
     a margin widen a shape rather than shrink the other one. *)
  let push (o : obj) d p =
    if Float.( > ) o.margin 0. then add p (scale d (0.5 *. o.margin)) else p
  in
  let wa = push a dir (a.support dir)
  and wb = push b opposite (b.support opposite) in
  { m = sub wa wb }

(* Closest point to the origin on the simplex, and the sub-simplex that carries
   it. [points] is non-empty and at most four vertices. The sub-simplex is what
   the next iteration keeps, so it has to be the *minimal* one: keeping an
   interior point would make the loop stall. *)
let closest (points : vertex array) : float array * vertex list * float array =
  let segment v0 v1 =
    let e = sub v1.m v0.m in
    let denom = dot e e in
    let t = if Float.( <= ) denom 1e-300 then 0. else -.dot v0.m e /. denom in
    if Float.( <= ) t 0.
    then [| 1. |], [ v0 ], v0.m
    else if Float.( >= ) t 1.
    then [| 1. |], [ v1 ], v1.m
    else [| 1. -. t; t |], [ v0; v1 ], add v0.m (scale e t)
  in
  let triangle v0 v1 v2 =
    (* The closest point of a triangle is the closest of its three edges or, when
       the origin's projection onto its plane lands inside it, that projection.
       This is the minimum over a handful of independently-correct pieces rather
       than a cascade of Voronoi-region tests: the cascade is shorter and I got
       two of its inequalities backwards twice, which the brute-force test caught
       and would have been invisible in the gate on a corpus that mostly touches. *)
    let a = v0.m
    and b = v1.m
    and c = v2.m in
    let interior =
      let ab = sub b a
      and ac = sub c a in
      let n = cross ab ac in
      let n2 = dot n n in
      if Float.( <= ) n2 1e-300
      then None
      else (
        let p = scale n (dot n a /. n2) in
        (* Inside when [p] is on the same side of every edge as the opposite
           vertex. *)
        let side u v w = dot (cross (sub v u) (sub p u)) (cross (sub v u) (sub w u)) in
        if
          Float.( >= ) (side a b c) 0.
          && Float.( >= ) (side b c a) 0.
          && Float.( >= ) (side c a b) 0.
        then (
          let w = cross (sub b p) (sub c p) in
          let w2 = dot w w in
          let weights =
            if Float.( <= ) w2 1e-300
            then [| 1.; 0.; 0. |]
            else (
              let wa = cross (sub b p) (sub c p)
              and wb = cross (sub c p) (sub a p)
              and wc = cross (sub a p) (sub b p) in
              let wa = dot wa n
              and wb = dot wb n
              and wc = dot wc n in
              let s = wa +. wb +. wc in
              let s = if Float.( <= ) (Float.abs s) 1e-300 then 1. else s in
              [| wa /. s; wb /. s; wc /. s |])
          in
          Some (weights, [ v0; v1; v2 ], p))
        else None)
    in
    let candidates =
      [ segment v0 v1; segment v1 v2; segment v0 v2 ]
      @
      match interior with
      | Some x -> [ x ]
      | None -> []
    in
    List.fold_left
      candidates
      ~init:(Option.value_exn (List.hd candidates))
      ~f:(fun ((_, _, bp) as best) ((_, _, p) as cand) ->
        if Float.( < ) (dot p p) (dot bp bp) then cand else best)
  in
  let det3 m =
    (m.(0) *. ((m.(4) *. m.(8)) -. (m.(5) *. m.(7))))
    -. (m.(1) *. ((m.(3) *. m.(8)) -. (m.(5) *. m.(6))))
    +. (m.(2) *. ((m.(3) *. m.(7)) -. (m.(4) *. m.(6))))
  in
  let det4 rows =
    let minor i j =
      let others =
        Array.filteri rows ~f:(fun r _ -> not (Int.equal r i)) |> Array.to_list
      in
      let strip row = Array.filteri row ~f:(fun c _ -> not (Int.equal c j)) in
      Array.concat (List.map others ~f:strip)
    in
    Array.foldi rows.(0) ~init:0. ~f:(fun j acc x ->
      let sign = if Int.equal (Int.rem j 2) 0 then 1. else -1. in
      acc +. (sign *. x *. det3 (minor 0 j)))
  in
  let tetra v0 v1 v2 v3 =
    (* The origin is inside when all four barycentric coordinates are
       non-negative. They come from a 4x4 solve -- the vertices against the
       origin, with the weights summing to one -- which is a statement about the
       tetra as a whole and does not care which way a face normal happens to
       point. The sign-of-face-products version of this is shorter and degenerate
       on flat simplices, which is most of them when two spheres are close. *)
    let rows =
      [| [| v0.m.(0); v1.m.(0); v2.m.(0); v3.m.(0) |]
       ; [| v0.m.(1); v1.m.(1); v2.m.(1); v3.m.(1) |]
       ; [| v0.m.(2); v1.m.(2); v2.m.(2); v3.m.(2) |]
       ; [| 1.; 1.; 1.; 1. |]
      |]
    in
    let det = det4 rows in
    let closest_face () =
      let best = ref None in
      List.iter
        [ [| v0; v1; v2 |]; [| v0; v1; v3 |]; [| v0; v2; v3 |]; [| v1; v2; v3 |] ]
        ~f:(fun face ->
          let w, kept, p = triangle face.(0) face.(1) face.(2) in
          let d2 = dot p p in
          match !best with
          | Some (b2, _, _, _) when Float.( <= ) b2 d2 -> ()
          | _ -> best := Some (d2, w, kept, p));
      let _, w, kept, p = Option.value_exn !best in
      w, kept, p
    in
    if Float.( <= ) (Float.abs det) 1e-300
    then closest_face ()
    else (
      let lambda j =
        let replacement = [| 0.; 0.; 0.; 1. |] in
        let cols =
          Array.init 4 ~f:(fun c ->
            if Int.equal c j then replacement else Array.map rows ~f:(fun r -> r.(c)))
        in
        (* The replaced matrix, fed back through the same determinant. *)
        let replaced =
          Array.init 4 ~f:(fun r -> Array.init 4 ~f:(fun c -> cols.(c).(r)))
        in
        det4 replaced /. det
      in
      let weights = Array.init 4 ~f:lambda in
      if Array.for_all weights ~f:(fun w -> Float.( >= ) w (-1e-12))
      then
        [| weights.(0); weights.(1); weights.(2); weights.(3) |], [ v0 ], [| 0.; 0.; 0. |]
      else closest_face ())
  in
  match Array.to_list points with
  | [] -> assert false
  | [ v ] -> [| 1. |], [ v ], v.m
  | [ v0; v1 ] -> segment v0 v1
  | [ v0; v1; v2 ] -> triangle v0 v1 v2
  | [ v0; v1; v2; v3 ] -> tetra v0 v1 v2 v3
  | _ -> assert false

let closest_point ps =
  let _, _, p = closest (Array.of_list (List.map ps ~f:(fun m -> { m }))) in
  p

(* [find] runs the GJK loop and reports what it found: whether the shapes touch,
   and the tetrahedron that contained the origin if one arose. The tetrahedron is
   EPA's starting polytope, so it is captured rather than discarded. *)
let find ?(iterations = 100) ?(tolerance = 1e-6) ?(trace = false) (a : obj) (b : obj) =
  let v0 = support a b [| 1.; 0.; 0. |] in
  let simplex = ref [ v0 ] in
  let closest_point = ref v0.m in
  let touching = ref false in
  (try
     for k = 0 to iterations - 1 do
       let d = scale !closest_point (-1.) in
       let gap = dot !closest_point !closest_point in
       if Float.( <= ) gap 1e-24
       then (
         touching := true;
         raise Stdlib.Exit);
       let v = support a b d in
       let diff = sub !closest_point v.m in
       if trace
       then
         Stdlib.Printf.printf
           "    it %d: |x| = %.6g, simplex %d, dot(x, s) = %.6g, gap %.6g\n"
           k
           (Float.sqrt gap)
           (List.length !simplex)
           (dot !closest_point v.m)
           (dot !closest_point diff);
       if Float.( < ) (dot !closest_point diff) (tolerance *. tolerance *. 0.5)
       then raise Stdlib.Exit;
       if Float.( > ) (dot !closest_point v.m) 0. then raise Stdlib.Exit;
       simplex := !simplex @ [ v ];
       let was = !simplex in
       let _, kept, p = closest (Array.of_list was) in
       simplex := kept;
       closest_point := p;
       (* A tetrahedron whose closest feature is the origin itself contains it,
          and is EPA's starting polytope. This has to be captured here: the next
          iteration exits at its first test, before it could see it. *)
       ignore was
     done
   with
   | Stdlib.Exit -> ());
  let inside =
    !touching || Float.( <= ) (Float.sqrt (dot !closest_point !closest_point)) tolerance
  in
  inside

let intersect ?iterations ?tolerance ?trace a b = find ?iterations ?tolerance ?trace a b

let of_geom shape ~pos ~mat ~size ~margin =
  let mat = Nx.create Nx.float64 [| 3; 3 |] mat
  and pos = Nx.create Nx.float64 [| 3 |] pos
  and size = Nx.create Nx.float64 [| 3 |] size in
  let support dir =
    Convex.support shape ~pos ~mat ~size ~dir:(Nx.create Nx.float64 [| 3 |] dir)
    |> Nx.to_array
  in
  { support; margin }
