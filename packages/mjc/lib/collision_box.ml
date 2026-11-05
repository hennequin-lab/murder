(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The kernel is written as the C engine writes it: three-vectors in float
   arrays, loops, early returns. Vectorising it would have meant expressing a
   separating-axis search with data-dependent branching as masks over fifteen
   candidates and two manifestly different contact generators, and the point of
   this file is to be checkable line by line against mjc_BoxBox. *)

let sep_eps = 1e-13
let par_eps = 1e-16
let sgn_eps = 1e-9
let dupe_eps = 1e-14
let edge_bias = 1e-6
let max_verts = 12

type contact =
  { dist : float
  ; pos : float array
  ; normal : float array
  }

type sat =
  { code : int
  ; sep : float
  ; separated : bool
  }

let m3 i =
  let r = i - (3 * (i / 3)) in
  if r < 0 then r + 3 else r

let sub3 a b = [| a.(0) -. b.(0); a.(1) -. b.(1); a.(2) -. b.(2) |]
let add3 a b = [| a.(0) +. b.(0); a.(1) +. b.(1); a.(2) +. b.(2) |]
let dot3 a b = (a.(0) *. b.(0)) +. (a.(1) *. b.(1)) +. (a.(2) *. b.(2))

(* [matᵀ v], with [mat] row-major 3x3: a world vector in the box's frame. *)
let mul_mat_t_vec3 mat v =
  [| (mat.(0) *. v.(0)) +. (mat.(3) *. v.(1)) +. (mat.(6) *. v.(2))
   ; (mat.(1) *. v.(0)) +. (mat.(4) *. v.(1)) +. (mat.(7) *. v.(2))
   ; (mat.(2) *. v.(0)) +. (mat.(5) *. v.(1)) +. (mat.(8) *. v.(2))
  |]

let mul_mat_vec3 mat v =
  [| (mat.(0) *. v.(0)) +. (mat.(1) *. v.(1)) +. (mat.(2) *. v.(2))
   ; (mat.(3) *. v.(0)) +. (mat.(4) *. v.(1)) +. (mat.(5) *. v.(2))
   ; (mat.(6) *. v.(0)) +. (mat.(7) *. v.(1)) +. (mat.(8) *. v.(2))
  |]

(* [rot] is the columns of [mat2] expressed in [mat1]'s frame: [mat1ᵀ mat2]. *)
let mul_mat_t_mat3 a b =
  Array.init 9 (fun k ->
    let r = k / 3
    and c = m3 k in
    (a.(r) *. b.(c)) +. (a.(3 + r) *. b.(3 + c)) +. (a.(6 + r) *. b.(6 + c)))

let norm3 v =
  let n = Float.sqrt (dot3 v v) in
  if n <= 0. then [| 0.; 0.; 0. |] else [| v.(0) /. n; v.(1) /. n; v.(2) /. n |]

let clip v lo hi = Float.max lo (Float.min hi v)

(* [clip_half_plane poly coord sign limit]: Sutherland-Hodgman against one
   half-plane [sign * v[coord] <= limit]. Vertices are [(x, y, z)] with [z]
   interpolated; when every vertex is already inside the polygon is returned
   untouched, which is the resting case and costs no copy. *)
let clip_half_plane (poly : float array array) coord sign limit =
  let nin = Array.length poly in
  let d = Array.init nin (fun k -> (sign *. poly.(k).(coord)) -. limit) in
  if Stdlib.Array.for_all (fun dk -> dk <= 0.) d
  then poly
  else (
    let out = ref [] in
    let nout = ref 0 in
    for k = 0 to nin - 1 do
      let k1 = if Int.equal (k + 1) nin then 0 else k + 1 in
      let dp = d.(k)
      and dq = d.(k1) in
      if dp <= 0. && !nout < max_verts
      then (
        out := poly.(k) :: !out;
        Stdlib.incr nout);
      let crosses = (dp < 0. && dq > 0.) || (dp > 0. && dq < 0.) in
      if crosses && !nout < max_verts
      then (
        let p = poly.(k)
        and q = poly.(k1) in
        let t = dp /. (dp -. dq) in
        out
        := [| p.(0) +. (t *. (q.(0) -. p.(0)))
            ; p.(1) +. (t *. (q.(1) -. p.(1)))
            ; p.(2) +. (t *. (q.(2) -. p.(2)))
           |]
           :: !out;
        Stdlib.incr nout)
    done;
    Array.of_list (List.rev !out))

let sat
      ~(pos1 : float array)
      ~(mat1 : float array)
      ~(size1 : float array)
      ~(pos2 : float array)
      ~(mat2 : float array)
      ~(size2 : float array)
      ~margin
  =
  let rot = mul_mat_t_mat3 mat1 mat2 in
  let rotabs = Array.map Float.abs rot in
  let pos21 = mul_mat_t_vec3 mat1 (sub3 pos2 pos1)
  and pos12 = mul_mat_t_vec3 mat2 (sub3 pos1 pos2) in
  let septol =
    margin
    +. (sep_eps
        *. (size1.(0) +. size1.(1) +. size1.(2) +. size2.(0) +. size2.(1) +. size2.(2)))
  in
  let sep_best = ref (-1e30)
  and code = ref (-1) in
  let separated = ref false in
  for i = 0 to 2 do
    let radius2 =
      (rotabs.((3 * i) + 0) *. size2.(0))
      +. (rotabs.((3 * i) + 1) *. size2.(1))
      +. (rotabs.((3 * i) + 2) *. size2.(2))
    in
    let sep = Float.abs pos21.(i) -. size1.(i) -. radius2 in
    if sep > septol then separated := true;
    if sep > !sep_best
    then (
      sep_best := sep;
      code := i)
  done;
  for j = 0 to 2 do
    let radius1 =
      (rotabs.(0 + j) *. size1.(0))
      +. (rotabs.(3 + j) *. size1.(1))
      +. (rotabs.(6 + j) *. size1.(2))
    in
    let sep = Float.abs pos12.(j) -. size2.(j) -. radius1 in
    if sep > septol then separated := true;
    if sep > !sep_best
    then (
      sep_best := sep;
      code := 3 + j)
  done;
  let sep_face = !sep_best
  and code_face = !code in
  for i = 0 to 2 do
    for j = 0 to 2 do
      let i1 = m3 (i + 1)
      and i2 = m3 (i + 2) in
      let ax1 = 0. -. rot.((3 * i2) + j)
      and ax2 = rot.((3 * i1) + j) in
      let norm2 = (ax1 *. ax1) +. (ax2 *. ax2) in
      if norm2 >= par_eps
      then (
        let inv = 1. /. Float.sqrt norm2 in
        let ax1 = ax1 *. inv
        and ax2 = ax2 *. inv in
        let radius1 = (size1.(i1) *. Float.abs ax1) +. (size1.(i2) *. Float.abs ax2) in
        let j1 = m3 (j + 1)
        and j2 = m3 (j + 2) in
        let a2_1 = (ax1 *. rot.((3 * i1) + j1)) +. (ax2 *. rot.((3 * i2) + j1))
        and a2_2 = (ax1 *. rot.((3 * i1) + j2)) +. (ax2 *. rot.((3 * i2) + j2)) in
        let radius2 = (size2.(j1) *. Float.abs a2_1) +. (size2.(j2) *. Float.abs a2_2) in
        let sep =
          Float.abs ((ax1 *. pos21.(i1)) +. (ax2 *. pos21.(i2))) -. radius1 -. radius2
        in
        if sep > septol then separated := true;
        if sep -. (edge_bias *. Float.abs sep) > !sep_best && sep > sep_face
        then (
          sep_best := sep;
          code := 6 + (3 * i) + j))
    done
  done;
  (* Face substitution: an edge axis within eight degrees of the best face
     axis duplicates it, and the face's manifold is better for the solver. *)
  if !code >= 6
  then (
    let i = (!code - 6) / 3
    and j = m3 (!code - 6) in
    let i1 = m3 (i + 1)
    and i2 = m3 (i + 2) in
    let axis = Array.make 3 0. in
    axis.(i1) <- 0. -. rot.((3 * i2) + j);
    axis.(i2) <- rot.((3 * i1) + j);
    let axis = norm3 axis in
    let face_dot =
      if code_face < 3
      then Float.abs axis.(code_face)
      else (
        let f = code_face - 3 in
        Float.abs
          ((axis.(0) *. rot.(0 + f))
           +. (axis.(1) *. rot.(3 + f))
           +. (axis.(2) *. rot.(6 + f))))
    in
    if face_dot > 0.99 && !sep_best < sep_face +. (0.05 *. Float.abs sep_face) +. 1e-15
    then (
      code := code_face;
      sep_best := sep_face));
  { code = !code; sep = !sep_best; separated = !separated }

let manifold
      ~(pos1 : float array)
      ~(mat1 : float array)
      ~(size1 : float array)
      ~(pos2 : float array)
      ~(mat2 : float array)
      ~(size2 : float array)
      ~margin
  =
  let rot = mul_mat_t_mat3 mat1 mat2 in
  let rotabs = Array.map Float.abs rot in
  let pos21 = mul_mat_t_vec3 mat1 (sub3 pos2 pos1)
  and pos12 = mul_mat_t_vec3 mat2 (sub3 pos1 pos2) in
  (* Stage 1: separating axes. *)
  let septol =
    margin
    +. (sep_eps
        *. (size1.(0) +. size1.(1) +. size1.(2) +. size2.(0) +. size2.(1) +. size2.(2)))
  in
  let sep_best = ref (-1e30)
  and code = ref (-1) in
  let separated = ref false in
  for i = 0 to 2 do
    let radius2 =
      (rotabs.((3 * i) + 0) *. size2.(0))
      +. (rotabs.((3 * i) + 1) *. size2.(1))
      +. (rotabs.((3 * i) + 2) *. size2.(2))
    in
    let sep = Float.abs pos21.(i) -. size1.(i) -. radius2 in
    if sep > septol then separated := true;
    if sep > !sep_best
    then (
      sep_best := sep;
      code := i)
  done;
  for j = 0 to 2 do
    let radius1 =
      (rotabs.(0 + j) *. size1.(0))
      +. (rotabs.(3 + j) *. size1.(1))
      +. (rotabs.(6 + j) *. size1.(2))
    in
    let sep = Float.abs pos12.(j) -. size2.(j) -. radius1 in
    if sep > septol then separated := true;
    if sep > !sep_best
    then (
      sep_best := sep;
      code := 3 + j)
  done;
  let sep_face = ref !sep_best
  and code_face = ref !code in
  for i = 0 to 2 do
    for j = 0 to 2 do
      let i1 = m3 (i + 1)
      and i2 = m3 (i + 2) in
      let ax1 = 0. -. rot.((3 * i2) + j)
      and ax2 = rot.((3 * i1) + j) in
      let norm2 = (ax1 *. ax1) +. (ax2 *. ax2) in
      if norm2 >= par_eps
      then (
        let inv = 1. /. Float.sqrt norm2 in
        let ax1 = ax1 *. inv
        and ax2 = ax2 *. inv in
        let radius1 = (size1.(i1) *. Float.abs ax1) +. (size1.(i2) *. Float.abs ax2) in
        let j1 = m3 (j + 1)
        and j2 = m3 (j + 2) in
        let a2_1 = (ax1 *. rot.((3 * i1) + j1)) +. (ax2 *. rot.((3 * i2) + j1))
        and a2_2 = (ax1 *. rot.((3 * i1) + j2)) +. (ax2 *. rot.((3 * i2) + j2)) in
        let radius2 = (size2.(j1) *. Float.abs a2_1) +. (size2.(j2) *. Float.abs a2_2) in
        let sep =
          Float.abs ((ax1 *. pos21.(i1)) +. (ax2 *. pos21.(i2))) -. radius1 -. radius2
        in
        if sep > septol then separated := true;
        if sep -. (edge_bias *. Float.abs sep) > !sep_best && sep > !sep_face
        then (
          sep_best := sep;
          code := 6 + (3 * i) + j))
    done
  done;
  if !separated || !code < 0
  then [||]
  else (
    (* Face substitution: an edge axis within eight degrees of the best face
       axis duplicates it, and the face's manifold is better for the solver. *)
    if !code >= 6
    then (
      let i = (!code - 6) / 3
      and j = m3 (!code - 6) in
      let i1 = m3 (i + 1)
      and i2 = m3 (i + 2) in
      let axis = Array.make 3 0. in
      axis.(i) <- 0.;
      axis.(i1) <- 0. -. rot.((3 * i2) + j);
      axis.(i2) <- rot.((3 * i1) + j);
      let axis = norm3 axis in
      let face_dot =
        if !code_face < 3
        then Float.abs axis.(!code_face)
        else (
          let f = !code_face - 3 in
          Float.abs
            ((axis.(0) *. rot.(0 + f))
             +. (axis.(1) *. rot.(3 + f))
             +. (axis.(2) *. rot.(6 + f))))
      in
      if
        face_dot > 0.99 && !sep_best < !sep_face +. (0.05 *. Float.abs !sep_face) +. 1e-15
      then (
        code := !code_face;
        sep_best := !sep_face));
    if !code >= 6
    then (
      (* Stage 2a: edge-edge. *)
      let i = (!code - 6) / 3
      and j = m3 (!code - 6) in
      let i1 = m3 (i + 1)
      and i2 = m3 (i + 2) in
      let j1 = m3 (j + 1)
      and j2 = m3 (j + 2) in
      let axis = Array.make 3 0. in
      axis.(i1) <- 0. -. rot.((3 * i2) + j);
      axis.(i2) <- rot.((3 * i1) + j);
      let axis = norm3 axis in
      let axis =
        if dot3 axis pos21 < 0. then [| -.axis.(0); -.axis.(1); -.axis.(2) |] else axis
      in
      let a2 =
        Array.init 3 (fun c ->
          (axis.(0) *. rot.(0 + c))
          +. (axis.(1) *. rot.(3 + c))
          +. (axis.(2) *. rot.(6 + c)))
      in
      let amb1 =
        if Float.abs axis.(i1) < sgn_eps
        then Some i1
        else if Float.abs axis.(i2) < sgn_eps
        then Some i2
        else None
      and amb2 =
        if Float.abs a2.(j1) < sgn_eps
        then Some j1
        else if Float.abs a2.(j2) < sgn_eps
        then Some j2
        else None
      in
      let d2 = [| rot.(0 + j); rot.(3 + j); rot.(6 + j) |] in
      let b = d2.(i) in
      let denom = 1. -. (b *. b) in
      let best_w1 = ref [| 0.; 0.; 0. |]
      and best_w2 = ref [| 0.; 0.; 0. |]
      and best_d2 = ref 1e30 in
      let n1 =
        match amb1 with
        | Some _ -> 2
        | None -> 1
      and n2 =
        match amb2 with
        | Some _ -> 2
        | None -> 1
      in
      for v1 = 0 to n1 - 1 do
        for v2 = 0 to n2 - 1 do
          let c1 = Array.make 3 0. in
          c1.(i1) <- (if axis.(i1) >= 0. then size1.(i1) else -.size1.(i1));
          c1.(i2) <- (if axis.(i2) >= 0. then size1.(i2) else -.size1.(i2));
          (match amb1 with
           | Some k when v1 = 1 -> c1.(k) <- -.c1.(k)
           | _ -> ());
          let cc = Array.make 3 0. in
          cc.(j1) <- (if a2.(j1) >= 0. then -.size2.(j1) else size2.(j1));
          cc.(j2) <- (if a2.(j2) >= 0. then -.size2.(j2) else size2.(j2));
          (match amb2 with
           | Some k when v2 = 1 -> cc.(k) <- -.cc.(k)
           | _ -> ());
          let c2 = add3 (mul_mat_vec3 rot cc) pos21 in
          let e = sub3 c2 c1 in
          let d1e = e.(i)
          and d2e = dot3 d2 e in
          let s = if denom < 1e-14 then 0. else (d1e -. (b *. d2e)) /. denom in
          let s = clip s (-.size1.(i)) size1.(i) in
          let t = clip ((b *. s) -. d2e) (-.size2.(j)) size2.(j) in
          let s = clip (d1e +. (b *. t)) (-.size1.(i)) size1.(i) in
          let p1 = Array.copy c1 in
          p1.(i) <- p1.(i) +. s;
          let p2 = add3 c2 (Array.map (fun x -> x *. t) d2) in
          let gap = sub3 p2 p1 in
          let gap2 = dot3 gap gap in
          if gap2 < !best_d2
          then (
            best_d2 := gap2;
            best_w1 := p1;
            best_w2 := p2)
        done
      done;
      let w1 = !best_w1
      and w2 = !best_w2 in
      let dist = dot3 (sub3 w2 w1) axis in
      if dist > septol
      then [||]
      else (
        let mid =
          [| 0.5 *. (w1.(0) +. w2.(0))
           ; 0.5 *. (w1.(1) +. w2.(1))
           ; 0.5 *. (w1.(2) +. w2.(2))
          |]
        in
        [| { dist
           ; pos = add3 (mul_mat_vec3 mat1 mid) pos1
           ; normal = norm3 (mul_mat_vec3 mat1 axis)
           }
        |]))
    else (
      (* Stage 2b: face contact. *)
      let ref1 = !code < 3 in
      let a = if ref1 then !code else !code - 3 in
      let sizeref = if ref1 then size1 else size2
      and sizeinc = if ref1 then size2 else size1
      and posref = if ref1 then pos1 else pos2
      and matref = if ref1 then mat1 else mat2
      and posoi = if ref1 then pos21 else pos12 in
      let rinc =
        if ref1
        then rot
        else
          Array.init 9 (fun k ->
            let r = k / 3
            and c = m3 k in
            rot.((3 * c) + r))
      in
      let sgn = if posoi.(a) >= 0. then 1. else -1. in
      let binc = ref 0 in
      for k = 1 to 2 do
        if Float.abs rinc.((3 * a) + k) > Float.abs rinc.((3 * a) + !binc) then binc := k
      done;
      let binc = !binc in
      let tinc = if sgn *. rinc.((3 * a) + binc) > 0. then -1. else 1. in
      let ax = m3 (a + 1)
      and ay = m3 (a + 2) in
      let bu = m3 (binc + 1)
      and bv = m3 (binc + 2) in
      let cx = Array.make 3 0.
      and du = Array.make 3 0.
      and dv = Array.make 3 0. in
      for r = 0 to 2 do
        let c = if r = 0 then ax else if r = 1 then ay else a in
        cx.(r) <- posoi.(c) +. (tinc *. sizeinc.(binc) *. rinc.((3 * c) + binc));
        du.(r) <- sizeinc.(bu) *. rinc.((3 * c) + bu);
        dv.(r) <- sizeinc.(bv) *. rinc.((3 * c) + bv)
      done;
      cx.(2) <- (sgn *. cx.(2)) -. sizeref.(a);
      du.(2) <- du.(2) *. sgn;
      dv.(2) <- dv.(2) *. sgn;
      let signs = [| 1., 1.; -1., 1.; -1., -1.; 1., -1. |] in
      let poly =
        Array.init 4 (fun k ->
          let su, sv = signs.(k) in
          [| cx.(0) +. (su *. du.(0)) +. (sv *. dv.(0))
           ; cx.(1) +. (su *. du.(1)) +. (sv *. dv.(1))
           ; cx.(2) +. (su *. du.(2)) +. (sv *. dv.(2))
          |])
      in
      let poly = clip_half_plane poly 0 1. sizeref.(ax) in
      let poly = clip_half_plane poly 0 (-1.) sizeref.(ax) in
      let poly = clip_half_plane poly 1 1. sizeref.(ay) in
      let poly = clip_half_plane poly 1 (-1.) sizeref.(ay) in
      let dupe2 =
        dupe_eps *. ((sizeref.(ax) *. sizeref.(ax)) +. (sizeref.(ay) *. sizeref.(ay)))
      in
      let accepted = ref [] in
      Stdlib.Array.iter
        (fun v ->
           if v.(2) <= margin
           then (
             let dupe =
               List.exists
                 (fun q ->
                    let dx = q.(0) -. v.(0)
                    and dy = q.(1) -. v.(1) in
                    (dx *. dx) +. (dy *. dy) < dupe2)
                 !accepted
             in
             if not dupe then accepted := v :: !accepted))
        poly;
      let accepted = Array.of_list (List.rev !accepted) in
      if Int.equal (Array.length accepted) 0
      then [||]
      else (
        let nsign = if ref1 then sgn else -.sgn in
        let normal =
          [| nsign *. matref.(0 + (3 * 0) + a)
           ; nsign *. matref.((3 * 1) + a)
           ; nsign *. matref.((3 * 2) + a)
          |]
        in
        Array.map
          (fun v ->
             let posc = Array.make 3 0. in
             posc.(ax) <- v.(0);
             posc.(ay) <- v.(1);
             posc.(a) <- sgn *. (sizeref.(a) +. (0.5 *. v.(2)));
             { dist = v.(2); pos = add3 (mul_mat_vec3 matref posc) posref; normal })
          accepted)))
