(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Curvature descriptions: every constructor's action is checked against the
   Hessian of the little loss it describes, computed by rune rather than from a
   hand-derived formula. The application takes the lane axis of a tangent
   batch — the contract the collector relies on — so each check covers both one
   lane and several. *)

open Windtrap

let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

let check_arr ?(eps = 1e-9) ~msg expected actual =
  let actual = to_arr actual in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e -> equal ~msg:(Printf.sprintf "%s[%d]" msg i) (float eps) e actual.(i))
    expected

let y3 () = vec [| 0.7; -1.3; 2.1 |]
let v3 () = vec [| 0.3; -0.6; 0.8 |]

(* The Hessian action, taken from rune's Hessian of [loss] as a function of a
   free [y]. *)
let hessian_action (loss : Nx.float64_t -> Nx.float64_t) y v =
  Nx.matmul (Rune.hessian' loss y) v

let one_lane t = Nx.reshape (Array.append [| 1 |] (Nx.shape t)) (Nx.contiguous t)
let lane batch i = Nx.slice [ Nx.I i ] batch

(* A deterministic lane batch [k; shape t]. *)
let lanes ~k t =
  let s = Nx.shape t in
  let n = Nx.numel t in
  let data =
    Array.init (k * n) (fun i ->
      let l = i / n
      and j = i mod n in
      float_of_int ((((j * 7) + (3 * l)) mod 11) - 5) /. 4.0)
  in
  Nx.create f64 (Array.append [| k |] s) data

(* [check_curv ~msg curv loss y] checks the description against [loss]: one
   lane against the Hessian action on a single direction, and [k] lanes against
   the batch of per-lane Hessian actions. *)
let check_curv
      ~msg
      ?(k = 3)
      (curv : Sofo.Curv.t)
      (loss : Nx.float64_t -> Nx.float64_t)
      y
      v
  =
  check_arr
    ~msg:(msg ^ " (one lane)")
    (to_arr (one_lane (hessian_action loss y v)))
    (Sofo.Curv.apply curv (one_lane v));
  let batch = lanes ~k y in
  let expected =
    Nx.stack ~axis:0 (List.init k (fun i -> hessian_action loss y (lane batch i)))
  in
  check_arr ~msg:(msg ^ " (batched)") (to_arr expected) (Sofo.Curv.apply curv batch)

let test_scale () =
  let s = 1.7 in
  let loss y = Nx.mul_s (Nx.sum (Nx.mul y y)) (0.5 *. s) in
  check_curv ~msg:"scale" (Sofo.Curv.scale s) loss (y3 ()) (v3 ())

let test_diag () =
  let w = vec [| 0.4; -1.1; 2.0 |] in
  let loss y = Nx.mul_s (Nx.sum (Nx.mul (Nx.mul w y) y)) 0.5 in
  check_curv ~msg:"diag" (Sofo.Curv.diag w) loss (y3 ()) (v3 ())

let test_softmax_ce () =
  (* The Hessian of −log softmax(y)[c] is diag p − p pᵀ at p = softmax y. *)
  let c = 2 in
  let ce y = Nx.neg (Nx.slice [ Nx.I c ] (Nx.log_softmax y)) in
  let p = Nx.softmax (y3 ()) in
  check_curv ~msg:"softmax_ce" (Sofo.Curv.softmax_ce p) ce (y3 ()) (v3 ());
  (* A loss averaged over rows scales its block. *)
  let scale = 0.25 in
  let scaled y = Nx.mul_s (ce y) scale in
  check_curv
    ~msg:"softmax_ce scaled"
    (Sofo.Curv.softmax_ce ~scale p)
    scaled
    (y3 ())
    (v3 ())

let test_hvp () =
  (* l = sum(y³)/3 has H = diag(2y); the closure is written at y's dtype and
     lifted over the directions by the library. *)
  let loss y = Nx.mul_s (Nx.sum (Nx.mul y (Nx.mul y y))) (1.0 /. 3.0) in
  let y = y3 () in
  let curv = Sofo.Curv.hvp ~y (fun v -> Nx.mul (Nx.mul_s v 2.0) y) in
  check_curv ~msg:"hvp" curv loss y (v3 ())

let tests =
  [ test "scale is s·I" test_scale
  ; test "diag is diag w" test_diag
  ; test "softmax_ce is diag p − p pᵀ per row" test_softmax_ce
  ; test "hvp lifts an arbitrary Hessian action" test_hvp
  ]

let () = run "sofo curv" tests
