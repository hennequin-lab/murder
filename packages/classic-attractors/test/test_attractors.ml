(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The generators of every attractor: shapes, dtypes, discard semantics, the
   snippets' permutation, and that each field stays on a bounded, non-trivial
   set. *)

open Base
open Nx
open Windtrap
open Classic_attractors

(* [dt] resolves each flow; the bounds are the ones the modules document. *)
let modules =
  [ "Lorenz", (module Lorenz : S), 0.01
  ; "Rossler", (module Rossler : S), 0.02
  ; "Thomas", (module Thomas : S), 0.02
  ; "Halvorsen", (module Halvorsen : S), 0.005
  ; "Aizawa", (module Aizawa : S), 0.01
  ; "Duffing", (module Duffing : S), 0.01
  ]

let equal_arrays ~msg a b = equal ~msg (array float_exact) a b

let rows ~stride x =
  let flat = to_array x in
  let n = Array.length flat / stride in
  Array.init n ~f:(fun i -> Array.sub flat ~pos:(i * stride) ~len:stride)

let shape_tests =
  List.concat_map modules ~f:(fun (name, (module M : S), dt) ->
    [ test (name ^ ": shapes") (fun () ->
        let x = M.simulate float32 ~dt ~n_steps:7 ~discard_first:3 4 in
        equal (array int) [| 7; 4; M.dim |] (shape x);
        let s = M.snippets float32 ~dt ~n_steps:7 ~discard_first:3 4 in
        equal (array int) [| 7; 4; M.dim |] (shape s))
    ; test (name ^ ": the same key replays") (fun () ->
        let x = M.simulate float32 ~key:(Rng.key 1) ~dt ~n_steps:4 3 in
        let y = M.simulate float32 ~key:(Rng.key 1) ~dt ~n_steps:4 3 in
        equal_arrays ~msg:"simulate" (to_array x) (to_array y);
        let x = M.snippets float32 ~key:(Rng.key 1) ~dt ~n_steps:4 3 in
        let y = M.snippets float32 ~key:(Rng.key 1) ~dt ~n_steps:4 3 in
        equal_arrays ~msg:"snippets" (to_array x) (to_array y))
    ; test (name ^ ": another key draws another run") (fun () ->
        let x = M.simulate float32 ~key:(Rng.key 1) ~dt ~n_steps:4 3 in
        let y = M.simulate float32 ~key:(Rng.key 2) ~dt ~n_steps:4 3 in
        is_true ~msg:"simulate" (not (Array.equal Float.equal (to_array x) (to_array y))))
    ; test (name ^ ": discard_first drops the head") (fun () ->
        let full = M.simulate float64 ~key:(Rng.key 3) ~dt ~n_steps:8 1 in
        let tail =
          M.simulate float64 ~key:(Rng.key 3) ~dt ~n_steps:5 ~discard_first:3 1
        in
        let head = slice [ R (3, 8); A; A ] full in
        equal_arrays ~msg:"the tail" (to_array head) (to_array tail))
    ; test (name ^ ": snippets permute the same chunks") (fun () ->
        let n = 5 in
        let chunks key =
          let x =
            M.snippets float64 ~key:(Rng.key key) ~dt ~n_steps:6 ~discard_first:10 n
          in
          let x = transpose ~axes:[ 1; 0; 2 ] x |> reshape [| n; 6 * M.dim |] in
          let rows = rows ~stride:(6 * M.dim) x in
          Array.sort rows ~compare:(Array.compare Float.compare);
          rows
        in
        is_true
          ~msg:"the chunk multiset"
          (Array.equal (Array.equal Float.equal) (chunks 4) (chunks 9)))
    ; test (name ^ ": random initial conditions stay finite") (fun () ->
        let x =
          M.simulate float32 ~key:(Rng.key 5) ~dt ~n_steps:300 ~discard_first:400 4
        in
        let flat = to_array x in
        is_true ~msg:"finite" (Array.for_all flat ~f:Float.is_finite);
        let lo = Array.fold flat ~init:Float.infinity ~f:Float.min in
        let hi = Array.fold flat ~init:Float.neg_infinity ~f:Float.max in
        is_true ~msg:"spread" Float.(hi - lo > 1e-2);
        is_true ~msg:"no blow-up" Float.(max (abs lo) (abs hi) < 1e4))
    ; test (name ^ ": the snippets are bounded") (fun () ->
        let x = M.snippets float32 ~dt ~n_steps:300 ~discard_first:1000 4 in
        let flat = to_array x in
        let lo = Array.fold flat ~init:Float.infinity ~f:Float.min in
        let hi = Array.fold flat ~init:Float.neg_infinity ~f:Float.max in
        is_true ~msg:"finite" (Array.for_all flat ~f:Float.is_finite);
        is_true ~msg:"bound" Float.(max (abs lo) (abs hi) < 100.))
    ; test (name ^ ": the field is dtype generic") (fun () ->
        let y =
          create
            float64
            [| 2; M.dim |]
            (Array.init (2 * M.dim) ~f:(fun i -> -0.5 +. (0.1 *. Float.of_int i)))
        in
        let y32 = cast float32 y in
        let at_tensor dt = scalar float64 dt in
        let f64 = M.field (at_tensor 0.) y |> cast float64 |> to_array in
        let f32 = M.field (at_tensor 0.) y32 |> cast float64 |> to_array in
        is_true
          ~msg:"float32 agrees with float64"
          (Array.for_all2_exn f64 f32 ~f:(fun a b -> Float.(abs (a - b) < 1e-4))))
    ; test (name ^ ": rejects impossible sizes") (fun () ->
        raises_match (Exn.invalid_arg ~sub:"n_steps") (fun () ->
          ignore (M.simulate float32 ~dt ~n_steps:0 1 : _));
        raises_match (Exn.invalid_arg ~sub:"run count") (fun () ->
          ignore (M.simulate float32 ~dt ~n_steps:1 0 : _));
        raises_match (Exn.invalid_arg ~sub:"discard_first") (fun () ->
          ignore (M.snippets float32 ~dt ~n_steps:1 ~discard_first:(-1) 1 : _));
        raises_match (Exn.invalid_arg ~sub:"dt") (fun () ->
          ignore (M.simulate float32 ~dt:0. ~n_steps:1 1 : _)))
    ])

let lorenz_centre =
  test "Lorenz: the attractor sits at z = 23.5" (fun () ->
    let x =
      Lorenz.simulate float64 ~key:(Rng.key 7) ~dt:0.01 ~n_steps:500 ~discard_first:500 4
    in
    let z = slice [ A; A; I 2 ] x in
    let mean = Nx.mean z |> Nx.item [] in
    is_true ~msg:(Printf.sprintf "mean z = %.2f" mean) Float.(mean > 10. && mean < 35.))

let () = Stdlib.exit (run "classic-attractors" (shape_tests @ [ lorenz_centre ]))
