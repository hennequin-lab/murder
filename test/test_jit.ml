(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The compiled (jitted) step against the eager one.

   [Compiled.create] wraps [prepare] and [finish] in [Rune.jit2] with the whole
   state, the gradients and the estimator threaded through input/output
   leaves, so a training loop traces once and replays. The only host call
   between the two halves is the estimator solve. *)

open Base
open Windtrap
open Symo

module Mlp = struct
  type 'a t =
    { w1 : 'a
    ; w2 : 'a
    }
  [@@deriving ptree]

  let dims : int list t = { w1 = [ 3; 2 ]; w2 = [ 1; 3 ] }
  let symmetries : Symmetry.spec list t = { w1 = [ Perm 0; Id ]; w2 = [ Id; Perm 0 ] }
  let surrogate_dim = 2
  let ensure_size_invariance = false
end

module M = Make (Mlp)

let xs = Nx.Rng.normal (Nx.Rng.key 7) Nx.float32 [| 8; 2 |]

let ys =
  let w = Nx.create Nx.float32 [| 2; 1 |] [| 1.0; -0.5 |] in
  Nx.add
    (Nx.matmul xs w)
    (Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 8) Nx.float32 [| 8; 1 |]) 0.05)

let mlp_loss (p : Nx.float32_t Mlp.t) =
  let h = Nx.relu (Nx.matmul xs (Nx.transpose p.w1)) in
  let out = Nx.matmul h (Nx.transpose p.w2) in
  let d = Nx.sub out ys in
  Nx.mean (Nx.mul d d)

let config : M.config =
  { learning_rate = Some 0.3; beta_1 = 0.9; beta_2 = 0.99; damping = 1e-4 }

let params () =
  { Mlp.w1 = Nx.Rng.normal (Nx.Rng.key 9) Nx.float32 [| 3; 2 |]
  ; w2 = Nx.Rng.normal (Nx.Rng.key 10) Nx.float32 [| 1; 3 |]
  }

let check_leaf ~msg ~tol (a : Nx.float32_t) (b : Nx.float32_t) =
  equal ~msg (array (float tol)) (Nx.to_array a) (Nx.to_array b)

let check_tree ~msg ~tol (a : Nx.float32_t Mlp.t) (b : Nx.float32_t Mlp.t) =
  Mlp.map2 (fun x y -> check_leaf ~msg ~tol x y) a b |> ignore

let check_state ~msg ~tol (a : M.state) (b : M.state) =
  M.State.map2 (fun x y -> check_leaf ~msg ~tol x y) a b |> ignore

let run_steps ~steps ~step =
  let state = ref (M.init ~config (params ())) in
  List.iter (List.range 0 steps) ~f:(fun _ ->
    let _, grads = Rune.value_and_grad M.ptree mlp_loss !state.M.State.theta in
    state := step ~state:!state ~grads);
  !state

let test_parity () =
  let compiled = M.Compiled.create ~config in
  let steps = 50 in
  let eager =
    run_steps ~steps ~step:(fun ~state ~grads -> M.step ~config ~state ~grads)
  in
  let jitted =
    run_steps ~steps ~step:(fun ~state ~grads ->
      M.Compiled.step compiled ~config ~state ~grads)
  in
  check_state ~msg:"jitted step matches eager" ~tol:1e-3 eager jitted

let test_state_threads () =
  (* The compiled closures must accept the whole state as input leaves and
     return the next one; a few more steps on the jitted state alone. *)
  let compiled = M.Compiled.create ~config in
  let s0 = M.init ~config (params ()) in
  let _, g0 = Rune.value_and_grad M.ptree mlp_loss s0.M.State.theta in
  let s1 = M.Compiled.step compiled ~config ~state:s0 ~grads:g0 in
  let _, g1 = Rune.value_and_grad M.ptree mlp_loss s1.M.State.theta in
  let s2 = M.Compiled.step compiled ~config ~state:s1 ~grads:g1 in
  is_true
    ~msg:"loss is finite after two jitted steps"
    (Float.is_finite (Nx.item [] (mlp_loss s2.M.State.theta)))

(* Tracing and compiling the two halves is a one-off cost: the first call pays
   it, every later call replays the compiled program. If calls retraced, ten
   steps would cost ten times the first; with one trace they cost a small
   fraction of it. *)
let test_no_retrace () =
  let compiled = M.Compiled.create ~config in
  let state = ref (M.init ~config (params ())) in
  let one_step () =
    let _, grads = Rune.value_and_grad M.ptree mlp_loss !state.M.State.theta in
    state := M.Compiled.step compiled ~config ~state:!state ~grads
  in
  let time f =
    let t0 = Unix.gettimeofday () in
    f ();
    Unix.gettimeofday () -. t0
  in
  let first = time one_step in
  let replayed = time (fun () -> List.iter (List.range 0 10) ~f:(fun _ -> one_step ())) in
  is_true ~msg:"ten replays cost less than the first trace" Float.(replayed < 5. *. first)

(* [random_transform] samples its permutations inside the trace from a key,
   so it compiles: the key is an input leaf and the same draws replay. *)
module Key = struct
  type 'a t = { key : 'a } [@@deriving ptree]
end

let test_random_transform_jit () =
  let theta = params () in
  let jitted =
    Rune.jit2
      (Nx.Ptree.instantiate (module Key))
      (Nx.Ptree.instantiate (module Mlp))
      (fun { Key.key } -> M.random_transform ~key theta)
  in
  List.iter (List.range 0 5) ~f:(fun seed ->
    let key = Nx.Rng.key (200 + seed) in
    check_tree
      ~msg:"jitted random transform"
      ~tol:1e-9
      (jitted { Key.key })
      (M.random_transform ~key theta))

let tests =
  [ test "eager and jitted steps agree over a run" test_parity
  ; test "state threads through compiled calls" test_state_threads
  ; test "compiled calls replay instead of retracing" test_no_retrace
  ; test "jitted random group element" test_random_transform_jit
  ]

let () = run "symo jit" tests
