(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The compiled (jitted) step against the eager one.

   [Compiled.create] wraps [prepare] and [finish] in [Rune.jit] with the whole
   state, the gradients and the estimator threaded through input/output
   leaves, so a training loop traces once and replays. The only host call
   between the two halves is the estimator solve. *)

open Base
open Nx
open Windtrap
open Symo

module Mlp = struct
  type ('a, _) t =
    { w1 : 'a
    ; w2 : 'a
    }
  [@@deriving ptree]

  let dims : (int list, _) t = { w1 = [ 3; 2 ]; w2 = [ 1; 3 ] }

  let symmetries : (Symmetry.spec list, _) t =
    { w1 = [ Perm 0; Id ]; w2 = [ Id; Perm 0 ] }
end

module M = Make (Mlp)

let relu x = maximum x (zeros_like x)
let xs = Nx.Rng.normal ~key:(Nx.Rng.key 7) float32 [| 8; 2 |]

let ys =
  let w = create float32 [| 2; 1 |] [| 1.0; -0.5 |] in
  Infix.((xs *@ w) + (Nx.Rng.normal ~key:(Nx.Rng.key 8) float32 [| 8; 1 |] *$ 0.05))

let mlp_loss (p : (Nx.host Nx.float32_t, Nx.host) Mlp.t) =
  let open Infix in
  let h = relu (xs *@ transpose p.w1) in
  let out = h *@ transpose p.w2 in
  let d = out - ys in
  mean (d * d)

let config : M.config =
  { learning_rate = Some 0.3; beta_1 = 0.9; beta_2 = 0.99; damping = 1e-4 }

let params () =
  { Mlp.w1 = Nx.Rng.normal ~key:(Nx.Rng.key 9) float32 [| 3; 2 |]
  ; w2 = Nx.Rng.normal ~key:(Nx.Rng.key 10) float32 [| 1; 3 |]
  }

let check_leaf ~msg ~tol (a : Nx.host Nx.float32_t) (b : Nx.host Nx.float32_t) =
  equal ~msg (array (float tol)) (to_array a) (to_array b)

let check_tree
      ~msg
      ~tol
      (a : (Nx.host Nx.float32_t, Nx.host) Mlp.t)
      (b : (Nx.host Nx.float32_t, Nx.host) Mlp.t)
  =
  Nx.Ptree.Payload.map2 (module Mlp) (fun _ x y -> check_leaf ~msg ~tol x y) a b |> ignore

let check_state ~msg ~tol (a : Nx.host M.state) (b : Nx.host M.state) =
  Nx.Ptree.Payload.map2 (module M.State) (fun _ x y -> check_leaf ~msg ~tol x y) a b
  |> ignore

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

(* The compiled step on a set of its own over the host's device: the state,
   the gradients and the compiled tables lie there, and the solve between the
   two halves runs on the host. It agrees with the eager step on the host. *)
module Other = (val Nx.devices (Nx.rigs Nx.Host.v))

let test_other_set () =
  let compiled = M.Compiled.create ~config in
  let other x = Nx.Ptree.place (module Mlp) Other.on x in
  let grads = other (snd (Rune.value_and_grad M.ptree mlp_loss (params ()))) in
  let state = M.init ~config (other (params ())) in
  let jitted = M.Compiled.step compiled ~config ~state ~grads in
  let state = M.init ~config (params ()) in
  let grads = snd (Rune.value_and_grad M.ptree mlp_loss (params ())) in
  let eager = M.step ~config ~state ~grads in
  let host x = Nx.Ptree.place (module M.State) Nx.Host.on x in
  check_state ~msg:"compiled step on another set" ~tol:1e-5 eager (host jitted)

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
    (Float.is_finite (item [] (mlp_loss s2.M.State.theta)))

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
let test_random_transform_jit () =
  let theta = params () in
  let jitted =
    Rune.jit
      Nx.Ptree.(Nx.Ptree.v Nx.Rng.walk @-> returns M.ptree)
      (fun key -> M.random_transform ~key theta)
  in
  List.iter (List.range 0 5) ~f:(fun seed ->
    let key = Nx.Rng.key (200 + seed) in
    check_tree
      ~msg:"jitted random transform"
      ~tol:1e-9
      (jitted key)
      (M.random_transform ~key theta))

let tests =
  [ test "eager and jitted steps agree over a run" test_parity
  ; test "compiled step on another set" test_other_set
  ; test "state threads through compiled calls" test_state_threads
  ; test "compiled calls replay instead of retracing" test_no_retrace
  ; test "jitted random group element" test_random_transform_jit
  ]

let () = Stdlib.exit (run "symo jit" tests)
