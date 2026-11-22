(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The compiled (jitted) step against the eager one.

   [Compiled.create] wraps the whole step in [Rune.jit]: the state, the
   gradients and the GGN samples are input leaves, the next state is the output
   leaf, so a training loop traces once and replays. The eager step is the same
   function without the jit. *)

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

let mlp_forward (p : (Nx.host Nx.float32_t, Nx.host) Mlp.t) =
  Infix.(relu (xs *@ transpose p.w1) *@ transpose p.w2)

let mlp_loss p =
  let open Infix in
  let d = mlp_forward p - ys in
  mean (d * d)

let config = M.default ~learning_rate:(Some 0.3) ~damping:1e-1

let params () =
  { Mlp.w1 = Nx.Rng.normal ~key:(Nx.Rng.key 9) float32 [| 3; 2 |]
  ; w2 = Nx.Rng.normal ~key:(Nx.Rng.key 10) float32 [| 1; 3 |]
  }

let samples params = M.ggn ~outputs:mlp_forward params

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

(* The whole training step: forward, backward, the GGN and the update. *)
let train_step state =
  let _, grads = Rune.value_and_grad M.ptree mlp_loss state.M.State.theta in
  M.step ~config ~state ~grads ~samples:(samples state.M.State.theta)

let run_steps ~steps ~step =
  let state = ref (M.init ~config (params ())) in
  List.iter (List.range 0 steps) ~f:(fun _ -> state := step !state);
  !state

let test_parity () =
  let compiled = M.Compiled.create ~config in
  let steps = 50 in
  let eager = run_steps ~steps ~step:train_step in
  let jitted =
    run_steps ~steps ~step:(fun state ->
      let _, grads = Rune.value_and_grad M.ptree mlp_loss state.M.State.theta in
      M.Compiled.step compiled ~state ~grads ~samples:(samples state.M.State.theta))
  in
  check_state ~msg:"jitted step matches eager" ~tol:1e-3 eager jitted

(* The Vega-like loop: the whole training step is one function of the state,
   and one [Rune.jit] compiles it. *)
let test_jitted_training_step () =
  let state_ptree : (Nx.host M.state, Nx.host) Nx.Ptree.t =
    Nx.Ptree.instantiate (module M.State)
  in
  let jitted = Rune.jit Nx.Ptree.(state_ptree @-> returns state_ptree) train_step in
  let steps = 20 in
  let eager = run_steps ~steps ~step:train_step in
  let jitted_state =
    List.fold
      (List.range 0 steps)
      ~init:(M.init ~config (params ()))
      ~f:(fun state _ -> jitted state)
  in
  check_state ~msg:"one jit for the whole training step" ~tol:1e-3 eager jitted_state

(* A set of its own over the host's device: the state, the gradients and the
   metric samples lie there, and the step runs there. It agrees with the eager
   step on the host. The one-sample metric keeps the data out of it. *)
module Other = (val Nx.devices (Nx.Devices.to_list Nx.Host.v))

let test_other_set () =
  let compiled = M.Compiled.create ~config in
  let other x = Nx.Ptree.place (module Mlp) Other.on x in
  let one_sample (g : (Nx.host Nx.float32_t, Nx.host) Mlp.t) =
    { Mlp.w1 = unsqueeze ~axes:[ 0 ] g.Mlp.w1; w2 = unsqueeze ~axes:[ 0 ] g.Mlp.w2 }
  in
  let host_grads = snd (Rune.value_and_grad M.ptree mlp_loss (params ())) in
  let state = M.init ~config (other (params ())) in
  let jitted =
    M.Compiled.step
      compiled
      ~state
      ~grads:(other host_grads)
      ~samples:(other (one_sample host_grads))
  in
  let state = M.init ~config (params ()) in
  let grads = host_grads in
  let eager = M.step ~config ~state ~grads ~samples:(one_sample grads) in
  let host x = Nx.Ptree.place (module M.State) Nx.Host.on x in
  check_state ~msg:"compiled step on another set" ~tol:1e-5 eager (host jitted)

let test_state_threads () =
  (* The compiled step accepts the whole state as input leaves and returns the
     next one; a few more steps on the jitted state alone. *)
  let compiled = M.Compiled.create ~config in
  let s0 = M.init ~config (params ()) in
  let _, g0 = Rune.value_and_grad M.ptree mlp_loss s0.M.State.theta in
  let s1 =
    M.Compiled.step compiled ~state:s0 ~grads:g0 ~samples:(samples s0.M.State.theta)
  in
  let _, g1 = Rune.value_and_grad M.ptree mlp_loss s1.M.State.theta in
  let s2 =
    M.Compiled.step compiled ~state:s1 ~grads:g1 ~samples:(samples s1.M.State.theta)
  in
  is_true
    ~msg:"loss is finite after two jitted steps"
    (Float.is_finite (item [] (mlp_loss s2.M.State.theta)))

(* Tracing and compiling the step is a one-off cost: the first call pays it,
   every later call replays the compiled program. If calls retraced, ten steps
   would cost ten times the first; with one trace they cost a small fraction of
   it. *)
let test_no_retrace () =
  let compiled = M.Compiled.create ~config in
  let state = ref (M.init ~config (params ())) in
  let one_step () =
    let _, grads = Rune.value_and_grad M.ptree mlp_loss !state.M.State.theta in
    state
    := M.Compiled.step
         compiled
         ~state:!state
         ~grads
         ~samples:(samples !state.M.State.theta)
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
      Nx.Ptree.(Nx.Rng.ptree @-> returns M.ptree)
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
  ; test "one jit for the whole training step" test_jitted_training_step
  ; test "compiled step on another set" test_other_set
  ; test "state threads through compiled calls" test_state_threads
  ; test "compiled calls replay instead of retracing" test_no_retrace
  ; test "jitted random group element" test_random_transform_jit
  ]

let () = Stdlib.exit (run "symo jit" tests)
