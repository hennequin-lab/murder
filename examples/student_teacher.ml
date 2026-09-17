(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Student-teacher regression with a two-layer MLP.

   A randomly initialised student is trained against a fixed teacher of the
   same architecture. Both have one hidden layer, so permuting the hidden
   units — the rows of [w1] and the columns of [w2] together — leaves the
   student's input-output map unchanged: the loss is invariant under the
   hidden-permutation group, which is exactly the symmetry SYMO exploits.

   The optimizer works on the parameter tree itself: gradients come from
   [Rune.value_and_grad] over the packed [S.ptree] traversal, and each step is
   the compiled [S.Compiled.step] (the eager [S.step] is the same three
   phases — prepare, host solve, finish — without [Rune.jit2]). *)

open Base
open Symo

module Mlp = struct
  type 'a t =
    { w1 : 'a (* hidden x input *)
    ; w2 : 'a (* output x hidden *)
    }
  [@@deriving ptree]

  let hidden = 32
  let input_dim = 8
  let dims : int list t = { w1 = [ hidden; input_dim ]; w2 = [ 1; hidden ] }

  (* Permuting hidden unit [h] permutes row [h] of [w1] and column [h] of
     [w2] by the same group element: one [Perm 0] group ties the two axes
     together. The other axes are [Id] (free, never transformed). *)
  let symmetries : Symmetry.spec list t =
    { w1 = [ Symmetry.Perm 0; Symmetry.Id ]; w2 = [ Symmetry.Id; Symmetry.Perm 0 ] }

  (* The surrogate network the estimator works on: free axes keep their
     dimension, the permuted hidden axis shrinks to 2. It must stay
     non-trivial (>= 2), or the estimated curvature vanishes. *)
  let surrogate_dim = 2
end

module S = Symo.Make (Mlp)

let batch = 128
let steps = 500
let print_every = 50
let inputs = Nx.Rng.normal (Nx.Rng.key 0) Nx.float32 [| batch; Mlp.input_dim |]

let init key =
  let w1 =
    Nx.Rng.normal (Nx.Rng.fold_in key 0) Nx.float32 [| Mlp.hidden; Mlp.input_dim |]
  in
  let w2 = Nx.Rng.normal (Nx.Rng.fold_in key 1) Nx.float32 [| 1; Mlp.hidden |] in
  { Mlp.w1 = Nx.mul_s w1 (1. /. Float.sqrt (Float.of_int Mlp.hidden))
  ; w2 = Nx.mul_s w2 0.5
  }

let forward (p : Nx.float32_t Mlp.t) x =
  let h = Nx.relu (Nx.matmul x (Nx.transpose p.w1)) in
  Nx.matmul h (Nx.transpose p.w2)

let teacher = init (Nx.Rng.key 42)
let targets = forward teacher inputs

let loss (p : Nx.float32_t Mlp.t) =
  let d = Nx.sub (forward p inputs) targets in
  Nx.mean (Nx.mul d d)

let () =
  let config : S.config =
    { learning_rate = Some 0.1; beta_1 = 0.9; beta_2 = 0.99; damping = 1e-4 }
  in
  let compiled = S.Compiled.create ~config in
  let state = ref (S.init ~config (init (Nx.Rng.key 7))) in
  Stdio.printf
    "student-teacher MLP: %d hidden units, %d inputs, %d examples\n"
    Mlp.hidden
    Mlp.input_dim
    batch;
  Stdio.printf "%8s  %12s\n" "step" "mse";
  Stdio.printf "%8d  %12.6f\n" 0 (Nx.item [] (loss !state.S.State.theta));
  for step = 1 to steps do
    let _, grads = Rune.value_and_grad S.ptree loss !state.S.State.theta in
    state := S.Compiled.step compiled ~config ~state:!state ~grads;
    if Int.equal (Int.rem step print_every) 0
    then Stdio.printf "%8d  %12.6f\n" step (Nx.item [] (loss !state.S.State.theta))
  done
