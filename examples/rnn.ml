(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Illustration of orbit averaging for an RNN model *)

open Base
open Nx
open Symo

let print s = Stdio.print_endline (Sexp.to_string_hum s)

module RNN = struct
  type 'a t =
    { b : 'a (* input x hidden *)
    ; w : 'a (* hidden x hidden *)
    ; c : 'a (* hidden x output *)
    ; bias : 'a (* hidden *)
    }
  [@@deriving ptree, sexp]

  let hidden = 256
  let input_dim = 16
  let output_dim = 64

  let dims =
    { b = [ input_dim; hidden ]
    ; w = [ hidden; hidden ]
    ; c = [ hidden; output_dim ]
    ; bias = [ hidden ]
    }

  let symmetries : Symmetry.spec list t =
    { b = [ Symmetry.Id; Symmetry.Perm 0 ]
    ; w = [ Symmetry.Perm 0; Symmetry.Perm 0 ]
    ; c = [ Symmetry.Perm 0; Symmetry.Id ]
    ; bias = [ Symmetry.Perm 0 ]
    }

  (* The surrogate network the estimator works on: free axes keep their
     dimension, the permuted hidden axis shrinks to 2. It must stay
     non-trivial (>= 2), or the estimated curvature vanishes. *)
  let surrogate_dim = 3

  let init () =
    let w = randn float32 [| hidden; hidden |] in
    let b = randn float32 [| input_dim; hidden |] in
    let c = randn float32 [| hidden; output_dim |] in
    let bias = zeros float32 [| hidden |] in
    let open Infix in
    { b = b /$ Float.sqrt (Float.of_int input_dim)
    ; w = (w *$ Float.(1.5 / sqrt (Float.of_int hidden)))
    ; c = c /$ Float.sqrt (Float.of_int hidden)
    ; bias
    }

  let forward ~dt (p : float32_t t) ~input =
    (* input = horizon x bs x input_dim *)
    let bs = dim 1 input in
    let open Infix in
    Rune.scan
      (module struct
        type t = float32_t [@@deriving ptree]
      end)
      ~f:(fun x u ->
        let o = tanh x in
        let x =
          (x *$ Float.(1. - dt)) + ((-x + (o *@ p.w) + (u *@ p.b) + p.bias) *$ dt)
        in
        x, o *@ p.c)
      ~init:(zeros float32 [| bs; hidden |])
      input
    |> snd
end

let teacher = RNN.init ()
let student = RNN.init ()
let dt = 0.04
let horizon = 100
let batch = 128
let input = randn float32 [| horizon; batch; RNN.input_dim |]
let targets = RNN.forward ~dt teacher ~input
let _ = Nx_io.save_txt "target" (slice [ A; I 0; A ] targets)

let loss (p : Nx.float32_t RNN.t) =
  let pred = RNN.forward ~dt p ~input in
  mean (square Infix.(pred - targets))

let _, grad = Rune.value_and_grad (Ptree.instantiate (module RNN)) loss student

module S = Symo.Make (RNN)

let _ = print [%message (S.surrogate_dims : int list RNN.t)]

let save_cov_for label theta =
  let s1 =
    Stdio.print_endline "computing first-order factors";
    let factors = S.First_order.factors_of_params theta in
    S.First_order.dense_of_factors factors
  in
  let s2 =
    Stdio.print_endline "computing second-order factors";
    let factors = S.Second_order.factors_of_pair ~symmetric:true theta theta in
    S.Second_order.dense_of_factors ~symmetric:true factors
  in
  let c2 = Infix.(s2 - (s1 *@ transpose s1)) in
  Nx_io.save_txt (Printf.sprintf "rnn_%s_c2" label) c2

let _ = save_cov_for "params" student
let _ = save_cov_for "grad" grad
