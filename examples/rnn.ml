(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Illustration of orbit averaging for an RNN model *)

open Base
open Nx
open Symo

let in_dir = Cmdargs.in_dir "-d"

module RNN = struct
  type 'a t =
    { b : 'a (* input x hidden *)
    ; w : 'a (* hidden x hidden *)
    ; c : 'a (* hidden x output *)
    ; bias : 'a (* hidden *)
    }
  [@@deriving ptree, sexp]

  let hidden = 8
  let input_dim = 4
  let output_dim = 3

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
  let surrogate_dim = 4
  let ensure_size_invariance = false

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
let _ = Nx_io.save_txt (in_dir "target") (slice [ A; I 0; A ] targets)

let loss (p : Nx.float32_t RNN.t) =
  let pred = RNN.forward ~dt p ~input in
  mean (square Infix.(pred - targets))

let value_and_grad_jit =
  let ptree = Ptree.instantiate (module RNN) in
  Rune.jit2 ~device:"CPU" ptree ptree (fun params ->
    let _, grad = Rune.value_and_grad ptree loss params in
    grad)

let grad = value_and_grad_jit student

module S = Symo.Make (RNN)

let uniformise_diag m =
  let open Infix in
  let d = recip (sqrt (diag m +$ 1e-12)) in
  reshape [| -1; 1 |] d * m * reshape [| 1; -1 |] d

let save_cov_for label theta =
  let s2 =
    let factors = S.Second_order.factors_of_pair ~symmetric:true theta theta in
    S.Second_order.dense_of_factors ~symmetric:true ~full:true factors
  in
  let s2_emp =
    let key = Rng.key 1985 in
    let flatten x =
      RNN.fold
        (fun _ acc x ->
           let x = reshape [| -1; 1 |] x in
           match acc with
           | None -> Some x
           | Some a -> Some (concatenate ~axis:0 [ a; x ]))
        None
        x
      |> Option.value_exn
    in
    List.range 0 10_000
    |> List.fold ~init:(scalar float32 0.) ~f:(fun acc i ->
      if Int.(i % 10 = 0) then Stdio.printf "\r%06i%!" i;
      let g = S.Orbit.random_transform ~key:(Rng.fold_in key i) grad |> flatten in
      Infix.(acc + (g *@ transpose g /$ Float.(of_int 10_000))))
  in
  Stdio.print_endline "";
  let s2 = uniformise_diag s2 in
  let s2_emp = uniformise_diag s2_emp in
  Nx_io.save_txt (in_dir (Printf.sprintf "rnn_%s_s2" label)) s2;
  Nx_io.save_txt (in_dir (Printf.sprintf "rnn_%s_s2_emp" label)) s2_emp;
  let cmap = Hugin.Cmap.cividis in
  Hugin.heatmap ~cmap ~data:s2 ~vmin:(-0.7) ~vmax:0.7 ()
  |> Hugin.xlim 0. 128.
  |> Hugin.ylim 0. 128.
  |> Hugin.no_axes
  |> Hugin.render_png
       ~width:512.
       ~height:512.
       (in_dir (Printf.sprintf "rnn_%s.png" label))

let _ = save_cov_for "params" student
let _ = save_cov_for "grad" grad
