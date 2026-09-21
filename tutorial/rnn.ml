(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Simple student-teacher trainign for an RNN *)

open Base
open Nx
open Symo

let print s = Stdio.print_endline (Sexp.to_string_hum s)
let in_dir = Cmdargs.in_dir "-d"

module RNN = struct
  module P = struct
    type 'a t =
      { b : 'a (* input x hidden *)
      ; w : 'a (* hidden x hidden *)
      ; c : 'a (* hidden x output *)
      ; bias : 'a (* hidden *)
      }
    [@@deriving ptree, sexp]

    let hidden = 128
    let input_dim = 16
    let output_dim = 32
    let dt = 0.1

    let dims =
      { b = [ input_dim; hidden ]
      ; w = [ hidden; hidden ]
      ; c = [ hidden; output_dim ]
      ; bias = [ hidden ]
      }

    let symmetries =
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
      ; w = (w *$ Float.(2. / sqrt (Float.of_int hidden)))
      ; c = c /$ Float.sqrt (Float.of_int hidden)
      ; bias
      }
  end

  let forward (p : float32_t P.t) ~input =
    (* input = horizon x bs x input_dim *)
    let bs = dim 1 input in
    let open Infix in
    Rune.scan
      (module struct
        type t = float32_t [@@deriving ptree]
      end)
      ~f:(fun x u ->
        let o = tanh x in
        let x = (x *$ Float.(1. - P.dt)) + (((o *@ p.w) + (u *@ p.b) + p.bias) *$ P.dt) in
        x, o *@ p.c)
      ~init:(zeros float32 [| bs; P.hidden |])
      input
    |> snd

  let forward_jit =
    Rune.jit
      (module struct
        type t = float32_t P.t * float32_t [@@deriving ptree]
      end)
      (fun (params, input) -> forward params ~input)

  let ptree = Ptree.instantiate (module P)
end

let teacher, student = Rng.with_key (Rng.key 42) @@ fun () -> RNN.P.init (), RNN.P.init ()
let horizon = 100
let full_batch = 1024
let batch = 128

let minibatch =
  let input =
    Rng.with_key (Rng.key 1)
    @@ fun () ->
    concatenate
      ~axis:0
      [ mul_s (randn float32 [| 10; full_batch; RNN.P.input_dim |]) 0.1
      ; zeros float32 [| horizon - 10; full_batch; RNN.P.input_dim |]
      ]
  in
  let target = RNN.forward_jit (teacher, input) in
  fun key bs ->
    let indices = Rng.permutation key full_batch |> slice [ R (0, bs) ] in
    take ~axis:1 ~indices input, take ~axis:1 ~indices target

let _ =
  let _, ys = minibatch (Rng.key 0) 1 in
  Nx_io.save_txt (in_dir "target_example") (slice [ A; I 0; A ] ys)

let loss (input, target) (p : Nx.float32_t RNN.P.t) =
  let pred = RNN.forward p ~input in
  mean (square Infix.(pred - target))

let value_and_grad_jit =
  Rune.jit2
    ~device:"CPU"
    (module struct
      type t = float32_t RNN.P.t * Rng.key [@@deriving ptree]
    end)
    (module struct
      type t = float32_t * float32_t RNN.P.t [@@deriving ptree]
    end)
    (fun (params, key) ->
       let data = minibatch key batch in
       Rune.value_and_grad RNN.ptree (loss data) params)

let train_with_symo () =
  let file = in_dir "loss_symo2" in
  Bos.Cmd.(v "rm" % "-f" % file) |> Bos.OS.Cmd.run |> ignore;
  let module S = Symo.Make (RNN.P) in
  Stdlib.Gc.full_major ();
  let config : S.config =
    { learning_rate = Some 0.001; beta_1 = 0.9; beta_2 = 0.99; damping = 1e-5 }
  in
  let compiled = S.Compiled.create ~config in
  let state = S.init ~config student in
  let rec loop i state key =
    let keys = Rng.split key ~n:2 in
    let loss_val, grads = value_and_grad_jit (state.S.State.theta, keys.(0)) in
    let mse = item [] loss_val in
    Stdio.printf "%8i  %.8f\n%!" i mse;
    if i % 10 = 0 then Nx_io.save_txt ~append:true file (create float32 [| 1 |] [| mse |]);
    let state = S.Compiled.step compiled ~config ~state ~grads in
    loop (i + 1) state keys.(1)
  in
  loop 0 state (Rng.key 1985)

let train_with_adam () =
  let file = in_dir "loss_adam" in
  Bos.Cmd.(v "rm" % "-f" % file) |> Bos.OS.Cmd.run |> ignore;
  let state = Vega.adam_init RNN.ptree student in
  let rec loop i params state key =
    let keys = Rng.split key ~n:2 in
    let loss_val, grads = value_and_grad_jit (params, keys.(0)) in
    let mse = item [] loss_val in
    Stdio.printf "%8i  %.8f\n%!" i mse;
    if i % 10 = 0 then Nx_io.save_txt ~append:true file (create float32 [| 1 |] [| mse |]);
    let params, state =
      Vega.adam_step RNN.ptree ~lr:(scalar float32 0.01) state ~params ~grads
    in
    loop (i + 1) params state keys.(1)
  in
  loop 0 student state (Rng.key 1985)

let _ = train_with_symo ()
