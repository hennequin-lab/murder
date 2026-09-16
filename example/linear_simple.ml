open Base
open Nx

let device = "CPU"
let d_in, d_out = 100, 3
let batch_size = 512
let max_iter = 10_000
let lr = 0.1
let n_tangents = 128
let damping : Sofo.Optim.damping = `Absolute 0.

module Model = struct
  module P = struct
    type t = float32_t [@@deriving ptree]
  end

  let init ~d_in ~d_out =
    let open Infix in
    randn float32 [| d_in; d_out |] /$ Float.(sqrt (of_int d_in))

  let forward (w : P.t) x =
    let open Infix in
    x *@ w
end

let student = Model.init ~d_in ~d_out

let minibatch =
  let open Infix in
  let teacher = Model.init ~d_in ~d_out in
  let input_cov_sqrt =
    let u, _ = qr (randn float32 [| d_in; d_in |]) in
    let lambda =
      Array.init d_in ~f:(fun i -> Float.(1. / (1. + square (of_int Int.(i + 1)))))
      |> create float32 [| d_in; 1 |]
      |> fun x -> x / mean x
    in
    sqrt lambda * u
  in
  fun ~key bs ->
    let x = Rng.normal key float32 [| bs; d_in |] *@ input_cov_sqrt in
    let y = Model.forward teacher x in
    x, y

module Aux = struct
  type t = Rng.key [@@deriving ptree]
end

module O = Sofo.Optim.Compiled (Model.P) (Aux)

let objective params key =
  let open Infix in
  let x, y = minibatch ~key batch_size in
  let y' = Model.forward params x in
  Sofo.mse y' y

(* JIT compilation machinery for a sketched objective *)
let sketch_step =
  Rune.jit2 ~device (module O.In) (module O.Out) (O.sketch ~k:n_tangents objective)

let rec loop ~i (params : Model.P.t) (state : Sofo.Optim.state) =
  if i >= max_iter
  then params
  else (
    let key_data = Rng.fold_in state.key 0 in
    let out = sketch_step { params; key = state.key; aux = key_data } in
    let params, state = O.update ~lr ~damping state params out in
    let loss = item [] out.loss in
    Stdio.printf "[%05i] loss = %.6f\n%!" i loss;
    loop ~i:(i + 1) params state)

let state = Sofo.Optim.init ~key:(Rng.key 1985) ()
let _ = loop ~i:0 student state
