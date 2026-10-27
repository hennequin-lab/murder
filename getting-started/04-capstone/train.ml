(* Capstone: train a small MLP with Rune and Vega.
 *
 * Run with:
 *   dune exec getting-started/04-capstone/train.exe
 *
 * The data are two Gaussian blobs; the model is a one-hidden-layer MLP with a
 * ReLU. The training step is an ordinary function: differentiate the loss with
 * Rune, then hand the gradients to Vega. *)

open Nx

type 'a params =
  { w1 : 'a
  ; b1 : 'a
  ; w2 : 'a
  ; b2 : 'a
  }
[@@deriving ptree]

module Params = struct
  type 'a t = 'a params

  let walk = walk_params
end

let params = Nx.Ptree.instantiate (module Params)
let relu x = maximum x (zeros_like x)
let sigmoid x = Infix.(ones_like x / (exp (-x) +$ 1.))

let forward p x =
  let open Infix in
  let h = relu ((x *@ p.w1) + p.b1) in
  sigmoid ((h *@ p.w2) + p.b2)

let accuracy p x y =
  let open Infix in
  let pred = forward p x in
  let correct = where (pred >$ 0.5) (ones_like y) (zeros_like y) in
  item [] (mean (cast float32 (correct = y)))

let init key =
  Nx.Rng.with_key key
  @@ fun () ->
  let open Infix in
  let scale k = Float.sqrt (2. /. float k) in
  { w1 = randn float32 [| 2; 16 |] *$ scale 2
  ; b1 = zeros float32 [| 16 |]
  ; w2 = randn float32 [| 16; 1 |] *$ scale 16
  ; b2 = zeros float32 [| 1 |]
  }

let make_data () =
  let n = 256 in
  let a =
    Nx.Rng.with_key (Nx.Rng.key 1) @@ fun () -> Infix.(randn float32 [| n; 2 |] +$ 1.5)
  in
  let b =
    Nx.Rng.with_key (Nx.Rng.key 2) @@ fun () -> Infix.(randn float32 [| n; 2 |] +$ -1.5)
  in
  let x = concatenate ~axis:0 [ a; b ] in
  let y = concatenate ~axis:0 [ zeros float32 [| n; 1 |]; ones float32 [| n; 1 |] ] in
  x, y

let steps = 400
let report_every = 50
let learning_rate = 0.01

let () =
  let x, y = make_data () in
  let loss p =
    let open Infix in
    let pred = forward p x in
    mean (square (pred - y))
  in
  let step (p, st) =
    let value, grads = Rune.value_and_grad params loss p in
    let p, st = Vega.adam_step params ~lr:(Vega.lr learning_rate) st ~params:p ~grads in
    p, st, item [] value
  in
  (* The accumulator is the pair [(params, optimizer state)]; each recursion
     replaces it with the result of one step. *)
  let rec train ~i (p, st) =
    if i > steps
    then p
    else (
      let p, st, value = step (p, st) in
      if i mod report_every = 0
      then Printf.printf "step %3d  loss %.5f  accuracy %.3f\n%!" i value (accuracy p x y);
      train ~i:(i + 1) (p, st))
  in
  let p0 = init (Nx.Rng.key 3) in
  let p = train ~i:1 (p0, Vega.adam_init params p0) in
  Printf.printf "\nFinal accuracy: %.3f\n%!" (accuracy p x y)
