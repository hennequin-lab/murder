(* Rune: fit a linear model by differentiating a typed parameter record.
 *
 * Run with:
 *   dune exec getting-started/02-rune/ex02_linear_regression.exe *)

type 'a params =
  { w : 'a
  ; b : 'a
  }

module Params = struct
  type 'a t = 'a params

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" leaf b in
    { w; b }
end

let params = Nx.Ptree.instantiate (module Params)
let learning_rate = 0.1
let steps = 200
let report_every = 40

let initial_params () =
  { w = Nx.zeros Nx.float32 [| 3; 1 |]; b = Nx.zeros Nx.float32 [| 1 |] }

let () =
  Nx.Rng.with_key (Nx.Rng.key 0)
  @@ fun () ->
  (* y = x @ w_true + b_true + a little noise. *)
  let w_true = Nx.create Nx.float32 [| 3; 1 |] [| 2.; -1.; 0.5 |] in
  let b_true = 0.3 in
  let x = Nx.randn Nx.float32 [| 64; 3 |] in
  let noise = Nx.mul_s (Nx.randn Nx.float32 [| 64; 1 |]) 0.01 in
  let y = Nx.add_s (Nx.add (Nx.matmul x w_true) noise) b_true in
  let loss p =
    let pred = Nx.add (Nx.matmul x p.w) p.b in
    Nx.mean (Nx.square (Nx.sub pred y))
  in
  let step p =
    let value, grads = Rune.value_and_grad params loss p in
    let descend leaf gradient = Nx.sub leaf (Nx.mul_s gradient learning_rate) in
    { w = descend p.w grads.w; b = descend p.b grads.b }, Nx.item [] value
  in
  (* Thread the parameters through one step per recursion. The recursive call
     is in tail position, so the stack does not grow with [steps]. *)
  let rec train ~i p =
    if i > steps
    then p
    else (
      let p, value = step p in
      if i mod report_every = 0 then Printf.printf "step %3d  loss %.6f\n%!" i value;
      train ~i:(i + 1) p)
  in
  let p = train ~i:1 (initial_params ()) in
  Printf.printf "\nw (expected ~[2.0; -1.0; 0.5]):\n  %s\n%!" (Nx.to_string p.w);
  Printf.printf "b (expected ~0.3):\n  %s\n%!" (Nx.to_string p.b)
