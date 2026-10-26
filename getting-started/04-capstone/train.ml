(* Capstone: train a small MLP with Rune and Vega.
 *
 * Run with:
 *   dune exec getting-started/04-capstone/train.exe
 *
 * The data are two Gaussian blobs; the model is a one-hidden-layer MLP with a
 * ReLU. The training step is an ordinary function: differentiate the loss with
 * Rune, then hand the gradients to Vega. *)

type 'a params =
  { w1 : 'a
  ; b1 : 'a
  ; w2 : 'a
  ; b2 : 'a
  }

module Params = struct
  type 'a t = 'a params

  let walk c { w1; b1; w2; b2 } =
    let open Nx.Ptree.Walk in
    let w1 = field c "w1" leaf w1 in
    let b1 = field c "b1" leaf b1 in
    let w2 = field c "w2" leaf w2 in
    let b2 = field c "b2" leaf b2 in
    { w1; b1; w2; b2 }
end

let params = Nx.Ptree.instantiate (module Params)
let relu x = Nx.maximum x (Nx.zeros_like x)
let sigmoid x = Nx.div (Nx.ones_like x) (Nx.add_s (Nx.exp (Nx.neg x)) 1.)

let forward p x =
  let h = relu (Nx.add (Nx.matmul x p.w1) p.b1) in
  sigmoid (Nx.add (Nx.matmul h p.w2) p.b2)

let accuracy p x y =
  let pred = forward p x in
  let correct = Nx.where (Nx.greater_s pred 0.5) (Nx.ones_like y) (Nx.zeros_like y) in
  Nx.item [] (Nx.mean (Nx.cast Nx.float32 (Nx.equal correct y)))

let init key =
  Nx.Rng.with_key key
  @@ fun () ->
  let scale k = Float.sqrt (2. /. float k) in
  { w1 = Nx.mul_s (Nx.randn Nx.float32 [| 2; 16 |]) (scale 2)
  ; b1 = Nx.zeros Nx.float32 [| 16 |]
  ; w2 = Nx.mul_s (Nx.randn Nx.float32 [| 16; 1 |]) (scale 16)
  ; b2 = Nx.zeros Nx.float32 [| 1 |]
  }

let make_data () =
  let n = 256 in
  let a =
    Nx.Rng.with_key (Nx.Rng.key 1)
    @@ fun () -> Nx.add_s (Nx.randn Nx.float32 [| n; 2 |]) 1.5
  in
  let b =
    Nx.Rng.with_key (Nx.Rng.key 2)
    @@ fun () -> Nx.add_s (Nx.randn Nx.float32 [| n; 2 |]) (-1.5)
  in
  let x = Nx.concatenate ~axis:0 [ a; b ] in
  let y =
    Nx.concatenate
      ~axis:0
      [ Nx.zeros Nx.float32 [| n; 1 |]; Nx.ones Nx.float32 [| n; 1 |] ]
  in
  x, y

let () =
  let x, y = make_data () in
  let p = init (Nx.Rng.key 3) in
  let st = Vega.adam_init params p in
  let loss p =
    let pred = forward p x in
    Nx.mean (Nx.square (Nx.sub pred y))
  in
  let p = ref p
  and st = ref st in
  for i = 1 to 400 do
    let l, g = Rune.value_and_grad params loss !p in
    let p', st' = Vega.adam_step params ~lr:(Vega.lr 0.01) !st ~params:!p ~grads:g in
    p := p';
    st := st';
    if i mod 50 = 0
    then
      Printf.printf
        "step %3d  loss %.5f  accuracy %.3f\n%!"
        i
        (Nx.item [] l)
        (accuracy !p x y)
  done;
  Printf.printf "\nFinal accuracy: %.3f\n%!" (accuracy !p x y)
