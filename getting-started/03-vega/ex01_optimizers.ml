(* Vega: optimizers that step whole parameter structures.
 *
 * Run with:
 *   dune exec getting-started/03-vega/ex01_optimizers.exe *)

let p = Nx.Ptree.tensor
let x0 () = Nx.create Nx.float32 [| 2 |] [| 5.; -3. |]

(* f(x) = 0.5 * ||x||^2, so grad f(x) = x. *)
let f x = Nx.mul_s (Nx.sum (Nx.square x)) 0.5
let steps = 50
let report_every = 10

let run ~name ~init ~step =
  Printf.printf "--- %s ---\n%!" name;
  let rec loop ~i (params, state) =
    if i > steps
    then ()
    else (
      let grads = params in
      let params, state = step state ~params ~grads in
      if i mod report_every = 0
      then
        Printf.printf
          "  step %2d  f = %.6f  x = %s\n%!"
          i
          (Nx.item [] (f params))
          (Nx.to_string params);
      loop ~i:(i + 1) (params, state))
  in
  let params = x0 () in
  loop ~i:1 (params, init p params);
  Printf.printf "\n%!"

let () =
  run ~name:"SGD (lr=0.1)" ~init:Vega.sgd_init ~step:(fun state ->
    Vega.sgd_step p ~lr:(Vega.lr 0.1) state);
  run ~name:"Adam (lr=0.01)" ~init:Vega.adam_init ~step:(fun state ->
    Vega.adam_step p ~lr:(Vega.lr 0.01) state);
  run ~name:"AdamW (lr=0.01, wd=0.01)" ~init:Vega.adamw_init ~step:(fun state ->
    Vega.adamw_step p ~lr:(Vega.lr 0.01) ~weight_decay:0.01 state)
