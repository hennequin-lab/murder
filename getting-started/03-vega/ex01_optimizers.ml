(* Vega: optimizers that step whole parameter structures.
 *
 * Run with:
 *   dune exec getting-started/03-vega/ex01_optimizers.exe *)

let p = Nx.Ptree.tensor
let x0 () = Nx.create Nx.float32 [| 2 |] [| 5.; -3. |]

(* f(x) = 0.5 * ||x||^2, so grad f(x) = x. *)
let f x = Nx.mul_s (Nx.sum (Nx.square x)) 0.5

let run name init step =
  Printf.printf "--- %s ---\n%!" name;
  let params = ref (x0 ()) in
  let st = ref (init p !params) in
  for i = 1 to 50 do
    let grads = !params in
    let params', st' = step !st ~params:!params ~grads in
    params := params';
    st := st';
    if i mod 10 = 0
    then
      Printf.printf
        "  step %2d  f = %.6f  x = %s\n%!"
        i
        (Nx.item [] (f !params))
        (Nx.to_string !params)
  done;
  Printf.printf "\n%!"

let () =
  run "SGD (lr=0.1)" Vega.sgd_init (fun st -> Vega.sgd_step p ~lr:(Vega.lr 0.1) st);
  run "Adam (lr=0.01)" Vega.adam_init (fun st -> Vega.adam_step p ~lr:(Vega.lr 0.01) st);
  run "AdamW (lr=0.01, wd=0.01)" Vega.adamw_init (fun st ->
    Vega.adamw_step p ~lr:(Vega.lr 0.01) ~weight_decay:0.01 st)
