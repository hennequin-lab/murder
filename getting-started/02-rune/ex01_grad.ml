(* Rune: differentiate an ordinary OCaml function.
 *
 * Run with:
 *   dune exec getting-started/02-rune/ex01_grad.exe *)

let f x = Nx.sum (Nx.square x)

let () =
  let x = Nx.create Nx.float32 [| 4 |] [| 1.; -2.; 3.; 0.5 |] in
  let g = Rune.grad' f x in
  Printf.printf "f(x) = %g\n%!" (Nx.item [] (f x));
  Printf.printf "grad f(x) = %s\n%!" (Nx.to_string g);
  Printf.printf "expected 2x = %s\n%!" (Nx.to_string (Nx.mul_s x 2.0));
  Printf.printf
    "max error = %g\n%!"
    (Nx.item [] (Nx.max (Nx.abs (Nx.sub g (Nx.mul_s x 2.0)))))
