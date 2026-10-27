(* Rune: differentiate an ordinary OCaml function.
 *
 * Run with:
 *   dune exec getting-started/02-rune/ex01_grad.exe *)

let f x = Nx.sum (Nx.square x)

let () =
  let x = Nx.create Nx.float32 [| 4 |] [| 1.; -2.; 3.; 0.5 |] in
  let g = Rune.grad' f x in
  let expected = Nx.mul_s x 2.0 in
  let max_error = Nx.sub g expected |> Nx.abs |> Nx.max |> Nx.item [] in
  Printf.printf "f(x) = %g\n%!" (Nx.item [] (f x));
  Printf.printf "grad f(x) = %s\n%!" (Nx.to_string g);
  Printf.printf "expected 2x = %s\n%!" (Nx.to_string expected);
  Printf.printf "max error = %g\n%!" max_error
