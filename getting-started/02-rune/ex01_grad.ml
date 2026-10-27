(* Rune: differentiate an ordinary OCaml function.
 *
 * Run with:
 *   dune exec getting-started/02-rune/ex01_grad.exe *)

open Nx

let f x = sum (square x)

let () =
  let open Infix in
  let x = create float32 [| 4 |] [| 1.; -2.; 3.; 0.5 |] in
  let g = Rune.grad' f x in
  let expected = x *$ 2.0 in
  let max_error = g - expected |> abs |> max |> item [] in
  Printf.printf "f(x) = %g\n%!" (item [] (f x));
  Printf.printf "grad f(x) = %s\n%!" (to_string g);
  Printf.printf "expected 2x = %s\n%!" (to_string expected);
  Printf.printf "max error = %g\n%!" max_error
