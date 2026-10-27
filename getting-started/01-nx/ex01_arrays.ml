(* Nx: arrays with a dtype and a shape.

   Run with:
   dune exec getting-started/01-nx/ex01_arrays.exe *)

open Nx

let shape_to_string x =
  shape x |> Array.to_list |> List.map string_of_int |> String.concat "x"

let show name x =
  Printf.printf "%-20s shape=%s  %s\n%!" name (shape_to_string x) (to_string x)

let () =
  let a = create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = create float32 [| 2; 3 |] [| 10.; 20.; 30.; 40.; 50.; 60. |] in
  let row = create float32 [| 1; 3 |] [| 1.; 0.; -1. |] in
  (* A [2x3] times a [3x2] is a [2x2]. *)
  let c = Infix.(a *@ transpose a) in
  List.iter
    (fun (name, x) -> show name x)
    [ "a", a
    ; "b", b
    ; ("a + b", Infix.(a + b))
    ; ("a * b", Infix.(a * b))
    ; ("a + row", Infix.(a + row))
    ; "sum along rows", sum ~axes:[ 1 ] a
    ; "mean", mean a
    ; "a @ a^T", c
    ];
  Printf.printf "\nElement [1;2] of a: %.1f\n%!" (item [ 1; 2 ] a);
  Printf.printf "The dtype is %s\n%!" (Nx_dtype.to_string (dtype a))
