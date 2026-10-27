(* Nx: indexing, slicing, reshape, and views.
 *
 * Run with:
 *   dune exec getting-started/01-nx/ex02_slicing.exe *)

open Nx

let show label x = Printf.printf "%-10s = %s\n%!" label (to_string x)

let () =
  let x = arange float32 0 12 1 |> reshape [| 3; 4 |] in
  Printf.printf "x =\n%s\n\n%!" (to_string x);
  let row = get [ 1 ] x in
  let column = slice [ A; I 2 ] x in
  let block = slice [ R (0, 2); R (1, 3) ] x in
  show "row 1" row;
  show "column 2" column;
  show "x[0:2,1:3]" block;
  Printf.printf "\n%!";
  (* [reshape] is a view when the layout allows it; [transpose] always is. *)
  let flat = reshape [| 12 |] x in
  let transposed = transpose x in
  show "flat" flat;
  Printf.printf "transposed =\n%s\n%!" (to_string transposed);
  (* Scalar reads copy to the host; use them at the edge of a computation. *)
  Printf.printf "x[2;3] = %.1f\n%!" (item [ 2; 3 ] x)
