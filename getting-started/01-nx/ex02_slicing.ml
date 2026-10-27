(* Nx: indexing, slicing, reshape, and views.
 *
 * Run with:
 *   dune exec getting-started/01-nx/ex02_slicing.exe *)

open Nx

let show label x = Printf.printf "%-10s = %s\n%!" label (Nx.to_string x)

let () =
  let x = Nx.arange Nx.float32 0 12 1 |> Nx.reshape [| 3; 4 |] in
  Printf.printf "x =\n%s\n\n%!" (Nx.to_string x);
  let row = Nx.get [ 1 ] x in
  let column = Nx.slice [ A; I 2 ] x in
  let block = Nx.slice [ R (0, 2); R (1, 3) ] x in
  show "row 1" row;
  show "column 2" column;
  show "x[0:2,1:3]" block;
  Printf.printf "\n%!";
  (* [reshape] is a view when the layout allows it; [transpose] always is. *)
  let flat = Nx.reshape [| 12 |] x in
  let transposed = Nx.transpose x in
  show "flat" flat;
  Printf.printf "transposed =\n%s\n%!" (Nx.to_string transposed);
  (* Scalar reads copy to the host; use them at the edge of a computation. *)
  Printf.printf "x[2;3] = %.1f\n%!" (Nx.item [ 2; 3 ] x)
