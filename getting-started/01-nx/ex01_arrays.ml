(* Nx: arrays with a dtype and a shape.

   Run with:
   dune exec getting-started/01-nx/ex01_arrays.exe *)

let show name x =
  Printf.printf
    "%-20s shape=%s  %s\n%!"
    name
    (Array.to_list (Nx.shape x) |> List.map string_of_int |> String.concat "x")
    (Nx.to_string x)

let () =
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.create Nx.float32 [| 2; 3 |] [| 10.; 20.; 30.; 40.; 50.; 60. |] in
  let row = Nx.create Nx.float32 [| 1; 3 |] [| 1.; 0.; -1. |] in
  show "a" a;
  show "b" b;
  show "a + b" (Nx.add a b);
  show "a * b" (Nx.mul a b);
  show "a + row" (Nx.add a row);
  show "sum along rows" (Nx.sum ~axes:[ 1 ] a);
  show "mean" (Nx.mean a);
  (* A 2x3 times a 3x2 is a 2x2. *)
  let c = Nx.matmul a (Nx.transpose a) in
  show "a @ a^T" c;
  Printf.printf "\nElement [1;2] of a: %.1f\n%!" (Nx.item [ 1; 2 ] a);
  Printf.printf "The dtype is %s\n%!" (Nx_dtype.to_string (Nx.dtype a))
