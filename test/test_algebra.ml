open Geom

module A =
  Algebra.Make
    (struct
      let signature = Signature.euclidean 3
    end)
    (struct
      type kind = Nx.float32_elt

      let dtype = Nx.float32
    end)

let failf fmt = Printf.ksprintf failwith fmt
let check name b = if not b then failf "FAIL %s" name
let approx a b = Float.abs (a -. b) < 1e-5

let mv_equal u v =
  let a = A.to_array u
  and b = A.to_array v in
  Array.length a = Array.length b && Array.for_all2 (fun x y -> approx x y) a b

let () =
  let e1 = A.e 1
  and e2 = A.e 2
  and e3 = A.e 3 in
  let e12 = A.gp e1 e2 in
  check "e1*e1=1" (approx (A.coeff (A.gp e1 e1) (Blade.of_int 0) |> Nx.item []) 1.);
  check "e3*e3=1" (approx (A.coeff (A.gp e3 e3) (Blade.of_int 0) |> Nx.item []) 1.);
  check "e1*e2=e12" (approx (A.coeff e12 (Blade.of_int 0b11) |> Nx.item []) 1.);
  check
    "e2*e1=-e12"
    (approx (A.coeff (A.gp e2 e1) (Blade.of_int 0b11) |> Nx.item []) (-1.));
  check "e1^e2=e12" (approx (A.coeff (A.op e1 e2) (Blade.of_int 0b11) |> Nx.item []) 1.);
  check "e1^e1=0" (approx (A.norm2 (A.op e1 e1) |> Nx.item []) 0.);
  check
    "e1.e12=e2"
    (approx (A.coeff (A.inner e1 e12) (Blade.of_int 0b10) |> Nx.item []) 1.);
  check
    "reverse e12=-e12"
    (approx (A.coeff (A.reverse e12) (Blade.of_int 0b11) |> Nx.item []) (-1.));
  check "norm e12=1" (approx (A.norm e12 |> Nx.item []) 1.);
  let x = A.of_array [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
  let y = A.of_array [| 0.5; -1.; 2.; 0.25; -0.5; 1.5; 2.5; -3. |] in
  let z = A.of_array [| -2.; 0.75; 1.; -0.5; 3.; -1.25; 0.5; 2. |] in
  check "assoc" (mv_equal (A.gp (A.gp x y) z) (A.gp x (A.gp y z)));
  let d = A.dual e1 in
  Printf.printf "dual e1: ";
  A.pp Format.std_formatter d;
  Format.pp_print_newline Format.std_formatter ();
  Printf.printf "e1*e2: ";
  A.pp Format.std_formatter e12;
  Format.pp_print_newline Format.std_formatter ();
  Printf.printf "OK\n"
