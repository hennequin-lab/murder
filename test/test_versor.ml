open Geom
module V = Versor.Make (E3)

let failf fmt = Printf.ksprintf failwith fmt
let check name b = if not b then failf "FAIL %s" name
let approx a b = Float.abs (a -. b) < 1e-3

let eq_mv a b =
  let x = E3.to_array a
  and y = E3.to_array b in
  Array.for_all2 approx x y

let () =
  let axis = Nx.create E3.dtype [| 3 |] [| 0.; 0.; 1. |] in
  let r = E3.rotor ~axis ~angle:(Nx.scalar E3.dtype 0.7) in
  let l = V.log r in
  check "exp(log rotor)" (eq_mv (V.exp l) r);
  check "exp_simple log" (eq_mv (V.exp_simple l) r);
  let r1 = E3.rotor ~axis ~angle:(Nx.scalar E3.dtype 1.4) in
  check "versor slerp 0" (eq_mv (V.slerp r r1 (Nx.scalar E3.dtype 0.)) r);
  check "versor slerp 1" (eq_mv (V.slerp r r1 (Nx.scalar E3.dtype 1.)) r1);
  let b = Nx.mul_s (E3.gp (E3.e 1) (E3.e 2)) (-0.35) in
  check "general exp simple" (eq_mv (V.exp b) (V.exp_simple b));
  Printf.printf "VERSOR OK\n"
