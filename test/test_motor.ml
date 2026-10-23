open Geom

let failf fmt = Printf.ksprintf failwith fmt
let check name b = if not b then failf "FAIL %s" name
let approx a b = Float.abs (a -. b) < 1e-4
let vec6 a b c d e f = Nx.create Motor.dtype [| 6 |] [| a; b; c; d; e; f |]

let check_motor name m1 m2 =
  let a = Motor.to_array m1
  and b = Motor.to_array m2 in
  check name (Array.for_all2 approx a b)

let check_vec name a b =
  check name (Array.length a = Array.length b && Array.for_all2 approx a b)

let () =
  let twists =
    [ vec6 0. 0. 0. 0. 0. 1.
    ; vec6 0. 0. 0.5 0. 0. 0.
    ; vec6 0.2 (-0.1) 0.5 0.3 0.1 (-0.2)
    ]
  in
  List.iter
    (fun tw ->
       let m = Motor.exp tw in
       let l = Motor.log m in
       check_vec "log(exp twist)" (Nx.to_array l) (Nx.to_array tw))
    twists;
  let m =
    Motor.mul
      (Motor.translate3 0.5 (-0.2) 0.3)
      (Motor.rotate
         ~axis:(Nx.create Motor.dtype [| 3 |] [| 0.; 0.; 1. |])
         ~angle:(Nx.scalar Motor.dtype 0.7))
  in
  check_motor "exp(log m)" (Motor.exp (Motor.log m)) m;
  let m0 = Motor.translate3 0. 0. 0.
  and m1 = Motor.translate3 1. 0. 0. in
  check_motor "slerp 0" (Motor.slerp m0 m1 (Nx.scalar Motor.dtype 0.)) m0;
  check_motor "slerp 1" (Motor.slerp m0 m1 (Nx.scalar Motor.dtype 1.)) m1;
  let tw = vec6 0.1 (-0.2) 0.3 0.2 0.1 (-0.1) in
  let left = Motor.log (Motor.mul (Motor.mul m (Motor.exp tw)) (Motor.inverse m)) in
  let right =
    Nx.matmul (Motor.adjoint m) (Nx.reshape [| 6; 1 |] tw) |> Nx.reshape [| 6 |]
  in
  check_vec "adjoint" (Nx.to_array left) (Nx.to_array right);
  Printf.printf "MOTOR OK\n"
