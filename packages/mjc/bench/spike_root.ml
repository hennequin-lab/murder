(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* `Rune.root` against the hand-written KKT rule, on the same box-constrained
   quadratic.

   The residual is square and its zero is the minimiser: on the free rows the
   gradient is zero, and on a row pinned at a bound the residual is the distance
   from that bound. The active set is a comparison of the argument, so it has no
   derivative, and the Jacobian `root` builds is exactly the matrix the hand
   rule factorizes.

   Run: dune exec packages/mjc/bench/spike_root.exe *)

open Base
module Printf = Stdlib.Printf

let f64 = Nx.float64

(* A synthetic row set of [n] rows: a diagonally dominant symmetric matrix, so
   the quadratic is strictly convex, with every row bounded below at zero and
   half of them bounded above as well. *)
let synthetic n =
  let k = Float.of_int n in
  let a =
    Array.init (n * n) ~f:(fun i ->
      let r = i / n
      and c = Int.rem i n in
      if Int.equal r c
      then k +. (2.0 *. Float.sin (Float.of_int (r + 1)))
      else 0.3 *. Float.cos (Float.of_int ((3 * r) + c)) /. k)
  in
  { Mjc.Solver.a = Nx.create f64 [| n; n |] a
  ; b =
      Nx.create
        f64
        [| n |]
        (Array.init n ~f:(fun i -> 1.5 *. Float.sin (Float.of_int ((2 * i) + 1))))
  ; lo = Nx.zeros f64 [| n |]
  ; hi =
      Nx.create
        f64
        [| n |]
        (Array.init n ~f:(fun i ->
           if Int.equal (Int.rem i 2) 0 then 1e300 else 0.3 *. Float.cos (Float.of_int i)))
  ; mask = Nx.ones f64 [| n |]
  ; bilateral = Nx.zeros f64 [| n |]
  ; lambda0 = Nx.zeros f64 [| n |]
  }

let qp_rows ?(b = [| 1.0; -2.0; -3.5 |]) () =
  let a = [| 4.0; 1.0; 0.0; 1.0; 3.0; 0.5; 0.0; 0.5; 2.0 |] in
  { Mjc.Solver.a = Nx.create f64 [| 3; 3 |] a
  ; b = Nx.create f64 [| 3 |] b
  ; lo = Nx.create f64 [| 3 |] [| 0.0; 0.0; 0.0 |]
  ; hi = Nx.create f64 [| 3 |] [| 1e300; 1e300; 0.25 |]
  ; mask = Nx.create f64 [| 3 |] [| 1.0; 1.0; 1.0 |]
  ; bilateral = Nx.create f64 [| 3 |] [| 0.0; 0.0; 0.0 |]
  ; lambda0 = Nx.create f64 [| 3 |] [| 0.0; 0.0; 0.0 |]
  }

(* Conjugate gradients on an abstract linear operator, which is what
   [Rune.root]'s [linear_solve] receives: the operator is traced once, so a
   solver may only apply it. *)
let op_cg ~iters op b =
  let min_val = 1e-300 in
  let x = ref (Nx.zeros_like b) in
  let r = ref b
  and p = ref b
  and rs = ref (Nx.dot b b) in
  for _ = 1 to iters do
    let ap = op !p in
    let alpha = Nx.div !rs (Nx.maximum_s (Nx.dot !p ap) min_val) in
    x := Nx.add !x (Nx.mul !p alpha);
    r := Nx.sub !r (Nx.mul ap alpha);
    let rs' = Nx.dot !r !r in
    let beta = Nx.div rs' (Nx.maximum_s !rs min_val) in
    p := Nx.add !r (Nx.mul !p beta);
    rs := rs'
  done;
  !x

(* [free] and the bound a row sits at, as functions of the argument: comparisons
   and selects, so their derivatives are zero and the residual's Jacobian is the
   active-set matrix. *)
let residual_of (r : Mjc.Solver.rows) lambda =
  let at_hi = Nx.greater_equal lambda r.Mjc.Solver.hi in
  let at_lo = Nx.less_equal lambda r.Mjc.Solver.lo in
  let at_bound =
    Nx.logical_and (Nx.less_s r.Mjc.Solver.bilateral 0.5) (Nx.logical_or at_hi at_lo)
  in
  let free =
    Nx.mul
      r.Mjc.Solver.mask
      (Nx.where at_bound (Nx.zeros_like lambda) (Nx.ones_like lambda))
  in
  let bnd = Nx.where at_hi r.Mjc.Solver.hi r.Mjc.Solver.lo in
  let grad = Nx.add (Nx.matmul r.Mjc.Solver.a lambda) r.Mjc.Solver.b in
  Nx.add (Nx.mul free grad) (Nx.mul (Nx.sub (Nx.ones_like free) free) (Nx.sub lambda bnd))

let solve_root ?linear_solve (r : Mjc.Solver.rows) =
  let solve () =
    Mjc.Solver.solve_rows ~implicit:false ~linear:`Cholesky ~iters:20 ~boxed:false r
  in
  Rune.root ?linear_solve Nx.Ptree.tensor ~residual:(residual_of r) solve

let time ?(warmup = 3) ?(reps = 20) f x =
  for _ = 1 to warmup do
    ignore (f x)
  done;
  let t0 = Unix.gettimeofday () in
  for _ = 1 to reps do
    ignore (f x)
  done;
  (Unix.gettimeofday () -. t0) /. Float.of_int reps

let () =
  let r = qp_rows () in
  let hand =
    Mjc.Solver.solve_rows ~implicit:true ~linear:`Cholesky ~iters:20 ~boxed:false r
  in
  let iters =
    if Array.length Stdlib.Sys.argv > 1
    then Stdlib.int_of_string Stdlib.Sys.argv.(1)
    else 3
  in
  let by_root = solve_root r
  and by_root_cg = solve_root ~linear_solve:(op_cg ~iters) r in
  Printf.printf
    "hand     %s\n%!"
    (Array.to_list (Nx.to_array hand)
     |> List.map ~f:(Printf.sprintf "%.10f")
     |> String.concat ~sep:" ");
  Printf.printf
    "root     %s\n%!"
    (Array.to_list (Nx.to_array by_root)
     |> List.map ~f:(Printf.sprintf "%.10f")
     |> String.concat ~sep:" ");
  Printf.printf
    "root(cg) %s\n%!"
    (Array.to_list (Nx.to_array by_root_cg)
     |> List.map ~f:(Printf.sprintf "%.10f")
     |> String.concat ~sep:" ");
  let grad_of solve =
    let _, g =
      Rune.value_and_grad
        Mjc.Solver.rows_ptree
        (fun (r : Mjc.Solver.rows) -> Nx.sum (solve r))
        r
    in
    Nx.to_array g.Mjc.Solver.b
  in
  let gh =
    grad_of (fun r ->
      Mjc.Solver.solve_rows ~implicit:true ~linear:`Cholesky ~iters:20 ~boxed:false r)
  in
  let gr = grad_of solve_root
  and grc = grad_of (solve_root ~linear_solve:(op_cg ~iters)) in
  Printf.printf
    "d/db hand     %s\n%!"
    (Array.to_list gh |> List.map ~f:(Printf.sprintf "%.12f") |> String.concat ~sep:" ");
  Printf.printf
    "d/db root     %s\n%!"
    (Array.to_list gr |> List.map ~f:(Printf.sprintf "%.12f") |> String.concat ~sep:" ");
  Printf.printf
    "d/db root(cg) %s\n%!"
    (Array.to_list grc |> List.map ~f:(Printf.sprintf "%.12f") |> String.concat ~sep:" ");
  let t_hand =
    time
      (fun r ->
         let _, g =
           Rune.value_and_grad
             Mjc.Solver.rows_ptree
             (fun (r : Mjc.Solver.rows) ->
                Nx.sum
                  (Mjc.Solver.solve_rows
                     ~implicit:true
                     ~linear:`Cholesky
                     ~iters:20
                     ~boxed:false
                     r))
             r
         in
         g)
      r
  in
  let t_root =
    time
      (fun r ->
         let _, g =
           Rune.value_and_grad Mjc.Solver.rows_ptree (fun r -> Nx.sum (solve_root r)) r
         in
         g)
      r
  in
  let t_rootc =
    time
      (fun r ->
         let _, g =
           Rune.value_and_grad
             Mjc.Solver.rows_ptree
             (fun r -> Nx.sum (solve_root ~linear_solve:(op_cg ~iters) r))
             r
         in
         g)
      r
  in
  Printf.printf
    "value_and_grad: hand %.3f ms, root %.3f ms, root+cg %.3f ms\n%!"
    (t_hand *. 1e3)
    (t_root *. 1e3)
    (t_rootc *. 1e3)

(* How the two scale with the row count: the hand rule factorizes once, while
   [root] has to build the Jacobian by applying the residual to every basis
   vector. *)
let () =
  if Array.length Stdlib.Sys.argv > 2 && String.equal Stdlib.Sys.argv.(2) "scale"
  then
    List.iter [ 3; 10; 30; 60 ] ~f:(fun n ->
      let r = synthetic n in
      let hand () =
        let _, g =
          Rune.value_and_grad
            Mjc.Solver.rows_ptree
            (fun r ->
               Nx.sum
                 (Mjc.Solver.solve_rows
                    ~implicit:true
                    ~linear:`Cholesky
                    ~iters:20
                    ~boxed:false
                    r))
            r
        in
        g
      in
      let by_root () =
        let _, g =
          Rune.value_and_grad Mjc.Solver.rows_ptree (fun r -> Nx.sum (solve_root r)) r
        in
        g
      in
      let t_hand = time hand ()
      and t_root = time by_root () in
      Printf.printf
        "%3d rows: hand %7.3f ms, root %7.3f ms, root/hand %.1f\n%!"
        n
        (t_hand *. 1e3)
        (t_root *. 1e3)
        (t_root /. t_hand))
