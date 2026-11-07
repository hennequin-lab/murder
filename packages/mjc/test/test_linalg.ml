(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The batch-friendly factorisation against the kernels it replaces.

   `Linalg`'s Cholesky is written out of elementwise operations and reductions
   so that eight lanes of it cost less than eight separate solves; the price is
   that it is not the library kernel. This test makes the price explicit and
   bounded: the factor must reproduce `Nx.cholesky`'s, the solve must reproduce
   `Nx.solve_triangular`'s, both must hold for a batch of matrices with a
   leading axis, and both must differentiate — the solver's rounds are traced,
   so a factorisation that cannot be differentiated is not a candidate. *)

open Base
open Windtrap
open Mjc

let f64 = Nx.float64

(* A symmetric positive definite matrix from a seeded generator: [A = B Bᵀ + n I]
   with [B] in [-1, 1], which is well conditioned and not diagonal. *)
let spd ~n ~batch =
  let state = Stdlib.Random.State.make [| 20260214 |] in
  let rnd () = Stdlib.Random.State.float state 1.0 in
  let lead = Array.append batch [| n; n |] in
  let numel = Array.fold lead ~init:1 ~f:( * ) in
  let b = Nx.create f64 lead (Array.init numel ~f:(fun _ -> (2.0 *. rnd ()) -. 1.0)) in
  let bt = Nx.matrix_transpose b in
  let eye = Nx.broadcast_to lead (Nx.eye f64 n) in
  Nx.add (Nx.matmul b bt) (Nx.mul_s eye (Float.of_int n))

let max_abs a = Array.fold (Nx.to_array (Nx.abs a)) ~init:0.0 ~f:Float.max

(* Relative error of a result whose reference may be large, with the same
   denominator the rest of the suite uses. *)
let rel expected actual =
  let e = Nx.to_array (Nx.reshape [| -1 |] expected)
  and a = Nx.to_array (Nx.reshape [| -1 |] actual) in
  Array.fold2_exn e a ~init:0.0 ~f:(fun acc x y ->
    if Float.equal x y
    then acc
    else (
      let den = Float.max (Float.abs x) (Float.abs y) in
      Float.max acc (Float.abs (x -. y) /. den)))

let rhs ~n ~batch =
  let lead = Array.append batch [| n; 1 |] in
  let numel = Array.fold lead ~init:1 ~f:( * ) in
  let state = Stdlib.Random.State.make [| 7 |] in
  Nx.create
    f64
    lead
    (Array.init numel ~f:(fun _ -> 2.0 *. Stdlib.Random.State.float state 1.0))

let test_factor_matches_kernel () =
  List.iter
    [ 1, [||]; 2, [||]; 5, [||]; 16, [||]; 40, [||]; 7, [| 3 |]; 7, [| 8 |] ]
    ~f:(fun (n, batch) ->
      let a = spd ~n ~batch in
      let ours = Linalg.cholesky a in
      let kernel = Nx.cholesky a in
      (* The factor is unique for a positive definite matrix up to the sign of
         its diagonal, which is positive in both, so this is a direct
         comparison and not a "reconstruct and compare" one. *)
      let e = rel kernel ours in
      if Float.( > ) e 1e-12
      then
        failwith
          (Printf.sprintf
             "cholesky n=%d batch=%d: relative error %.3g"
             n
             (Array.fold batch ~init:1 ~f:( * ))
             e);
      (* And the defining property, which does not depend on the kernel being
         right: L Lᵀ = A. *)
      let recon = Nx.matmul ours (Nx.matrix_transpose ours) in
      let e = rel a recon in
      if Float.( > ) e 1e-12
      then failwith (Printf.sprintf "cholesky n=%d: L Lᵀ is off by %.3g" n e))

(* The blocked path against the unblocked recurrence and the kernel, at the
   row counts the solver meets: `nefcmax` is 48 to 202 in the corpus, and the
   block is sixteen, so these are one to thirteen panels. *)
let test_blocked_matches_unblocked () =
  List.iter
    [ 17, [||]; 32, [||]; 48, [||]; 76, [||]; 202, [||]; 76, [| 3 |] ]
    ~f:(fun (n, batch) ->
      let a = spd ~n ~batch in
      let blocked = Linalg.cholesky a in
      let unblocked = Linalg.cholesky ~block:n a in
      (* The two orders sum the same terms in a different order, so they agree
         to accumulated rounding rather than exactly: each of the [n] steps
         brings a few ulps, which is [`n * eps] with a small constant. The
         threshold is that bound, not a number picked to make this pass. *)
      let tol = 1e-14 *. Float.of_int n in
      let e = rel unblocked blocked in
      if Float.( > ) e tol
      then
        failwith
          (Printf.sprintf
             "blocked cholesky n=%d: against unblocked, %.3g (tol %.1e)"
             n
             e
             tol);
      (* And against the kernel, which shares neither order. *)
      let e = rel (Nx.cholesky a) blocked in
      if Float.( > ) e tol
      then
        failwith
          (Printf.sprintf
             "blocked cholesky n=%d: against the kernel, %.3g (tol %.1e)"
             n
             e
             tol);
      (* The defining property again, at a size where the blocked path is what
         ran. *)
      let recon = Nx.matmul blocked (Nx.matrix_transpose blocked) in
      let e = rel a recon in
      if Float.( > ) e 1e-11
      then failwith (Printf.sprintf "blocked cholesky n=%d: L Lᵀ is off by %.3g" n e);
      (* And it is still a factor one can solve with. *)
      let b = rhs ~n ~batch in
      let x = Linalg.solve blocked (Nx.reshape (Array.append batch [| n |]) b) in
      let e = rel b (Nx.matmul a (Nx.reshape (Array.append batch [| n; 1 |]) x)) in
      if Float.( > ) e 1e-10
      then failwith (Printf.sprintf "blocked cholesky n=%d: A x is off by %.3g" n e))

let test_solve_matches_kernel () =
  List.iter
    [ 1, [||]; 3, [||]; 12, [||]; 33, [||]; 6, [| 5 |] ]
    ~f:(fun (n, batch) ->
      let a = spd ~n ~batch in
      let b = rhs ~n ~batch in
      let ours = Linalg.solve_spd a b in
      (* The reference is the kernel's own path, so that a mistake in the
         recurrence cannot cancel against the same mistake in the test. *)
      let kernel =
        Nx.solve_triangular
          ~triangle:Lower
          ~transpose:Transpose
          (Nx.cholesky a)
          (Nx.solve_triangular ~triangle:Lower (Nx.cholesky a) b)
      in
      let e = rel kernel ours in
      if Float.( > ) e 1e-11
      then
        failwith
          (Printf.sprintf "solve n=%d: against the kernel, relative error %.3g" n e);
      (* And against the definition: A x = b. *)
      let e = rel b (Nx.matmul a ours) in
      if Float.( > ) e 1e-11
      then failwith (Printf.sprintf "solve n=%d: A x is off by %.3g" n e))

let test_differentiable () =
  (* The solver's rounds are traced, so both arguments have to differentiate.
     A quadratic loss in the solution has a closed form, so the check is
     against central differences on the same graph rather than against a
     formula the test would have to rederive. *)
  let n = 6 in
  let a = spd ~n ~batch:[||] in
  let b = rhs ~n ~batch:[||] in
  let loss v =
    Nx.sum (Nx.mul (Linalg.solve_spd a (Nx.add b v)) (Linalg.solve_spd a (Nx.add b v)))
  in
  let g = Rune.grad Nx.Ptree.tensor loss b in
  let analytic = Nx.to_array (Nx.reshape [| -1 |] g) in
  let eps = 1e-6 in
  let value v = Nx.item [] (loss (Nx.add b v)) in
  let numeric =
    Array.init (n * 1) ~f:(fun k ->
      let e =
        Nx.create
          f64
          [| n; 1 |]
          (Array.init (n * 1) ~f:(fun i -> if Int.equal i k then eps else 0.0))
      in
      (value e -. value (Nx.mul_s e (-1.0))) /. (2.0 *. eps))
  in
  let worst =
    Array.fold2_exn analytic numeric ~init:0.0 ~f:(fun acc x y ->
      Float.max acc (Float.abs (x -. y) /. Float.max 1.0 (Float.abs x)))
  in
  if Float.( > ) worst 1e-6
  then failwith (Printf.sprintf "gradient: worst relative error %.3g" worst);
  (* The matrix, too: a loss in the factor's entries differentiates through the
     recurrence, which is what makes the solver's gradient correct. *)
  let loss_a v =
    Nx.sum (Nx.mul (Linalg.solve_spd (Nx.add a v) b) (Linalg.solve_spd (Nx.add a v) b))
  in
  let g = Rune.grad Nx.Ptree.tensor loss_a (Nx.zeros_like a) in
  if Float.( > ) (max_abs g) 1e300
  then failwith "gradient with respect to A is not finite";
  if not (Float.equal (max_abs g) (max_abs g))
  then failwith "gradient with respect to A is NaN"

let () =
  Stdlib.exit
    (run
       "mjc linalg"
       [ test "factor matches the kernel, batched or not" test_factor_matches_kernel
       ; test
           "the blocked factor matches the unblocked one"
           test_blocked_matches_unblocked
       ; test "solve matches the kernel, batched or not" test_solve_matches_kernel
       ; test "factor and solve differentiate" test_differentiable
       ])
