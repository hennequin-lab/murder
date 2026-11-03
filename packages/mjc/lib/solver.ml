(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base

let max_iterations = 8

type linear =
  [ `Cholesky
  | `Cg
  ]

(* Conjugate gradients on a symmetric positive definite system, with a fixed
   number of iterations: exact in [n] steps, and a compiled program cannot
   test for convergence anyway. [min_val] keeps a zero right-hand side from
   dividing by zero. *)
let cg ~iters a b =
  let min_val = 1e-300 in
  let x = ref (Nx.zeros_like b) in
  let r = ref b in
  let p = ref b in
  let rs = ref (Nx.dot b b) in
  for _ = 1 to iters do
    let ap = Nx.matmul a !p in
    let alpha = Nx.div !rs (Nx.maximum_s (Nx.dot !p ap) min_val) in
    x := Nx.add !x (Nx.mul !p alpha);
    r := Nx.sub !r (Nx.mul ap alpha);
    let rs' = Nx.dot !r !r in
    let beta = Nx.div rs' (Nx.maximum_s !rs min_val) in
    p := Nx.add !r (Nx.mul !p beta);
    rs := rs'
  done;
  !x

let forward ?(linear = `Cholesky) (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  let nb = t.nefcmax in
  let rows = Nx.arange_f Nx.float64 0.0 (Float.of_int nb) 1.0 in
  let mask = Nx.less rows (Nx.cast Nx.float64 d.nefc) in
  let zero = Nx.zeros_like rows in
  let select x = Nx.where mask x zero in
  (* A = J M^-1 J^T + diag R, on the fixed buffer. *)
  let l = Mass.factor d.mass_matrix in
  let jt = Nx.matrix_transpose d.efc_J in
  let solved = Mass.solve l jt in
  let a0 = Nx.matmul d.efc_J solved in
  let weight = Nx.where mask d.efc_R (Nx.ones_like rows) in
  let a = Nx.add a0 (Nx.diag weight) in
  (* b = J qacc_smooth - aref, zero on the rows that do not exist. *)
  let b = select (Nx.sub (Nx.matmul d.efc_J d.qacc) d.efc_aref) in
  (* Active-set Newton on the quadratic. A limit row can only push, so a row
     is in the free set while its force is positive or its reduced gradient is
     negative; the frozen rows are pinned to zero by an identity block, which
     keeps the system square and positive definite. The first iteration takes
     every row free, so a cold start reaches the unconstrained optimum before
     any row is frozen. *)
  let lambda = ref (select d.efc_force) in
  let iters = Int.min (Int.of_float (Nx.item [] p.iterations)) max_iterations in
  for i = 1 to iters do
    let grad = Nx.add (Nx.matmul a !lambda) b in
    let free =
      if i = 1
      then mask
      else
        Nx.logical_and
          mask
          (Nx.logical_or (Nx.greater_s !lambda 0.0) (Nx.less_s grad 0.0))
    in
    let ff = Nx.where free (Nx.ones_like rows) (Nx.zeros_like rows) in
    let outer = Nx.matmul (Nx.reshape [| nb; 1 |] ff) (Nx.reshape [| 1; nb |] ff) in
    let frozen = Nx.diag (Nx.sub (Nx.ones_like ff) ff) in
    let am = Nx.add (Nx.mul a outer) frozen in
    let rhs = Nx.where free (Nx.neg grad) (Nx.zeros_like grad) in
    let step =
      match linear with
      | `Cholesky -> Mass.solve (Mass.factor am) rhs
      | `Cg -> cg ~iters:nb am rhs
    in
    lambda
    := Nx.where free (Nx.maximum_s (Nx.add !lambda step) 0.0) (Nx.zeros_like !lambda)
  done;
  let qfrc_constraint = Nx.matmul jt !lambda in
  let qacc = Nx.add d.qacc (Mass.solve l qfrc_constraint) in
  { d with efc_force = !lambda; qfrc_constraint; qacc }
