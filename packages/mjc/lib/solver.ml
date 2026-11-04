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
     negative; equality rows are always free, since they pull and push, and a
     friction row is free while its force is inside the loss. The
     frozen rows are pinned to zero by an identity block, which keeps the
     system square and positive definite. The first iteration takes every row
     free, so a cold start reaches the unconstrained optimum before any row is
     frozen. *)
  (* Equality rows are two-sided: their force takes either sign, so they stay
     free and their step is never projected. *)
  let of_type k = Nx.equal d.efc_type (Nx.scalar Nx.int32 (Stdlib.Int32.of_int k)) in
  let bilateral = of_type Constraint.type_equality in
  let friction =
    Nx.logical_or
      (of_type Constraint.type_friction_dof)
      (of_type Constraint.type_friction_tendon)
  in
  let floss = d.efc_frictionloss in
  (* The rows are the multipliers of a convex quadratic with bounds: equality
     rows are free, limit rows are non-negative, friction rows sit in
     [-loss, +loss]. A primal active-set method solves it: a Newton step over
     the free block, a ratio test that stops the step at the first bound it
     would cross, and the blocking rows held at their bound until a solve has
     moved the others. Holding is what stops the cycling a plain
     project-after-the-step falls into. *)
  (* Only a friction row has an upper bound, so only it can block a step from
     above; a model without one keeps the cheaper step. *)
  let boxed = Array.exists t.dof_frictionloss ~f:(fun x -> Float.( > ) x 0.0) in
  let lambda = ref (select d.efc_force) in
  let held = ref (Nx.zeros_like rows) in
  let tiny = 1e-300 in
  let big = 1e300 in
  let hi = Nx.where friction floss (Nx.full_like rows big) in
  let lo =
    Nx.where
      friction
      (Nx.neg floss)
      (Nx.where bilateral (Nx.full_like rows (-.big)) (Nx.zeros_like rows))
  in
  let slack x bound = Nx.mul_s (Nx.abs bound) (x *. 1e-12) in
  let iters = Int.min (Int.of_float (Nx.item [] p.iterations)) max_iterations in
  for i = 1 to iters do
    let grad = Nx.add (Nx.matmul a !lambda) b in
    let at_hi = Nx.greater_equal !lambda (Nx.sub hi (slack 1.0 hi)) in
    let at_lo = Nx.less_equal !lambda (Nx.add lo (slack 1.0 lo)) in
    let at_bound =
      Nx.logical_and (Nx.logical_not bilateral) (Nx.logical_or at_hi at_lo)
    in
    (* A row at a bound wants back in when its gradient points inward: the
       gradient is the multiplier, and its sign is the KKT condition. *)
    let wants_in =
      Nx.logical_or
        (Nx.logical_and at_hi (Nx.greater_s grad 0.0))
        (Nx.logical_and at_lo (Nx.less_s grad 0.0))
    in
    let releasable = Nx.less_s !held 0.5 in
    let free =
      if i = 1
      then mask
      else
        Nx.logical_and
          mask
          (Nx.logical_or
             bilateral
             (Nx.logical_and
                releasable
                (Nx.logical_or (Nx.logical_not at_bound) wants_in)))
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
    let headroom b = Nx.sub b !lambda in
    let upward =
      Nx.where
        (Nx.logical_and (Nx.greater_s step 0.0) (Nx.greater_s (headroom hi) 0.0))
        (Nx.div (headroom hi) (Nx.maximum_s step tiny))
        (Nx.full_like rows big)
    in
    let downward =
      Nx.where
        (Nx.logical_and (Nx.less_s step 0.0) (Nx.less_s (headroom lo) 0.0))
        (Nx.div (headroom lo) (Nx.minimum_s step (-.tiny)))
        (Nx.full_like rows big)
    in
    let ratio = Nx.min (Nx.minimum upward downward) in
    (* The first pass is the unconstrained optimum, clamped into the box: a
       feasible start is what makes the ratio test meaningful afterwards. *)
    let alpha =
      if i = 1 || not boxed then Nx.scalar Nx.float64 1.0 else Nx.minimum_s ratio 1.0
    in
    let blocked =
      if not boxed
      then Nx.zeros_like rows
      else
        Nx.where
          (Nx.logical_and
             (Nx.less_s ratio 1.0)
             (Nx.logical_or (Nx.less_equal upward ratio) (Nx.less_equal downward ratio)))
          (Nx.ones_like rows)
          (Nx.zeros_like rows)
    in
    let stepped = Nx.add !lambda (Nx.mul step alpha) in
    let clamped = Nx.where bilateral stepped (Nx.minimum (Nx.maximum stepped lo) hi) in
    lambda := Nx.where free clamped (Nx.minimum (Nx.maximum !lambda lo) hi);
    held := blocked
  done;
  let qfrc_constraint = Nx.matmul jt !lambda in
  let qacc = Nx.add d.qacc (Mass.solve l qfrc_constraint) in
  { d with efc_force = !lambda; qfrc_constraint; qacc }
