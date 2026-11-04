(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base

let max_iterations = 12

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
  let big = 1e300 in
  let hi = Nx.where friction floss (Nx.full_like rows big) in
  let lo =
    Nx.where
      friction
      (Nx.neg floss)
      (Nx.where bilateral (Nx.full_like rows (-.big)) (Nx.zeros_like rows))
  in
  let slack x bound = Nx.mul_s (Nx.abs bound) (x *. 1e-12) in
  let lambda = ref (select d.efc_force) in
  (* Only a box-constrained row set stalls the active set badly enough to
     need a warm start: the corpus's equality and limit rows converge from
     the warmstart force alone, and these sweeps are unrolled, so they cost
     tracing time in every compiled step. *)
  let boxed = Array.exists t.dof_frictionloss ~f:(fun x -> Float.( > ) x 0.0) in
  if boxed
  then (
    (* Gauss-Seidel warm start. The active set below is exact once it finds the
       right one, but a near-degenerate problem can stall it. Coordinate descent
       on the same quadratic with the same bounds is monotone and cannot cycle,
       so its answer is used as the starting point: it gets the active set right
       even when it has not converged, and Newton then finishes it exactly. Each
       sweep walks the rows in order, so this is the sequential form. *)
    let polish_sweeps = 8 in
    for _ = 1 to polish_sweeps do
      for r = 0 to nb - 1 do
        let row = Nx.slice [ I r; A ] a in
        let grad_r = Nx.add (Nx.dot row !lambda) (Nx.get [ r ] b) in
        let curvature = Nx.get [ r; r ] a in
        let fresh =
          Nx.sub (Nx.get [ r ] !lambda) (Nx.div grad_r curvature)
          |> fun x -> Nx.minimum (Nx.maximum x (Nx.get [ r ] lo)) (Nx.get [ r ] hi)
        in
        lambda := Nx.set [ I r ] fresh !lambda
      done
    done);
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
    let free =
      if i = 1
      then mask
      else
        Nx.logical_and
          mask
          (Nx.logical_or bilateral (Nx.logical_or (Nx.logical_not at_bound) wants_in))
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
    (* Accept the step only if it lowers the objective. A full Newton step can
       cross a bound or land on a wrong active set; the ratio test that would
       stop it exactly can cycle on a near-degenerate problem, so instead the
       step is tried at a few fractions and the largest one that improves is
       taken. The objective is the quadratic the rows minimise, so every
       iteration strictly decreases it and the iteration cannot cycle. *)
    let cost v =
      let av = Nx.matmul a v in
      Nx.add (Nx.mul_s (Nx.dot v av) 0.5) (Nx.dot b v)
    in
    let into_bounds v = Nx.where bilateral v (Nx.minimum (Nx.maximum v lo) hi) in
    let chosen = ref !lambda in
    let best = ref (cost !lambda) in
    let taken = ref (Nx.scalar Nx.float64 0.0) in
    List.iter [ 1.0; 0.5; 0.25; 0.125 ] ~f:(fun frac ->
      let candidate = into_bounds (Nx.add !lambda (Nx.mul_s step frac)) in
      let value = cost candidate in
      let take = Nx.logical_and (Nx.less_s !taken 0.5) (Nx.less value !best) in
      chosen := Nx.where take candidate !chosen;
      best := Nx.where take value !best;
      taken := Nx.where take (Nx.scalar Nx.float64 1.0) !taken);
    lambda := !chosen
  done;
  let qfrc_constraint = Nx.matmul jt !lambda in
  let qacc = Nx.add d.qacc (Mass.solve l qfrc_constraint) in
  { d with efc_force = !lambda; qfrc_constraint; qacc }
