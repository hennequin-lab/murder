(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx

(* The schedule runs `iters × inner_rounds` rounds whatever the problem needs;
   MuJoCo's Newton exits on tolerance, so a model asking for eight pays nothing
   for the unused ones and we pay 2x. Four passes is where the corpus stops
   changing: `G4_ITERS=4` reproduces the gate's numbers *exactly* on hopper,
   walker and humanoid (same worst error, same shadowing horizon), while 2 fails
   and 1 fails badly. So four is the cap, and `Step.config ?iters` is how a
   caller says otherwise. *)
let max_iterations = 4

type linear =
  [ `Cholesky (** {!Mass.factor} and {!Mass.solve}: the kernels. *)
  | `Batched
    (** {!Linalg}: the same factorisation written out of elementwise
        operations, so that a map over the lanes of a step keeps one lane's
        cost instead of paying the batched kernel's, which loses past about
        forty rows (`doc/perf.md` §5). Its own accuracy is 1e-12 against the
        kernel on well-conditioned matrices (`test/test_linalg.ml`), and it is
        not the default: on the corpus's stiffest contact state it moves the
        solved force by 1e-4, which is a conditioning question but not one a
        parity test will sign off on. *)
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

(* {1 The row solve}

   The rows minimise a convex quadratic [½ λᵀ A λ + bᵀ λ] over a box, and
   everything the minimiser depends on is in this record. It is a record rather
   than six arguments because it is also the *input structure of a derivative*:
   {!solve_rows} attaches an implicit rule to it, and the rule's tangents are a
   value of the same shape. *)

type ('s, 'd) f = ('s, 'd) Rigid.f

(* Polymorphic in its arrays, as the model's parameter record is, so the ptree
   derivation has a dtype to be polymorphic over. *)
type ('a, 'd) rows_ =
  { a : 'a (** [[nb;nb]], [J M⁻¹ Jᵀ + diag R], symmetric positive definite. *)
  ; b : 'a (** [[nb]], [J qacc_smooth - aref]. *)
  ; lo : 'a (** [[nb]], the lower bound of each row's force. *)
  ; hi : 'a (** [[nb]], the upper bound. *)
  ; mask : 'a (** [[nb]], [1] on the rows that exist. *)
  ; bilateral : 'a (** [[nb]], [1] on the rows that pull as well as push. *)
  ; lambda0 : 'a (** [[nb]], the warm start. *)
  }
[@@deriving ptree]

type ('s, 'd) rows = (('s, 'd) Rigid.f, 'd) rows_

(* The derived walk, instantiated at the model's dtype and place, as
   [Model.ptree] is. *)
module Rows_desc = struct
  type ('a, 'd) t = ('a, 'd) rows_

  let walk = walk_rows_
end

(* A function, not a value: a description is polymorphic in the device only
   until the value restriction fixes it, and a compiled program may run on any
   device. Each call instantiates it afresh at the brand in use. *)
let rows_ptree () = Nx.Ptree.instantiate (module Rows_desc)

(* The active set at a solution: a row is free when it is strictly inside its
   box, and an equality row is always free. This is the KKT active set, and it
   is what the implicit derivative freezes — not the iteration's last mask,
   which at a converged solve is the same thing, and at an unconverged one is
   the very thing that would make the derivative depend on the budget. *)
let free_at (r : ('s, 'a) rows) (lambda : ('s, 'a) f) =
  let big = 1e300 in
  let one = Nx.ones_like lambda in
  let at_hi = Nx.greater_equal lambda (Nx.sub r.hi (Nx.mul_s (Nx.abs r.hi) 1e-12)) in
  let at_lo = Nx.less_equal lambda (Nx.add r.lo (Nx.mul_s (Nx.abs r.lo) 1e-12)) in
  let at_bound = Nx.logical_and (Nx.less_s r.bilateral 0.5) (Nx.logical_or at_hi at_lo) in
  ignore big;
  ignore one;
  Nx.mul r.mask (Nx.where at_bound (Nx.zeros_like lambda) (Nx.ones_like lambda))

(* The matrix the free block solves, kept square and positive definite by an
   identity block on the pinned rows. *)
let system (r : ('s, 'a) rows) (free : ('s, 'a) f) =
  let nb = (Nx.shape r.a).(0) in
  let outer = Nx.matmul (Nx.reshape [| nb; 1 |] free) (Nx.reshape [| 1; nb |] free) in
  Nx.add (Nx.mul r.a outer) (Nx.diag (Nx.sub (Nx.ones_like free) free))

(* The iteration itself: a Gauss-Seidel warm start when the rows are box
   constrained, then active-set Newton with a monotone line search. *)
(* Active-set rounds per outer iteration. The search is one {!Rune.iterate} of
   at most [iters * inner_rounds] steps, so the traced body is ONE round
   whatever the budgets are: the compile cost is flat in both, and exactness no
   longer has to be traded against the compiled step's size. Four rounds made a
   two-row limit solve at a violently accelerated humanoid state stall until the
   next boundary rebuild; eight solve every recorded state of the corpus to
   MuJoCo's optimum, and that is the cap.

   The cap is a budget, not a schedule: the loop stops as soon as the KKT
   conditions hold ([kkt_tolerance]), which on the corpus is after one or two
   rounds. The budget still bounds the work, because a compiled program unrolls
   nothing and a loop must terminate; the difference between the two is the
   difference between [doc/perf.md] §3's "a third to a half of a step is rounds
   that were never needed" and paying for the rounds taken. *)

let inner_rounds = 8

(** [kkt_tolerance] is the residual at which the iteration stops, relative to
    the driving term's own magnitude: the largest violation may be
    [kkt_tolerance] times [1 + ‖b‖∞]. The violation is the largest way a row
    breaks the KKT conditions of the box-constrained quadratic — a free row
    whose gradient is not zero, or a row held at a bound whose gradient points
    back into the box.

    The test is on the gradient, not on the cost improvement, because a blocked
    round makes no progress and is not a converged one: it pins a row and the
    next round rebuilds the set. A gradient residual cannot be fooled that way.

    [1e-12] is a measurement. The round solves its free block exactly, so at the
    correct working set the next round's step is zero and the iterate is already
    the answer; the loop returns the iterate it certified, and what that costs is
    the last digits. Over the 160 recorded states of the five models
    `test/test_constraint.ml` checks, the forces agree with a solve that spends
    its whole budget to [2.3e-16] relative and the accelerations to [2.2e-12] —
    four orders inside the [1e-8] a step is gated at and six inside the [1e-6] a
    row force is. Every threshold from [1e-8] down to [1e-14] passes the
    constraint and dynamics parity suites. It is tight enough that the corpus
    cannot tell the two apart, and loose enough that a badly conditioned contact
    — where the exact solve's floor is [1e-13] to [1e-11] of [‖b‖∞] — still
    certifies instead of paying its budget. *)
let kkt_tolerance = 1e-12

(** A row set and its brand, named so that the loop's traced body can annotate
    the values it carries: an annotation inside the body would otherwise have to
    name a concrete device. *)
let iterate
      (type d)
      ?(stop = kkt_tolerance)
      ~(linear : linear)
      ~iters
      ~boxed
      (r : ('s, d) rows)
  : ('s, d) f
  =
  let nb = (Nx.shape r.a).(0) in
  let at = Nx.placement r.a in
  let dt = Nx.dtype r.a in
  let rows_idx = Nx.arange_f_at at dt 0.0 (Float.of_int nb) 1.0 in
  let zero = Nx.zeros_like rows_idx in
  let mask = Nx.greater_s r.mask 0.5 in
  let bilateral = Nx.greater_s r.bilateral 0.5 in
  let slack x bound = Nx.mul_s (Nx.abs bound) (x *. 1e-12) in
  let lambda = ref (Nx.where mask r.lambda0 zero) in
  if boxed
  then (
    let polish_sweeps = 8 in
    for _ = 1 to polish_sweeps do
      for i = 0 to nb - 1 do
        let row = Nx.slice [ I i; A ] r.a in
        let grad_i = Nx.add (Nx.dot row !lambda) (Nx.get [ i ] r.b) in
        let curvature = Nx.get [ i; i ] r.a in
        let fresh =
          Nx.sub (Nx.get [ i ] !lambda) (Nx.div grad_i curvature)
          |> fun x -> Nx.minimum (Nx.maximum x (Nx.get [ i ] r.lo)) (Nx.get [ i ] r.hi)
        in
        lambda := Nx.set [ I i ] fresh !lambda
      done
    done);
  (* The loop. A round's phase says which working-set rule it starts from: 0 is
     the first one, which takes every live row; 1 a later block boundary, which
     rebuilds from the gradient; 2 a round inside a block, which keeps the
     carried set. The phase is a function of the round's index, which the loop
     carries and the boundary rule reads — schedule, not data, but the loop has
     no row to read it from. *)
  let length = iters * inner_rounds in
  if Int.( <= ) length 0
  then !lambda
  else (
    let period = Float.of_int inner_rounds in
    let last = Float.of_int (length - 1) in
    let mask_f = Nx.where mask (Nx.ones_like rows_idx) zero in
    let cost v = Nx.add (Nx.mul_s (Nx.dot v (Nx.matmul r.a v)) 0.5) (Nx.dot r.b v) in
    (* The scale the residual is measured against: the driving term, which is
       the constraint-space acceleration the rows are asked to supply. Adding
       one keeps an unloaded problem's threshold absolute rather than zero. *)
    let threshold = Nx.mul_s (Nx.add_s (Nx.max (Nx.abs r.b)) 1.0) stop in
    let round (lambda, free, idx, done_) =
      (* The working set this round starts from. *)
      let grad = Nx.add (Nx.matmul r.a lambda) r.b in
      let at_hi = Nx.greater_equal lambda (Nx.sub r.hi (slack 1.0 r.hi)) in
      let at_lo = Nx.less_equal lambda (Nx.add r.lo (slack 1.0 r.lo)) in
      let at_bound =
        Nx.logical_and (Nx.logical_not bilateral) (Nx.logical_or at_hi at_lo)
      in
      let wants_in =
        Nx.logical_or
          (Nx.logical_and at_hi (Nx.greater_s grad 0.0))
          (Nx.logical_and at_lo (Nx.less_s grad 0.0))
      in
      let rebuilt =
        Nx.logical_and
          mask
          (Nx.logical_or bilateral (Nx.logical_or (Nx.logical_not at_bound) wants_in))
      in
      let as_float x = Nx.where x (Nx.ones_like lambda) (Nx.zeros_like lambda) in
      let is_first = Nx.equal_s idx 0.0 in
      let at_boundary = Nx.logical_or is_first (Nx.equal_s (Nx.mod_s idx period) 0.0) in
      let wide v = Nx.broadcast_to [| nb |] v in
      let free_start_f =
        Nx.where
          (wide at_boundary)
          (Nx.where (wide is_first) mask_f (as_float rebuilt))
          free
      in
      (* free_start_f is 0/1-valued already; the round works on the boolean view. *)
      let free_start = Nx.greater_s free_start_f 0.5 in
      (* One round: Newton on the free block, exact ratio test, pin. *)
      let am = system r (as_float free_start) in
      let rhs = Nx.where free_start (Nx.neg grad) zero in
      let step =
        match linear with
        | `Cholesky -> Mass.solve (Mass.factor am) rhs
        | `Batched ->
          (* {!Linalg.solve} takes a *factor*, not a matrix: passing it [am] solved
             with an unfactored matrix, which is why this path diverged outright
             (hopper reached worst 2.0 over a thousand steps) while the blocked
             factorisation itself was as accurate as the kernel on every corpus
             state. {!Linalg.solve_spd} factors first. *)
          Linalg.solve_spd am (Nx.reshape [| nb; 1 |] rhs)
          |> fun x -> Nx.reshape [| nb |] x
        | `Cg -> cg ~iters:nb am rhs
      in
      let target = Nx.where (Nx.greater_s step 0.0) r.hi r.lo in
      let pushing = Nx.logical_and (Nx.logical_not bilateral) free_start in
      let outside =
        Nx.logical_or
          (Nx.logical_and (Nx.greater_s step 0.0) (Nx.greater (Nx.add lambda step) r.hi))
          (Nx.logical_and (Nx.less_s step 0.0) (Nx.less (Nx.add lambda step) r.lo))
      in
      (* A row whose step is zero is not blocking anything: it is either pinned
         or it has nothing to move. Letting it into the ratio would divide zero
         by zero, and a NaN alpha leaves the iterate exactly where it was. *)
      let cross = Nx.logical_and pushing outside in
      let ratio =
        Nx.where cross (Nx.div (Nx.sub target lambda) step) (Nx.full_like rows_idx 1.0)
      in
      let alpha =
        Nx.minimum_s
          (Nx.maximum_s (Nx.min (Nx.minimum ratio (Nx.full_like rows_idx 1.0))) 0.0)
          1.0
      in
      let candidate = Nx.add lambda (Nx.mul step alpha) in
      let candidate =
        Nx.where bilateral candidate (Nx.minimum (Nx.maximum candidate r.lo) r.hi)
      in
      let lambda' = Nx.where (Nx.less (cost candidate) (cost lambda)) candidate lambda in
      let blocked = Nx.logical_and cross (Nx.less_equal ratio (Nx.add_s alpha 1e-12)) in
      let free' = as_float (Nx.logical_and free_start (Nx.logical_not blocked)) in
      (* The KKT test, on the set a rebuild would take rather than on the one
         this round works with: [rebuilt] is a function of [lambda] alone, so a
         zero residual on it says [lambda] solves the problem, whatever set the
         round happened to carry. *)
      let violation =
        Nx.where
          rebuilt
          (Nx.abs grad)
          (Nx.where
             at_lo
             (Nx.maximum (Nx.neg grad) zero)
             (Nx.where at_hi (Nx.maximum grad zero) (Nx.abs grad)))
      in
      let converged = Nx.less_equal (Nx.max violation) threshold in
      (* A certified iterate is the answer: take no step past it. The test is on
         the set a rebuild would take, and a round in the middle of a block
         works with the set it carried, so its step can move a converged [lambda]
         by a couple of rounding units. Stopping before that step is what makes
         the loop's result the iterate it certified. *)
      let lambda' = Nx.where converged lambda lambda' in
      let done' =
        Nx.logical_or done_ (Nx.logical_or converged (Nx.greater_equal_s idx last))
      in
      lambda', free', Nx.add_s idx 1.0, done'
    in
    let carry =
      Nx.Ptree.quad Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.tensor
    in
    let final_lambda, _free, _idx, _done =
      Rune.iterate
        carry
        ~max:length
        ~until:(fun (_, _, _, done_) -> done_)
        round
        (!lambda, mask_f, Nx.scalar_at at dt 0.0, Nx.scalar_at at Nx.bool false)
    in
    final_lambda)

(* The implicit rule. On the free block the minimiser solves a *linear* system,
   [A_FF λ_F = -b_F], so its derivative is
   [dλ_F = -A_FF⁻¹ (dA λ + db)_F] with the pinned rows held at their bound —
   the implicit function theorem at the KKT point, not the derivative of the
   iteration. That is what makes the gradient independent of how many
   iterations the caller asked for. *)
let implicit_tangent (r : ('s, 'a) rows) ~lambda (d : ('s, 'a) rows) : ('s, 'a) f =
  let free = free_at r lambda in
  let am = system r free in
  let solve v = Mass.solve (Mass.factor am) v in
  let rhs = Nx.neg (Nx.add (Nx.matmul d.a lambda) d.b) in
  Nx.where (Nx.greater_s free 0.5) (solve rhs) (Nx.zeros_like lambda)

let implicit_transpose (r : ('s, 'a) rows) ~lambda (w : ('s, 'a) f) : ('s, 'a) rows =
  let free = free_at r lambda in
  let am = system r free in
  (* [A] is symmetric, so the transpose of the free block is its own solve. *)
  let z =
    Nx.where
      (Nx.greater_s free 0.5)
      (Nx.neg (Mass.solve (Mass.factor am) w))
      (Nx.zeros_like w)
  in
  let zero = Nx.zeros_like lambda in
  { a = Nx.outer z lambda
  ; b = z
  ; lo = zero
  ; hi = zero
  ; mask = zero
  ; bilateral = zero
  ; lambda0 = zero
  }

(* [solve_rows ?implicit ~linear ~iters ~boxed r] is the minimiser. Its
   derivative is the implicit rule; [~implicit:false] differentiates through the
   iteration instead, which is what the cost sheet measures the rule against. *)
let solve_rows
      (type d)
      ?(implicit = true)
      ?stop
      ~(linear : linear)
      ~iters
      ~boxed
      (r : ('s, d) rows)
  : ('s, d) f
  =
  if not implicit
  then iterate ?stop ~linear ~iters ~boxed r
  else
    (* The rule computes the value and the tangent map in one visit: the
       iteration runs once, and the [λ] the tangent is taken at is the one the
       rule returns, which is what keeps it legal — a [λ] computed outside and
       closed over would be a value the differentiation tracks, used by a rule
       that does not know it. *)
    Rune.custom_jvp
      (rows_ptree ())
      Nx.Ptree.tensor
      (fun r ->
         let lambda = iterate ?stop ~linear ~iters ~boxed r in
         ( lambda
         , fun d ->
             Rune.custom_transpose
               (rows_ptree ())
               Nx.Ptree.tensor
               (fun d -> implicit_tangent r ~lambda d)
               ~transpose:(fun w -> implicit_transpose r ~lambda w)
               d ))
      r

(* [iters] is the iteration budget the model's option asks for, capped at
   [max_iterations]. It is read from the model when the caller does not say,
   which is what a host caller wants; the differentiable entry point passes it
   explicitly, because reading a parameter tensor is a host read and a traced
   parameter cannot be read. *)
let forward
      ?(linear = `Cholesky)
      ?(implicit = true)
      ?iters
      ?rows
      ?stop
      (m : ('s, 'a) Model.m)
      (d : ('s, 'a) Data.d)
  : ('s, 'a) Data.d
  =
  let t = m.topo
  and p = m.params in
  (* The row buffer is the model's `nefcmax`, which is a bound chosen when the
     model was written; this step's rows are `nefc`, which is usually far
     smaller -- hopper has one row and a sixty-four row buffer, and the
     factorisation is cubic in the buffer. [rows] is how many the caller
     compiled for: the solve is `rows x rows` and a step that needs more raises
     rather than being truncated. *)
  let nb = Stdlib.min (Option.value rows ~default:t.nefcmax) t.nefcmax in
  let at = Nx.placement d.efc_J in
  let lead = Array.length (Nx.shape d.efc_J) - 2 in
  let head = List.init lead ~f:(fun _ -> A) in
  let take1 x = Nx.slice (head @ [ R (0, nb) ]) x in
  let take2 x = Nx.slice (head @ [ R (0, nb); A ]) x in
  Nx.check
    Nx.Ptree.unit
    (Nx.less_equal d.nefc (Nx.scalar_at at Nx.int32 (Stdlib.Int32.of_int nb)))
    ()
    (fun _ () ->
       Stdlib.invalid_arg
         (Printf.sprintf
            "Solver.forward: this step needs more than the %d rows it was compiled for \
             (nefcmax is %d). Raise the row budget, or the model's nefcmax if that is \
             what is short."
            nb
            t.nefcmax));
  let dt = Nx.dtype d.qpos in
  let rows_idx = Nx.arange_f_at at dt 0.0 (Float.of_int nb) 1.0 in
  let mask = Nx.less rows_idx (Nx.cast dt d.nefc) in
  let zero = Nx.zeros_like rows_idx in
  let select x = Nx.where mask x zero in
  (* A = J M^-1 J^T + diag R, on the fixed buffer. *)
  let l = Mass.factor d.mass_matrix in
  let j = take2 d.efc_J in
  let jt = Nx.matrix_transpose j in
  let solved = Mass.solve l jt in
  let a0 = Nx.matmul j solved in
  let weight = Nx.where mask (take1 d.efc_R) (Nx.ones_like rows_idx) in
  let a = Nx.add a0 (Nx.diag weight) in
  (* b = J qacc_smooth - aref, zero on the rows that do not exist. *)
  let b = select (Nx.sub (Nx.matmul j d.qacc) (take1 d.efc_aref)) in
  let of_type k =
    Nx.equal (take1 d.efc_type) (Nx.scalar_at at Nx.int32 (Stdlib.Int32.of_int k))
  in
  let bilateral = of_type Constraint.type_equality in
  let friction =
    Nx.logical_or
      (of_type Constraint.type_friction_dof)
      (of_type Constraint.type_friction_tendon)
  in
  let floss = take1 d.efc_frictionloss in
  (* The rows are the multipliers of a convex quadratic with bounds: equality
     rows are free, limit rows are non-negative, friction rows sit in
     [-loss, +loss]. *)
  let big = 1e300 in
  let hi = Nx.where friction floss (Nx.full_like rows_idx big) in
  let lo =
    Nx.where
      friction
      (Nx.neg floss)
      (Nx.where bilateral (Nx.full_like rows_idx (-.big)) (Nx.zeros_like rows_idx))
  in
  let boxed = Array.exists t.dof_frictionloss ~f:(fun x -> Float.( > ) x 0.0) in
  let iters =
    match iters with
    | Some n -> n
    | None -> Int.min (Int.of_float (Nx.item [] p.iterations)) max_iterations
  in
  ignore p;
  let rows =
    { a
    ; b
    ; lo
    ; hi
    ; mask = Nx.where mask (Nx.ones_like rows_idx) zero
    ; bilateral = Nx.where bilateral (Nx.ones_like rows_idx) zero
    ; lambda0 = select (take1 d.efc_force)
    }
  in
  let lambda = solve_rows ~implicit ?stop ~linear ~iters ~boxed rows in
  let qfrc_constraint = Nx.matmul jt lambda in
  let qacc = Nx.add d.qacc (Mass.solve l qfrc_constraint) in
  (* Dead rows are left at zero, so a stale force from a previous step cannot
     leak into a warm start. *)
  let efc_force = Nx.set (head @ [ R (0, nb) ]) lambda (Nx.zeros_like d.efc_force) in
  { d with efc_force; qfrc_constraint; qacc }
