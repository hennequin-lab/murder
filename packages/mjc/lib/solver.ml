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

(* {1 The row solve}

   The rows minimise a convex quadratic [½ λᵀ A λ + bᵀ λ] over a box, and
   everything the minimiser depends on is in this record. It is a record rather
   than six arguments because it is also the *input structure of a derivative*:
   {!solve_rows} attaches an implicit rule to it, and the rule's tangents are a
   value of the same shape. *)

type f = Nx.host Nx.float64_t

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

type rows = (Nx.host Nx.float64_t, Nx.host) rows_

(* The derived walk, instantiated at the model's dtype and place, as
   [Model.ptree] is. *)
module Rows_desc = struct
  type ('a, 'd) t = ('a, 'd) rows_

  let walk = walk_rows_
end

let rows_ptree : (rows, Nx.host) Nx.Ptree.t = Nx.Ptree.instantiate (module Rows_desc)

(* The active set at a solution: a row is free when it is strictly inside its
   box, and an equality row is always free. This is the KKT active set, and it
   is what the implicit derivative freezes — not the iteration's last mask,
   which at a converged solve is the same thing, and at an unconverged one is
   the very thing that would make the derivative depend on the budget. *)
let free_at (r : rows) (lambda : f) =
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
let system (r : rows) (free : f) =
  let nb = (Nx.shape r.a).(0) in
  let outer = Nx.matmul (Nx.reshape [| nb; 1 |] free) (Nx.reshape [| 1; nb |] free) in
  Nx.add (Nx.mul r.a outer) (Nx.diag (Nx.sub (Nx.ones_like free) free))

(* The iteration itself: a Gauss-Seidel warm start when the rows are box
   constrained, then active-set Newton with a monotone line search. *)
(* The iteration itself: a Gauss-Seidel warm start when the rows are box
   constrained, then active-set Newton with a monotone line search. *)
(* Active-set rounds per outer iteration. The whole search is one
   {!Rune.scan} of [iters * inner_rounds] steps, so the traced body is ONE
   round whatever the budgets are: the compile cost is flat in both, and
   exactness no longer has to be traded against the compiled step's size.
   Four rounds made a two-row limit solve at a violently accelerated humanoid
   state stall until the next boundary rebuild; eight solve every recorded
   state of the corpus to MuJoCo's optimum, and that is the budget.

   The scan does not make the round free at run time: a step costs roughly
   what one unrolled round cost, so a longer schedule costs correspondingly
   more, and the cost sheet in [doc/grad-report.md] carries both budgets'
   numbers. Converging early out of the scan — {!Rune.iterate}, whose body is
   traced once too but stops when the KKT test holds instead of after the
   budget — is the obvious next step and belongs to M7. *)

let iterate ~(linear : linear) ~iters ~boxed (r : rows) : f =
  let nb = (Nx.shape r.a).(0) in
  let rows_idx = Nx.arange_f Nx.float64 0.0 (Float.of_int nb) 1.0 in
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
  (* The scan. A step's phase says which working-set rule it starts from:
     0 is the first one, which takes every live row; 1 a later block
     boundary, which rebuilds from the gradient; 2 a round inside a block,
     which keeps the carried set. Whatever the phase, the same round then
     runs. The phase vector is host-computed — it is schedule, not data. *)
  let length = iters * inner_rounds in
  (* The scan folds over the rows of this array; each row is the phase of one
     round, as a [1]-shaped value. *)
  let phases =
    Nx.create
      Nx.float64
      [| length; 1 |]
      (Array.init length ~f:(fun i ->
         if Int.equal i 0
         then 0.0
         else if Int.equal (Int.rem i inner_rounds) 0
         then 1.0
         else 2.0))
  in
  let mask_f = Nx.where mask (Nx.ones_like rows_idx) zero in
  let cost v = Nx.add (Nx.mul_s (Nx.dot v (Nx.matmul r.a v)) 0.5) (Nx.dot r.b v) in
  let round (lambda, free) (phase : Nx.host Nx.float64_t) : _ * _ =
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
    (* The schedule is arithmetic, not data: phase 0 is the first boundary —
       which takes every live row — phase 1 a later boundary, which rebuilds
       from the gradient, phase 2 a round inside a block, which keeps the set
       the scan carried in. The comparisons are broadcast elementwise ops, not
       a branch on a scalar, which a traced step cannot take. *)
    let p = Nx.broadcast_to [| nb; 1 |] phase in
    let is_first = Nx.equal_s p 0.0 in
    let at_boundary = Nx.logical_or is_first (Nx.equal_s p 1.0) in
    (* A phase is one value per step; the masks it decides cover every row. *)
    let flag f = Nx.broadcast_to [| nb |] (Nx.slice [ I 0; A ] f) in
    let free_start_f =
      Nx.where
        (flag at_boundary)
        (Nx.where (flag is_first) mask_f (as_float rebuilt))
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
    (lambda', free'), Nx.reshape [| 1 |] (cost lambda')
  in
  let carry_ptree = Nx.Ptree.pair Nx.Ptree.tensor Nx.Ptree.tensor in
  let (final_lambda, _free), _outs =
    Rune.scan
      carry_ptree
      Nx.Ptree.tensor
      Nx.Ptree.tensor
      round
      ~init:(!lambda, mask_f)
      phases
  in
  final_lambda

(* The implicit rule. On the free block the minimiser solves a *linear* system,
   [A_FF λ_F = -b_F], so its derivative is
   [dλ_F = -A_FF⁻¹ (dA λ + db)_F] with the pinned rows held at their bound —
   the implicit function theorem at the KKT point, not the derivative of the
   iteration. That is what makes the gradient independent of how many
   iterations the caller asked for. *)
let implicit_tangent (r : rows) ~lambda (d : rows) : f =
  let free = free_at r lambda in
  let am = system r free in
  let solve v = Mass.solve (Mass.factor am) v in
  let rhs = Nx.neg (Nx.add (Nx.matmul d.a lambda) d.b) in
  Nx.where (Nx.greater_s free 0.5) (solve rhs) (Nx.zeros_like lambda)

let implicit_transpose (r : rows) ~lambda (w : f) : rows =
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
let solve_rows ?(implicit = true) ~(linear : linear) ~iters ~boxed (r : rows) : f =
  if not implicit
  then iterate ~linear ~iters ~boxed r
  else
    (* The rule computes the value and the tangent map in one visit: the
       iteration runs once, and the [λ] the tangent is taken at is the one the
       rule returns, which is what keeps it legal — a [λ] computed outside and
       closed over would be a value the differentiation tracks, used by a rule
       that does not know it. *)
    Rune.custom_jvp
      rows_ptree
      Nx.Ptree.tensor
      (fun r ->
         let lambda = iterate ~linear ~iters ~boxed r in
         ( lambda
         , fun d ->
             Rune.custom_transpose
               rows_ptree
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
      (m : Model.host)
      (d : Data.host)
  : Data.host
  =
  let t = m.topo
  and p = m.params in
  let nb = t.nefcmax in
  let rows_idx = Nx.arange_f Nx.float64 0.0 (Float.of_int nb) 1.0 in
  let mask = Nx.less rows_idx (Nx.cast Nx.float64 d.nefc) in
  let zero = Nx.zeros_like rows_idx in
  let select x = Nx.where mask x zero in
  (* A = J M^-1 J^T + diag R, on the fixed buffer. *)
  let l = Mass.factor d.mass_matrix in
  let jt = Nx.matrix_transpose d.efc_J in
  let solved = Mass.solve l jt in
  let a0 = Nx.matmul d.efc_J solved in
  let weight = Nx.where mask d.efc_R (Nx.ones_like rows_idx) in
  let a = Nx.add a0 (Nx.diag weight) in
  (* b = J qacc_smooth - aref, zero on the rows that do not exist. *)
  let b = select (Nx.sub (Nx.matmul d.efc_J d.qacc) d.efc_aref) in
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
    ; lambda0 = select d.efc_force
    }
  in
  let lambda = solve_rows ~implicit ~linear ~iters ~boxed rows in
  let qfrc_constraint = Nx.matmul jt lambda in
  let qacc = Nx.add d.qacc (Mass.solve l qfrc_constraint) in
  { d with efc_force = lambda; qfrc_constraint; qacc }
