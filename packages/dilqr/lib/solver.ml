(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The iLQR solver, in two flavors:

   - [solve] is the classic adaptive iLQR: Levenberg-Marquardt regularization
     with Cholesky retry, Armijo backtracking line search, convergence checks.
     All control flow is host-side, so it runs eagerly only — do not place it
     inside Rune.jit. Its time loops are scans, which run as loops when eager.

   - [solve_fixed] runs at most [max_iters] iterations, stopping when the
     batch has converged, with a recursive backtracking line search and an
     optional adaptive damping. Its control flow is data-dependent but stays in
     the graph: every loop is a [Rune.iterate] with a tensor test and every
     branch a [Rune.cond], so there is no host read of a traced tensor, the
     whole solve is a pure tensor computation that Rune.jit traces and compiles
     into loops, and it runs a batch under one step.

   Every function takes the problem as a value, and works at any dtype: the
   dtype phantom is a type parameter, never a module member. *)

open Base
open Nx

type 'a t =
  { xs : 'a array (** states, length [horizon + 1] *)
  ; us : 'a array (** controls, length [horizon] *)
  ; cost : 'a (** scalar tensor: running + final loss *)
  ; iters : 'a
    (** the number of iterations the solve ran, as a scalar of the problem's
        dtype. A batch runs the iterations of its slowest member, so this is
        how much of a [max_iters] budget it spent; the adaptive solve reports
        the updates it applied. *)
  }

(* [stack ts] is the array [ts] as one tensor with its length leading. *)
let stack (ts : ('a, 'b, 'd) Nx.t array) : ('a, 'b, 'd) Nx.t =
  Nx.stack ~axis:0 (Array.to_list ts)

(* [unstack len t] is the rows of [t], as views. *)
let unstack len (t : ('a, 'b, 'd) Nx.t) : ('a, 'b, 'd) Nx.t array =
  Array.init len ~f:(fun k -> slice [ I k ] t)

(* [is_batched x0 us0] is [true] when the state and every control carry a
   leading batch axis. *)
let is_batched x0 us0 =
  Array.length (shape x0) > 1 || Array.exists us0 ~f:(fun u -> Array.length (shape u) > 1)

(* [ks p x] is the step index of each step, a scalar row per step, beside
   [x]. *)
let ks p x = arange_at (beside x) p.Problem.dtype 0 p.Problem.horizon 1
let dyn p theta ~k x u = p.Problem.dynamics theta ~k ~x ~u

(* [states p theta x0 us] integrates the dynamics over the controls [us]
   ([horizon; m]), returning the state path ([horizon + 1; n]) as one scan. *)
let states p theta (x0 : ('w, 'd) Problem.value) (us : ('w, 'd) Problem.value)
  : ('w, 'd) Problem.value
  =
  let _, x =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      (fun x (u, k) ->
         let x' = dyn p theta ~k x u in
         x', x')
      ~init:x0
      (us, ks p us)
  in
  concatenate ~axis:0 [ unsqueeze ~axes:[ 0 ] x0; x ]

(* [cost_of p theta xs us] is the running + final loss along the trajectory
   ([horizon + 1; n]) under the controls ([horizon; m]), as one scan. *)
let cost_of p theta (xs : ('w, 'd) Problem.value) (us : ('w, 'd) Problem.value)
  : ('w, 'd) Problem.value
  =
  let _, losses =
    Rune.scan
      Nx.Ptree.unit
      Nx.Ptree.(pair (pair tensor tensor) tensor)
      Nx.Ptree.tensor
      (fun () ((x, u), k) -> (), p.Problem.running_loss theta ~k ~x ~u)
      ~init:()
      ((slice [ R (0, p.Problem.horizon) ] xs, us), ks p us)
  in
  Infix.(p.Problem.final_loss theta ~x:(slice [ I p.Problem.horizon ] xs) + sum losses)

(* [rollout_unbatched p theta x0 us] integrates the dynamics, returning the
   state path and the total cost (running + final, a scalar tensor). *)
let rollout_unbatched
      p
      theta
      (x0 : ('w, 'd) Problem.value)
      (us : ('w, 'd) Problem.value array)
  : ('w, 'd) Problem.value array * ('w, 'd) Problem.value
  =
  let us = stack us in
  let xs = states p theta x0 us in
  unstack (p.Problem.horizon + 1) xs, cost_of p theta xs us

(* The structure of a rollout, the states as a list, for the [Rune.vmap] that
   batches it. *)
let rollout_structure () = Nx.Ptree.(pair (list tensor) tensor)

(* [rollout p theta x0 us] is {!rollout_unbatched}; batched inputs are rolled
   out member by member. *)
let rollout p theta x0 us0 =
  if not (is_batched x0 us0)
  then rollout_unbatched p theta x0 us0
  else (
    let inputs : ('p, ('w, 'd) Problem.value) Problem.Inputs.t =
      { Problem.Inputs.theta; x0; us0 }
    in
    let xs, cost =
      Rune.vmap
        Nx.Ptree.(Problem.inputs p @-> returns (rollout_structure ()))
        (fun (i : ('p, ('w, 'd) Problem.value) Problem.Inputs.t) ->
           let xs, cost = rollout_unbatched p i.theta i.x0 i.us0 in
           Array.to_list xs, cost)
        inputs
    in
    Array.of_list xs, cost)

(* The rows of one step's linearization: the state, the control and the step
   index. *)
module Step_rows = struct
  type ('a, _) t =
    { x : 'a
    ; u : 'a
    ; k : 'a
    }
  [@@deriving ptree]
end

(* Functions, not values: a structure built with [instantiate] is an
   application, so a top-level binding of one is weakly polymorphic and would
   pin this module to the first dtype it is used at. *)
let step_rows () = Nx.Ptree.instantiate (module Step_rows)
let block_rows () = Nx.Ptree.instantiate (module Lqr.Rows)

(* [step_blocks p theta row] linearizes dynamics and losses at one step, one
   [jacfwd'] or [grad'] pass per block. *)
let step_blocks p theta (row : (('w, 'd) Problem.value, 'd) Step_rows.t)
  : (('w, 'd) Problem.value, 'd) Lqr.Rows.t
  =
  let x = row.x
  and u = row.u
  and k = row.k in
  let a = Rune.jacfwd' (fun x -> dyn p theta ~k x u) x in
  let b = Rune.jacfwd' (fun u -> dyn p theta ~k x u) u in
  let lx = Rune.grad' (fun x -> p.Problem.running_loss theta ~k ~x ~u) x in
  let lu = Rune.grad' (fun u -> p.Problem.running_loss theta ~k ~x ~u) u in
  let lxx =
    Rune.jacfwd'
      (fun x -> Rune.grad' (fun y -> p.Problem.running_loss theta ~k ~x:y ~u) x)
      x
  in
  let luu =
    Rune.jacfwd'
      (fun u -> Rune.grad' (fun v -> p.Problem.running_loss theta ~k ~x ~u:v) u)
      u
  in
  let lux =
    Rune.jacfwd'
      (fun x -> Rune.grad' (fun u -> p.Problem.running_loss theta ~k ~x ~u) u)
      x
  in
  { Lqr.Rows.a
  ; Lqr.Rows.b
  ; Lqr.Rows.lx
  ; Lqr.Rows.lu
  ; Lqr.Rows.lxx
  ; Lqr.Rows.luu
  ; Lqr.Rows.lux
  }

(* [blocks_t p theta xs us] computes the linearization of dynamics and losses
   along the trajectory ([horizon + 1; n]) under the controls ([horizon; m]),
   as one scan. *)
let blocks_t p theta (xs : ('w, 'd) Problem.value) (us : ('w, 'd) Problem.value)
  : (float, 'w, 'd) Lqr.tape
  =
  let rows =
    { Step_rows.x = slice [ R (0, p.Problem.horizon) ] xs; u = us; k = ks p us }
  in
  let _, blocks =
    Rune.scan
      Nx.Ptree.unit
      (step_rows ())
      (block_rows ())
      (fun () row -> (), step_blocks p theta row)
      ~init:()
      rows
  in
  let xf = slice [ I p.Problem.horizon ] xs in
  let fx = Rune.grad' (fun x -> p.Problem.final_loss theta ~x) xf in
  let fxx =
    Rune.jacfwd' (fun x -> Rune.grad' (fun y -> p.Problem.final_loss theta ~x:y) x) xf
  in
  { Lqr.blocks; fx; fxx }

(* [blocks_unbatched p theta xs us] linearizes dynamics and losses along the
   trajectory, from per-step arrays. *)
let blocks_unbatched
      p
      theta
      (xs : ('w, 'd) Problem.value array)
      (us : ('w, 'd) Problem.value array)
  : (float, 'w, 'd) Lqr.tape
  =
  blocks_t p theta (stack xs) (stack us)

(* The structure of a tape, as the blocks and the terminal pair, for the
   [Rune.vmap] that batches a linearization. *)
let tape_structure () = Nx.Ptree.(pair (block_rows ()) (pair tensor tensor))

let blocks p theta (xs : ('w, 'd) Problem.value array) (us : ('w, 'd) Problem.value array)
  : (float, 'w, 'd) Lqr.tape
  =
  let batched =
    Array.exists xs ~f:(fun x -> Array.length (shape x) > 1)
    || Array.exists us ~f:(fun u -> Array.length (shape u) > 1)
  in
  if not batched
  then blocks_unbatched p theta xs us
  else (
    (* Stack after the batch axes so that each member is a [horizon; n] row
       for the vmap. *)
    let batch = Array.length (shape xs.(0)) - 1 in
    let xs = Nx.stack ~axis:batch (Array.to_list xs)
    and us = Nx.stack ~axis:batch (Array.to_list us) in
    let blocks, (fx, fxx) =
      Rune.vmap
        Nx.Ptree.(pair p.params (pair tensor tensor) @-> returns (tape_structure ()))
        (fun (theta, (xs, us)) ->
           let { Lqr.blocks; fx; fxx } = blocks_t p theta xs us in
           blocks, (fx, fxx))
        (theta, (xs, us))
    in
    { Lqr.blocks; fx; fxx })

(* The per-step tangents of the first-order data: what a materialized tangent
   (an [Lqr.tangent]) holds, and what {!tangent} emits. *)
module Tangent_data_rows = struct
  type ('a, _) t =
    { da : 'a
    ; db : 'a
    ; df : 'a
    ; dlx : 'a
    ; dlu : 'a
    }
  [@@deriving ptree]
end

let tangent_data_rows () = Nx.Ptree.instantiate (module Tangent_data_rows)

(* [step_tangent p theta ~dtheta row] is the θ-tangent of the first-order data
   at one step. One forward pass, with the Jacobians and gradients the data are
   defined by taken inside it: the tangents of [A], [B], [f] and of the loss
   gradient, which is what a tangent of the LQR problem consumes. The tangents
   of the loss Hessians are never needed (see [Lqr]). *)
let step_tangent p theta ~dtheta (row : (('w, 'd) Problem.value, 'd) Step_rows.t)
  : (('w, 'd) Problem.value, 'd) Tangent_data_rows.t
  =
  let x = row.x
  and u = row.u
  and k = row.k in
  let derivs th =
    let f = dyn p th ~k x u in
    let a = Rune.jacfwd' (fun x -> dyn p th ~k x u) x in
    let b = Rune.jacfwd' (fun u -> dyn p th ~k x u) u in
    let lx = Rune.grad' (fun x -> p.Problem.running_loss th ~k ~x ~u) x in
    let lu = Rune.grad' (fun u -> p.Problem.running_loss th ~k ~x ~u) u in
    f, (a, (b, (lx, lu)))
  in
  let df, (da, (db, (dlx, dlu))) =
    snd
      (Rune.jvp
         p.Problem.params
         Nx.Ptree.(pair tensor (pair tensor (pair tensor (pair tensor tensor))))
         derivs
         theta
         dtheta)
  in
  { Tangent_data_rows.da; db; df; dlx; dlu }

(* [tangent_t p theta ~dtheta xs us] is the tangent of the first-order data
   along the stacked trajectory, as one scan over the horizon. *)
let tangent_t p theta ~dtheta (xs : ('w, 'd) Problem.value) (us : ('w, 'd) Problem.value)
  : (float, 'w, 'd) Lqr.tangent
  =
  let rows =
    { Step_rows.x = slice [ R (0, p.Problem.horizon) ] xs; u = us; k = ks p us }
  in
  let _, t =
    Rune.scan
      Nx.Ptree.unit
      (step_rows ())
      (tangent_data_rows ())
      (fun () row -> (), step_tangent p theta ~dtheta row)
      ~init:()
      rows
  in
  let xf = slice [ I p.Problem.horizon ] xs in
  let dfx =
    snd
      (Rune.jvp
         p.Problem.params
         Nx.Ptree.tensor
         (fun th -> Rune.grad' (fun x -> p.Problem.final_loss th ~x) xf)
         theta
         dtheta)
  in
  { Lqr.da = t.Tangent_data_rows.da; db = t.db; df = t.df; dlx = t.dlx; dlu = t.dlu; dfx }

(* [tangent p theta ~dtheta xs us] is {!tangent_t} at a per-step trajectory.
   The trajectory is one member's: a batched problem is mapped over the batch
   by the solve that calls this, so the whole call sees one member. *)
let tangent
      p
      theta
      ~dtheta
      (xs : ('w, 'd) Problem.value array)
      (us : ('w, 'd) Problem.value array)
  : (float, 'w, 'd) Lqr.tangent
  =
  tangent_t p theta ~dtheta (stack xs) (stack us)

(* [flip x] reverses the horizon axis. *)
let flip x = Nx.flip ~axes:[ 0 ] x

(* The per-step inputs of the streaming tangent backward pass: the trajectory
   point the step differentiates at, the tape and costate rows it reads, and the
   tangent trajectory. Unlike {!Tangent_data_rows} it carries no tangent: the
   step computes those, and drops them once it has used them. *)
module Tangent_stream_rows = struct
  type ('a, _) t =
    { x : 'a
    ; u : 'a
    ; k : 'a
    ; g : 'a
    ; a : 'a
    ; b : 'a
    ; lxx : 'a
    ; lux : 'a
    ; lx : 'a
    ; fb : 'a
    }
  [@@deriving ptree]
end

let tangent_stream_rows () = Nx.Ptree.instantiate (module Tangent_stream_rows)

(* [step_tangent_data p theta ~dtheta row] is the θ-tangent of the first-order
   data at one step, without the dynamics value: what the tangent backward pass
   consumes, and nothing else. *)
let step_tangent_data
      p
      theta
      ~dtheta
      (row : (('w, 'd) Problem.value, 'd) Tangent_stream_rows.t)
  : ('w, 'd) Problem.value
    * (('w, 'd) Problem.value * (('w, 'd) Problem.value * ('w, 'd) Problem.value))
  =
  let x = row.x
  and u = row.u
  and k = row.k in
  let derivs th =
    let a = Rune.jacfwd' (fun x -> dyn p th ~k x u) x in
    let b = Rune.jacfwd' (fun u -> dyn p th ~k x u) u in
    let lx = Rune.grad' (fun x -> p.Problem.running_loss th ~k ~x ~u) x in
    let lu = Rune.grad' (fun u -> p.Problem.running_loss th ~k ~x ~u) u in
    a, (b, (lx, lu))
  in
  snd
    (Rune.jvp
       p.Problem.params
       Nx.Ptree.(pair tensor (pair tensor (pair tensor tensor)))
       derivs
       theta
       dtheta)

(* [tangent_solve p theta ~dtheta ~dx0 ~xs ~us] is the tangent of the local
   model's LQR solve at the stacked trajectory, along the tangent of the
   problem's parameters and with initial state tangent [dx0].

   The same three sweeps [Lqr.solve_jvp] runs over a materialized [Lqr.tangent],
   but the tangents of the first-order data are computed at each step as the
   sweep reaches it and never stored: the only direction-multiplied tensors are
   the tangent trajectory, the tangent feedforward and the tangent control, so a
   batch of [k] directions costs [O (k · horizon · (n + m))] rather than the
   [O (k · horizon · (n² + n·m))] the [Lqr.tangent] tape would take.

   The tape itself, its Riccati factors and the costate are shared: they do not
   depend on the direction, so under [Rune.vmap] they are constants of the map
   and are computed once for the whole batch. *)
let tangent_solve
      p
      theta
      ~dtheta
      ~(dx0 : ('w, 'd) Problem.value)
      (xs : ('w, 'd) Problem.value array)
      (us : ('w, 'd) Problem.value array)
  : (float, 'w, 'd) Lqr.solution
  =
  let horizon = p.Problem.horizon in
  let xs = stack xs
  and us = stack us in
  let tape = blocks_t p theta xs us in
  let common = Lqr.backward_common tape in
  let x0s = slice [ R (0, horizon) ] xs in
  (* 1. The tangent trajectory, driven by the tangent of the dynamics. *)
  let _, g_next =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair (step_rows ()) tensor)
      Nx.Ptree.tensor
      (fun g (row, a) ->
         let df =
           snd
             (Rune.jvp
                p.Problem.params
                Nx.Ptree.tensor
                (fun th -> dyn p th ~k:row.Step_rows.k row.x row.u)
                theta
                dtheta)
         in
         let g' = Infix.(Lqr.matvec a g + df) in
         g', g')
      ~init:dx0
      ({ Step_rows.x = x0s; u = us; k = ks p us }, tape.Lqr.blocks.a)
  in
  let g = concatenate ~axis:0 [ unsqueeze ~axes:[ 0 ] dx0; g_next ] in
  (* 2. The tangent backward pass, against the primal's factors. *)
  let rows =
    { Tangent_stream_rows.x = flip x0s
    ; u = flip us
    ; k = flip (ks p us)
    ; g = flip (slice [ R (0, horizon) ] g)
    ; a = flip tape.Lqr.blocks.a
    ; b = flip tape.Lqr.blocks.b
    ; lxx = flip tape.Lqr.blocks.lxx
    ; lux = flip tape.Lqr.blocks.lux
    ; lx = flip (Lqr.costate tape)
    ; fb = flip common.Lqr.feedback
    }
  in
  let dfx =
    snd
      (Rune.jvp
         p.Problem.params
         Nx.Ptree.tensor
         (fun th ->
            Rune.grad' (fun x -> p.Problem.final_loss th ~x) (slice [ I horizon ] xs))
         theta
         dtheta)
  in
  let e_last = Infix.(dfx + Lqr.matvec tape.Lqr.fxx (slice [ I horizon ] g)) in
  let e0, q_hat =
    Rune.scan
      Nx.Ptree.tensor
      (tangent_stream_rows ())
      Nx.Ptree.tensor
      (fun e row ->
         let da, (db, (dlx, dlu)) = step_tangent_data p theta ~dtheta row in
         let q, e' =
           Lqr.tangent_step
             ~e
             { Lqr.Tangent_rows.a = row.Tangent_stream_rows.a
             ; b = row.b
             ; da
             ; db
             ; lxx = row.lxx
             ; lux = row.lux
             ; lx = row.lx
             ; dlx
             ; dlu
             ; g = row.g
             ; fb = row.fb
             }
         in
         e', q)
      ~init:e_last
      rows
  in
  (* 3. The tangent gain rollout. *)
  let _, (du, v_next) =
    Rune.scan
      Nx.Ptree.tensor
      (Nx.Ptree.instantiate (module Lqr.Tangent_roll_rows))
      Nx.Ptree.(pair tensor tensor)
      (fun v row ->
         let du, v' = Lqr.tangent_roll_step ~v row in
         v', (du, v'))
      ~init:(zeros_like dx0)
      { Lqr.Tangent_roll_rows.a = tape.Lqr.blocks.a
      ; b = tape.Lqr.blocks.b
      ; chol = common.Lqr.quu_chol
      ; fb = common.Lqr.feedback
      ; q = flip q_hat
      }
  in
  let v = concatenate ~axis:0 [ unsqueeze ~axes:[ 0 ] (zeros_like dx0); v_next ] in
  let xs = Infix.(g + v) in
  if not p.Problem.free_initial
  then { Lqr.xs; us = du }
  else (
    (* The initial state is a variable, so the tangent also satisfies its
       stationarity: the pinned sweep leaves [e0] there, and the correction is
       the model's step in [x₀] with the closed loop's response. *)
    let dx1, du1 = Lqr.initial_response ~common ~tape ~residual:e0 in
    { Lqr.xs = Infix.(xs + dx1); us = Infix.(du + du1) })

(* ---------------------------------------------------------------------------
   The fixed-iteration solve

   One iteration is a Riccati pass at the current damping strength, then a
   backtracking line search that tries the trial steps in order and stops at
   the first the batch accepts. It costs the levels its slowest member needs:
   each level is a [Rune.cond] branch, and a branch not taken computes nothing,
   eagerly or compiled.

   The batch moves at one step, so it is limited by the member that has to
   backtrack furthest. A member whose modelled decrease has become negligible
   has converged: it is frozen and does not vote, so it cannot veto the steps
   of the others. When no step is accepted the damping is raised and the pass
   retried; a full step accepted lowers it again. The iteration loop stops once
   every member has converged, or once an iteration could not buy a step at any
   damping: the batch spends the iterations of its slowest member out of
   [max_iters], and {!t.iters} says how many.

   All of it is traced: every loop is a [Rune.iterate] with a tensor test,
   every branch a [Rune.cond], and everything carried between iterations is a
   tensor.
   --------------------------------------------------------------------------- *)

(* A trial's rollout. *)
module Fixed_step = struct
  type ('a, _) t =
    { xs : 'a
    ; us : 'a
    ; cost : 'a
    }
  [@@deriving ptree]
end

let fixed_step () = Nx.Ptree.instantiate (module Fixed_step)

type ('w, 'd) fixed_step = (('w, 'd) Problem.value, 'd) Fixed_step.t

(* What a line search returns: the step it accepted — zero when it accepted
   none — and the trajectory and cost that go with it, the nominal ones when it
   accepted none. *)
module Fixed_search = struct
  type ('a, _) t =
    { alpha : 'a
    ; xs : 'a
    ; us : 'a
    ; cost : 'a
    }
  [@@deriving ptree]
end

let fixed_search () = Nx.Ptree.instantiate (module Fixed_search)

type ('w, 'd) fixed_search = (('w, 'd) Problem.value, 'd) Fixed_search.t

(* What one Riccati pass gives the control flow, beside the trajectory it was
   run at. [d1], [d2] and [scale] hold one scalar per member. *)
module Fixed_gains = struct
  type ('a, _) t =
    { ff : 'a (** the feedforward gain [k_k], [horizon; m] *)
    ; fb : 'a (** the feedback gain [K_k], [horizon; m; n] *)
    ; d : 'a (** the step in the initial state, [n]; zero when it is pinned *)
    ; d1 : 'a (** the modelled first-order decrease, per member *)
    ; d2 : 'a (** the curvature the gains used, per member *)
    ; scale : 'a (** the value-Hessian scale of the pass, per member *)
    }
  [@@deriving ptree]
end

let fixed_gains () = Nx.Ptree.instantiate (module Fixed_gains)

type ('w, 'd) fixed_gains = (('w, 'd) Problem.value, 'd) Fixed_gains.t

(* One attempt at an iteration: a damping strength and the line search it
   bought. The retry loop carries one of these and raises the strength until
   the search accepts or the retries run out. [scale] is zero until a pass has
   measured one, and [attempts] counts the passes of this iteration. *)
module Fixed_attempt = struct
  type ('a, _) t =
    { beta : 'a
    ; delta : 'a
    ; scale : 'a
    ; alpha : 'a
    ; xs : 'a
    ; us : 'a
    ; cost : 'a
    ; d1 : 'a
    ; attempts : 'a
    }
  [@@deriving ptree]
end

let fixed_attempt () = Nx.Ptree.instantiate (module Fixed_attempt)

type ('w, 'd) fixed_attempt = (('w, 'd) Problem.value, 'd) Fixed_attempt.t

(* The solve's state between iterations. *)
module Fixed_iter = struct
  type ('a, _) t =
    { xs : 'a
    ; us : 'a
    ; cost : 'a
    ; d1 : 'a
    ; alpha : 'a
    ; beta : 'a
    ; delta : 'a
    ; scale : 'a
    ; count : 'a
    }
  [@@deriving ptree]
end

let fixed_iter () = Nx.Ptree.instantiate (module Fixed_iter)

type ('w, 'd) fixed_iter = (('w, 'd) Problem.value, 'd) Fixed_iter.t

(* What a batch changes about the control flow, and nothing else: a reduction
   over members, and putting a per-member quantity beside the axes it meets. A
   single member is the degenerate case of both, and the code below is the same
   for either. *)
type 'd layout =
  { members : 'd Nx.bool_t -> 'd Nx.bool_t
    (** one flag per member, or none, reduced to one for the whole batch *)
  ; tape_like : 'v 's. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
    (** a per-member scalar, shaped for the tape's rows ([B; m; m]) *)
  ; state_like : 'd Nx.bool_t -> 'd Nx.bool_t
    (** a per-member mask, shaped for a stacked trajectory ([B; K; n]) *)
  }

let single_layout (type d) : d layout =
  let tape_like : 'v 's. ('v, 's, d) Nx.t -> ('v, 's, d) Nx.t = fun x -> x in
  let state_like : d Nx.bool_t -> d Nx.bool_t = fun flags -> flags in
  { members = (fun flags -> flags); tape_like; state_like }

let batch_layout (type d) (b : int) : d layout =
  let tape_like : 'v 's. ('v, 's, d) Nx.t -> ('v, 's, d) Nx.t =
    fun x -> reshape [| b; 1; 1 |] x
  in
  let state_like : d Nx.bool_t -> d Nx.bool_t =
    fun flags -> reshape [| b; 1; 1 |] flags
  in
  { members = (fun flags -> Nx.all ~axes:[ 0 ] flags); tape_like; state_like }

(* The per-member tensor work: one member's rollout, Riccati pass and trial.
   The dynamics and the losses are the problem's own, written for one member. *)

let fixed_rollout_lane p theta x0 us =
  let xs = states p theta x0 us in
  xs, cost_of p theta xs us

(* [fixed_scale common] is the scale of the value Hessian a pass left: the
   largest magnitude on its diagonal, per member. Damping is a fraction of it,
   so members whose curvature differs by orders of magnitude are damped alike,
   and the scale costs one reduction of an [n; n] block the pass computed
   anyway. *)
let fixed_scale (common : (float, 'w, 'd) Lqr.common) =
  Nx.max (Nx.abs (Nx.diagonal common.Lqr.value_hessian))

(* [initial_step p ~common ~value_grad] is the step the local model takes in the
   initial state — zero when the problem pins it — and the decrease it buys,
   which the line search's model must count: the value function's own, in
   [d1]'s units. The model is quadratic in [x₀], so the step is Newton's, and
   it is the one {!Lqr.initial_deviation} reports. *)
let initial_step p ~(common : (float, 'w, 'd) Lqr.common) ~value_grad =
  if p.Problem.free_initial
  then (
    let d = Lqr.initial_deviation ~common ~value_grad in
    d, Infix.(sum ~axes:[ -1 ] (value_grad * d)))
  else zeros_like value_grad, zeros_like (sum ~axes:[ -1 ] value_grad)

(* [stacked_x0 xs] is the state a stacked trajectory starts from: step 0, the
   horizon being the second-to-last axis, so a batch's leading axes are kept. *)
let stacked_x0 (xs : ('a, 'b, 'd) Nx.t) =
  let axis = Array.length (shape xs) - 2 in
  let spec = Array.create ~len:(axis + 1) Nx.A in
  spec.(axis) <- Nx.I 0;
  slice (Array.to_list spec) xs

let fixed_backward_lane p theta xs us mu =
  let tape = blocks_t p theta xs us in
  let common = Lqr.backward_common ~mu tape in
  let gains, d1, d2, value_grad =
    Lqr.backward_lin ~common ~tape ~lin:(Lqr.lin_of_tape tape)
  in
  let d, decrease = initial_step p ~common ~value_grad in
  { Fixed_gains.ff = gains.Lqr.feedforward
  ; fb = gains.feedback
  ; d
  ; d1 = (if p.Problem.free_initial then Infix.(d1 + decrease) else d1)
  ; d2
  ; scale = fixed_scale common
  }

let fixed_trial_lane p theta d xs us ff fb alpha =
  (* The trial starts from the nominal's own initial state, moved by the model's
     step in [x₀] at the step size the search is trying. *)
  let x0 = Infix.(stacked_x0 xs + (d * alpha)) in
  let xs_t, us_t =
    Lqr.rollout_tensor
      ~dyn:(dyn p theta)
      ~x0
      ~xs
      ~us
      ~gains:{ Lqr.feedback = fb; feedforward = ff }
      ~alpha
  in
  { Fixed_step.xs = xs_t; us = us_t; cost = cost_of p theta xs_t us_t }

(* The kernels, as one record: one call per pass or per trial, over the stacked
   trajectory, never one per step. *)
type ('w, 'p, 'd) fixed_kernels =
  { rollout :
      'p
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value * ('w, 'd) Problem.value
  ; backward :
      'p
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> ('w, 'd) fixed_gains
  ; trial :
      'p
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> ('w, 'd) fixed_step
  }

let single_fixed_kernels (p : ('w, 'p, 'd) Problem.t) : ('w, 'p, 'd) fixed_kernels =
  { rollout = (fun theta x0 us -> fixed_rollout_lane p theta x0 us)
  ; backward = (fun theta xs us mu -> fixed_backward_lane p theta xs us mu)
  ; trial =
      (fun theta d xs us ff fb alpha -> fixed_trial_lane p theta d xs us ff fb alpha)
  }

let batched_fixed_kernels (p : ('w, 'p, 'd) Problem.t) : ('w, 'p, 'd) fixed_kernels =
  let rollout =
    Rune.vmap
      Nx.Ptree.(
        pair p.Problem.params (pair tensor tensor) @-> returns (pair tensor tensor))
      (fun (theta, (x0, us)) -> fixed_rollout_lane p theta x0 us)
  in
  let backward =
    Rune.vmap
      Nx.Ptree.(
        pair p.Problem.params (pair tensor (pair tensor tensor))
        @-> returns (fixed_gains ()))
      (fun (theta, (xs, (us, mu))) -> fixed_backward_lane p theta xs us mu)
  in
  let trial =
    Rune.vmap
      Nx.Ptree.(
        pair
          p.Problem.params
          (pair tensor (pair tensor (pair tensor (pair tensor (pair tensor tensor)))))
        @-> returns (fixed_step ()))
      (fun (theta, (d, (xs, (us, (ff, (fb, alpha)))))) ->
         fixed_trial_lane p theta d xs us ff fb alpha)
  in
  { rollout = (fun theta x0 us -> rollout (theta, (x0, us)))
  ; backward = (fun theta xs us mu -> backward (theta, (xs, (us, mu))))
  ; trial =
      (fun theta d xs us ff fb alpha -> trial (theta, (d, (xs, (us, (ff, (fb, alpha)))))))
  }

(* [unstack_horizon len t] is [t]'s horizon sliced into [len] per-step
   tensors, batch axes kept: [t]'s last two axes are the horizon and the step
   dimension. *)
let unstack_horizon len (t : ('a, 'b, 'd) Nx.t) =
  let axis = Array.length (shape t) - 2 in
  Array.init len ~f:(fun k ->
    let spec = Array.create ~len:(axis + 1) Nx.A in
    spec.(axis) <- Nx.I k;
    slice (Array.to_list spec) t)

(* The solve. [us] is the stacked initial controls, batch axes leading
   ([B; K; m]) for a batch and [K; m] for one member. *)
let solve_fixed_with
      (type w p d)
      (p : (w, p, d) Problem.t)
      (layout : d layout)
      (kernels : (w, p, d) fixed_kernels)
      ?(alphas = Linesearch.schedule ~n:8 ())
      ?(damping = Damping.default)
      ?(c1 = 1e-2)
      ?(conv_threshold = 1e-6)
      ~(max_iters : int)
      (theta : p)
      (x0 : (w, d) Problem.value)
      (us : (w, d) Problem.value)
  : (w, d) Problem.value t
  =
  let scalar v = scalar_like x0 v in
  let zero = scalar 0.0 in
  let xs, cost0 = kernels.rollout theta x0 us in
  (* A per-member zero: a stacked quantity's batch axis is the shape of the
     members' costs, and is empty for one member. *)
  let member_zero = zeros_like cost0 in
  let member v = broadcast_to (shape cost0) (scalar v) in
  (* What the damping choice means for the loop: a strength that is the same
     for every pass, or a schedule that adapts it. *)
  let fixed_mu, reg =
    match damping with
    | Damping.Off -> None, None
    | Damping.Fixed mu -> (if Float.equal mu 0.0 then None else Some (scalar mu)), None
    | Damping.Adaptive schedule -> None, Some schedule
  in
  (* An adaptive schedule has to be able to raise the damping, so it is the
     only mode that retries a failed line search; a fixed strength that bought
     no step buys none at a retry either. *)
  let max_retries =
    match reg with
    | None -> 1
    | Some _ -> 12
  in
  let beta0, delta0 =
    match reg with
    | None -> member_zero, member 1.0
    | Some r -> member r.Regularisation.mu, member r.Regularisation.delta
  in
  (* The strength a pass is run at: a fixed one is one scalar for the batch; an
     adaptive one is a fraction of the member's own value-Hessian scale, and is
     the raw fraction until a pass has measured one (the first pass uses the
     schedule's own units, as the adaptive solve does). *)
  let strength beta scale =
    match fixed_mu with
    | Some mu -> mu
    | None ->
      (match reg with
       | None -> zero
       | Some _ ->
         let known = Nx.greater_s scale 0.0 in
         layout.tape_like (Nx.where known Infix.(beta * scale) beta))
  in
  (* A member has converged when its model's first-order decrease is negligible
     against its own cost, the adaptive solve's test: it is second order in the
     control gradient, so it fires well before the gradient itself vanishes. *)
  let converged d1 cost =
    let scale = Nx.maximum_s (Nx.abs cost) 1.0 in
    Nx.less_equal (Nx.abs d1) Infix.(scale *$ conv_threshold)
  in
  (* The Levenberg-Marquardt schedule, in tensor arithmetic so that it rides
     through the traced loops: [increase] at least doubles the growth factor,
     [decrease] at most halves it, and the strength floors at [floor]. *)
  let increase beta delta =
    let delta' = Nx.maximum_s Infix.(delta *$ 2.0) 2.0 in
    delta', Nx.maximum_s Infix.(beta * delta') Regularisation.floor
  in
  let decrease beta delta =
    let delta' = Nx.minimum_s Infix.(delta /$ 2.0) 0.5 in
    delta', Nx.maximum_s Infix.(beta * delta') Regularisation.floor
  in
  (* Recursive backtracking. A level rolls a trial out and tests every member
     that is still active at it; a member that has converged takes no step and
     does not vote, so it cannot veto the others. The batch accepts a level
     when every active member satisfies Armijo there, and only then does the
     recursion stop: the batch moves at the step its slowest member can take. *)
  let rec search
            ~active
            ~(nominal : (w, d) fixed_step)
            ~(gains : (w, d) fixed_gains)
            ~cost
            ~d1
            ~d2
    = function
    | [] ->
      { Fixed_search.alpha = zero
      ; xs = nominal.Fixed_step.xs
      ; us = nominal.us
      ; cost = nominal.cost
      }
    | alpha :: tl ->
      let alpha_t = broadcast_to (shape cost) (scalar alpha) in
      let trial =
        kernels.trial
          theta
          gains.Fixed_gains.d
          nominal.xs
          nominal.us
          gains.ff
          gains.fb
          alpha_t
      in
      let model = Infix.((alpha_t * d1) + (alpha_t * alpha_t * d2 *$ 0.5)) in
      let armijo = Nx.less_equal trial.Fixed_step.cost Infix.(cost + (model *$ c1)) in
      let ok = layout.members (Nx.logical_or (Nx.logical_not active) armijo) in
      Rune.cond
        (fixed_search ())
        ok
        (fun () ->
           { Fixed_search.alpha = scalar alpha
           ; xs = trial.xs
           ; us = trial.us
           ; cost = trial.cost
           })
        (fun () -> search ~active ~nominal ~gains ~cost ~d1 ~d2 tl)
  in
  (* One pass and the line search it buys. The scale is the pass's own, kept
     from the last finite one; the first finite scale converts the strength to
     relative units without moving the damping — [beta] was the strength, and
     [beta / scale] times the scale is that same strength. *)
  let attempt ~(nominal : (w, d) fixed_step) ~beta ~delta ~scale ~attempts =
    let gains = kernels.backward theta nominal.xs nominal.us (strength beta scale) in
    let active = Nx.logical_not (converged gains.Fixed_gains.d1 nominal.cost) in
    let found =
      search ~active ~nominal ~gains ~cost:nominal.cost ~d1:gains.d1 ~d2:gains.d2 alphas
    in
    let valid = Nx.logical_and (Nx.isfinite gains.scale) (Nx.greater_s gains.scale 0.0) in
    let fresh = Nx.logical_and valid (Nx.less_equal_s scale 0.0) in
    let scale' = Nx.where valid gains.scale scale in
    let beta = Nx.where fresh Infix.(beta / gains.scale) beta in
    (* Raise the damping when no step was bought, lower it after a full step,
       keep it after a backtracked one. *)
    let delta', beta' =
      match reg with
      | None -> delta, beta
      | Some _ ->
        let accepted = Nx.greater_s found.Fixed_search.alpha 0.0 in
        let full = Nx.greater_equal_s found.alpha 1.0 in
        let delta_up, beta_up = increase beta delta in
        let delta_dn, beta_dn = decrease beta delta in
        ( Nx.where accepted (Nx.where full delta_dn delta) delta_up
        , Nx.where accepted (Nx.where full beta_dn beta) beta_up )
    in
    (* A member that has converged keeps its trajectory: it takes no step,
       whatever the batch's. *)
    let state_like = layout.state_like active in
    { Fixed_attempt.beta = beta'
    ; delta = delta'
    ; scale = scale'
    ; alpha = found.alpha
    ; xs = Nx.where state_like found.xs nominal.xs
    ; us = Nx.where state_like found.us nominal.us
    ; cost = Nx.where active found.cost nominal.cost
    ; d1 = gains.d1
    ; attempts = Infix.(attempts +$ 1.0)
    }
  in
  (* The retry loop of one iteration: damping is raised until the line search
     accepts, or the retries run out. *)
  let retry ~(nominal : (w, d) fixed_step) ~beta ~delta ~scale =
    let until (a : (w, d) fixed_attempt) =
      Nx.logical_or
        (Nx.greater_s a.Fixed_attempt.alpha 0.0)
        (Nx.greater_equal_s a.attempts (Float.of_int max_retries))
    in
    Rune.iterate
      (fixed_attempt ())
      ~max:max_retries
      ~until
      (fun a ->
         attempt ~nominal ~beta:a.beta ~delta:a.delta ~scale:a.scale ~attempts:a.attempts)
      { Fixed_attempt.beta
      ; delta
      ; scale
      ; alpha = zero
      ; xs = nominal.Fixed_step.xs
      ; us = nominal.us
      ; cost = nominal.cost
      ; d1 = member_zero
      ; attempts = zero
      }
  in
  let step (s : (w, d) fixed_iter) =
    let nominal = { Fixed_step.xs = s.xs; us = s.us; cost = s.cost } in
    let a = retry ~nominal ~beta:s.beta ~delta:s.delta ~scale:s.scale in
    { Fixed_iter.xs = a.xs
    ; us = a.us
    ; cost = a.cost
    ; d1 = a.d1
    ; alpha = a.alpha
    ; beta = a.beta
    ; delta = a.delta
    ; scale = a.scale
    ; count = Infix.(s.count +$ 1.0)
    }
  in
  let until (s : (w, d) fixed_iter) =
    let finished =
      Nx.logical_or (layout.members (converged s.d1 s.cost)) (Nx.less_equal_s s.alpha 0.0)
    in
    Nx.logical_or
      (Nx.logical_and (Nx.greater_s s.count 0.0) finished)
      (Nx.greater_equal_s s.count (Float.of_int max_iters))
  in
  let state =
    Rune.iterate
      (fixed_iter ())
      ~max:max_iters
      ~until
      step
      { Fixed_iter.xs
      ; us
      ; cost = cost0
      ; d1 = member_zero
      ; alpha = zero
      ; beta = beta0
      ; delta = delta0
      ; scale = member_zero
      ; count = zero
      }
  in
  { xs = unstack_horizon (p.Problem.horizon + 1) state.xs
  ; us = unstack_horizon p.Problem.horizon state.us
  ; cost = state.cost
  ; iters = state.count
  }

(* The fixed-iteration solve. Batched inputs — a leading batch axis on the
   parameters, the initial state and every initial control — are solved as one
   batch: the members share the step and the damping schedule, the batch stops
   when the last of them has converged or could not be moved, and the dynamics
   and the losses stay written for one member. *)
let solve_fixed p ?alphas ?damping ?c1 ?conv_threshold ~max_iters theta x0 us0 =
  if not (is_batched x0 us0)
  then (
    let us = stack us0 in
    solve_fixed_with
      p
      single_layout
      (single_fixed_kernels p)
      ?alphas
      ?damping
      ?c1
      ?conv_threshold
      ~max_iters
      theta
      x0
      us)
  else (
    let b = (shape x0).(0) in
    let us = Nx.stack ~axis:1 (Array.to_list us0) in
    solve_fixed_with
      p
      (batch_layout b)
      (batched_fixed_kernels p)
      ?alphas
      ?damping
      ?c1
      ?conv_threshold
      ~max_iters
      theta
      x0
      us)

(* Scalar-read helper for the adaptive solver. *)
let item_f t = item [] t

let log p (fmt : ('a, Stdlib.Format.formatter, unit) format) : 'a =
  if p.Problem.verbose
  then Stdlib.Format.printf (Stdlib.( ^^ ) (Stdlib.( ^^ ) "dilqr: " fmt) "\n%!")
  else Stdlib.Format.ifprintf Stdlib.Format.err_formatter fmt

(* The traceable kernels of the adaptive solve, as one record. The eager and
   the compiled implementations differ only in these three functions: the
   host-side control flow above them — regularization retries, Armijo
   backtracking, convergence — is unchanged. Every tensor is the stacked form
   the solver keeps between iterations, so a kernel is one call per iteration
   (or per line-search trial), not one per step. *)
type ('w, 'p, 'd) kernels =
  { rollout :
      'p
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value * ('w, 'd) Problem.value
    (** [rollout theta x0 us] is the state path and the total cost. *)
  ; backward :
      'p
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> (float, 'w, 'd) Lqr.gains
         * ('w, 'd) Problem.value
         * ('w, 'd) Problem.value
         * ('w, 'd) Problem.value
    (** [backward theta xs us mu] linearizes at the trajectory and runs the
        damped Riccati pass: the gains, the expected-decrease coefficients, and
        the model's step in the initial state ([n]; zero when it is pinned). *)
  ; trial :
      'p
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value
      -> (float, 'w, 'd) Lqr.gains
      -> ('w, 'd) Problem.value
      -> ('w, 'd) Problem.value * ('w, 'd) Problem.value * ('w, 'd) Problem.value
    (** [trial theta d xs us gains alpha] rolls out one line-search step from
        the trajectory's own initial state, moved by [d] at [alpha], and
        evaluates its cost. *)
  }

(* The type of the compiled adaptive solve: {!solve} with the problem
   already given. *)
type ('w, 'p, 'd) compiled =
  ?max_iter:int
  -> ?conv_threshold:float
  -> ?c1:float
  -> ?ls_depth:int
  -> ?damping:Damping.t
  -> 'p
  -> ('w, 'd) Problem.value
  -> ('w, 'd) Problem.value array
  -> ('w, 'd) Problem.value t

(* [eager_kernels p] runs the kernels as eager tensor code. *)
let eager_kernels (p : ('w, 'p, 'd) Problem.t) : ('w, 'p, 'd) kernels =
  { rollout =
      (fun theta x0 us ->
        let xs = states p theta x0 us in
        xs, cost_of p theta xs us)
  ; backward =
      (fun theta xs us mu ->
        let tape = blocks_t p theta xs us in
        let common = Lqr.backward_common ~mu tape in
        let gains, d1, d2, value_grad =
          Lqr.backward_lin ~common ~tape ~lin:(Lqr.lin_of_tape tape)
        in
        let d, decrease = initial_step p ~common ~value_grad in
        gains, (if p.Problem.free_initial then Infix.(d1 + decrease) else d1), d2, d)
  ; trial =
      (fun theta d xs us gains alpha ->
        let x0 = Infix.(stacked_x0 xs + (d * alpha)) in
        let xs_t, us_t =
          Lqr.rollout_tensor ~dyn:(dyn p theta) ~x0 ~xs ~us ~gains ~alpha
        in
        xs_t, us_t, cost_of p theta xs_t us_t)
  }

(* [compiled_kernels ?beam ?parallel p] is {!eager_kernels} with each kernel
   compiled by [Rune.jit]. The compiled functions are values: build them once
   and every later call replays their programs for the same shapes and
   placements. [mu] and [alpha] are inputs, not captured floats, so a new
   regularization or a new trial step does not retrace. *)
let compiled_kernels ?beam ?parallel (p : ('w, 'p, 'd) Problem.t) : ('w, 'p, 'd) kernels =
  let rollout =
    Rune.jit
      ?beam
      ?parallel
      Nx.Ptree.(
        pair p.Problem.params (pair tensor tensor) @-> returns (pair tensor tensor))
      (fun (theta, (x0, us)) ->
         let xs = states p theta x0 us in
         xs, cost_of p theta xs us)
  in
  let backward =
    Rune.jit
      ?beam
      ?parallel
      Nx.Ptree.(
        pair p.Problem.params (pair tensor (pair tensor tensor))
        @-> returns (pair (pair (pair tensor tensor) tensor) (pair tensor tensor)))
      (fun (theta, (xs, (us, mu))) ->
         let tape = blocks_t p theta xs us in
         let common = Lqr.backward_common ~mu tape in
         let gains, d1, d2, value_grad =
           Lqr.backward_lin ~common ~tape ~lin:(Lqr.lin_of_tape tape)
         in
         let d, decrease = initial_step p ~common ~value_grad in
         let d1 = if p.Problem.free_initial then Infix.(d1 + decrease) else d1 in
         ((gains.Lqr.feedback, gains.feedforward), d), (d1, d2))
  in
  let trial =
    Rune.jit
      ?beam
      ?parallel
      Nx.Ptree.(
        pair
          p.Problem.params
          (pair tensor (pair tensor (pair tensor (pair (pair tensor tensor) tensor))))
        @-> returns (pair (pair tensor tensor) tensor))
      (fun (theta, (d, (xs, (us, ((feedback, feedforward), alpha))))) ->
         let gains = { Lqr.feedback; feedforward } in
         let x0 = Infix.(stacked_x0 xs + (d * alpha)) in
         let xs_t, us_t =
           Lqr.rollout_tensor ~dyn:(dyn p theta) ~x0 ~xs ~us ~gains ~alpha
         in
         (xs_t, us_t), cost_of p theta xs_t us_t)
  in
  { rollout =
      (fun theta x0 us ->
        let xs, cost = rollout (theta, (x0, us)) in
        xs, cost)
  ; backward =
      (fun theta xs us mu ->
        let ((feedback, feedforward), d), (d1, d2) = backward (theta, (xs, (us, mu))) in
        { Lqr.feedback; feedforward }, d1, d2, d)
  ; trial =
      (fun theta d xs us gains alpha ->
        let (xs_t, us_t), cost =
          trial (theta, (d, (xs, (us, ((gains.Lqr.feedback, gains.feedforward), alpha)))))
        in
        xs_t, us_t, cost)
  }

(* The adaptive, host-driven solve. Requires untraced tensors: it reads
   scalars on the host and retries the Cholesky factorization on
   non-positive-definite [Quu]. Unbatched: a batch would need one line search
   per member, which is host control flow. Use {!solve_fixed} for batches.
   [kernels] is where the tensor work happens: {!eager_kernels} for the plain
   solve, {!compiled_kernels} for the compiled one. *)
let solve_with
      (type w p d)
      (p : (w, p, d) Problem.t)
      (kernels : (w, p, d) kernels)
      ?(max_iter = 20)
      ?(conv_threshold = 1e-6)
      ?(c1 = 1e-2)
      ?(ls_depth = 8)
      ?(damping = Damping.default)
      (theta : p)
      (x0 : (w, d) Problem.value)
      (us0 : (w, d) Problem.value array)
  : (w, d) Problem.value t
  =
  if is_batched x0 us0
  then
    failwith
      "dilqr: the adaptive solve is unbatched; use Solver.solve_fixed or solve_fixed for \
       batched problems";
  let us = ref (stack us0) in
  let xs, cost0 = kernels.rollout theta x0 !us in
  let xs = ref xs in
  let cost_t = ref cost0 in
  let cost = ref (item_f cost0) in
  (* The damping choice: a strength to use at every pass, and the schedule to
     adapt it when there is one. *)
  let fixed_mu =
    match damping with
    | Damping.Off -> Some 0.0
    | Damping.Fixed mu -> Some mu
    | Damping.Adaptive _ -> None
  in
  let reg =
    ref
      (match damping with
       | Damping.Adaptive schedule -> Some schedule
       | _ -> None)
  in
  let mu_value () =
    match !reg with
    | Some schedule -> schedule.Regularisation.mu
    | None -> Option.value fixed_mu ~default:0.0
  in
  let mu () = scalar_like x0 (mu_value ()) in
  let finite d1 d2 = item [] (all (isfinite d1)) && item [] (all (isfinite d2)) in
  (* Raise the strength, and say whether there was a schedule to raise: a
     pinned strength has nothing to retry. *)
  let raise_damping () =
    match !reg with
    | None -> false
    | Some schedule ->
      reg := Some (Regularisation.increase schedule);
      true
  in
  let lower_damping () =
    match !reg with
    | None -> ()
    | Some schedule -> reg := Some (Regularisation.decrease schedule)
  in
  let rec backward attempts =
    if attempts > 30
    then
      failwith
        "dilqr: Solver.solve: Quu could not be regularized to positive definiteness";
    let gains, d1, d2, d = kernels.backward theta !xs !us (mu ()) in
    (* A non-positive-definite [Quu] gives a NaN Cholesky factor, and the NaN
       runs through the sweep into the gains and the expected-decrease
       accumulators. *)
    if finite d1 d2
    then gains, d1, d2, d
    else if raise_damping ()
    then (
      log p "cholesky failed, mu -> %g" (mu_value ());
      backward (attempts + 1))
    else
      failwith
        "dilqr: Solver.solve: Quu is not positive definite and the damping is pinned; \
         pass ~damping:(Damping.Adaptive ...) or a larger Damping.Fixed"
  in
  let converged c c' =
    Float.(Float.abs ((c -. c') /. Float.max (Float.abs c) 1.0) < conv_threshold)
  in
  (* How many updates the solve applied, for the result's [iters]. *)
  let updates = ref 0 in
  let rec loop (iter : int) (c : float) (attempts : int) =
    if iter >= max_iter
    then ()
    else (
      let gains, d1t, d2t, d = backward 0 in
      let d1 = item_f d1t
      and d2 = item_f d2t in
      (* Converged: the modelled first-order decrease is negligible against
         the cost scale (it is second order in the control gradient, so this
         fires well before the gradient itself vanishes). *)
      if Float.(Float.abs d1 <= conv_threshold *. Float.max 1.0 (Float.abs c))
      then ()
      else (
        let accepted =
          let rec try_alphas = function
            | [] -> None
            | alpha :: tl ->
              let alpha_t = scalar_like x0 alpha in
              let xs_t, us_t, cost_t' = kernels.trial theta d !xs !us gains alpha_t in
              let fv : float = item_f cost_t' in
              let model = (alpha *. d1) +. (0.5 *. alpha *. alpha *. d2) in
              if Float.(fv <= c +. (c1 *. model))
              then Some (xs_t, us_t, cost_t', fv, alpha)
              else try_alphas tl
          in
          try_alphas (Linesearch.schedule ~n:ls_depth ())
        in
        match accepted with
        | None ->
          (* No trial decreased enough: damp harder and retry the same
             iteration. If even the largest damping cannot buy a decreasing
             step — or there is no schedule to raise — stop: the solve is stuck
             at a point its quadratic model cannot improve. *)
          log p "line search failed at iter %d, mu -> %g" iter (mu_value ());
          if attempts >= 12 || not (raise_damping ())
          then ()
          else loop iter c (attempts + 1)
        | Some (xs_t, us_t, cost_t', fv, alpha) ->
          log p "iter %d: %g -> %g (alpha %g, mu %g)" iter c fv alpha (mu_value ());
          (* Decay the damping only when the full step was accepted;
             backtracked steps keep it. *)
          if Float.equal alpha (List.hd_exn (Linesearch.schedule ~n:ls_depth ()))
          then lower_damping ();
          xs := xs_t;
          us := us_t;
          cost_t := cost_t';
          Int.incr updates;
          if converged c fv
          then (
            log p "converged after %d iterations: %g -> %g" iter c fv;
            ())
          else loop (iter + 1) fv 0))
  in
  loop 0 !cost 0;
  { xs = unstack (p.Problem.horizon + 1) !xs
  ; us = unstack p.Problem.horizon !us
  ; cost = !cost_t
  ; iters = scalar_like x0 (Float.of_int !updates)
  }

let solve p = solve_with p (eager_kernels p)

(* The adaptive solve with its traceable kernels compiled. Build the function
   once and reuse it: the programs are keyed by the shapes and placements of
   the inputs, so repeated calls replay them. [beam] and [parallel] are
   [Rune.jit]'s search options. *)
let compile ?beam ?parallel p =
  let kernels = compiled_kernels ?beam ?parallel p in
  fun ?max_iter ?conv_threshold ?c1 ?ls_depth ?damping theta x0 us0 ->
    solve_with p kernels ?max_iter ?conv_threshold ?c1 ?ls_depth ?damping theta x0 us0
