(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The iLQR solver, in two flavors:

   - [solve] is the classic adaptive iLQR: Levenberg-Marquardt regularization
     with Cholesky retry, Armijo backtracking line search, convergence checks.
     All control flow is host-side, so it runs eagerly only — do not place it
     inside Rune.jit. Its time loops are scans, which run as loops when eager.

   - [solve_fixed] runs a fixed number of iterations with a fixed
     regularization, and its line search is a masked argmin over a fixed trial
     schedule. There is no data-dependent control flow and no host read of a
     traced tensor, and every time loop is a [Rune.scan], so the whole solve
     is a pure tensor computation that Rune.jit can trace and compile into
     loops: one step is traced whatever the horizon.

   Every function takes the problem as a value, and works at any dtype: the
   dtype phantom is a type parameter, never a module member. *)

open Base

type 'a t =
  { xs : 'a array (** states, length [horizon + 1] *)
  ; us : 'a array (** controls, length [horizon] *)
  ; cost : 'a (** scalar tensor: running + final loss *)
  }

(* [stack ts] is the array [ts] as one tensor with its length leading. *)
let stack (ts : ('a, 'b) Nx.t array) : ('a, 'b) Nx.t = Nx.stack ~axis:0 (Array.to_list ts)

(* [unstack len t] is the rows of [t], as views. *)
let unstack len (t : ('a, 'b) Nx.t) : ('a, 'b) Nx.t array =
  Array.init len ~f:(fun k -> Nx.slice [ I k ] t)

(* [is_batched x0 us0] is [true] when the state and every control carry a
   leading batch axis. *)
let is_batched x0 us0 =
  Array.length (Nx.shape x0) > 1
  || Array.exists us0 ~f:(fun u -> Array.length (Nx.shape u) > 1)

(* [ks p] is the step index of each step, a scalar row per step. *)
let ks p = Nx.arange p.Problem.dtype 0 p.Problem.horizon 1
let dyn p theta ~k x u = p.Problem.dynamics theta ~k ~x ~u

(* [states p theta x0 us] integrates the dynamics over the controls [us]
   ([horizon; m]), returning the state path ([horizon + 1; n]) as one scan. *)
let states p theta (x0 : 'w Problem.value) (us : 'w Problem.value) : 'w Problem.value =
  let _, x =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      ~f:(fun x (u, k) ->
        let x' = dyn p theta ~k x u in
        x', x')
      ~init:x0
      (us, ks p)
  in
  Nx.concatenate ~axis:0 [ Nx.unsqueeze ~axes:[ 0 ] x0; x ]

(* [cost_of p theta xs us] is the running + final loss along the trajectory
   ([horizon + 1; n]) under the controls ([horizon; m]), as one scan. *)
let cost_of p theta (xs : 'w Problem.value) (us : 'w Problem.value) : 'w Problem.value =
  let _, losses =
    Rune.scan
      Nx.Ptree.unit
      Nx.Ptree.(pair (pair tensor tensor) tensor)
      Nx.Ptree.tensor
      ~f:(fun () ((x, u), k) -> (), p.Problem.running_loss theta ~k ~x ~u)
      ~init:()
      ((Nx.slice [ R (0, p.Problem.horizon) ] xs, us), ks p)
  in
  Nx.add
    (p.Problem.final_loss theta ~x:(Nx.slice [ I p.Problem.horizon ] xs))
    (Nx.sum losses)

(* [rollout_unbatched p theta x0 us] integrates the dynamics, returning the
   state path and the total cost (running + final, a scalar tensor). *)
let rollout_unbatched p theta (x0 : 'w Problem.value) (us : 'w Problem.value array)
  : 'w Problem.value array * 'w Problem.value
  =
  let us = stack us in
  let xs = states p theta x0 us in
  unstack (p.Problem.horizon + 1) xs, cost_of p theta xs us

(* The structure of a rollout, for the [Rune.vmap] that batches it. *)
let rollout_structure () =
  Nx.Ptree.(
    iso
      (fun (xs, cost) -> Array.of_list xs, cost)
      (fun (xs, cost) -> Array.to_list xs, cost)
      (pair (list tensor) tensor))

(* [rollout p theta x0 us] is {!rollout_unbatched}; batched inputs are rolled
   out member by member. *)
let rollout p theta x0 us0 =
  if not (is_batched x0 us0)
  then rollout_unbatched p theta x0 us0
  else (
    let inputs : ('p, 'w Problem.value) Problem.Inputs.t =
      { Problem.Inputs.theta; x0; us0 }
    in
    Rune.vmap
      Nx.Ptree.(Problem.inputs p @-> returns (rollout_structure ()))
      (fun (i : ('p, 'w Problem.value) Problem.Inputs.t) ->
         rollout_unbatched p i.theta i.x0 i.us0)
      inputs)

(* The rows of one step's linearization: the state, the control and the step
   index. *)
module Step_rows = struct
  type 'a t =
    { x : 'a
    ; u : 'a
    ; k : 'a
    }

  let walk c { x; u; k } =
    let open Nx.Ptree.Walk in
    let x = field c "x" leaf x in
    let u = field c "u" leaf u in
    let k = field c "k" leaf k in
    { x; u; k }
end

(* Functions, not values: a structure built with [instantiate] is an
   application, so a top-level binding of one is weakly polymorphic and would
   pin this module to the first dtype it is used at. *)
let step_rows () = Nx.Ptree.instantiate (module Step_rows)
let block_rows () = Nx.Ptree.instantiate (module Lqr.Rows)

(* [step_blocks p theta row] linearizes dynamics and losses at one step, one
   [jacfwd'] or [grad'] pass per block. *)
let step_blocks p theta (row : 'w Problem.value Step_rows.t) : 'w Problem.value Lqr.Rows.t
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
let blocks_t p theta (xs : 'w Problem.value) (us : 'w Problem.value)
  : (float, 'w) Lqr.tape
  =
  let rows =
    { Step_rows.x = Nx.slice [ R (0, p.Problem.horizon) ] xs; u = us; k = ks p }
  in
  let _, blocks =
    Rune.scan
      Nx.Ptree.unit
      (step_rows ())
      (block_rows ())
      ~f:(fun () row -> (), step_blocks p theta row)
      ~init:()
      rows
  in
  let xf = Nx.slice [ I p.Problem.horizon ] xs in
  let fx = Rune.grad' (fun x -> p.Problem.final_loss theta ~x) xf in
  let fxx =
    Rune.jacfwd' (fun x -> Rune.grad' (fun y -> p.Problem.final_loss theta ~x:y) x) xf
  in
  { Lqr.blocks; fx; fxx }

(* [blocks_unbatched p theta xs us] linearizes dynamics and losses along the
   trajectory, from per-step arrays. *)
let blocks_unbatched p theta (xs : 'w Problem.value array) (us : 'w Problem.value array)
  : (float, 'w) Lqr.tape
  =
  blocks_t p theta (stack xs) (stack us)

(* The structure of a tape, for the [Rune.vmap] that batches a
   linearization. *)
let tape_structure () =
  Nx.Ptree.(
    iso
      (fun (blocks, (fx, fxx)) -> { Lqr.blocks; fx; fxx })
      (fun { Lqr.blocks; fx; fxx } -> blocks, (fx, fxx))
      (pair (block_rows ()) (pair tensor tensor)))

let blocks p theta (xs : 'w Problem.value array) (us : 'w Problem.value array)
  : (float, 'w) Lqr.tape
  =
  let batched =
    Array.exists xs ~f:(fun x -> Array.length (Nx.shape x) > 1)
    || Array.exists us ~f:(fun u -> Array.length (Nx.shape u) > 1)
  in
  if not batched
  then blocks_unbatched p theta xs us
  else (
    (* Stack after the batch axes so that each member is a [horizon; n] row
       for the vmap. *)
    let batch = Array.length (Nx.shape xs.(0)) - 1 in
    let xs = Nx.stack ~axis:batch (Array.to_list xs)
    and us = Nx.stack ~axis:batch (Array.to_list us) in
    Rune.vmap
      Nx.Ptree.(pair p.params (pair tensor tensor) @-> returns (tape_structure ()))
      (fun (theta, (xs, us)) -> blocks_t p theta xs us)
      (theta, (xs, us)))

(* The per-step tangents of the first-order data: what a materialized tangent
   (an [Lqr.tangent]) holds, and what {!tangent} emits. *)
module Tangent_data_rows = struct
  type 'a t =
    { da : 'a
    ; db : 'a
    ; df : 'a
    ; dlx : 'a
    ; dlu : 'a
    }

  let walk c { da; db; df; dlx; dlu } =
    let open Nx.Ptree.Walk in
    let da = field c "da" leaf da in
    let db = field c "db" leaf db in
    let df = field c "df" leaf df in
    let dlx = field c "dlx" leaf dlx in
    let dlu = field c "dlu" leaf dlu in
    { da; db; df; dlx; dlu }
end

let tangent_data_rows () = Nx.Ptree.instantiate (module Tangent_data_rows)

(* [step_tangent p theta ~dtheta row] is the θ-tangent of the first-order data
   at one step. One forward pass, with the Jacobians and gradients the data are
   defined by taken inside it: the tangents of [A], [B], [f] and of the loss
   gradient, which is what a tangent of the LQR problem consumes. The tangents
   of the loss Hessians are never needed (see [Lqr]). *)
let step_tangent p theta ~dtheta (row : 'w Problem.value Step_rows.t)
  : 'w Problem.value Tangent_data_rows.t
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
let tangent_t p theta ~dtheta (xs : 'w Problem.value) (us : 'w Problem.value)
  : (float, 'w) Lqr.tangent
  =
  let rows =
    { Step_rows.x = Nx.slice [ R (0, p.Problem.horizon) ] xs; u = us; k = ks p }
  in
  let _, t =
    Rune.scan
      Nx.Ptree.unit
      (step_rows ())
      (tangent_data_rows ())
      ~f:(fun () row -> (), step_tangent p theta ~dtheta row)
      ~init:()
      rows
  in
  let xf = Nx.slice [ I p.Problem.horizon ] xs in
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
let tangent p theta ~dtheta (xs : 'w Problem.value array) (us : 'w Problem.value array)
  : (float, 'w) Lqr.tangent
  =
  tangent_t p theta ~dtheta (stack xs) (stack us)

(* [flip x] reverses the horizon axis. *)
let flip x = Nx.flip ~axes:[ 0 ] x

(* The per-step inputs of the streaming tangent backward pass: the trajectory
   point the step differentiates at, the tape and costate rows it reads, and the
   tangent trajectory. Unlike {!Tangent_data_rows} it carries no tangent: the
   step computes those, and drops them once it has used them. *)
module Tangent_stream_rows = struct
  type 'a t =
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

  let walk c { x; u; k; g; a; b; lxx; lux; lx; fb } =
    let open Nx.Ptree.Walk in
    let x = field c "x" leaf x in
    let u = field c "u" leaf u in
    let k = field c "k" leaf k in
    let g = field c "g" leaf g in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    let lxx = field c "lxx" leaf lxx in
    let lux = field c "lux" leaf lux in
    let lx = field c "lx" leaf lx in
    let fb = field c "fb" leaf fb in
    { x; u; k; g; a; b; lxx; lux; lx; fb }
end

let tangent_stream_rows () = Nx.Ptree.instantiate (module Tangent_stream_rows)

(* [step_tangent_data p theta ~dtheta row] is the θ-tangent of the first-order
   data at one step, without the dynamics value: what the tangent backward pass
   consumes, and nothing else. *)
let step_tangent_data p theta ~dtheta (row : 'w Problem.value Tangent_stream_rows.t)
  : 'w Problem.value * ('w Problem.value * ('w Problem.value * 'w Problem.value))
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
      ~(dx0 : 'w Problem.value)
      (xs : 'w Problem.value array)
      (us : 'w Problem.value array)
  : (float, 'w) Lqr.solution
  =
  let horizon = p.Problem.horizon in
  let xs = stack xs
  and us = stack us in
  let tape = blocks_t p theta xs us in
  let common = Lqr.backward_common tape in
  let x0s = Nx.slice [ R (0, horizon) ] xs in
  (* 1. The tangent trajectory, driven by the tangent of the dynamics. *)
  let _, g_next =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair (step_rows ()) tensor)
      Nx.Ptree.tensor
      ~f:(fun g (row, a) ->
        let df =
          snd
            (Rune.jvp
               p.Problem.params
               Nx.Ptree.tensor
               (fun th -> dyn p th ~k:row.Step_rows.k row.x row.u)
               theta
               dtheta)
        in
        let g' = Nx.add (Lqr.matvec a g) df in
        g', g')
      ~init:dx0
      ({ Step_rows.x = x0s; u = us; k = ks p }, tape.Lqr.blocks.a)
  in
  let g = Nx.concatenate ~axis:0 [ Nx.unsqueeze ~axes:[ 0 ] dx0; g_next ] in
  (* 2. The tangent backward pass, against the primal's factors. *)
  let rows =
    { Tangent_stream_rows.x = flip x0s
    ; u = flip us
    ; k = flip (ks p)
    ; g = flip (Nx.slice [ R (0, horizon) ] g)
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
            Rune.grad' (fun x -> p.Problem.final_loss th ~x) (Nx.slice [ I horizon ] xs))
         theta
         dtheta)
  in
  let e_last = Nx.add dfx (Lqr.matvec tape.Lqr.fxx (Nx.slice [ I horizon ] g)) in
  let _, q_hat =
    Rune.scan
      Nx.Ptree.tensor
      (tangent_stream_rows ())
      Nx.Ptree.tensor
      ~f:(fun e row ->
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
      ~f:(fun v row ->
        let du, v' = Lqr.tangent_roll_step ~v row in
        v', (du, v'))
      ~init:(Nx.zeros_like dx0)
      { Lqr.Tangent_roll_rows.a = tape.Lqr.blocks.a
      ; b = tape.Lqr.blocks.b
      ; chol = common.Lqr.quu_chol
      ; fb = common.Lqr.feedback
      ; q = flip q_hat
      }
  in
  let v =
    Nx.concatenate ~axis:0 [ Nx.unsqueeze ~axes:[ 0 ] (Nx.zeros_like dx0); v_next ]
  in
  { Lqr.xs = Nx.add g v; us = du }

(* [pick ~w ~batched] selects one slice along axis 0 of a batched trajectory
   with a one-hot weight vector [w]. Pure tensor ops. *)
let pick ~w ~(batched : ('a, 'b) Nx.t) : ('a, 'b) Nx.t =
  let extra =
    Array.create ~len:(Array.length (Nx.shape batched) - Array.length (Nx.shape w)) 1
  in
  let w = Nx.reshape (Array.append (Nx.shape w) extra) w in
  Nx.sum ~axes:[ 0 ] (Nx.mul batched w)

(* Scalar-read helper for the adaptive solver. *)
let item_f t = Nx.item [] t

let log p (fmt : ('a, Stdlib.Format.formatter, unit) format) : 'a =
  if p.Problem.verbose
  then Stdlib.Format.printf (Stdlib.( ^^ ) (Stdlib.( ^^ ) "dilqr: " fmt) "\n%!")
  else Stdlib.Format.ifprintf Stdlib.Format.err_formatter fmt

(* The traceable, fixed-iteration solve. [alphas] is the host-side trial
   schedule (e.g. [1.; 0.5; ...]); the acceptance test over trials is
   realized with tensor ops, so the result never depends on host reads.
   A non-PD regularized [Quu] surfaces as NaNs, which the acceptance mask
   rejects. *)
let solve_fixed_unbatched
      p
      ?(alphas = Linesearch.schedule ~n:8 ())
      ?(mu = 0.0)
      ?(c1 = 1e-2)
      ~(iters : int)
      (theta : 'p)
      (x0 : 'w Problem.value)
      (us0 : 'w Problem.value array)
  : 'w Problem.value t
  =
  let alphas = Array.of_list alphas in
  (* A host float becomes a scalar tensor for the LQR core; zero means no
     damping terms at all, which is cheaper than regularizing by zero. *)
  let mu = if Float.equal mu 0.0 then None else Some (Nx.scalar_like x0 mu) in
  let us = ref (stack us0) in
  let xs = ref (states p theta x0 !us) in
  let cost = ref (cost_of p theta !xs !us) in
  for _ = 1 to iters do
    let tape = blocks_t p theta !xs !us in
    let gains, d1, d2 = Lqr.backward ?mu tape ~lin:(Lqr.lin_of_tape tape) in
    let n_alpha = Array.length alphas in
    (* Placeholders, every element is written below. *)
    let us_trials = Array.create ~len:n_alpha x0 in
    let xs_trials = Array.create ~len:n_alpha x0 in
    let costs = Array.create ~len:n_alpha !cost in
    for i = 0 to n_alpha - 1 do
      let xs_t, us_t =
        Lqr.rollout ~dyn:(dyn p theta) ~x0 ~xs:!xs ~us:!us ~gains ~alpha:alphas.(i)
      in
      us_trials.(i) <- us_t;
      xs_trials.(i) <- xs_t;
      costs.(i) <- cost_of p theta xs_t us_t
    done;
    let us_all = Nx.stack ~axis:0 (Array.to_list us_trials) in
    let xs_all = Nx.stack ~axis:0 (Array.to_list xs_trials) in
    let costs = Nx.stack ~axis:0 (Array.to_list costs) in
    (* Modelled cost change: alpha*d1 + 0.5*alpha^2*d2, <= 0 for useful
       trials. Accept trials that beat [cost + c1 * model]. *)
    let alphas_t =
      Nx.stack ~axis:0 (Array.to_list (Array.map ~f:(Nx.scalar_like !cost) alphas))
    in
    let pred =
      Nx.add
        (Nx.mul alphas_t d1)
        (Nx.mul (Nx.mul_s (Nx.mul alphas_t alphas_t) (0.5 *. c1)) d2)
    in
    let accept = Nx.less_equal costs (Nx.add !cost (Nx.mul_s pred c1)) in
    (* [any] is true only when a trial was accepted. The argmin of an
       all-infinity score vector returns 0, so testing [sum w] instead would
       select trial 0 — possibly a non-finite rejected rollout — whenever no
       trial is accepted. Mask the rejected trials' costs and trajectories
       before any weighted sum: a NaN rejected trial must not reach the
       result, nor the backward pass through [pick]. *)
    let any = Nx.any accept in
    let costs = Nx.where accept costs (Nx.full_like costs 0.0) in
    let scores = Nx.where accept costs (Nx.full_like costs Float.infinity) in
    let best = Nx.cast Nx.int32 (Nx.argmin ~axis:0 scores) in
    let w = Nx.cast (Nx.dtype !cost) (Nx.equal (Nx.arange Nx.int32 0 n_alpha 1) best) in
    let mask =
      Nx.reshape
        (Array.append
           [| n_alpha |]
           (Array.create ~len:(Array.length (Nx.shape us_all) - 1) 1))
        accept
    in
    let us_all = Nx.where mask us_all (Nx.broadcast_to (Nx.shape us_all) !us) in
    let xs_all = Nx.where mask xs_all (Nx.broadcast_to (Nx.shape xs_all) !xs) in
    let us_sel = pick ~w ~batched:us_all in
    let xs_sel = pick ~w ~batched:xs_all in
    let cost_sel = Nx.sum (Nx.mul costs w) in
    let us' = Nx.where any us_sel !us in
    let xs' = Nx.where any xs_sel !xs in
    let cost' = Nx.where any cost_sel !cost in
    us := us';
    xs := xs';
    cost := cost'
  done;
  { xs = unstack (p.Problem.horizon + 1) !xs
  ; us = unstack p.Problem.horizon !us
  ; cost = !cost
  }

(* The structure of a solution: the states, the controls and the cost, for the
   [Rune.vmap] that turns the unbatched solve into a batched one. *)
let solution_structure () =
  Nx.Ptree.(
    iso
      (fun (xs, (us, cost)) -> { xs = Array.of_list xs; us = Array.of_list us; cost })
      (fun { xs; us; cost } -> Array.to_list xs, (Array.to_list us, cost))
      (pair (list tensor) (pair (list tensor) tensor)))

(* The fixed-iteration solve. Batched inputs (a leading batch axis on the
   parameters, the initial state and every initial control) are solved by
   [Rune.vmap]-ing the unbatched solve over the batch: every member gets its
   own line search and its own trajectory, and the dynamics and losses are
   written for one member. *)
let solve_fixed p ?alphas ?mu ?c1 ~iters theta x0 us0 =
  if not (is_batched x0 us0)
  then solve_fixed_unbatched p ?alphas ?mu ?c1 ~iters theta x0 us0
  else (
    let inputs : ('p, 'w Problem.value) Problem.Inputs.t =
      { Problem.Inputs.theta; x0; us0 }
    in
    Rune.vmap
      Nx.Ptree.(Problem.inputs p @-> returns (solution_structure ()))
      (fun (i : ('p, 'w Problem.value) Problem.Inputs.t) ->
         solve_fixed_unbatched p ?alphas ?mu ?c1 ~iters i.theta i.x0 i.us0)
      inputs)

(* The adaptive, host-driven solve. Requires untraced tensors: it reads
   scalars on the host and retries the Cholesky factorization on
   non-positive-definite [Quu]. Unbatched: a batch would need one line search
   per member, which is host control flow. Use {!solve_fixed} for batches. *)
let solve
      p
      ?(max_iter = 20)
      ?(conv_threshold = 1e-6)
      ?(c1 = 1e-2)
      ?(ls_depth = 8)
      ?(reg = Regularisation.init)
      (theta : 'p)
      (x0 : 'w Problem.value)
      (us0 : 'w Problem.value array)
  : 'w Problem.value t
  =
  if is_batched x0 us0
  then
    failwith
      "dilqr: the adaptive solve is unbatched; use Solver.solve_fixed or \
       solve_ilqr_fixed for batched problems";
  let us = ref (stack us0) in
  let xs = ref (states p theta x0 !us) in
  let cost = ref (item_f (cost_of p theta !xs !us)) in
  let reg = ref reg in
  let backward_adaptive tape =
    let rec retry attempts =
      if attempts > 30
      then failwith "dilqr: Quu could not be regularized to positive definiteness";
      let mu = Nx.scalar_like x0 !reg.Regularisation.mu in
      let gains, d1, d2 = Lqr.backward ~mu tape ~lin:(Lqr.lin_of_tape tape) in
      (* A non-positive-definite [Quu] gives a NaN Cholesky factor, and the
         NaN runs through the sweep into the gains and the expected-decrease
         accumulators. *)
      if Nx.item [] (Nx.all (Nx.isfinite d1)) && Nx.item [] (Nx.all (Nx.isfinite d2))
      then gains, d1, d2
      else (
        reg := Regularisation.increase !reg;
        log p "cholesky failed, mu -> %g" !reg.Regularisation.mu;
        retry (attempts + 1))
    in
    retry 0
  in
  let converged c c' =
    Float.(Float.abs ((c -. c') /. Float.max (Float.abs c) 1.0) < conv_threshold)
  in
  let rec loop (iter : int) (c : float) (attempts : int) =
    if iter >= max_iter
    then ()
    else (
      let tape = blocks_t p theta !xs !us in
      let gains, d1t, d2t = backward_adaptive tape in
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
              let xs_t, us_t =
                Lqr.rollout ~dyn:(dyn p theta) ~x0 ~xs:!xs ~us:!us ~gains ~alpha
              in
              let cost_t = cost_of p theta xs_t us_t in
              let fv : float = item_f cost_t in
              let model = (alpha *. d1) +. (0.5 *. alpha *. alpha *. d2) in
              if Float.(fv <= c +. (c1 *. model))
              then Some (xs_t, us_t, fv, alpha)
              else try_alphas tl
          in
          try_alphas (Linesearch.schedule ~n:ls_depth ())
        in
        match accepted with
        | None ->
          (* No trial decreased enough: regularize harder and retry the
             same iteration. If even the largest regularization cannot
             buy a decreasing step, stop — the solve is stuck at a point
             its quadratic model cannot improve. *)
          log p "line search failed at iter %d, mu -> %g" iter !reg.Regularisation.mu;
          if attempts >= 12
          then ()
          else (
            reg := Regularisation.increase !reg;
            loop iter c (attempts + 1))
        | Some (xs_t, us_t, fv, alpha) ->
          log
            p
            "iter %d: %g -> %g (alpha %g, mu %g)"
            iter
            c
            fv
            alpha
            !reg.Regularisation.mu;
          (* Decay the regularization only when the full step was
             accepted; backtracked steps keep it. *)
          if Float.equal alpha (List.hd_exn (Linesearch.schedule ~n:ls_depth ()))
          then reg := Regularisation.decrease !reg;
          xs := xs_t;
          us := us_t;
          if converged c fv
          then (
            log p "converged after %d iterations: %g -> %g" iter c fv;
            ())
          else loop (iter + 1) fv 0))
  in
  loop 0 !cost 0;
  { xs = unstack (p.Problem.horizon + 1) !xs
  ; us = unstack p.Problem.horizon !us
  ; cost = cost_of p theta !xs !us
  }
