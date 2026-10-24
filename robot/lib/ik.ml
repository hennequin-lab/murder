(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx
module R = Robot

type state =
  { q : float32_t
  ; lambda : float32_t
  ; iterations : int
  }

let q s = s.q
let lambda s = s.lambda
let iterations s = s.iterations
let prod a = Array.fold_left ( * ) 1 a

let batch_shape x =
  let s = shape x in
  Array.sub s 0 (Array.length s - 1)

let init ?(lambda = 1e-3) ~q:q0 () =
  let batch = batch_shape q0 in
  let lambda =
    match batch with
    | [||] -> Nx.full float32 [||] lambda
    | _ -> Nx.full float32 batch lambda
  in
  { q = q0; lambda; iterations = 0 }

(* The normal equations of one batch, collapsed to three axes so the batched
   transpose and solve have one shape to deal with. *)
let normal_equations ~lambda error jacobian =
  let batch = Array.sub (shape jacobian) 0 (Array.length (shape jacobian) - 2) in
  let bsz = prod batch in
  let dim = (shape jacobian).(Array.length (shape jacobian) - 2)
  and dof = (shape jacobian).(Array.length (shape jacobian) - 1) in
  let j = reshape [| bsz; dim; dof |] jacobian in
  let e = reshape [| bsz; dim |] error in
  let jt = transpose ~axes:[ 0; 2; 1 ] j in
  let damping =
    mul (reshape [| bsz; 1; 1 |] (reshape [| bsz |] lambda)) (eye float32 dof)
  in
  let h = add (matmul jt j) damping in
  let g = matmul jt (reshape [| bsz; dim; 1 |] e) in
  h, reshape [| bsz; dof |] (matmul (inv h) g), batch

let batch_of x = Array.sub (shape x) 0 (Array.length (shape x) - 1)

(* [matvec m v] is [m v] over the last two axes, batched: [J dq] for a Jacobian
   and a step. *)
let matvec m v =
  let shape_m = shape m in
  let batch = Array.sub shape_m 0 (Array.length shape_m - 2) in
  let bsz = prod batch in
  let rows = shape_m.(Array.length shape_m - 2)
  and cols = shape_m.(Array.length shape_m - 1) in
  let out =
    matmul
      (reshape [| bsz; rows; cols |] m)
      (reshape [| bsz; cols; 1 |] (reshape (Array.append batch [| cols |]) v))
  in
  reshape (Array.append batch [| rows |]) out

let project ?lower ?upper q =
  match lower, upper with
  | None, None -> q
  | Some lo, None -> maximum lo q
  | None, Some hi -> minimum hi q
  | Some lo, Some hi -> minimum hi (maximum lo q)

let iterate ?(step_limit = 0.5) ?lower ?upper task state =
  let q0 = state.q in
  let batch = batch_of q0 in
  let error = Task.residual task q0 in
  let jacobian = Task.jacobian task q0 in
  let _, step, _ = normal_equations ~lambda:state.lambda error jacobian in
  (* The trust region, per sample. *)
  let dof = (shape q0).(Array.length batch) in
  let step = reshape (Array.append batch [| dof |]) step in
  let largest = max ~axes:[ Array.length batch ] (abs step) in
  let cap =
    div (scalar float32 step_limit) (reshape (Array.append batch [| 1 |]) largest)
  in
  let scale = minimum_s cap 1. in
  let delta = mul scale step in
  let candidate = project ?lower ?upper (sub q0 delta) in
  (* The gain ratio, not the bare improvement: the linear model predicts
     [before - |r + J dq|^2], and a step is kept only where the residual fell by
     a fair fraction of what the model promised. Comparing the two residual
     values alone lets rounding noise accept steps that change nothing, which
     shrinks the damping, which enlarges the next step, which also changes
     nothing — a stall that looks like progress. *)
  let before = Task.squared task q0
  and after = Task.squared task candidate in
  let predicted =
    let linear = Nx.add error (matvec jacobian delta) in
    sub before (Nx.sum ~axes:[ Array.length batch ] (mul linear linear))
  in
  let actual = sub before after in
  let gain = div actual (maximum_s predicted 1e-20) in
  let better = logical_and (greater_s actual 0.) (greater_s gain 1e-3) in
  let better_col =
    if Array.length batch = 0 then better else reshape (Array.append batch [| 1 |]) better
  in
  let q = where better_col candidate q0 in
  let lambda =
    let grow = mul_s state.lambda 4.
    and shrink = mul_s state.lambda 0.5 in
    maximum_s (minimum_s (where better shrink grow) 1e8) 1e-12
  in
  { q; lambda; iterations = state.iterations + 1 }

type result =
  { q : float32_t
  ; residual : float32_t
  ; iterations : int
  ; lambda : float32_t
  ; converged : bool_t
  }

let all_converged converged = Nx.item [] (Nx.all converged)

let solve
      ?(max_iter = 100)
      ?(tolerance = 1e-6)
      ?lambda:lambda0
      ?step_limit
      ?lower
      ?upper
      task
      ~q:q0
  =
  let state = ref (init ?lambda:lambda0 ~q:q0 ()) in
  let converged () = less_s (Task.norm task (q !state)) tolerance in
  let continue = ref true in
  while !continue && iterations !state < max_iter do
    state := iterate ?step_limit ?lower ?upper task !state;
    continue := not (all_converged (converged ()))
  done;
  { q = q !state
  ; residual = Task.norm task (q !state)
  ; iterations = iterations !state
  ; lambda = lambda !state
  ; converged = converged ()
  }
