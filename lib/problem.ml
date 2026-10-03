(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The problem description, as a value.

    A problem carries everything the solver needs to know about the task:
    the dtype, the state and control dimensions, the horizon, the dynamics,
    the running and final losses, and the structure of its parameters. All
    of it is data, so problems are built, stored and passed around like any
    other value, and nothing is a module or a functor.

    [params] is the structure of the problem's parameters at [value], built
    with [Nx.Ptree.instantiate (module Params)]. The solver walks it to
    build the structure of its inputs, and [Rune.grad] differentiates with
    respect to it. *)

open Base

(** The concrete tensor type: states, controls and losses at dtype ['w]. *)
type 'w value = (float, 'w) Nx.t

(** The solve inputs: problem parameters, initial state, initial control
    guess. A structure of tensors, so that a differentiation with respect to
    the solve inputs sees [theta], [x0] and every control of [us0] as its
    own leaf with its own name. *)
module Inputs = struct
  type ('p, 'a) t =
    { theta : 'p (** problem parameters *)
    ; x0 : 'a (** initial state, shape [n] *)
    ; us0 : 'a array (** initial control guess, length [horizon] *)
    }
end

(** The problem description. It is all data: the solver takes one of these,
    and the transformations take the parameter structure it carries. *)
type ('w, 'p) t =
  { dtype : (float, 'w) Nx.dtype (** the dtype, for scalar creation *)
  ; n : int (** state dimension *)
  ; m : int (** control dimension *)
  ; horizon : int (** number of control steps; states run from 0 to [horizon] *)
  ; verbose : bool (** per-iteration diagnostics on stdout *)
  ; params : 'p Nx.Ptree.t
    (** the parameter structure at [value], built with
        [Nx.Ptree.instantiate (module Params)] *)
  ; dynamics : 'p -> k:'w value -> x:'w value -> u:'w value -> 'w value
    (** [x_{k+1} = dynamics theta ~k ~x ~u], shape [n]. [k] is the step
        index as a scalar tensor, so a time-varying problem indexes its
        parameters with it. *)
  ; running_loss : 'p -> k:'w value -> x:'w value -> u:'w value -> 'w value
    (** scalar loss at step [k] *)
  ; final_loss : 'p -> x:'w value -> 'w value (** scalar terminal loss *)
  }

(** [inputs problem] is the structure of the solve inputs: the parameters at
    [theta], walked with the problem's own structure, then the initial state
    and every control of the initial guess. *)
let inputs (type w p) (problem : (w, p) t) : (p, w value) Inputs.t Nx.Ptree.t =
  let module I = struct
    type 'a t = (p, 'a) Inputs.t

    let walk c ({ Inputs.theta; x0; us0 } : (p, 'a) Inputs.t) : (p, 'b) Inputs.t =
      let open Nx.Ptree.Walk in
      let theta = field c "theta" (structure problem.params) theta in
      let x0 = field c "x0" leaf x0 in
      let us0 =
        field
          c
          "us0"
          (fun c us0 ->
             ignore (int c (Array.length us0));
             Array.mapi us0 ~f:(fun i u -> index c i leaf u))
          us0
      in
      { Inputs.theta; x0; us0 }
  end
  in
  Nx.Ptree.instantiate (module I)

(** [taus_of problem xs us] stacks the trajectory into a [(K+1); n+m]
    tensor: row k is (x_k; u_k) for k < K, row K is (x_K; 0). This is the
    output of the differentiable solve; a loss on the solve takes its
    cotangent from here. *)
let taus_of p xs us =
  let zero_u = Nx.zeros_like us.(0) in
  let rows =
    Array.init (p.horizon + 1) ~f:(fun k ->
      if k < p.horizon
      then Nx.concatenate ~axis:0 [ xs.(k); us.(k) ]
      else Nx.concatenate ~axis:0 [ xs.(k); zero_u ])
  in
  Nx.stack ~axis:0 (Array.to_list rows)

(** [split_taus problem taus] is ([xs], [us]) with [xs] of length
    [horizon + 1] and [us] of length [horizon]. *)
let split_taus p taus =
  let row k = Nx.slice [ I k ] taus in
  let xs = Array.create ~len:(p.horizon + 1) taus in
  let us = Array.create ~len:p.horizon taus in
  for k = 0 to p.horizon - 1 do
    xs.(k) <- Nx.slice [ R (0, p.n) ] (row k);
    us.(k) <- Nx.slice [ R (p.n, p.n + p.m) ] (row k)
  done;
  xs.(p.horizon) <- Nx.slice [ R (0, p.n) ] (row p.horizon);
  xs, us

(** A differentiable loss evaluated on the solve output: the problem's own
    running + final losses along the trajectory. Useful for self-derivation
    checks and as a building block for user losses. *)
let loss_of_taus p theta taus =
  let xs, us = split_taus p taus in
  let acc = ref (p.final_loss theta ~x:xs.(p.horizon)) in
  for k = 0 to p.horizon - 1 do
    acc
    := Nx.add
         !acc
         (p.running_loss
            theta
            ~k:(Nx.scalar p.dtype (Float.of_int k))
            ~x:xs.(k)
            ~u:us.(k))
  done;
  !acc
