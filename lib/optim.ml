(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Taylor optimizer (Eq. 10 of the paper) on the orbit machinery.

   The step is split so that only the estimator solve is not traceable:
   prepare (traceable)               solve (host)          finish (traceable)
   momentum, orbit averages,    ->   damped symmetric   ->  factors of H_inv,
   dense S_w/S_g, EMA, betas         powers, eigh           apply to g_avg, shift

   The solve reads the two small dense surrogates on the host, wherever the
   parameters live, and its result joins them in [finish].

   [prepare] and [finish] touch neither [Nx.item] of a traced value nor a
   data-dependent branch, so [Compiled] can wrap them in [Rune.jit] with the
   whole state threaded through input/output leaves. *)

open Base

module Config = struct
  type t =
    { learning_rate : float option
    ; beta_1 : float
    ; beta_2 : float
    ; damping : float
    }
end

type config = Config.t

(* [bump_tensor beta_t beta] decays a bias-correction counter by [beta] with a
   floor, so the debiasing denominator never reaches zero. *)
let bump_tensor beta_t beta =
  Nx.maximum (Nx.mul_s beta_t beta) (Nx.scalar Nx.float32 1e-4)

let ema ~beta x avg = Nx.add (Nx.mul_s avg beta) (Nx.mul_s x (1. -. beta))
let debias beta_t x = Nx.div x (Nx.sub (Nx.scalar Nx.float32 1.) beta_t)

module type S = sig
  type 'a tree
  type config = Config.t

  module State : sig
    type 'a t =
      { theta : 'a tree
      ; g_avg : 'a tree
      ; sigma_g_avg : 'a
      ; beta_1_t : 'a
      ; beta_2_t : 'a
      }

    include Nx.Ptree.S with type 'a t := 'a t
  end

  type state = Nx.float32_t State.t

  module Mid : sig
    type 'a t =
      { theta : 'a tree
      ; g_avg : 'a tree
      ; g_avg_debias : 'a tree
      ; sigma_w : 'a
      ; sigma_g : 'a
      ; sigma_g_avg : 'a
      ; beta_1_t : 'a
      ; beta_2_t : 'a
      }

    include Nx.Ptree.S with type 'a t := 'a t
  end

  type mid = Nx.float32_t Mid.t

  val init : config:config -> Nx.float32_t tree -> state
  val prepare : config:config -> state:state -> grads:Nx.float32_t tree -> mid
  val solve : damping:float -> mid -> Nx.float32_t
  val finish : config:config -> mid:mid -> Nx.float32_t -> state
  val step : config:config -> state:state -> grads:Nx.float32_t tree -> state

  module Compiled : sig
    type t

    val create : config:config -> t
    val step : t -> config:config -> state:state -> grads:Nx.float32_t tree -> state
  end
end

module Make (M : Nx.Ptree.S) (O : Orbit.S with type 'a t = 'a M.t) :
  S with type 'a tree = 'a M.t = struct
  module P = Nx.Ptree.Payload

  type 'a tree = 'a M.t
  type config = Config.t

  open Config

  module State = struct
    type 'a t =
      { theta : 'a M.t
      ; g_avg : 'a M.t
      ; sigma_g_avg : 'a
      ; beta_1_t : 'a
      ; beta_2_t : 'a
      }
    [@@deriving ptree]
  end

  type state = Nx.float32_t State.t

  module Mid = struct
    type 'a t =
      { theta : 'a M.t
      ; g_avg : 'a M.t
      ; g_avg_debias : 'a M.t
      ; sigma_w : 'a
      ; sigma_g : 'a
      ; sigma_g_avg : 'a
      ; beta_1_t : 'a
      ; beta_2_t : 'a
      }
    [@@deriving ptree]
  end

  type mid = Nx.float32_t Mid.t

  let dense_size =
    P.fold
      (module M)
      (fun _ d acc -> acc + List.fold d ~init:1 ~f:Int.( * ))
      O.surrogate_dims
      0

  let init ~config theta =
    { State.theta
    ; g_avg = P.map (module M) (fun _ x -> Nx.zeros_like x) theta
    ; sigma_g_avg = Nx.zeros Nx.float32 [| dense_size; dense_size |]
    ; beta_1_t = Nx.scalar Nx.float32 config.beta_1
    ; beta_2_t = Nx.scalar Nx.float32 config.beta_2
    }

  (* Everything the host solve needs, plus the pieces [finish] carries into
     the next state. *)
  let prepare ~config ~state ~grads =
    let g_avg =
      P.map2
        (module M)
        (fun _ g avg -> ema ~beta:config.beta_1 g avg)
        grads
        state.State.g_avg
    in
    let g_avg_debias =
      P.map (module M) (fun _ g -> debias state.State.beta_1_t g) g_avg
    in
    let theta_star = O.First_order.orbit_average state.State.theta in
    let g_star = O.First_order.orbit_average g_avg_debias in
    let delta_w =
      P.map2 (module M) (fun _ t ts -> Nx.sub t ts) state.State.theta theta_star
    in
    let delta_g = P.map2 (module M) (fun _ g gs -> Nx.sub g gs) g_avg_debias g_star in
    let sigma_w =
      O.Second_order.dense_of_factors (O.Second_order.factors_of_pair delta_w delta_w)
    in
    let sigma_g_avg =
      O.Second_order.dense_of_factors (O.Second_order.factors_of_pair delta_g delta_g)
      |> fun s -> ema ~beta:config.beta_2 s state.State.sigma_g_avg
    in
    let sigma_g = debias state.State.beta_2_t sigma_g_avg in
    { Mid.theta = state.State.theta
    ; g_avg
    ; g_avg_debias
    ; sigma_w
    ; sigma_g
    ; sigma_g_avg
    ; beta_1_t = bump_tensor state.State.beta_1_t config.beta_1
    ; beta_2_t = bump_tensor state.State.beta_2_t config.beta_2
    }

  let solve ~damping mid =
    let host = Nx.place Nx.Placement.host in
    Solve.hessian_inverse ~damping (host mid.Mid.sigma_w) (host mid.Mid.sigma_g)

  let finish ~config ~mid hessian_inv =
    let factors = O.Second_order.factors_of_dense hessian_inv in
    let delta = O.Second_order.apply ~factors mid.Mid.g_avg in
    let theta =
      match config.learning_rate with
      | None -> mid.Mid.theta
      | Some lr ->
        P.map2 (module M) (fun _ t d -> Nx.sub t (Nx.mul_s d lr)) mid.Mid.theta delta
    in
    { State.theta
    ; g_avg = mid.Mid.g_avg
    ; sigma_g_avg = mid.Mid.sigma_g_avg
    ; beta_1_t = mid.Mid.beta_1_t
    ; beta_2_t = mid.Mid.beta_2_t
    }

  let step ~config ~state ~grads =
    let mid = prepare ~config ~state ~grads in
    let hessian_inv = solve ~damping:config.damping mid in
    finish ~config ~mid hessian_inv

  (* ------------------------------------------------------------------------
     JIT

     [prepare] and [finish] are compiled once per config; the state, the
     gradients and the estimator cross the boundary as input/output leaves, so
     a training loop never retraces. The only host call between them is
     [Solve.hessian_inverse].
     ------------------------------------------------------------------------ *)

  module Compiled = struct
    module Input = struct
      type 'a t =
        { state : 'a State.t
        ; grads : 'a M.t
        }
      [@@deriving ptree]
    end

    type input = Nx.float32_t Input.t

    module Finish = struct
      type 'a t =
        { mid : 'a Mid.t
        ; hessian_inv : 'a
        }
      [@@deriving ptree]
    end

    type finish_input = Nx.float32_t Finish.t

    (* [Nx.Ptree.instantiate] returns a structure with weak dtype parameters;
       pinning each to the working dtype at the value level lets the jit take
       them as input/output structures. *)
    let state_ptree : state Nx.Ptree.t = Nx.Ptree.instantiate (module State)
    let mid_ptree : mid Nx.Ptree.t = Nx.Ptree.instantiate (module Mid)
    let input_ptree : input Nx.Ptree.t = Nx.Ptree.instantiate (module Input)
    let finish_ptree : finish_input Nx.Ptree.t = Nx.Ptree.instantiate (module Finish)

    type t =
      { prepare : input -> mid
      ; finish : finish_input -> state
      }

    let create ~config =
      let prepare_fn (i : input) =
        prepare ~config ~state:i.Input.state ~grads:i.Input.grads
      in
      let finish_fn (f : finish_input) =
        finish ~config ~mid:f.Finish.mid f.Finish.hessian_inv
      in
      { prepare = Rune.jit Nx.Ptree.(input_ptree @-> returns mid_ptree) prepare_fn
      ; finish = Rune.jit Nx.Ptree.(finish_ptree @-> returns state_ptree) finish_fn
      }

    let step t ~config ~state ~grads =
      let mid = t.prepare { Input.state; grads } in
      let hessian_inv = solve ~damping:config.damping mid in
      t.finish { Finish.mid; hessian_inv }
  end
end
