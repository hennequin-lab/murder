(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Taylor optimizer (Eq. 10 of the paper) on the orbit machinery.

   The step is split so that only the estimator solve is not traceable:
     prepare (traceable)               solve (host)          finish (traceable)
     momentum, orbit averages,    ->   damped symmetric   ->  factors of H_inv,
     dense S_w/S_g, EMA, betas         powers, svd64          apply to g_avg, shift

   [prepare] and [finish] touch neither [Nx.item] of a traced value nor a
   data-dependent branch, so [Compiled] can wrap them in [Rune.jit2] with the
   whole state threaded through input/output leaves. *)

open Base

type config =
  { learning_rate : float option
  ; beta_1 : float
  ; beta_2 : float
  ; damping : float
  }

(* [bump_tensor beta_t beta] decays a bias-correction counter by [beta] with a
   floor, so the debiasing denominator never reaches zero. *)
let bump_tensor beta_t beta =
  Nx.maximum (Nx.mul_s beta_t beta) (Nx.scalar Nx.float32 1e-4)

let ema ~beta x avg = Nx.add (Nx.mul_s avg beta) (Nx.mul_s x (1. -. beta))
let debias beta_t x = Nx.div x (Nx.sub (Nx.scalar Nx.float32 1.) beta_t)

module Make (M : Nx.Ptree.Uniform) (O : Orbit.S with type 'a t = 'a M.t) = struct
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
    M.fold (fun _ acc d -> acc + List.fold d ~init:1 ~f:Int.( * )) 0 O.surrogate_dims

  let init ~config theta =
    { State.theta
    ; g_avg = M.map (fun x -> Nx.zeros Nx.float32 (Nx.shape x)) theta
    ; sigma_g_avg = Nx.zeros Nx.float32 [| dense_size; dense_size |]
    ; beta_1_t = Nx.scalar Nx.float32 config.beta_1
    ; beta_2_t = Nx.scalar Nx.float32 config.beta_2
    }

  (* Everything the host solve needs, plus the pieces [finish] carries into
     the next state. *)
  let prepare ~config ~state ~grads =
    let g_avg =
      M.map2 (fun g avg -> ema ~beta:config.beta_1 g avg) grads state.State.g_avg
    in
    let g_avg_debias = M.map (debias state.State.beta_1_t) g_avg in
    let theta_star = O.First_order.orbit_average state.State.theta in
    let g_star = O.First_order.orbit_average g_avg_debias in
    let delta_w = M.map2 (fun t ts -> Nx.sub t ts) state.State.theta theta_star in
    let delta_g = M.map2 (fun g gs -> Nx.sub g gs) g_avg_debias g_star in
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

  let solve ~damping mid = Solve.hessian_inverse ~damping mid.Mid.sigma_w mid.Mid.sigma_g

  let finish ~config ~mid hessian_inv =
    let factors = O.Second_order.factors_of_dense hessian_inv in
    let delta = O.Second_order.apply ~factors mid.Mid.g_avg in
    let theta =
      match config.learning_rate with
      | None -> mid.Mid.theta
      | Some lr -> M.map2 (fun t d -> Nx.sub t (Nx.mul_s d lr)) mid.Mid.theta delta
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

  (* Debug dumps of the dense surrogates and the estimator, opt-in. *)
  let debug_save path x = Nx_io.save_npy path x

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

    (* [Nx.Ptree.instantiate] returns a package with weak dtype parameters;
       pinning each to the working dtype at the value level lets the jit take
       them as input/output structures. *)
    let state_ptree : (module Nx.Ptree.S with type t = state) =
      Nx.Ptree.instantiate (module State)

    let mid_ptree : (module Nx.Ptree.S with type t = mid) =
      Nx.Ptree.instantiate (module Mid)

    let input_ptree : (module Nx.Ptree.S with type t = input) =
      Nx.Ptree.instantiate (module Input)

    let finish_ptree : (module Nx.Ptree.S with type t = finish_input) =
      Nx.Ptree.instantiate (module Finish)

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
      { prepare = Rune.jit2 input_ptree mid_ptree prepare_fn
      ; finish = Rune.jit2 finish_ptree state_ptree finish_fn
      }

    let step t ~config ~state ~grads =
      let mid = t.prepare { Input.state; grads } in
      let hessian_inv = solve ~damping:config.damping mid in
      t.finish { Finish.mid; hessian_inv }
  end
end
