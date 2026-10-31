(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Taylor optimizer (Eq. 10 of the paper) on the orbit machinery.

   The step is split so that only the estimator solve is not traceable:
   prepare (traceable)               solve (host)          finish (traceable)
   momentum, orbit averages,    ->   blockwise damped   ->  factors of H_inv,
   factors of S_w/S_g, EMA           powers, eigh          apply to g_avg, shift

   The curvature operators are carried as their block coordinates (see
   {!Blocks}), packed into one flat tensor per operator so the state and the
   jit boundary stay tensor-valued. The host solve works on the assembled
   blocks — one small matrix per isotypic kind — so it is both cheaper and
   exact across sizes, unlike a surrogate built from the same factors.

   [prepare] and [finish] touch neither [Nx.item] of a traced value nor a
   data-dependent branch, so [Compiled] can wrap them in [Rune.jit] with the
   whole state threaded through input/output leaves. *)

open Base
open Nx

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
let bump_tensor beta_t beta = maximum Infix.(beta_t *$ beta) (scalar_like beta_t 1e-4)
let ema ~beta x avg = Infix.((avg *$ beta) + (x *$ (1. -. beta)))
let debias beta_t x = Infix.(x / (scalar_like beta_t 1. - beta_t))

module type S = sig
  type ('a, 'd) tree
  type config = Config.t

  module State : sig
    type ('a, 'd) t =
      { theta : ('a, 'd) tree
      ; g_avg : ('a, 'd) tree
      ; sigma_g_avg : 'a
      ; beta_1_t : 'a
      ; beta_2_t : 'a
      }

    include Nx.Ptree.S with type ('a, 'd) t := ('a, 'd) t
  end

  type 'd state = ('d Nx.float32_t, 'd) State.t

  module Mid : sig
    type ('a, 'd) t =
      { theta : ('a, 'd) tree
      ; g_avg : ('a, 'd) tree
      ; g_avg_debias : ('a, 'd) tree
      ; sigma_w : 'a
      ; sigma_g : 'a
      ; sigma_g_avg : 'a
      ; beta_1_t : 'a
      ; beta_2_t : 'a
      }

    include Nx.Ptree.S with type ('a, 'd) t := ('a, 'd) t
  end

  type 'd mid = ('d Nx.float32_t, 'd) Mid.t

  val init : config:config -> ('d Nx.float32_t, 'd) tree -> 'd state

  val prepare
    :  config:config
    -> state:'d state
    -> grads:('d Nx.float32_t, 'd) tree
    -> 'd mid

  (** [solve ~damping mid] runs on the host: it places [mid]'s packed factors
      there and its result beside them. *)
  val solve : damping:float -> 'd mid -> 'd Nx.float32_t

  val finish : config:config -> mid:'d mid -> 'd Nx.float32_t -> 'd state

  val step
    :  config:config
    -> state:'d state
    -> grads:('d Nx.float32_t, 'd) tree
    -> 'd state

  module Compiled : sig
    type 'd t

    val create : config:config -> 'd t

    val step
      :  'd t
      -> config:config
      -> state:'d state
      -> grads:('d Nx.float32_t, 'd) tree
      -> 'd state
  end
end

module Make (M : Nx.Ptree.S) (O : Orbit.S with type ('a, 'd) t = ('a, 'd) M.t) :
  S with type ('a, 'd) tree = ('a, 'd) M.t = struct
  module P = Nx.Ptree.Payload

  type ('a, 'd) tree = ('a, 'd) M.t
  type config = Config.t

  open Config

  let map f x = P.map (module M) f x
  let map2 f x y = P.map2 (module M) f x y

  module State = struct
    type ('a, 'd) t =
      { theta : ('a, 'd) M.t
      ; g_avg : ('a, 'd) M.t
      ; sigma_g_avg : 'a
      ; beta_1_t : 'a
      ; beta_2_t : 'a
      }
    [@@deriving ptree]
  end

  type 'd state = ('d Nx.float32_t, 'd) State.t

  module Mid = struct
    type ('a, 'd) t =
      { theta : ('a, 'd) M.t
      ; g_avg : ('a, 'd) M.t
      ; g_avg_debias : ('a, 'd) M.t
      ; sigma_w : 'a
      ; sigma_g : 'a
      ; sigma_g_avg : 'a
      ; beta_1_t : 'a
      ; beta_2_t : 'a
      }
    [@@deriving ptree]
  end

  type 'd mid = ('d Nx.float32_t, 'd) Mid.t

  let init ~config theta =
    let zero = map (fun _ x -> zeros_like x) theta in
    let zero_factors = O.Second_order.pack (O.Second_order.factors_of_pair zero zero) in
    { State.theta
    ; g_avg = zero
    ; sigma_g_avg = zero_factors
    ; beta_1_t = scalar_like zero_factors config.beta_1
    ; beta_2_t = scalar_like zero_factors config.beta_2
    }

  (* Everything the host solve needs, plus the pieces [finish] carries into
     the next state. *)
  let prepare ~config ~state ~grads =
    let g_avg =
      map2 (fun _ g avg -> ema ~beta:config.beta_1 g avg) grads state.State.g_avg
    in
    let g_avg_debias = map (fun _ g -> debias state.State.beta_1_t g) g_avg in
    let theta_star = O.First_order.orbit_average state.State.theta in
    let g_star = O.First_order.orbit_average g_avg_debias in
    let delta_w = map2 (fun _ t ts -> Infix.(t - ts)) state.State.theta theta_star in
    let delta_g = map2 (fun _ g gs -> Infix.(g - gs)) g_avg_debias g_star in
    let sigma_w = O.Second_order.pack (O.Second_order.factors_of_pair delta_w delta_w) in
    let sigma_g_avg =
      O.Second_order.pack (O.Second_order.factors_of_pair delta_g delta_g)
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

  (* The host solve: unpack the two packed factor lists on the host, assemble
     the blocks, apply Eq. 10 blockwise, and pack the result beside [mid]. *)
  let solve ~damping mid =
    let host = place Nx.Host.on in
    let sigma_w = O.Second_order.unpack (host mid.Mid.sigma_w) in
    let sigma_g = O.Second_order.unpack (host mid.Mid.sigma_g) in
    let blocks_w = O.Second_order.blocks_of_factors sigma_w in
    let blocks_g = O.Second_order.blocks_of_factors sigma_g in
    Solve.hessian_inverse_blocks ~damping blocks_w blocks_g
    |> O.Second_order.factors_of_blocks
    |> O.Second_order.pack
    |> place (beside mid.Mid.sigma_w)

  let finish ~config ~mid hessian_inv =
    let factors = O.Second_order.unpack hessian_inv in
    let delta = O.Second_order.apply ~factors mid.Mid.g_avg in
    let theta =
      match config.learning_rate with
      | None -> mid.Mid.theta
      | Some lr -> map2 (fun _ t d -> Infix.(t - (d *$ lr))) mid.Mid.theta delta
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
     [Solve.hessian_inverse_blocks].
     ------------------------------------------------------------------------ *)

  module Compiled = struct
    module Input = struct
      type ('a, 'd) t =
        { state : ('a, 'd) State.t
        ; grads : ('a, 'd) M.t
        }
      [@@deriving ptree]
    end

    type 'd input = ('d Nx.float32_t, 'd) Input.t

    module Finish = struct
      type ('a, 'd) t =
        { mid : ('a, 'd) Mid.t
        ; hessian_inv : 'a
        }
      [@@deriving ptree]
    end

    type 'd finish_input = ('d Nx.float32_t, 'd) Finish.t

    type 'd t =
      { prepare : 'd input -> 'd mid
      ; finish : 'd finish_input -> 'd state
      }

    (* A description is at one dtype and one set, so [create] makes them at
       the working dtype and the caller's set, for the jit to take them as
       input/output structures. *)
    let create (type d) ~config : d t =
      let state_ptree : (d state, d) Nx.Ptree.t = Nx.Ptree.instantiate (module State) in
      let mid_ptree : (d mid, d) Nx.Ptree.t = Nx.Ptree.instantiate (module Mid) in
      let input_ptree : (d input, d) Nx.Ptree.t = Nx.Ptree.instantiate (module Input) in
      let finish_ptree : (d finish_input, d) Nx.Ptree.t =
        Nx.Ptree.instantiate (module Finish)
      in
      let prepare_fn (i : d input) =
        prepare ~config ~state:i.Input.state ~grads:i.Input.grads
      in
      let finish_fn (f : d finish_input) =
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
