(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The damped-GGN optimizer on the orbit machinery.

   The whole step is one traceable function:

     θ ← θ - η · P · ḡ,   P = (S_g + λI)^{-1}  (default)  or  (S_g^{1/2} + λI)^{-1}
     S_g = EMA_β₂( mean over the output elements of R2(∇f_i, ∇f_i) )

   {!ggn} builds the samples from the model's outputs: one per-output
   gradient per element of the output, whose mean orbit-averaged outer
   product is the GGN. [λ] is absolute and one scalar for every isotypic kind
   and leaf: the [`Inverse] form adds it to each block's spectrum
   (gradient-squared units), the [`Root] form to its square root (gradient
   units, where Adam's [epsilon] sits). Both give a positive definite [P]
   whose gain is capped at [1/λ], so the step is bounded and always a descent
   direction.

   The operator is carried as its block coordinates (see {!Blocks}), packed
   into one flat tensor so the state and the jit boundary stay tensor-valued.
   The solve works on the assembled blocks — one small matrix per isotypic
   kind — so it is both cheaper and exact across sizes, unlike a surrogate
   built from the same factors.

   Nothing in the step reads a host value of a traced tensor or branches on
   data, so [Compiled] jits the whole step into one program. *)

open Base
open Nx

module Config = struct
  (** [form] is the shape of the damped inverse: [`Inverse], the default, is
      [(S_g + λI)^{-1}], with [λ] in gradient-squared units, and [`Root] is
      [(S_g^{1/2} + λI)^{-1}], with [λ] in gradient units. *)
  type form =
    [ `Inverse
    | `Root
    ]

  type t =
    { learning_rate : float option
    ; beta_1 : float
    ; beta_2 : float
    ; damping : float
      (** Absolute damping [λ]: it is added to every isotypic block's spectrum,
          or to its square root, depending on {!form}, and caps the
          preconditioner's gain ([1/λ]). It must be positive. *)
    ; form : form
    }

  (** [default ~learning_rate ~damping] is [beta_1 = 0.9], [beta_2 = 0.99] and
      [`Inverse]: the damped GGN step. *)
  let default ~learning_rate ~damping =
    { learning_rate; beta_1 = 0.9; beta_2 = 0.99; damping; form = `Inverse }
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
      ; beta_2_t : 'a
      }

    include Nx.Ptree.S with type ('a, 'd) t := ('a, 'd) t
  end

  type 'd state = ('d Nx.float32_t, 'd) State.t

  module Mid : sig
    type ('a, 'd) t =
      { theta : ('a, 'd) tree
      ; g_avg : ('a, 'd) tree
      ; sigma_g : 'a
      ; sigma_g_avg : 'a
      ; beta_2_t : 'a
      }

    include Nx.Ptree.S with type ('a, 'd) t := ('a, 'd) t
  end

  type 'd mid = ('d Nx.float32_t, 'd) Mid.t

  val init : config:config -> ('d Nx.float32_t, 'd) tree -> 'd state

  (** [default ~learning_rate ~damping] is [beta_1 = 0.9], [beta_2 = 0.99] and
      [`Inverse]: the damped GGN step. *)
  val default : learning_rate:float option -> damping:float -> config

  (** [ggn ~outputs params] is the metric's samples: the gradient of each
      element of [outputs params] with respect to [params], one sample per
      element, with the output's leading axis. Their mean second-order orbit
      average is the GGN (up to the output element count, which the absolute
      damping absorbs). The Jacobian is one forward pass and one backward pass
      per output element, and it traces. *)
  val ggn
    :  outputs:(('d Nx.float32_t, 'd) tree -> 'd Nx.float32_t)
    -> ('d Nx.float32_t, 'd) tree
    -> ('d Nx.float32_t, 'd) tree

  (** [prepare ~config ~state ~grads ~samples] is the traceable metric half of
      the step: the momentum, the metric averaged over [samples] and EMA'd with
      [beta_2], and the debiased metric. *)
  val prepare
    :  config:config
    -> state:'d state
    -> grads:('d Nx.float32_t, 'd) tree
    -> samples:('d Nx.float32_t, 'd) tree
    -> 'd mid

  (** [solve ~config mid] forms the damped inverse of [mid]'s packed [S_g] on
      its blocks, under [config.form], and packs it beside them. It reads no
      host value, so it traces. *)
  val solve : config:config -> 'd mid -> 'd Nx.float32_t

  val finish : config:config -> mid:'d mid -> 'd Nx.float32_t -> 'd state

  (** [step ~config ~state ~grads ~samples] is the whole step, eager. *)
  val step
    :  config:config
    -> state:'d state
    -> grads:('d Nx.float32_t, 'd) tree
    -> samples:('d Nx.float32_t, 'd) tree
    -> 'd state

  (** The whole step as one [Rune.jit] program. *)
  module Compiled : sig
    type 'd t

    (** [create ~config] compiles the step once. The config is a constant of
        the program: changing it (or the damping) means a new [create]. *)
    val create : config:config -> 'd t

    val step
      :  'd t
      -> state:'d state
      -> grads:('d Nx.float32_t, 'd) tree
      -> samples:('d Nx.float32_t, 'd) tree
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
      ; beta_2_t : 'a
      }
    [@@deriving ptree]
  end

  type 'd state = ('d Nx.float32_t, 'd) State.t

  module Mid = struct
    type ('a, 'd) t =
      { theta : ('a, 'd) M.t
      ; g_avg : ('a, 'd) M.t
      ; sigma_g : 'a
      ; sigma_g_avg : 'a
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
    ; beta_2_t = scalar_like zero_factors config.beta_2
    }

  let default = Config.default

  (* [representative t] is one of the tree's tensors, to place a constant
     beside. *)
  let representative t =
    let r = ref None in
    ignore
      (P.map (module M) (fun _ x -> if Option.is_none !r then r := Some x) t
       : (unit, 'd) M.t);
    Option.value_exn !r

  (* The GGN's samples: [Rune.vjp] records one forward pass and returns the
     pullback, which the map applies to each output element's unit cotangent.
     The result carries one sample per output element along a new leading
     axis. *)
  let ggn ~outputs params =
    let ptree : (('d Nx.float32_t, 'd) M.t, 'd) Nx.Ptree.t =
      Nx.Ptree.instantiate (module M)
    in
    let y, pullback = Rune.vjp ptree Nx.Ptree.tensor outputs params in
    let m = Array.fold (shape y) ~init:1 ~f:Int.( * ) in
    let basis =
      reshape
        (Array.append [| m |] (shape y))
        (eye_at (beside (representative params)) float32 m)
    in
    Rune.vmap Nx.Ptree.(tensor @-> returns ptree) (fun c -> pullback c) basis

  (* [sample_count samples] is the length of the metric tree's leading axis. *)
  let sample_count samples =
    let count = ref (-1) in
    ignore
      (P.map
         (module M)
         (fun _ x ->
            if Int.(!count < 0) then count := dim 0 x;
            ())
         samples
       : (unit, 'd) M.t);
    if Int.(!count < 1)
    then invalid_arg "Symo.Optim: the metric samples hold no leading sample axis";
    !count

  (* [sample_metric samples] is the mean over the samples of the second-order
     orbit average of their outer products: the orbit average of the GGN when
     the samples are {!ggn}'s per-output gradients. Projection and factor
     estimation are linear, so the mean commutes with them and one packed
     tensor accumulates it. *)
  let sample_metric samples =
    let n = sample_count samples in
    let sample i = P.map (module M) (fun _ x -> slice [ Nx.I i ] x) samples in
    let estimate i =
      O.Second_order.pack (O.Second_order.factors_of_pair (sample i) (sample i))
    in
    let total =
      List.fold (List.range 1 n) ~init:(estimate 0) ~f:(fun acc i ->
        Infix.(acc + estimate i))
    in
    Infix.(total /$ Float.of_int n)

  (* Everything the solve needs, plus the pieces [finish] carries into the
     next state. *)
  let prepare ~config ~state ~grads ~samples =
    let g_avg =
      map2 (fun _ g avg -> ema ~beta:config.beta_1 g avg) grads state.State.g_avg
    in
    let sigma_g_avg =
      sample_metric samples |> fun s -> ema ~beta:config.beta_2 s state.State.sigma_g_avg
    in
    let sigma_g = debias state.State.beta_2_t sigma_g_avg in
    { Mid.theta = state.State.theta
    ; g_avg
    ; sigma_g
    ; sigma_g_avg
    ; beta_2_t = bump_tensor state.State.beta_2_t config.beta_2
    }

  (* The solve: unpack the packed factors of [S_g], assemble its blocks, apply
     the damped inverse in the config's form, and pack the preconditioner. *)
  let solve ~config mid =
    let sigma_g = O.Second_order.unpack mid.Mid.sigma_g in
    let blocks = O.Second_order.blocks_of_factors sigma_g in
    (match config.form with
     | `Inverse -> Solve.damped_inverse_blocks ~damping:config.damping blocks
     | `Root -> Solve.damped_inverse_root_blocks ~damping:config.damping blocks)
    |> O.Second_order.factors_of_blocks
    |> O.Second_order.pack

  let finish ~config ~mid preconditioner =
    let factors = O.Second_order.unpack preconditioner in
    let delta = O.Second_order.apply ~factors mid.Mid.g_avg in
    let theta =
      match config.learning_rate with
      | None -> mid.Mid.theta
      | Some lr -> map2 (fun _ t d -> Infix.(t - (d *$ lr))) mid.Mid.theta delta
    in
    { State.theta
    ; g_avg = mid.Mid.g_avg
    ; sigma_g_avg = mid.Mid.sigma_g_avg
    ; beta_2_t = mid.Mid.beta_2_t
    }

  let step ~config ~state ~grads ~samples =
    let mid = prepare ~config ~state ~grads ~samples in
    let preconditioner = solve ~config mid in
    finish ~config ~mid preconditioner

  (* ------------------------------------------------------------------------
     JIT

     The whole step is one program: the state, the gradients and the GGN
     samples cross the boundary as input leaves and the next state comes back
     as the output leaf, so a training loop traces once and replays.
     ------------------------------------------------------------------------ *)

  module Compiled = struct
    module Input = struct
      type ('a, 'd) t =
        { state : ('a, 'd) State.t
        ; grads : ('a, 'd) M.t
        ; samples : ('a, 'd) M.t
        }
      [@@deriving ptree]
    end

    type 'd input = ('d Nx.float32_t, 'd) Input.t
    type 'd t = { run : 'd input -> 'd state }

    (* A description is at one dtype and one set, so [create] makes them at
       the working dtype and the caller's set, for the jit to take them as
       input/output structures. *)
    let create (type d) ~config : d t =
      let state_ptree : (d state, d) Nx.Ptree.t = Nx.Ptree.instantiate (module State) in
      let input_ptree : (d input, d) Nx.Ptree.t = Nx.Ptree.instantiate (module Input) in
      let run_fn (i : d input) =
        step ~config ~state:i.Input.state ~grads:i.Input.grads ~samples:i.Input.samples
      in
      { run = Rune.jit Nx.Ptree.(input_ptree @-> returns state_ptree) run_fn }

    let step t ~state ~grads ~samples = t.run { Input.state; grads; samples }
  end
end
