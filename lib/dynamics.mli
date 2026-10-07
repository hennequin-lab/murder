(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Input-driven dynamics.

    A dynamics integrates one step: [x' = f(x, u)], with [x] of shape [[n]] and
    [u] of shape [[m]], or batches of rows with the batch axes leading. The
    step index [k] is a tensor, so a time-varying dynamics indexes its
    parameters with it — and a time-invariant one ignores it. Only the
    forward map is given: [dilqr] obtains the Jacobians it needs with
    [Rune.jacfwd'], so the reference implementation's analytic
    [dyn_x]/[dyn_u] (and their hand-written derivatives of [phi]) are gone.

    A dynamics is a value whose parameters are passed into [dyn]; the
    constructors here are the architectures the reference implementation
    ships. A model whose dynamics is new is a record literal with a
    [structure], its parameters and a [dyn]. *)

open Types

type ('w, 'p) t =
  { n : int (** state dimension *)
  ; m : int (** input dimension *)
  ; structure : 'p Nx.Ptree.t (** the structure of the dynamics' parameters *)
  ; params : 'p (** the parameters a fresh model starts from *)
  ; dyn : 'p -> k:'w value -> x:'w value -> u:'w value -> 'w value
    (** [dyn p ~k ~x ~u] is the next state. *)
  }

(** [integrate d p ~x0 ~u] rolls the dynamics out over the leading (time) axis
    of [u] and returns the state path, of shape [[T + 1; n]] against
    [[T; m]] controls (batch axes follow time on both). One {!Rune.scan}
    stands for the horizon, so the rollout traces and compiles. *)
val integrate : ('w, 'p) t -> 'p -> x0:'w value -> u:'w value -> 'w value

(** {1:linear Linear} *)

(** A linear system in discrete time:
    [x' = x A + (k = 0 ? u : dt_over_tau · u)] with [A = I + dt_over_tau (W − I)]
    and an optional input projection [b]: [u] is replaced by [u @ b] when [b]
    is present. [W] is what is learned; [A] has the eigenvalues of a
    continuous-time system with the given spectral abscissa after one step at
    the given ratio of step size to time constant. *)
module Linear : sig
  (** The parameters of a linear dynamics over payload ['a]. {!structure}
      gives their [Nx.Ptree.t]. *)
  type 'a params =
    { dt_over_tau : float [@ptree.skip] (** step size over time constant; not learned *)
    ; w : 'a (** [[n; n]] *)
    ; b : 'a option (** [[m; n]], absent when the input needs no projection *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype: call it where the
      structure is used. A top-level [let s = structure ()] would pin it to
      the first dtype, as any application does; write [let s () = structure ()]
      instead. *)
  val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

  (** [make ~key ~alpha ~n ~m dtype] is a linear system with spectral
      abscissa [alpha] ([1.] by default) and step ratio [dt_over_tau = 0.1].
      [W] starts at [alpha · I]; [b] is drawn from a standard normal when
      [n ≠ m] and is absent otherwise ([key] defaults to the current random
      scope).

      Raises [Invalid_argument] if [n] or [m] is not positive. *)
  val make
    :  ?key:Nx.Rng.t
    -> ?alpha:float
    -> n:int
    -> m:int
    -> (float, 'w) Nx.dtype
    -> ('w, (float, 'w) Nx.t params) t

  (** [matrix p] is [A = I + dt_over_tau (W − I)]. *)
  val matrix : (float, 'w) Nx.t params -> 'w value

  val dyn : (float, 'w) Nx.t params -> k:'w value -> x:'w value -> u:'w value -> 'w value
end

(** {1:linear_nonlinear Linear–nonlinear} *)

(** The tanh RNN of the reference implementation: a linear part [a0], a
    non-linearity through a hidden layer [a1, bias1, a2, bias2], and an input
    term,

    {[
     x' = x + (x A0 − x + φ (x A1 + bias1) A2 + bias2) · d + u'
    ]}
    {[
     u' = (k = 0 ? u : d · u)
    ]}

    with [d = dt_over_tau] a learned per-dimension vector, [u] projected by
    [b] when present, and [φ] the activation passed to {!make}. This is the
    architecture of the paper's Lorenz and neural experiments. *)
module Linear_nonlinear : sig
  (** The parameters of a linear–nonlinear dynamics over payload ['a].
      {!structure} gives their [Nx.Ptree.t]. *)
  type 'a params =
    { dt_over_tau : 'a (** [[n]] *)
    ; a0 : 'a (** [[n; n]] *)
    ; a1 : 'a (** [[n; nh]] *)
    ; a2 : 'a (** [[nh; n]] *)
    ; bias1 : 'a (** [[nh]] *)
    ; bias2 : 'a (** [[n]] *)
    ; b : 'a option (** [[m; n]] *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype: call it where the
      structure is used. A top-level [let s = structure ()] would pin it to
      the first dtype, as any application does; write [let s () = structure ()]
      instead. *)
  val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

  (** [make ~key ~radius ~dt ~tau ~n ~nh ~m ~phi dtype] initializes

      - [a1] from a normal of standard deviation [1 / sqrt n] (as the
        reference implementation's [Linear_nonlinear]),
      - [a2] from a normal of standard deviation [radius / sqrt nh],
      - [a0] at [0.5 · I], [bias1] uniformly on [−1, 1] and [bias2] at zero,
      - [b] from a normal of standard deviation [1 / sqrt m], and
      - [dt_over_tau] at [dt / tau] in every dimension.

      [radius] defaults to [0.1], [key] to the current random scope. Raises
      [Invalid_argument] if [n], [nh] or [m] is not positive. *)
  val make
    :  ?key:Nx.Rng.t
    -> ?radius:float
    -> dt:float
    -> tau:float
    -> n:int
    -> nh:int
    -> m:int
    -> phi:('w value -> 'w value)
    -> (float, 'w) Nx.dtype
    -> ('w, (float, 'w) Nx.t params) t

  (** The forward map. [phi] is the activation; the constructor captures it in
      the dynamics' [dyn], and exposing the map lets a caller use it without
      building a {!type-t}. *)
  val dyn
    :  phi:('w value -> 'w value)
    -> (float, 'w) Nx.t params
    -> k:'w value
    -> x:'w value
    -> u:'w value
    -> 'w value
end

(** {1:gnode Gated nonlinear ODE} *)

(** A gated recurrent unit in discrete time: a decay gate [g] and a proposed
    state [h],

    {[
     g x = φ (x G1 + g_bias1)
    ]}
    {[
     h x = φ (x H1 + h_bias1) H2 + h_bias2
    ]}
    {[
     x' = x + d · g x · (h x − x) + u'
    ]}

    with [u'] as in {!module:Linear_nonlinear} and the same [k = 0]
    convention. [φ] is the activation of both the gate and the proposal, as in
    the reference implementation. *)
module Gnode : sig
  (** The parameters of a gated dynamics over payload ['a]. {!structure}
      gives their [Nx.Ptree.t]. *)
  type 'a params =
    { dt_over_tau : float [@ptree.skip] (** step size over time constant; not learned *)
    ; g_a1 : 'a (** [[n; n]] *)
    ; g_bias1 : 'a (** [[n]] *)
    ; h_a1 : 'a (** [[n; nh]] *)
    ; h_a2 : 'a (** [[nh; n]] *)
    ; h_bias1 : 'a (** [[nh]] *)
    ; h_bias2 : 'a (** [[n]] *)
    ; b : 'a option (** [[m; n]] *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype: call it where the
      structure is used. A top-level [let s = structure ()] would pin it to
      the first dtype, as any application does; write [let s () = structure ()]
      instead. *)
  val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

  (** [make ~key ~radius ~m ~dt ~tau ~n ~nh ~phi dtype] initializes [g_a1]
      and [g_bias1] at zero, [h_a1] from a normal of standard deviation
      [1 / sqrt n], [h_a2] from one of standard deviation [radius / sqrt nh],
      the biases at zero, and [b] from a standard normal when [m] is given.

      [radius] defaults to [0.1], [key] to the current random scope. [m]
      defaults to [n], with [b] then absent. Raises [Invalid_argument] if [n]
      or [nh] is not positive. *)
  val make
    :  ?key:Nx.Rng.t
    -> ?radius:float
    -> ?m:int
    -> dt:float
    -> tau:float
    -> n:int
    -> nh:int
    -> phi:('w value -> 'w value)
    -> (float, 'w) Nx.dtype
    -> ('w, (float, 'w) Nx.t params) t

  val dyn
    :  phi:('w value -> 'w value)
    -> (float, 'w) Nx.t params
    -> k:'w value
    -> x:'w value
    -> u:'w value
    -> 'w value
end
