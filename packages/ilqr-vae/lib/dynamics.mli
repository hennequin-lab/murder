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

(** [integrate d p ~x0 ~u] rolls the dynamics out over the horizon of [u]:
    [u] is [[batch…; T; m]], [x0] is [[batch…; n]], and the state path is
    [[batch…; T + 1; n]]. Trial axes lead, the horizon follows them. One
    {!Rune.scan} stands for the horizon, so the rollout traces and compiles. *)
val integrate : ('w, 'p) t -> 'p -> x0:'w value -> u:'w value -> 'w value

(** [integrate_samples d p ~x0 ~u] is {!integrate} with a sample axis after
    the horizon: [u] is [[batch…; T; n_samples; m]], [x0] is
    [[batch…; n_samples; n]], and the state path is
    [[batch…; T + 1; n_samples; n]]. *)
val integrate_samples : ('w, 'p) t -> 'p -> x0:'w value -> u:'w value -> 'w value

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

(** {1:mgu Minimal gated unit} *)

(** The minimal gated unit of Heck & Salem (2017), written as the paper's
    Appendix B.2 states it, with the control input entering the candidate
    additively and never the forget gate:

    {[
      f_t  = \phi (z_{t-1} U_f + b_f)
    ]}
    {[
      \hat z_t = \phi ((f_t \odot z_{t-1}) U_h + b_h) + u_t W
    ]}
    {[
      z_t  = (1 - f_t) \odot z_{t-1} + f_t \odot \hat z_t
    ]}

    with the same activation [\phi] for the gate and the candidate, as in the
    paper ([tanh] there; {!make} takes it as an argument, so [relu] or any
    other activation can be used). Unlike the other gated units here it
    applies no step-size scaling to the input, since the recurrence is the
    paper's own discrete-time one.

    {b The biases.} The paper's equations omit them; [b_f] is nevertheless
    necessary, because with the zero initial state of the encoder
    ([z_0 = 0]) a biasless gate is [\phi 0 = 0] for both [tanh] and [relu],
    so the state, and the gradient of every parameter through it, would stay
    at the origin forever. {!make} therefore initializes [b_f] to a positive
    value and learns it like any other parameter.

    {b Stability.} [tanh] keeps the gate in [(-1, 1)]: the state's linear
    factor [1 - f_t] then lies in [(0, 2)] and can amplify the state only
    when the gate goes negative. [relu] keeps the gate non-negative but
    unbounded above, so [1 - f_t] can be arbitrarily negative; a ReLU gate
    relies on the learned parameters staying near the stable regime. *)
module Mgu : sig
  (** The parameters of a minimal gated unit over payload ['a]. {!structure}
      gives their [Nx.Ptree.t]. *)
  type 'a params =
    { uf : 'a (** [[n; n]]; the state-to-gate weight *)
    ; uh : 'a (** [[n; n]]; applied to [f_t \odot z_{t-1}] *)
    ; w : 'a (** [[m; n]]; the additive input weight *)
    ; bf : 'a (** [[n]]; the gate bias *)
    ; bh : 'a (** [[n]]; the candidate bias *)
    }
  [@@deriving ptree]

  (** [structure ()] is the [Nx.Ptree.t] of [params]. It is a function, not a
      value, so that it stays polymorphic in the dtype: call it where the
      structure is used. A top-level [let s = structure ()] would pin it to
      the first dtype, as any application does; write [let s () = structure ()]
      instead. *)
  val structure : unit -> ('a, 'b) Nx.t params Nx.Ptree.t

  (** [make ~key ~bias_f ~n ~m ~phi dtype] initializes [uf] and [uh] at zero
      (weak initial dynamics, easily controlled by the inputs), [w] from a
      normal of standard deviation [1 / sqrt m], [bh] at zero and [bf] at
      [bias_f] in every dimension. [phi] is the activation of both the gate
      and the candidate.

      [bias_f] defaults to [0.5]. [key] defaults to the current random scope.
      Raises [Invalid_argument] if [n] or [m] is not positive. *)
  val make
    :  ?key:Nx.Rng.t
    -> ?bias_f:float
    -> n:int
    -> m:int
    -> phi:('w value -> 'w value)
    -> (float, 'w) Nx.dtype
    -> ('w, (float, 'w) Nx.t params) t

  (** The forward map of {!make}, with the activation exposed. *)
  val dyn
    :  phi:('w value -> 'w value)
    -> (float, 'w) Nx.t params
    -> k:'w value
    -> x:'w value
    -> u:'w value
    -> 'w value
end
