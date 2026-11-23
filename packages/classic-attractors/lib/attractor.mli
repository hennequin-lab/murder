(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Classic-attractors authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The interface of a classic continuous-time attractor.

    An attractor is a smooth vector field on a small state space, autonomous or
    periodically driven. The generators integrate it with classical
    fourth-order Runge-Kutta ([Jera.Ode.rk4]), one step per [dt], at float64
    times, and return host tensors of the dtype asked for. The field, its
    parameters and its bounds are documented in each module's interface. *)

module type S = sig
  (** [dim] is the dimension of the state space. *)
  val dim : int

  (** [field t y] is the derivative of the state [y] at time [t], broadcast
      over the rows of [y]: a state is [[n; dim]] and the field answers the
      same. [t] is a scalar, in any float dtype and on any set; a field that
      does not read time ignores it. This is the vector field the generators
      integrate, for callers that solve or roll out their own way. *)
  val field : (float, 't, 'd) Nx.t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t

  (** [simulate dt ~n_steps n] is [n] independent runs of [n_steps] states,
      [dt] apart, as a tensor of shape [[n_steps; n; dim]]:

      - the runs start from [n] initial conditions drawn uniformly in the
        attractor's bounds ([key] selects the draw, [Nx.Rng.key 0] by default);
      - each is integrated for [n_steps + discard_first] states and its first
        [discard_first] states are dropped, so the answer still spans
        [n_steps] states ([discard_first] defaults to [0]);
      - the first state of a run is at the initial condition, and consecutive
        states are one RK4 step of size [dt] apart.

      The runs are independent: the initial conditions differ, so the
      trajectories are unrelated after a few Lyapunov times.

      Raises [Invalid_argument] if [n_steps < 1], [n < 1], [discard_first < 0],
      or [dt] is not finite and positive. *)
  val simulate
    :  ?discard_first:int
    -> ?key:Nx.host Nx.Rng.key
    -> (float, 's) Nx.dtype
    -> dt:float
    -> n_steps:int
    -> int
    -> (float, 's, Nx.host) Nx.t

  (** [snippets dt ~n_steps n] is [n] non-overlapping chunks of [n_steps]
      consecutive states of one long run, as a tensor of shape
      [[n_steps; n; dim]]:

      - the run starts from the attractor's classic initial condition and is
        integrated for [n * n_steps + discard_first] states, of which the first
        [discard_first] are dropped ([discard_first] defaults to [0]);
      - the remaining states are cut into [n] consecutive chunks of [n_steps]
        states, with no state shared between chunks;
      - the chunks are returned in a uniformly random order ([key] selects it,
        [Nx.Rng.key 0] by default), so a mini-batch is not a run of consecutive
        chunks.

      The chunks come from one trajectory, so they are correlated at lags
      shorter than their spacing; {!simulate} draws independent initial
      conditions instead.

      Raises [Invalid_argument] as {!simulate} does. *)
  val snippets
    :  ?discard_first:int
    -> ?key:Nx.host Nx.Rng.key
    -> (float, 's) Nx.dtype
    -> dt:float
    -> n_steps:int
    -> int
    -> (float, 's, Nx.host) Nx.t
end
