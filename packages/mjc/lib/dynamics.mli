(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Differentiable dynamics.

    This is the entry point a learner calls. A {!t} is a model compiled once —
    its topology, its integration scheme, its solver budget, its damping and
    every other structural decision read from the model and held as host values
    — and then {!run} is a straight line of tensor operations from one {!Data.t}
    to the next.

    The distinction from {!Step} is what [run] may read. A step may read a
    model parameter with [Nx.item] or [Nx.to_array], which is fine for a value
    and impossible for a *differentiated* one: a parameter a transformation
    tracks cannot be read, and it must not be, because a solver budget or an
    integrator's scheme is structure rather than a number to differentiate.
    Compiling takes those reads out of the program, and what is left can be
    differentiated with respect to a mass, a gain or a geom size.

    The corpus's own gradient tests take the derivative of the step with respect
    to the *state*; that works either way. The derivative with respect to a
    parameter is what needs this module. *)

(** {1:type The closure} *)

(** The type for compiled dynamics: a model's structure, read once. *)
type t

(** [compile m] is [m]'s *structure*: its topology, its integration scheme, its
    solver budget, whether any dof is damped, whether any actuator has a
    velocity term. Every one of those is a decision about the model, not a
    number in it, and each is read from [m] here rather than inside [run] — a
    program that a differentiation traces may not read a parameter, and a budget
    is not a derivative anyway.

    The model is [compile]'s argument and also [run]'s: compile the structure
    from a host model, and pass the model whose parameters you mean to
    differentiate at run time. *)
val compile : Model.host -> t

(** [run t m d] is [d] one time step forward, with [t]'s structure and [m]'s
    parameters. [m] may be a model whose parameters a differentiation tracks:
    everything it contributes is tensor code. *)
val run : t -> Model.host -> Data.host -> Data.host

(** [dt t] is the model's timestep. *)
val dt : t -> float

(** [horizon t m d n] is [d] after [n] steps. *)
val horizon : t -> Model.host -> Data.host -> int -> Data.host

(** {1:metric The metric} *)

(** [mass t m d] is the joint-space mass matrix at [d], [[nv; nv]], which is the
    [M] of the [M]-weighted inner product a differential-dynamic-programming
    solver needs: [⟨a, b⟩ = aᵀ M b].

    It is the mass matrix the step itself factors, so it costs no more than a
    step does to recompute, and it is differentiable in the same way. *)
val mass : t -> Model.host -> Data.host -> Nx.host Nx.float64_t

(** [inner t m d a b] is [aᵀ M b] at [d]. *)
val inner
  :  t
  -> Model.host
  -> Data.host
  -> Nx.host Nx.float64_t
  -> Nx.host Nx.float64_t
  -> Nx.host Nx.float64_t
