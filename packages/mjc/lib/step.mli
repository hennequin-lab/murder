(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Integration: one step of the semi-implicit Euler scheme.

    The order is MuJoCo's [mj_advance]: the velocity is advanced with the
    acceleration the forward pass just computed, then the position with the
    new velocity, then the clock. A model whose joints have no limits, no
    contacts and no equalities has no constraints, so its [qacc] is the whole
    answer and this is [mj_step]. *)

(** {1:step Step} *)

(** The type for a compiled step: a model's constants, read once, and the
    whole pipeline as one program. *)
type t

(** The type for the structural decisions a step makes about a model. They are
    host values — a timestep, a budget, a scheme — because a program that a
    differentiation traces may not read them from a parameter tensor. *)
type config

(** [config m] reads them from [m]. *)
val config : Nx.host Model.model -> config

(** [config_dt c] is the timestep [c] holds. *)
val config_dt : config -> float

(** [step_with c m d] is one step with [c]'s structure and [m]'s *parameters*,
    read as tensors: it is the function a gradient with respect to a model
    parameter goes through, and it never reads one. *)
val step_with : config -> Nx.host Model.model -> Nx.host Data.data -> Data.host

(** [compile m] is [m]'s step, ready to run. The time step and the damping
    test are read here, not per call, because a traced program may not read a
    value from the data it is tracing. *)
val compile : Nx.host Model.model -> t

(** [run t d] is [d] one time step forward, through the compiled program. The
    first call traces and compiles, the later ones reuse it. *)
val run : t -> Nx.host Data.data -> Data.host

(** [dt t] is the model's time step, which the compiled program holds. *)
val dt : t -> float

(** [advance m d] is [d] one time step forward from a forward pass that has
    already run: [qvel] gains [dt * qacc], [qpos] then gains [dt] times the new
    [qvel] through {!Pose.integrate_pos}, and [time] gains [dt]. *)
val advance : Nx.host Model.model -> Nx.host Data.data -> Data.host

(** [step m d] is [advance m (Forward.forward m d)]. *)
val step : Nx.host Model.model -> Nx.host Data.data -> Data.host
