(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Point Jacobians and their time derivatives.

    A Jacobian here is the map from joint velocities to the spatial velocity
    of a point rigidly attached to a body, in world coordinates and about the
    subtree centre of mass of the body's root, which is the convention
    MuJoCo's {!Data.cdof} uses. It is the input to every constraint row that
    couples two points, which in v1 means the equality constraints. *)

(** {1:types Types} *)

(** A [float64] tensor on the host. *)
type 'd f = 'd Nx.float64_t

(** [body_dof_masks ~at t] is the `[nbody; nv]` table of which degrees of
    freedom move which body. It is structure, so a caller that asks for many
    points at once -- the contact stage does, once per candidate -- should build
    it once and pass it as [~masks]. *)
val body_dof_masks : at:'a Nx.Placement.t -> Topo.t -> 'a Nx.float64_t

(** {1:jacobians Jacobians} *)

(** [point m d ~body ~pos] is [jacp, jacr]: the translational and rotational
    Jacobians of the world position [pos] rigidly attached to [body], each of
    shape [[nv;3]]. Rows outside the body's ancestor chain are zero, which is
    what makes the product with a velocity a world velocity of that point. *)
val point
  :  ?masks:'a Nx.float64_t
  -> 'a Model.model
  -> 'a Data.data
  -> body:int
  -> pos:'a f
  -> 'a f * 'a f

(** [points ~masks m d ~bodies ~positions] is the same pair of Jacobians for a
    batch of points: [bodies] a `[k]` array of body indices and [positions]
    `[k; 3]`, giving `[k; nv; 3]` each. A caller with many points -- the contact
    stage, one per candidate -- uses this instead of calling {!point} in a loop,
    which is the difference between a program that compiles and one that does
    not. *)
val points
  :  ?masks:'a Nx.float64_t
  -> 'a Model.model
  -> 'a Data.data
  -> bodies:int array
  -> positions:'a Nx.float64_t
  -> 'a Nx.float64_t * 'a Nx.float64_t

(** [point_dot m d ~body ~pos] is the time derivative of {!point} along the
    current velocity, [dq/dt] held constant: [Jdot] in MuJoCo's notation. *)
val point_dot
  :  ?masks:'a Nx.float64_t
  -> 'a Model.model
  -> 'a Data.data
  -> body:int
  -> pos:'a f
  -> 'a f * 'a f

(** [jdotv m d ~body ~pos] is [Jdot qvel] for the same point, shape [[6]]:
    translation first, then rotation, the same order as {!Data.cvel} uses. *)
val jdotv : 'a Model.model -> 'a Data.data -> body:int -> pos:'a f -> 'a f
