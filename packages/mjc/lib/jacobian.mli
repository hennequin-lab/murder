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
