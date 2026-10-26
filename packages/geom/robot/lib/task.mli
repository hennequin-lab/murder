(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Task residuals.

    A task is a map from a configuration to a residual vector and its
    derivative: the value a solver consumes and the value a paper plots. Nothing
    here reduces a task to a scalar or hides it inside a solver — {!Geom_robot.Ik}
    is one consumer, and a learnt model or a QP is another.

    {2:families Two families}

    - {b Pose tasks} ({!pose}): the residual is the Lie-algebra error
      [log (M_target^-1 M(q))], six numbers, expressed in the frame's body
      coordinates. It pairs with the {e body} Jacobian, which is what
      {!Robot.jacobian} gives with [`Body], and the pairing is
      enforced here rather than left to the caller.
    - {b Geometric tasks} ({!primitive}): a reference object attached to a frame,
      carried by the motion, contracted with a target object, and compacted to
      the blades that can be nonzero. A point on a plane is one residual; a point
      on a circle is five; a tool line through a point is three. The same code
      covers point, line, plane, circle, sphere and point pair, which is the
      whole point of doing this in a geometric algebra.

    {2:incidence Representation, and why this is a contraction}

    [geom] stores objects in their lower-grade representative — points, spheres
    and planes are grade 1, lines, circles and point pairs grade 2 — and
    incidence there is the contraction [X dot O = 0]. GAFRO stores the same
    objects in the outer-product nullspace and writes the error as
    [X_d and M X M~]; that is the same constraint in the other representation,
    and for the lower-grade one the contraction is the operator that vanishes
    exactly on the object. So a task here is

    {[
     E(q) = X_d dot (M(q) X M(q)~)
    ]}

    with the contraction taken in the order that keeps its grade — the
    lower-grade factor on the left, since [a dot b] vanishes above it — then
    compacted. A point on a plane gives
    the signed distance, a point on a sphere the power; the redundancy of the
    line and circle cases is the price of one formula, and the per-row
    {!weight} is how it is managed.

    {2:batch Batching}

    Every task maps [q] of shape [batch_shape @ [|dof|]] to a residual of shape
    [batch_shape @ [|dim|]] and a Jacobian of shape
    [batch_shape @ [|dim; dof|]], so a batch of targets is a tensor, not a loop.
    The closures a {!custom} task installs must do the same. *)

(** {1:types Types} *)

(** The type for tasks. A task holds the model it belongs to, the dimension of
    its residual and the two maps that produce it. *)
type t

(** [model t] is the model [t] is defined on. *)
val model : t -> Robot.t

(** [dim t] is the length of the residual vector. *)
val dim : t -> int

(** [label t] names the task for printing and error messages. *)
val label : t -> string

(** {1:constructors Constructors} *)

(** [pose model ~frame ~target] is the pose task: drive the named frame onto
    [target]. The residual is [log (target^-1 M(q))] — six numbers in the
    frame's body coordinates — and the Jacobian is that frame's body Jacobian
    (see {!jacobian}). It is the same convention on both sides: a step [dq]
    changes the body twist by [J dq], and the error is the body twist that
    cancels the error. *)
val pose : Robot.t -> frame:Robot.frame -> target:Geom.Motor.t -> t

(** [primitive model ~frame ~reference ~target] is the geometric task: the
    object [reference], attached to the named frame and carried by its motion,
    is contracted with [target], and the blades that can be nonzero become the
    residual.

    Both objects are normalized at construction ([?normalize] defaults to [true])
    so that the residual does not depend on the representative's scale: points
    by {!Geom.Cga.normalize_point}, planes by {!Geom.Cga.normalize_plane},
    spheres by {!Geom.Cga.normalize_sphere}, and lines, circles and point pairs
    by their Clifford magnitude.

    The useful pairs are the point-to-object tasks — a point on a plane, a
    sphere, a line or a circle, or on one of two points — and the dual ones where
    the moving object is the line (a tool axis through a point) or the plane. *)
val primitive
  :  ?normalize:bool
  -> Robot.t
  -> frame:Robot.frame
  -> reference:Geom.Cga.t
  -> target:Geom.Cga.t
  -> t

(** [custom ?label model ~dim ~residual ~jacobian] is a task defined by its two
    maps, for the residuals this module does not know about. [residual q] must
    have shape [batch_shape @ [|dim|]] and [jacobian q] shape
    [batch_shape @ [|dim; dof|]], and the Jacobian must be the derivative of the
    residual: nothing here checks that. *)
val custom
  :  ?label:string
  -> Robot.t
  -> dim:int
  -> residual:(Robot.q -> Nx.float32_t)
  -> jacobian:(Robot.q -> Nx.float32_t)
  -> t

(** {1:combining Combining} *)

(** [weight t w] scales row [i] of the residual and of the Jacobian by
    [w.(i)]: the per-component weight a geometric task needs, since its
    components have different units.

    Raises [Invalid_argument] if [Array.length w <> dim t]. *)
val weight : t -> float array -> t

(** [scale t s] is [weight t [| s; s; ... |]], the uniform version. *)
val scale : t -> float -> t

(** [stack ts] is the concatenation of the tasks' residual vectors, whose
    Jacobian stacks the same way. The tasks must share a model.

    Raises [Invalid_argument] if the list is empty or the models differ. *)
val stack : t list -> t

(** {1:evaluating Evaluating} *)

(** [residual t q] is the residual vector, of shape [batch_shape @ [|dim|]]. *)
val residual : t -> Robot.q -> Nx.float32_t

(** [jacobian t q] is the Jacobian of the task, of shape
    [batch_shape @ [|dim; dof|]].

    For a {!primitive} task it is the exact derivative of {!residual}. For a
    {!pose} task it is the {e body Jacobian} of the frame: the linearisation of
    the pose error in the group, [M(q)^-1 dM/dq], rather than the derivative of
    its logarithm. The two agree to first order and coincide at the solution, so
    the iteration {!Geom_robot.Ik.iterate} builds from this pair is Gauss-Newton
    on the pose and converges quadratically; it is also the Jacobian every
    operational-space controller uses, because a joint step that changes the
    body twist by [J dq] is exactly what a task-space controller wants.

    The exact derivative of [log] would need the differential of the [SE(3)]
    logarithm, [d log_E]; that is a refinement, and it changes the step only
    where the error is already large, where a trust region dominates anyway. *)
val jacobian : t -> Robot.q -> Nx.float32_t

(** [squared t q] is the sum of the squares of the residual, of shape
    [batch_shape]: the least-squares objective the residual defines, and the
    scalar a demo logs. *)
val squared : t -> Robot.q -> Nx.float32_t

(** [norm t q] is its square root, of shape [batch_shape]. *)
val norm : t -> Robot.q -> Nx.float32_t

(** [pp ppf t] prints the task: its label, its dimension and its model's
    coordinate count. *)
val pp : Format.formatter -> t -> unit
