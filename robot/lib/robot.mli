(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Robot models and their kinematics.

    [Robot] is the model layer of the robotics stack sketched in
    [ROBOTICS.md]: links, joints and frames, {e batched} forward kinematics, and
    the differential kinematics a task residual or an optimizer consumes.

    {2:conventions Conventions}

    These are the definitions everything here is written against. Every one of
    them is checked in the test suite; do not mix them silently with another
    library's.

    - A motion [M_ab] maps coordinates from frame [b] to frame [a]:
      [p_a = M_ab p_b M_ab~]. Composition is [M_ac = M_ab * M_bc], so the motion
      applied first is the right operand, as with matrices acting on column
      vectors.
    - The model is a tree rooted at link [0]. Every other link is the child of
      exactly one joint, and the joints are given in a topological order.
    - Joint [j] hangs off link [parent] and drives link [child]. Its local
      motion, in the parent link's frame, is

    {[
    M_fixed_j * exp (q_j * g_j)
    ]}

    where [M_fixed_j] is the joint's {e origin} and [g_j] its unit screw, both
    fixed at construction. The child's world pose is therefore

    {[
    M_world_child = M_world_parent * M_fixed_j * exp (q_j * g_j).
    ]}

    A joint frame therefore coincides with its child link's frame when
    [q_j = 0], and an axis that does not pass through the joint origin is
    described by putting the origin where the axis is.
    - A frame is a named rigid transform on a link:
      [M_world_frame = M_world_link * M_pose].
    - A {e screw} is a twist [(omega, v)] with the angular part first, exactly
      as {!Motor.exp} and {!Motor.log} take and return it. The linear part is
      the velocity of the point at the origin of the frame the screw is
      expressed in, so a body point [p] moves with [v + omega x p].
    - The configuration [q] has shape [batch_shape @ [|dof|]], with one column
      per {e moving} joint, in the order the joints were given to {!create},
      fixed joints skipped. Any number of leading batch axes is allowed, and
      every function here maps a batch of configurations to a batch of motions
      with the same batch axes.

    {2:algebra Which motion algebra}

    {!Make} is a functor over the module it computes motions with, and
    {!Float32} — the packed dual quaternion {!Motor} — is the instance this
    library exposes as [Robot]. A float64 instance is a second {!MOTION}
    implementation and a second [Make] application; nothing in this module
    assumes [float32] except the instance it happens to ship. This is the seam
    that keeps the model layer independent of the scalar-precision decision
    still open for the algebra modules ({!Motor}, {!Cga}, {!Pga} are fixed to
    [float32]).

    {2:status Status}

    Implemented: model construction and validation, forward kinematics for all
    links and frames, the analytic motor Jacobian, the geometric Jacobian, and
    the tangent map that gives forward-mode AD the closed form. Not implemented
    yet: joint limits in the solver (the data is here), floating bases,
    multi-DOF joints, dynamics and model I/O. *)

(** {1:motion_interface The motion interface} *)

(** The motions a robot model is built from.

    An implementation is the packed eight-component dual quaternion algebra of
    {!Motor} at some scalar kind. Everything the model layer needs is here: the
    product, the inverse, the exponential that turns a unit screw and a
    coordinate into a joint motion, and the components of a motion that a
    Jacobian needs to re-reference. *)
module type MOTION = sig
  (** The scalar kind of the motions. *)
  type kind

  (** The type for motions. The last axis has length eight. *)
  type t = (float, kind) Nx.t

  (** The data type descriptor of the scalar kind. *)
  val dtype : (float, kind) Nx.dtype

  (** [one] is the identity motion. *)
  val one : t

  (** [mul m1 m2] applies [m2] first and [m1] second, as {!Motor.mul}. *)
  val mul : t -> t -> t

  (** [inverse m] is the inverse motion, as {!Motor.inverse}. *)
  val inverse : t -> t

  (** [normalize m] removes the drift of a long product, as {!Motor.normalize}. *)
  val normalize : t -> t

  (** [translate v] is the translation by the vector [v], as {!Motor.translate}. *)
  val translate : (float, kind) Nx.t -> t

  (** [translate3 x y z] is [translate] of the unbatched vector [(x, y, z)]. *)
  val translate3 : float -> float -> float -> t

  (** [rotate ~axis ~angle] is the rotation by [angle] radians about [axis], as
      {!Motor.rotate}. *)
  val rotate : axis:(float, kind) Nx.t -> angle:(float, kind) Nx.t -> t

  (** [exp twist] is the screw motion of the twist [(omega, v)], as
      {!Motor.exp}. *)
  val exp : (float, kind) Nx.t -> t

  (** [log m] is the twist whose [exp] is [m], as {!Motor.log}. *)
  val log : t -> (float, kind) Nx.t

  (** [adjoint m] is the 6 by 6 adjoint acting on twists, as {!Motor.adjoint}. *)
  val adjoint : t -> (float, kind) Nx.t

  (** [translation m] is the position of the frame [m], as
      {!Motor.translation}. *)
  val translation : t -> (float, kind) Nx.t

  (** [rotation m] is the 3 by 3 rotation of [m], as {!Motor.rotation}. *)
  val rotation : t -> (float, kind) Nx.t
end

(** The packed dual quaternion of {!Motor}: the [float32] instance. *)
module Float32 : MOTION with type kind = Nx.float32_elt

(** {1:models Models} *)

module Make (Mo : MOTION) : sig
  (** {2:types Types} *)

  (** The type for motions of this instance. *)
  type motion = Mo.t

  (** The type for twists: six-vectors [(omega, v)], on the last axis. *)
  type twist = (float, Mo.kind) Nx.t

  (** The type for configurations: [batch_shape @ [|dof|]], on the last axis. *)
  type q = (float, Mo.kind) Nx.t

  (** A joint axis or direction, in the joint frame. It is normalised when the
      model is built. *)
  type axis = float * float * float

  (** The kinds of joint this layer models.

      - [Fixed] contributes no coordinate;
      - [Revolute a] turns by one coordinate about the axis [a] through the
        joint origin, the coordinate in radians and right-handed about [a];
      - [Prismatic a] slides by one coordinate along the direction [a], the
        coordinate in the length unit of the model.

      A continuous joint is a [Revolute] without limits. A floating base or a
      helical joint is not a kind yet: the model stores every joint as a unit
      screw, so a general one-dimensional screw is a one-line addition. *)
  type kind =
    | Fixed
    | Revolute of axis
    | Prismatic of axis

  (** The type for a named frame of a model. *)
  type frame

  (** The type for robot models. *)
  type t

  (** The description of one joint, as given to {!create}.

      [origin] is the fixed motion from the parent link's frame to the joint
      frame, so it carries the link geometry: for a serial arm, the translation
      from the previous joint to this one. [limits] are the lower and upper
      bounds of the coordinate, in radians or in length units; [None], the
      default when the field is omitted at construction, means unbounded. *)
  type joint_spec =
    { name : string
    ; kind : kind
    ; parent : string
    ; child : string
    ; origin : motion
    ; limits : (float * float) option
    }

  (** The description of one frame: a name, the link it is attached to, and its
      fixed pose on that link. *)
  type frame_spec =
    { name : string
    ; link : string
    ; pose : motion
    }

  (** [create ~links ~joints ?frames ()] is the model with these links, joints
      and frames, or a message naming the first fault found.

      [links] are names, and [links] whatever the list, its head is the root:
      the fixed base of the robot. Every other link must appear exactly once as
      the [child] of a joint, and each joint's [parent] must be the root or a
      link some earlier joint drives, so that a single pass over [joints]
      computes every world pose. Names must be non-empty and distinct within
      their namespace; a joint axis must be non-zero.

      Validation is part of construction, so a [t] carries the invariants the
      kinematics relies on and no later function checks them again. *)
  val create
    :  links:string list
    -> joints:joint_spec list
    -> ?frames:frame_spec list
    -> unit
    -> (t, string) result

  (** {2:inspection Inspection} *)

  (** [dof m] is the number of scalar coordinates, that is, the length of [q]'s
      last axis. *)
  val dof : t -> int

  (** [link_count m] is the number of links, including the root. *)
  val link_count : t -> int

  (** [joint_count m] is the number of joints, fixed ones included. *)
  val joint_count : t -> int

  (** [link_names m] are the link names, indexed as the model indexes them. *)
  val link_names : t -> string array

  (** [joint_names m] are the joint names, in model order. *)
  val joint_names : t -> string array

  (** [kind_of_joint m j] is the kind of joint [j]. *)
  val kind_of_joint : t -> int -> kind

  (** [limits m j] are the bounds of joint [j]'s coordinate, or
      [neg_infinity, infinity] when it has none. *)
  val limits : t -> int -> float * float

  (** [frame_names m] are the frame names, indexed as the model indexes them. *)
  val frame_names : t -> string array

  (** [frame m name] is the frame called [name].

      Raises [Invalid_argument] if the model has no such frame; the message
      lists the frames it has. *)
  val frame : t -> string -> frame

  (** [pp ppf m] prints the model: its links, joints, kinds, limits and
      frames. *)
  val pp : Format.formatter -> t -> unit

  (** {2:forward Forward kinematics}

      Every function here is a batch map: a [q] of shape [batch_shape @ [|dof|]]
      gives motions of shape [batch_shape @ ...], with no host loop over the
      batch and no branch on a coordinate's value. The loops are over the model,
      whose size is static. *)

  (** [joint_motion m j q] is the local motion of joint [j]: the identity
      [origin] for a fixed joint, and [origin * exp (q_j * g_j)] otherwise, of
      shape [batch_shape @ [|8|]]. *)
  val joint_motion : t -> int -> q -> motion

  (** [link_motors m q] are the world poses of every link, of shape
      [batch_shape @ [|link_count; 8|]]. Link [i] is row [i], and the root is
      the identity. *)
  val link_motors : t -> q -> motion

  (** [frame_motors m q] are the world poses of every frame, of shape
      [batch_shape @ [|frame_count; 8|]], in the order of {!frame_names}. *)
  val frame_motors : t -> q -> motion

  (** [pose m ~frame:f q] is the world pose of frame [f], of shape
      [batch_shape @ [|8|]]. *)
  val pose : t -> frame:frame -> q -> motion

  (** [relative m ~from:a ~to_:b q] is [pose a^(-1) * pose b], the pose of
      frame [b] expressed in frame [a], of shape [batch_shape @ [|8|]]. It is
      the motion that carries coordinates of [b] into [a], and the natural
      quantity for a task that asks where one frame is seen from another. *)
  val relative : t -> from:frame -> to_:frame -> q -> motion

  (** {2:differential Differential kinematics} *)

  (** [motion_jacobian m ~frame:f q] is the derivative of the frame's motion
      with respect to the coordinates, of shape
      [batch_shape @ [|8; dof|]]: column [i] is

      {[
      d M_frame / d q_i
      = ((M_parent_i * M_fixed_i * g_i * M_fixed_i) ^ (-1 * M_parent_i) ^ -1) * M_frame
      ]}

      which is the packed form of the geometric Jacobian's column acting on the
      pose. It is the {e analytic} Jacobian: the derivative of the eight
      coefficients, useful directly for a residual that is polynomial in the
      motion and for the chain rule into a task error.

      There is one column per coordinate of [q], in [q]'s order, so the result
      multiplies a whole [dq]: a coordinate whose joint is not on the frame's
      chain contributes a zero column.

      For a joint with more than one coordinate this formula would need the
      left-trivialised derivative of [exp], which is the identity only for the
      one-dimensional screws this layer models. *)
  val motion_jacobian : t -> frame:frame -> q -> motion

  (** [jacobian m ?space ~frame:f q] is the geometric Jacobian of the frame: the
      matrix of shape [batch_shape @ [|6; dof|]] whose column [i] is the twist
      the coordinate [q_i] contributes to the frame's velocity, so that
      [jacobian * qdot] is the frame's velocity.

      Every column is a screw [(omega, v)] whose linear part is the velocity of
      the point at the {e frame's} origin, not the world origin, and therefore
      [v = v_origin + omega x p] for [p] the frame's position. [space] says in
      which frame the coordinates of the twist are expressed:

      - [`World] (the default): world axes, so [omega] is the frame's angular
        velocity as seen from the world and [v] is [dp/dt] for [p] the frame's
        position;
      - [`Body]: the frame's own axes, so [omega] is the frame's angular
        velocity in its own coordinates and [v] is the velocity of its origin,
        likewise. It is the classical {e body} Jacobian, the matrix that maps
        joint velocities to [log (M^-1 M(q + h qdot)) / h] as [h] tends to zero.

      Both are the same physical velocity in different coordinates — [`Body] is
      [`World] with each three-vector rotated, not carried by the frame's
      adjoint, which would move the reference point as well. The test suite
      checks [`World] against a finite difference of {!pose} and [`Body] against
      the right-trivialised difference of the logarithm, which is what fixes the
      signs and the reference points.

      {b Which one a pose error pairs with.} An error [e = log (M_target^-1 M)]
      lives in the frame's body coordinates, so the step that cancels it is
      [dq = -J_body^+ e]; [dq = -J_world^+ e] mixes coordinates, and a solver
      built on it converges linearly at best, or not at all. {!log} of the
      {e other} difference, [log (M M_target^-1)], is a spatial twist referenced
      to the world origin, which is a third convention and not [`World] — use
      the body pairing unless there is a reason not to.

      As for {!motion_jacobian}, there is one column per coordinate of [q] and a
      joint off the frame's chain contributes zeros, so [J * qdot] is the
      frame's velocity for a full-length [qdot]. *)
  val jacobian : t -> ?space:[ `Body | `World ] -> frame:frame -> q -> twist

  (** {2:ad Differentiable kinematics}

      The kinematics is already traceable — it is pure {!Nx} — so
      [Rune.jacfwd], [Rune.jvp] and [Rune.grad] differentiate {!pose} as it
      stands. They differentiate the closed form of [exp] and its neighbours,
      which is where the branches are.

      {!pose_tangent} is the same derivative written once, analytically, from
      {!motion_jacobian}; {!pose_diff} attaches it to {!pose} as a
      {!Rune.custom_jvp} rule. Use {!pose_diff} where the derivative is what
      matters: forward mode then evaluates the closed form instead of
      differentiating through it, reverse mode transposes the closed form's
      operations, and higher orders differentiate the rule rather than the
      branches. This is the arrangement the design note argues for: automatic
      differentiation for composition, analytic derivatives as its leaves. *)

  (** [pose_tangent m ~frame:f q dq] is the tangent of {!pose} at [q] along
      [dq]: the sum over [i] of [motion_jacobian]-column [i] times [dq_i], of
      shape [batch_shape @ [|8|]]. It is linear in [dq], which is what a
      tangent map must be. *)
  val pose_tangent : t -> frame:frame -> q -> q -> motion

  (** [pose_diff m ~frame:f] is {!pose} with the derivative {!pose_tangent}
      attached through {!Rune.custom_jvp}. Its values are {!pose}'s, its
      first derivative is the closed form of {!pose_tangent} — not of the
      branches {!pose} would differentiate — and higher derivatives
      differentiate the rule's code. Both modes use it, and it is traceable, so
      it can replace {!pose} anywhere a derivative is taken. *)
  val pose_diff : t -> frame:frame -> q -> motion

  (** {2:standard Models of standard robots} *)

  (** [planar2 ?l1 ?l2 ()] is a two-link planar arm in the [xy] plane: two
      revolute joints about [z], links along [x], and a frame [ee] at the tip of
      the second link. Both lengths default to [1.].

      Its forward kinematics has a closed form, which the test suite states and
      checks: [x = l1 cos q0 + l2 cos (q0 + q1)], [y = l1 sin q0 + l2 sin (q0 + q1)]. *)
  val planar2 : ?l1:float -> ?l2:float -> unit -> t

  (** [six_dof ?lengths ()] is the six-joint arm the visualization demo draws:
      alternating [z] and [y] axes, each joint offset by the previous link's
      length along [z], and a frame [ee] at the tip. [lengths] has six entries
      and defaults to the demo's. *)
  val six_dof : ?lengths:float array -> unit -> t
end

(** {1:instance The float32 instance}

    [Geom_robot.Robot] {e is} {!Make} applied to {!Float32}: the model layer on
    the packed dual quaternion of {!Motor}. The functor is the seam for another
    scalar kind; the instance is what the rest of the stack uses. *)
include module type of Make (Float32)
