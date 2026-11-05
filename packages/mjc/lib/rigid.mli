(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Rigid-body math on tensors: quaternions, rotations and 3-vectors.

    Quaternions are [w; x; y; z], the order MuJoCo stores in [qpos] and
    [body_quat], and the product is the Hamilton product. A rotation maps a
    vector in the rotated frame to the parent frame, so [quat_rot_vec q v]
    with [q = body_quat] takes a vector in body coordinates to the parent's.

    Every function takes a rank-2 tensor whose last axis holds the components
    and whose first axis is a batch of bodies, joints or geoms. Under
    {!Rune.vmap} and {!Rune.grad} the shapes are the ones written here: the
    transformation removes the batch axis it adds. *)

(** {1:types Types} *)

(** The type for float tensors. [('s,'d) f] is a tensor at storage dtype ['s]
    on the device set ['d]. *)
type ('s, 'd) f = (float, 's, 'd) Nx.t

(** {1:quaternions Quaternions} *)

(** [quat_mul a b] is the Hamilton product [a b], shape [[n;4]] from two
    [[n;4]] operands. *)
val quat_mul : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f

(** [quat_mul_axis q v] is [q] times the pure quaternion [(0, v)]: the
    rotation of [q] applied to [v], in quaternion form. *)
val quat_mul_axis : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f

(** [quat_deriv q omega] is the time derivative of [q] when the angular
    velocity is [omega]: [0.5 * (0, omega) * q], which is the side MuJoCo
    multiplies on. *)
val quat_deriv : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f

(** [quat_conj q] is [q]'s conjugate [[w;-x;-y;-z]], the inverse rotation for a
    unit quaternion. *)
val quat_conj : ('s, 'd) f -> ('s, 'd) f

(** [quat_normalize q] is [q] divided by its norm. A zero quaternion is left
    as it is, so that a caller can detect it; no kernel produces one. *)
val quat_normalize : ('s, 'd) f -> ('s, 'd) f

(** [quat_rot_vec q v] is [v] rotated by [q], for a unit [q]: [[n;4]] and
    [[n;3]] give [[n;3]]. *)
val quat_rot_vec : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f

(** [quat_to_mat q] is the rotation matrix of [q], row-major, [[n;4]] to
    [[n;9]]: element [[i;3*j+k]] is row [j], column [k] of the matrix that
    takes a vector in the rotated frame to the parent frame. *)
val quat_to_mat : ('s, 'd) f -> ('s, 'd) f

(** [quat_of_mat m] is the quaternion of the rotation matrix [m] ([[n;9]]), in
    MuJoCo's [mju_mat2Quat] convention: the same branch, chosen by which
    component is largest, so the representative matches the engine's. *)
val quat_of_mat : ('s, 'd) f -> ('s, 'd) f

(** [quat_of_axis_angle ~axis ~angle] is the rotation of [angle] radians about
    [axis], a unit 3-vector: [[n;3]] and [[n]] give [[n;4]]. *)
val quat_of_axis_angle : axis:('s, 'd) f -> angle:('s, 'd) f -> ('s, 'd) f

(** [quat_of_rotvec r] is the rotation whose axis is [r]'s direction and whose
    angle is its norm: [[n;3]] to [[n;4]]. A zero vector gives the identity.
    Near zero the sine ratio is evaluated by its series, so the result is
    smooth in [r] at [r = 0]. *)
val quat_of_rotvec : ('s, 'd) f -> ('s, 'd) f

(** [quat_integrate q w ~scale] is [q] advanced by the angular velocity [w]
    for [scale], the same map as MuJoCo's [mju_quatIntegrate]: normalize [q],
    then multiply it on the right by the rotation of [scale * w]. *)
val quat_integrate : ('s, 'd) f -> ('s, 'd) f -> scale:float -> ('s, 'd) f

(** {1:vectors Vectors and matrices} *)

(** [vec_dot a b] is the inner product along the last axis, [[n;3]] to [[n]]. *)
val vec_dot : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f

(** [vec_cross a b] is the cross product along the last axis, [[n;3]] to
    [[n;3]]. *)
val vec_cross : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f

(** [vec_norm a] is the Euclidean norm along the last axis, [[n;3]] to
    [[n]]. *)
val vec_norm : ('s, 'd) f -> ('s, 'd) f

(** [safe_sqrt x] is [sqrt x] with a floor under its argument, so that its
    derivative stays finite at zero: [sqrt]'s own rule divides by the root, and
    a select that masks the value out does not mask an infinite cotangent —
    [0 * inf] is a NaN, and it reaches the reverse pass of every model whose
    frame is built from a zero normal. [safe_sqrt_exact] is the same root with
    the value it would have had: exactly zero at zero, [sqrt x] everywhere a
    model's geometry reaches. Use it wherever a length, a normalization or a
    norm is computed under a mask. *)
val safe_sqrt : ('s, 'd) f -> ('s, 'd) f

(** See {!safe_sqrt}. *)
val safe_sqrt_exact : ('s, 'd) f -> ('s, 'd) f

(** [vec_normalize a] is [a] divided by its norm; a zero vector stays zero. *)
val vec_normalize : ('s, 'd) f -> ('s, 'd) f

(** [mat_vec m v] is the matrix-vector product of the row-major [[n;9]]
    matrices [m] with the [[n;3]] vectors [v], giving [[n;3]]. *)
val mat_vec : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f

(** {1:spatial Spatial 6-vectors}

    A spatial motion is a 3-vector of rotation followed by a 3-vector of
    translation; a spatial force is a 3-vector of torque followed by a
    3-vector of force. Each function here takes rank-1 tensors of six
    components. *)

(** [dot6 a b] is the inner product of two 6-vectors. *)
val dot6 : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f

(** [cross_motion vel v] is the motion cross product of [vel] with the motion
    [v], MuJoCo's [mji_crossMotion]. *)
val cross_motion : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f

(** [cross_force vel f] is the force cross product of the motion [vel] with
    the force [f], MuJoCo's [mji_crossForce]. *)
val cross_force : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f

(** [mul_inert_vec i v] is a ten-entry inertia times a motion 6-vector,
    MuJoCo's [mju_mulInertVec]: six rotational entries, three first moments,
    the mass. *)
val mul_inert_vec : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f
