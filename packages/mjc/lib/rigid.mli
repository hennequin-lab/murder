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

(** [vec_normalize a] is [a] divided by its norm; a zero vector stays zero. *)
val vec_normalize : ('s, 'd) f -> ('s, 'd) f

(** [mat_vec m v] is the matrix-vector product of the row-major [[n;9]]
    matrices [m] with the [[n;3]] vectors [v], giving [[n;3]]. *)
val mat_vec : ('s, 'd) f -> ('s, 'd) f -> ('s, 'd) f
