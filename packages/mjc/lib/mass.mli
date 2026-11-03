(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The position stage's inertia quantities: the subtree centre of mass, the
    centre-of-mass inertia of each body, the motion subspace of each degree of
    freedom, and the mass matrix.

    The layout follows MuJoCo's [mj_comPos] and [mj_crb] so that the numbers
    can be compared array by array: [cinert] holds, per body, a symmetric
    rotational inertia as six numbers (xx, yy, zz, xy, xz, yz), the first
    moment as three, and the mass, all about the centre of mass of the body's
    kinematic tree. [cdof] holds, per degree of freedom, an angular 3-vector
    and a linear 3-vector in the same frame. *)

(** {1:stage The position stage} *)

(** [forward m d] is [d] with [subtree_com], [com], [cinert], [cdof] and
    [mass_matrix] computed. [com] is [subtree_com] of the world body.

    Reads [d.xpos d.xquat d.xipos d.ximat d.xanchor d.xaxis] and the model's
    masses, inertias, joint axes and armature. *)
val forward : Model.host -> Data.host -> Data.host

(** {1:factor Factorization} *)

(** The type for a factored mass matrix. *)
type factor

(** [factor m] is the Cholesky factor of the symmetric positive definite
    matrix [m], which [solve] reuses. It takes the matrix rather than the data
    so that a step can factor [M + h diag(B)], the implicit-damping metric,
    with the same code. [Nx.cholesky] raises on a matrix that is not positive
    definite. *)
val factor : Nx.host Nx.float64_t -> factor

(** [solve f b] is [M^-1 b] using the factor [f]. *)
val solve : factor -> Nx.host Nx.float64_t -> Nx.host Nx.float64_t
