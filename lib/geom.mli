(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Geometric algebra for robotics.

    [Geom] is a small, batched Clifford-algebra library for the Raven stack. It
    provides one generic core and three concrete models:

    - {!E3}: the Euclidean algebra [Cl(3, 0, 0)], the rotation algebra;
    - {!Pga}: plane-based projective geometry [Cl(3, 0, 1)], with planes,
      lines and points;
    - {!Cga}: the conformal model of [RR^3], [Cl(4, 1)], with points, spheres,
      planes, lines, circles, point pairs and the full Möbius group.

    The core {!Algebra} is a functor over a metric signature and a real scalar
    kind. A multivector is an {!Nx.t} with the blade axis last and ordinary
    batch axes in front, so the algebra composes with the rest of Raven and is
    compatible with [Rune.jit] and [Rune.grad]. The models fix the scalar kind
    to [float32].

    {2 A first expression}

    The following is the shape of a computation once the implementation lands:

    {[
      open Geom

      let x = Cga.point3 1. 2. 3. in
      let s =
        Cga.sphere_of_point ~center:(Cga.point3 0. 0. 0.)
          ~radius:(Nx.scalar Nx.float32 2.)
      in
      let on_sphere = Cga.incident x s in
      let moved = Cga.apply_motor (Motor.translate3 0. 0. 1.) x in
      ...
    ]}

    The mathematics and the sign conventions are in [doc/notes.typ]; the design
    and the milestones are in [PLAN.md].

    {2 Scalars and structures}

    The core is polymorphic in the scalar kind ({!module-type-SCALAR}), while the
    model modules are instantiated at [float32]. A value is an ordinary {!Nx.t},
    so it is already a leaf for [Nx.Ptree] and for the optimizers in Vega; no
    custom [walk] is needed. Record types that appear later (for example in the
    visualization package) use [ppx_ptree] rather than a hand-written
    traversal. *)

(** {1:modules Modules} *)

(** Basis blades and their index arithmetic. *)
module Blade = Blade

(** Metric signatures. *)
module Signature = Signature

(** The dense algebra functor and its module types. *)
module Algebra = Algebra

(** Exponential and logarithm of versors, over an algebra. *)
module Versor = Versor

(** Packed rigid motions: the eight-component dual quaternions shared by
    {!Cga} and {!Pga}. *)
module Motor = Motor

(** The Euclidean algebra [Cl(3, 0, 0)]. *)
module E3 = E3

(** Plane-based projective geometric algebra [Cl(3, 0, 1)]. *)
module Pga = Pga

(** The conformal model of [RR^3], [Cl(4, 1)]. *)
module Cga = Cga

(** {1:module_types Module types} *)

(** The interface of a scalar kind. *)
module type SCALAR = Algebra.SCALAR

(** The interface of a signature parameter for {!Algebra.Make}. *)
module type SIGNATURE = Algebra.SIGNATURE

(** The interface of a concrete algebra. *)
module type ALGEBRA = Algebra.ALGEBRA

(** The interface of the exponential layer of an algebra. *)
module type VERSOR = Versor.VERSOR
