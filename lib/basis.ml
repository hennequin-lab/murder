(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The symbolic basis of the invariant subspace, as produced by
   {!Compiler.basis_of_spec}. A basis carries its label, whether it was
   symmetrized, its components, and the axes tied by each group id.

   [components] are the basis elements (each with one factor), [group_axes]
   records which axes must be transformed by the same group element (one list
   per group id), [group_ids] is the group id of each entry of [group_axes]
   (sorted), and [label] is the human-readable symmetry specification. *)

open Base

type t =
  { label : string
  ; symmetric : bool
  ; components : Component.t list
  ; group_axes : Index.t list list
  ; group_ids : int list
  }

let equal (a : t) (b : t) = Poly.equal a b
let compare (a : t) (b : t) = Poly.compare a b
