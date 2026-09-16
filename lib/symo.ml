(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The public surface of the library: the symbolic front end and the
   compiler now; the orbit machinery and optimizer as the port progresses (see
   PLAN.md, M0-M1). *)

module Sides = Sides
module Index = Index
module Symmetry = Symmetry
module Term = Term
module Component = Component
module Basis = Basis
module Delta = Delta
module Contract = Contract
module Compiler = Compiler
module Orbit = Orbit
module Solve = Solve
module Optim = Optim

(* [Make (M)] is the old [Symo.Make]: the orbit machinery and the optimizer for
   a model [M] whose parameter tree carries [dims] and [symmetries]. *)
module Make (M : Orbit.Model) = struct
  module Orbit = Orbit.Make (M)

  (* The packed traversal of the parameter tree, for [Rune.grad] and
     [Rune.jit]. *)
  let ptree : (module Nx.Ptree.S with type t = Nx.float32_t M.t) =
    Nx.Ptree.instantiate (module M)

  include Orbit
  include Optim.Make (M) (Orbit)
end
