(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The two sides of a commutation equation (the left and right copies an
   orbit average pairs).

   A parameter tensor's axes are split into a left side and a right side, and
   an orbit average pairs a left object with a right object. The same record
   carries a tensor's dimensions and its symmetry specification: [left] and
   [right] are lists of the same length as the number of axes on each side. *)

type 'a t =
  { left : 'a
  ; right : 'a
  }
[@@deriving equal, compare]

let map ~f t = { left = f t.left; right = f t.right }
