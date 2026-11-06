(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Contact detection.

    The port follows MuJoCo's two phases. Which geoms may touch at all is a
    property of the model — the [contype]/[conaffinity] bitmask, the body
    filter, the table of primitive pairs — so the candidate list is built once,
    from host data, while the program is traced. The distance itself is
    geometry and stays tensors. *)

(** One pair of geoms that may touch, and which contact of the pair it is. *)
type cand =
  { g1 : int
  ; g2 : int
  ; slot : int
  }

(** [candidates m] is the geom pairs v1 can solve, in MuJoCo's order. It is host
    data and is empty when the model disables contacts. *)
val candidates : 'a Model.model -> cand array

(** [contacts m d] is [d] with the contacts of the current pose: [ncon] and
    the [contact_*] arrays, compacted to the contacts that actually exist.

    v1 computes plane-sphere, plane-capsule, sphere-sphere and sphere-capsule.
    Any other pair a model can produce is a port bug, not a silent omission:
    the candidate list is only built from those four. *)
val contacts : Nx.host Model.model -> Nx.host Data.data -> Nx.host Data.data
