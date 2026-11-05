(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Box-box manifolds as tensor code, batch over pairs.

    {!Mjc.Collision} computes every other pair as one expression over the whole
    candidate list, so that [Rune.jit] and [vmap] see a straight line. Box-box
    is the exception in the C engine — a fifteen-axis search with early exits
    and two manifestly different contact generators — and it is expressed here
    as masks over those fifteen axes and a fixed twelve-slot clip, which is the
    same program in straight-line form.

    The reference is {!Mjc.Collision_box}, a statement-for-statement host port
    that matches the C engine on the [boxes2] corpus; this module is checked
    against it on random poses, which is the only way to know that a
    straight-line version of a branching algorithm is the same algorithm.

    Every tuning constant and every tie-break is the host kernel's: the
    separation slack, the edge bias, the face substitution, and "first axis
    wins a tie" — a tidier formulation that picks the last is a different
    engine. *)

(** {1:types Batches} *)

(** The type for a batch of pairs: [pos] is [[n;3]], [mat] [[n;9]] row-major and
    [size] [[n;3]], the layout of [Data.geom_xpos], [Data.geom_xmat] and
    [Params.geom_size] sliced to the pairs. *)
type batch =
  { pos1 : Nx.host Nx.float64_t
  ; mat1 : Nx.host Nx.float64_t
  ; size1 : Nx.host Nx.float64_t
  ; pos2 : Nx.host Nx.float64_t
  ; mat2 : Nx.host Nx.float64_t
  ; size2 : Nx.host Nx.float64_t
  ; margin : Nx.host Nx.float64_t (** [[n]] *)
  }

(** The type for the separating-axis decision, per pair.

    [code] is [0..2] a face of one box, [3..5] a face of the other, [6 + 3i + j]
    an edge cross; [sep] is the separation along the winner. [separated] pairs
    have no contact and the rest is meaningless for them. *)
type sat =
  { code : Nx.host Nx.float64_t (** [[n]], as a float for arithmetic. *)
  ; sep : Nx.host Nx.float64_t (** [[n]] *)
  ; separated : Nx.host Nx.float64_t (** [[n]], 1. is separated. *)
  }

(** [sat b] is the separating-axis result of each pair in [b]. *)
val sat : batch -> sat

(** {1:manifold Manifold} *)

(** The type for a batch of manifolds: up to twelve contacts per pair. *)
type manifold =
  { dist : Nx.host Nx.float64_t (** [[n;12]], signed distance along the normal. *)
  ; pos : Nx.host Nx.float64_t (** [[n;12;3]], world. *)
  ; normal : Nx.host Nx.float64_t (** [[n;12;3]], unit, from box 1 to box 2. *)
  ; live : Nx.host Nx.float64_t (** [[n;12]], 1. where the slot holds a contact. *)
  }

(** [manifold b] is the contacts of each pair in [b].

    A pair whose winning axis is an edge cross makes one contact, in slot 0. A
    pair whose winner is a face takes the vertices of the clipped polygon, in
    the order the candidates are generated rather than the engine's clip order:
    the geometry is the engine's, the row order is not, which reaches the solver
    only through the warm start. *)
val manifold : batch -> manifold

(** [debug b] is [(max face separation, max edge separation, tolerance)], for a
    test that has to see why the separating-axis decision differs. *)
val debug : batch -> Nx.host Nx.float64_t * Nx.host Nx.float64_t * Nx.host Nx.float64_t

(** Twelve: a four-gon clipped by four half-planes has at most eight vertices,
    and the C engine allows twelve. *)
val max_contacts : int
