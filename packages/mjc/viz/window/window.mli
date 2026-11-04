(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The interactive window: V1 of the viewer plan.

    A fixed-step loop that steps a model, draws its scene and lets the mouse
    and the keyboard steer. Everything it draws comes from {!Mjc_viz.Scene},
    so the stills backend and the window show the same picture.

    {b Not built yet.} The workspace resolves its libraries from `dune.lock`,
    and raylib is not in it (see `raylib.md` in this directory for the one step
    that adds it). Until then this directory is a stanza dune cannot build. *)

(** The type for a run: the model, its state, and how to step it. *)
type t

(** [open_ model] is a viewer for [model], at its reference state. *)
val open_ : Mjc.Model.host -> t

(** [run ?steps_per_frame ?frames t] is the interactive loop. It returns when
    the window closes or escape is pressed. [steps_per_frame] defaults to [1]
    and [frames] to unlimited: the bound exists for the headless check. *)
val run : ?steps_per_frame:int -> ?frames:int -> t -> unit
