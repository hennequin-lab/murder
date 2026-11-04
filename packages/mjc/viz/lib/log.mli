(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A run, recorded.

    The viewer's third mode: {e follow}. A simulation writes one fixed-stride
    record per frame; a viewer in another process (or the same one, later)
    reads the newest complete record and can jump to any frame in constant
    time. That is what makes "it diverged at step 40 000" a jump rather than a
    replay, and it is why the format is fixed-stride and why the record holds
    world-space geom poses instead of joint positions: a reader that had to run
    kinematics would need the engine to be as fast as the writer's.

    {2:format Format}

    All numbers are little-endian IEEE doubles, so a record is a [float array]
    and reading one is a positioned read of [stride] of them.

    Header, 96 bytes:

    {v
      magic    8   "MJLOG\0\0\0"
      version  u32 1
      ngeom    u32
      nconmax  u32
      stride   u32  floats per record
      reserved u32  zero
      name    64   the model's name, zero padded
    v}

    Record, [stride] doubles:

    {v
      step            1
      time            1
      ncon            1   as a double; counts are small
      nefc            1
      geom_xpos       3 * ngeom
      geom_xmat       9 * ngeom
      contact_pos     3 * nconmax
      contact_frame   9 * nconmax   the normal, then two tangents
      contact_normal  3 * nconmax
      contact_dist    1 * nconmax
      contact_force   1 * nconmax   the normal force of the contact
    v}

    [stride = 4 + 12 ngeom + 17 nconmax]. A record is written whole or not at
    all — a torn record is detected by a short read, not by a checksum — so a
    reader that races a writer sees the previous frame or the next one. *)

(** {1:records Records} *)

(** The type for one recorded frame: what a viewer needs to draw it, and
    nothing that would need the engine to interpret. *)
type frame =
  { step : int64
  ; time : float
  ; ncon : int
  ; nefc : int
  ; geom : float array
    (** [[12 * ngeom]]: per geom its position and its row-major rotation,
        which is the layout {!Scene.of_poses} takes. *)
  ; contact_pos : float array (** [[3 * ncon]] *)
  ; contact_frame : float array (** [[9 * ncon]]: the normal, then two tangents. *)
  ; contact_normal : float array (** [[3 * ncon]], unit, from geom 1 towards 2. *)
  ; contact_dist : float array (** [[ncon]], signed distance. *)
  ; contact_force : float array (** [[ncon]], the force summed over the rows. *)
  }

(** {1:writing Writing} *)

(** The type for a log being written. *)
type writer

(** [create ~path model] is a writer that has written [path]'s header and no
    record. The file is truncated.

    Raises [Sys_error] if [path] cannot be created. *)
val create : path:string -> Mjc.Model.host -> writer

(** [append w ~step d] writes one record from [d]: the geom poses, the contact
    geometry, and one number per contact — the normal force, summed over the
    contact's constraint rows.

    It reads the arrays it needs and allocates one buffer per call; a caller
    that writes every step should write every few frames instead. *)
val append : writer -> step:int64 -> Mjc.Data.host -> unit

(** [close w] closes the file. Further appends raise. *)
val close : writer -> unit

(** {1:reading Reading} *)

(** The type for an open log. *)
type t

(** [open_ path] is the log at [path], positioned at its first record.

    It is [Error message] when the file is not a log, when its version is not
    this one, or when its length is not a whole number of records. *)
val open_ : string -> (t, string) result

(** [length t] is the number of records, from the file's length. *)
val length : t -> int

(** [frame t i] is record [i], read with one positioned read: constant time
    whatever [i] is.

    Raises [Invalid_argument] if [i] is outside the log's range. *)
val frame : t -> int -> frame

(** [model_shape t] is [(ngeom, nconmax)] as the header recorded them, which a
    caller checks against the model it intends to draw the frames with. *)
val model_shape : t -> int * int

(** [name t] is the model name from the header. *)
val name : t -> string

(** [close t] closes the file. *)
val close_reader : t -> unit

(** [divergence ?limit t] is the first frame whose geometry has left [limit] —
    a geom position that is not finite, or is further than [limit] from the
    origin — or [None] if every frame is inside it. [limit] defaults to
    [1e3].

    It scans, so call it when a run has stopped rather than per drawn frame. On
    a 4 000-frame log of a small model the scan is a few milliseconds, and it
    answers the question the viewer exists for: did the run diverge, and at
    which step. *)
val divergence : ?limit:float -> t -> int option

(** {1:scene Replay} *)

(** [scene model t i] is frame [i] as a scene: the geoms at their recorded
    poses, the recorded contacts, and a time. Overlays that need velocities or
    forces are empty, because a record does not carry them.

    Raises [Invalid_argument] if [t]'s shape does not match [model]. *)
val scene : Mjc.Model.host -> t -> int -> Scene.t

(** [pp] formats a summary: the name, the frame count and the shape. *)
val pp : Format.formatter -> t -> unit
