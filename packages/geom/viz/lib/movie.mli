(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Frame sequences and MP4 export.

    A movie is a scene sampled on a regular timeline. The library renders the
    frames itself and delegates video encoding to the system [ffmpeg]; it does
    not encode video in process. [ffmpeg] must be on [PATH] for {!to_mp4}.

    {b Frames.} Frame [i] is rendered at [time = i / fps], so a
    [duration]-second movie at [fps] frames per second has
    [int_of_float (duration * fps)] frames, with at least one. The page size
    of every frame is the camera's page size; {!spec.density} sets the raster
    resolution. *)

(** The type for a movie specification. *)
type spec =
  { duration : float (** duration in seconds *)
  ; fps : int (** frames per second; must be positive *)
  ; scene : Draw.scene (** the scene to sample *)
  ; density : float (** raster pixels per page point, as in {!Frame.create} *)
  }

(** [frame_count spec] is the number of frames,
    [max 1 (int_of_float (duration * fps))].

    Raises [Invalid_argument] if [fps] is not positive. *)
val frame_count : spec -> int

(** [frame_at spec i] is frame [i], rendered at [time = i / fps] with
    [spec.density] and the camera's page size. Index [i] is not checked
    against {!frame_count}; frames beyond the count are simply later times.

    Raises [Invalid_argument] if [fps] is not positive. *)
val frame_at : spec -> int -> Frame.t

(** [render_png_sequence spec ~dir] creates [dir] if necessary and writes
    [frame-000000.png], [frame-000001.png], … for every frame. Existing files
    with those names are overwritten. *)
val render_png_sequence : spec -> dir:string -> unit

(** [to_mp4 ?crf ?frames_dir spec ~path] renders the frame sequence and encodes
    it to [path] with

    {v ffmpeg -framerate fps -i frame-%06d.png -c:v libx264 -pix_fmt yuv420p -crf crf v}

    [crf] is the x264 constant rate factor, [18] by default (high quality).
    The frames go to a temporary directory that is removed on success and on
    encoder failure, unless [frames_dir] is given, in which case that
    directory is used and kept.

    Raises [Failure] if the frame directory cannot be created or if [ffmpeg]
    exits with a non-zero status. *)
val to_mp4 : ?crf:int -> ?frames_dir:string -> spec -> path:string -> unit
