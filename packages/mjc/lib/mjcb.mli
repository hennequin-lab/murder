(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The [.mjcb] model interchange format.

    A [.mjcb] file is a compiled model: everything a step reads, and nothing
    the compiler computed at run time. It is written by a development-time
    converter that loads MJCF with MuJoCo and calls [mj_setConst], and read by
    [mjc] at run time. The format is binary, little-endian and self-describing
    enough to reject a file written by a newer version with a named field
    rather than a crash.

    A file holds, in one payload:

    - a header: magic, version, the MuJoCo release that compiled the model,
      the topology counts, and the budgets ([nconmax], [nefcmax],
      [max_geom_pairs]);
    - the topology arrays of {!Topo.t}, as [int32];
    - the object names, as UTF-8 strings;
    - the parameter tensors of {!Model.Params.t}, each with its name, rank,
      shape and [float64] payload;
    - a trailing magic, so a truncated file is detected where it is written
      rather than where a tensor is read.

    The full field list, and which milestone reads which field, is in
    [doc/mjcb.md]. *)

(** {1:version Version} *)

(** [version] is the format version this implementation writes and the newest
    it reads. *)
val version : int

(** {1:byte_strings Byte strings} *)

(** [of_string ~source s] is the model [s] describes. [source] names the
    origin in error messages and defaults to ["<string>"]. *)
val of_string : ?source:string -> string -> (Model.host, string) result

(** [to_string m] is the bytes of [m], ready to write. *)
val to_string : Model.host -> (string, string) result

(** {1:files Files} *)

(** [read path] is the model stored at [path]. *)
val read : string -> (Model.host, string) result

(** [write path m] writes [m] to [path], replacing it. *)
val write : string -> Model.host -> (unit, string) result
