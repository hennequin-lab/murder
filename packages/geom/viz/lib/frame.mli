(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pages, rasterisation and file output.

    A frame attaches a raster density to a Hugin picture. The page is measured
    in page points, the same units as {!Camera.width} and {!Camera.height};
    the density is the number of raster pixels per point. SVG and PDF are
    vector formats and ignore the density.

    {b Output.} {!png}, {!svg} and {!pdf} return the encoded bytes; the
    [save_*] functions write them to a file. PNG goes through Hugin's raster
    backend, SVG and PDF through its vector backends, so the three files show
    the same page. *)

(** The type for a page ready to rasterise or save. *)
type t

(** [create ?density ~width ~height picture] is a page of [width × height]
    page points holding [picture]. [density] (default [1.]) is the number of
    raster pixels per point; use [1.5] or [2.] for PNG output. *)
val create : ?density:float -> width:float -> height:float -> Hugin_vg.Picture.t -> t

(** [rgba t] is the rendered page as an unsigned 8-bit RGBA tensor of shape
    [[|h; w; 4|]], with [w × h] the page size multiplied by the density and
    rounded to the nearest integer. The first axis points down. Use it to
    post-process the image; {!png} is the encoded form. *)
val rgba : t -> Nx.uint8_t

(** [png t] is the PNG encoding of the page at the frame density. *)
val png : t -> string

(** [svg t] is the SVG encoding of the page. SVG is vector output: the density
    is ignored. *)
val svg : t -> string

(** [pdf t] is the PDF encoding of the page. PDF is vector output: the density
    is ignored. *)
val pdf : t -> string

(** [save_png t path] writes {!png} to [path]. The file is truncated. *)
val save_png : t -> string -> unit

(** [save_svg t path] writes {!svg} to [path]. *)
val save_svg : t -> string -> unit

(** [save_pdf t path] writes {!pdf} to [path]. *)
val save_pdf : t -> string -> unit
