(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Drawing a {!Scene.t} to a picture.

    The backend here is [geom.viz]'s painter's algorithm, which is exact enough
    for debugging a physics engine and needs no display, no GPU and no
    dependency the repository does not already pin. It is the V0 backend: a
    window comes later.

    Shapes that [Geom_viz.Primitive] does not have are decomposed. A capsule
    is drawn as its axis plus a sphere at each end, which is what a capsule
    *is*; a box as its twelve edges; a cylinder as its two rims and four
    silhouette lines. Edges rather than faces, because a debug view wants to
    see through a solid. *)

(** {1:camera Camera} *)

(** The type for camera settings. [None] fields are derived from the scene's
    bounding box, so a still can be taken with no arguments. *)
type camera =
  { eye : Scene.p3 option
  ; target : Scene.p3 option
  ; up : Scene.p3
  ; fov_y : float
  ; width : float
  ; height : float
  }

(** [default_camera] looks at the scene from front-right-above, with a 45°
    vertical field of view on a 960×540 page. *)
val default_camera : camera

(** [with_size ~width ~height c] is [c] on a page of that many points. *)
val with_size : width:float -> height:float -> camera -> unit -> camera

(** {1:drawing Drawing} *)

(** [scene ?camera s] is the picture of [s] ready to rasterise. *)
val scene : ?camera:camera -> Scene.t -> Geom_viz.Draw.scene

(** [png ?density ?camera s] is the encoded PNG of [s]. [density] defaults to
    [1.5], the pixel per point of a retina-class still. *)
val png : ?density:float -> ?camera:camera -> Scene.t -> string

(** [save_png ?density ?camera path s] writes [png] to [path]. *)
val save_png : ?density:float -> ?camera:camera -> string -> Scene.t -> unit
