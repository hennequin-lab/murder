(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scenes, styles and the painter's algorithm.

    A {!scene} is a camera, a background colour and a list of {!entity}
    values. An entity pairs a {!Primitive.t} with a {!style}, a paint layer and
    a motion. {!picture_at} evaluates every motion at a time, projects the
    resulting primitives with {!Camera.project}, sorts them and builds a Hugin
    picture. {!render} wraps that picture in a renderable page.

    {b Depth order.} Lower layer numbers paint first. Inside one layer,
    entities are sorted by depth and painted far to near. Depth is one number
    per primitive, taken at a representative point, so the algorithm is a
    painter's algorithm: intersecting surfaces and lines through solids can
    sort incorrectly. Split an object into pieces when that matters. *)

(** The type for a colour. This is Hugin's colour type, in linear RGB. *)
type color = Hugin_gg.Color.t

(** The type for styles. *)
type style =
  { fill : color option (** interior colour; [None] draws no interior *)
  ; stroke : color option (** outline colour; [None] draws no outline *)
  ; width : float (** outline width in page points *)
  ; dash : float list (** dash pattern in page points; [[]] is a solid line *)
  ; opacity : float (** opacity in [[0, 1]] applied to the filled and stroked result *)
  }

(** [default] is a blue fill with a dark outline and a 1.5 point stroke. *)
val default : style

(** [solid ?alpha c] is a fill of colour [c] with no outline. [alpha] defaults
    to [1.]. *)
val solid : ?alpha:float -> color -> style

(** [outline ?width ?dash c] is an outline of colour [c] with no fill.
    [width] defaults to [1.5] and [dash] to [[]]. *)
val outline : ?width:float -> ?dash:float list -> color -> style

(** The type for an entity: a primitive, how to paint it, when it paints and
    how it moves. *)
type entity

(** The type for a scene: a camera, a background and a list of entities. *)
type scene

(** [static ?layer ?style p] is the primitive [p] at rest. [layer] defaults
    to [1] and [style] to {!default}. *)
val static : ?layer:int -> ?style:style -> Primitive.t -> entity

(** [moving ?layer ?style ~motion p] is [p] transformed by the rigid motion
    [motion t] at time [t]. [motion] is evaluated once per picture. *)
val moving
  :  ?layer:int
  -> ?style:style
  -> motion:(float -> Geom.Motor.t)
  -> Primitive.t
  -> entity

(** [moving_with ?layer ?style ~motion p] is [p] replaced by
    [motion t p] at time [t]. It is the hook for recomputing an object per
    frame, as the kinematic-chain demo does, instead of transforming a fixed
    one. *)
val moving_with
  :  ?layer:int
  -> ?style:style
  -> motion:(float -> Primitive.t -> Primitive.t)
  -> Primitive.t
  -> entity

(** [create ?background camera entities] is the scene drawn with [camera] and
    the given entities. [background] defaults to a near-white
    [(0.98, 0.98, 0.98)]. *)
val create : ?background:color -> Camera.t -> entity list -> scene

(** [add e s] is [s] with [e] appended to its entity list. *)
val add : entity -> scene -> scene

(** [camera s] is the camera of [s]. *)
val camera : scene -> Camera.t

(** [background s] is the background colour of [s]. *)
val background : scene -> color

(** [picture_at s ~time] draws [s] at [time] and returns the page-sized Hugin
    picture. The page is [Camera.width camera × Camera.height camera] points.
    The picture contains the background rectangle and every visible entity,
    already ordered by layer and depth. *)
val picture_at : scene -> time:float -> Hugin_vg.Picture.t

(** [render s ~time] is {!picture_at} wrapped in a
    [Hugin_vg.Renderable.v width height] value. Use {!Frame.create} to attach a
    raster density and save it. *)
val render : scene -> time:float -> Hugin_vg.Renderable.t
