(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Cameras and the PGA image plane.

    A camera is a look-at camera with a perspective or orthographic projection.
    Its projection is a projective-geometry construction, not a 4×4 matrix:

    - {!view} is the rigid motion that carries world coordinates into camera
      coordinates;
    - {!eye_point} is the world-space [Pga.t] point of the eye;
    - {!image_plane} is the world-space [Pga.t] plane of the image
      plane;
    - a perspective projection is the meet of the view ray (the line through
      the eye and the world point) with the image plane.

    {2:camera-coordinates Camera coordinates}

    In camera coordinates the eye is at the origin, [+x] points right, [+y]
    points up and [+z] points toward the viewer. The camera looks down [-z] and
    the image plane is [z = -f], where [f = (height / 2) / tan (fov_y / 2)] is
    the focal length in page points. A point [(x, y, z)] with [z < 0] projects
    to [(f x / (-z), f y / (-z))] on the image plane. The meet performs the
    perspective division: {!project} does not divide by hand.

    {2:page Page coordinates}

    Page coordinates are the coordinates the renderer draws in. The origin is
    the top-left corner of the [width × height] page, [x] grows right and [y]
    grows down. [page_x = width / 2 + image_x] and
    [page_y = height / 2 - image_y]. Depths are positive in front of the eye.

    {2:orthographic Orthographic mode}

    With [~ortho:true] the camera keeps the same eye, view motor and image
    plane, but projects by the uniform scale
    [image_x = ortho_scale * x_cam, image_y = ortho_scale * y_cam] instead of
    intersecting the view ray with the image plane. The default [ortho_scale]
    is [height / 4]. [~ortho] is useful for textbook figures and exact GA
    diagrams; {!ray} then returns view lines parallel to the view axis. *)

(** The type for cameras. *)
type t

(** [create ~eye ~target ~up ~fov_y ~width ~height ()] is the look-at camera
    at [eye] looking at [target] with [up] as the up direction.

    [fov_y] is the vertical field of view in radians; [width] and [height] are
    the page size in page points. The image plane is at the focal length
    [height / 2 / tan (fov_y / 2)] in front of the eye.

    [near] (default [0.1]) and [far] (default [1000.]) bound the depth range
    that {!project} accepts, in world units. [ortho] selects the orthographic
    projection (default [false]); [ortho_scale] is its world-to-page scale
    (default [height / 4]).

    Raises [Invalid_argument] if [fov_y] is outside [(0, pi)], if [up] is
    parallel to the view direction, if a size is not finite and positive, or
    if [near] / [far] / [ortho_scale] are not finite and positive with
    [near < far]. *)
val create
  :  ?ortho:bool
  -> ?near:float
  -> ?far:float
  -> ?ortho_scale:float
  -> eye:Primitive.p3
  -> target:Primitive.p3
  -> up:Primitive.p3
  -> fov_y:float
  -> width:float
  -> height:float
  -> unit
  -> t

(** {1:parameters Camera parameters} *)

(** [eye c] is the position of the eye in world coordinates. *)
val eye : t -> Primitive.p3

(** [right c] is the unit world-space camera [x] axis. *)
val right : t -> Primitive.p3

(** [up c] is the unit world-space camera [y] axis. The component of the [up]
    argument along the view direction is removed by {!create}. *)
val up : t -> Primitive.p3

(** [forward c] is the unit world-space direction from the eye toward the
    target, that is, the camera's [-z] axis. *)
val forward : t -> Primitive.p3

(** [focal c] is the focal length in page points. *)
val focal : t -> float

(** [width c] is the page width in page points. *)
val width : t -> float

(** [height c] is the page height in page points. *)
val height : t -> float

(** [near c] is the near distance: points with a camera-space depth at most
    [near] are rejected by {!project}. *)
val near : t -> float

(** [far c] is the far distance: points with a camera-space depth at least
    [far] are rejected by {!project}. *)
val far : t -> float

(** [is_orthographic c] is [true] for an orthographic camera. *)
val is_orthographic : t -> bool

(** [ortho_scale c] is the uniform world-to-page scale of an orthographic
    camera. It is ignored by a perspective camera. *)
val ortho_scale : t -> float

(** {1:pga The camera as a PGA object} *)

(** [view c] is the rigid motion that maps world coordinates into camera
    coordinates: [R (p - eye)] for the rotation [R] whose rows are
    {!right}, {!up} and the negated {!forward}. *)
val view : t -> Geom.Motor.t

(** [eye_point c] is the world-space [Pga.t] point of the eye. It is the point
    shared by every perspective view ray. *)
val eye_point : t -> Geom.Pga.t

(** [image_plane c] is the world-space [Pga.t] plane of the image plane, the
    plane [z_cam = -focal]. It is the image of the local plane
    [Pga.plane3 0. 0. 1. (-.focal)] under the inverse view motor. *)
val image_plane : t -> Geom.Pga.t

(** [ray c ~x ~y] is the world-space [Pga.t] line through the page point
    [(x, y)]. For a perspective camera it is the line through the eye and the
    image point of that page point; for an orthographic camera it is the line
    parallel to the view axis through the page point's image point. The
    direction of the line ([Pga.line_direction]) points from the eye into the
    scene.

    [x] and [y] are page coordinates: points, not pixels, with the origin at
    the top-left corner. Use {!ray} for picking, or to draw a line along a
    view ray. *)
val ray : t -> x:float -> y:float -> Geom.Pga.t

(** {1:projection Projection} *)

(** [project c p] is [Some (x, y, depth)] when the world point [p] is visible:
    the page coordinates of [p] with [y] down, and its depth [(- z_cam)],
    positive in front. It is [None] when the point is behind the eye or
    outside the open depth interval [(near, far)].

    For a perspective camera the projection is computed by the PGA
    construction: [p] is carried to camera coordinates by {!view}, the line
    through the camera-space eye and the point is built with
    [Pga.line_through], and its meet with the local image plane is a point
    whose coordinates are the image coordinates. The orthographic camera uses
    its scale instead.

    {!project_pinhole} computes the same result more cheaply. *)
val project : t -> Primitive.p3 -> (float * float * float) option

(** [project_image c p] is {!project} before the page mapping: the image
    coordinates [x] right and [y] up with the principal point at the origin,
    and the depth. For a perspective camera these are the coordinates of the
    meet with the image plane; for an orthographic camera they are the
    [ortho_scale]-scaled camera coordinates. *)
val project_image : t -> Primitive.p3 -> (float * float * float) option

(** [project_pinhole c p] is the closed-form evaluation of {!project}:
    [x_img = f x_cam / (-z_cam)], [y_img = f y_cam / (-z_cam)], with the
    orthographic scale in orthographic mode. It is mathematically identical to
    {!project} but avoids the multivector allocations, so it is the right
    choice in a per-point loop. The test suite asserts that the two agree. *)
val project_pinhole : t -> Primitive.p3 -> (float * float * float) option
