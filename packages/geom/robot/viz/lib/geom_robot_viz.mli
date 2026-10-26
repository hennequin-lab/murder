(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Robot scenes for [geom-viz].

    A robot model is a tree of links with named frames; a picture of one is a
    list of [geom-viz] {!Geom_viz.Draw.entity} values, one per link, joint and
    marker. This module is the whole of the second step: it reads poses from
    {!Geom_robot.Robot} — batched [link_motors] calls, never a loop over the
    batch — and hands the renderer nothing but {!Geom_viz.Primitive.t} values.
    It knows no algebra of its own.

    {b Coordinates.} Everything here is host-side and eager: the renderer is a
    scalar CPU renderer, so a picture of a robot is a handful of unbatched
    [Robot] calls whose results are read into {!Geom_viz.Primitive.p3}. That is
    the deliberate boundary — the tensor work stays in [geom_robot], and only
    the numbers cross.

    {[
      let model = Geom_robot.Robot.planar2 () in
      let t = Geom_robot_viz.make model in
      let q = Nx.create Nx.float32 [| 2 |] [| 0.4; 0.8 |] in
      let scene =
        Geom_viz.Draw.create
          (Geom_robot_viz.camera ())
          (Geom_robot_viz.stage t :: Geom_robot_viz.entities t q)
      in
      (* Geom_viz.Frame.save_png (Geom_viz.Frame.create picture) "robot.png" *)
    ]}

    {b What a robot looks like here.} Links are thick segments between
    consecutive joint origins, joints are spheres, the base is a disc, frames
    are triads and a pose target is a dashed circle across its approach axis.
    There are no meshes and no capsules: the point is to see the kinematics, and
    a stick figure shows a Jacobian's geometry better than a mesh does. *)

(** {1 Styles} *)

(** The styles of the parts of a robot. Each field is a
    {!Geom_viz.Draw.style}. *)
type style =
  { link : Geom_viz.Draw.style
  ; joint : Geom_viz.Draw.style
  ; base : Geom_viz.Draw.style
  ; tool : Geom_viz.Draw.style
  ; target : Geom_viz.Draw.style
  ; trail : Geom_viz.Draw.style
  }

(** [default_style] is a dark slate stick figure with amber joints, a violet
    dashed target and a matching trail: readable on a light page, and
    distinguishable in greyscale. *)
val default_style : style

(** {1 Robots} *)

(** The type for a drawable robot: a model, the styles of its parts and the
    sizes of its markers. *)
type t

(** [make ?style ?joint_radius ?base_radius model] is the drawable robot for
    [model]. [joint_radius] (default [0.07]) scales the spheres: the root's is
    [1.4] times it and the others are exactly it. [base_radius] (default
    [0.22]) is the radius of the disc under the root. *)
val make
  :  ?style:style
  -> ?joint_radius:float
  -> ?base_radius:float
  -> Geom_robot.Robot.t
  -> t

(** [model t] is the model [t] draws. *)
val model : t -> Geom_robot.Robot.t

(** {1 One configuration} *)

(** [links t q] are the link segments, one between each pair of consecutive
    joint origins, of shape [[nlinks - 1]]. *)
val links : t -> Geom_robot.Robot.q -> Geom_viz.Primitive.t list

(** [joints t q] are the joint spheres, one per link, the root's larger. *)
val joints : t -> Geom_robot.Robot.q -> Geom_viz.Primitive.t list

(** [base t] is the disc drawn under the root link. *)
val base : t -> Geom_viz.Primitive.t

(** [tool t ~frame ?len q] is the triad of the named frame, with axes of length
    [len] (default [0.16]). *)
val tool
  :  t
  -> frame:Geom_robot.Robot.frame
  -> ?len:float
  -> Geom_robot.Robot.q
  -> Geom_viz.Primitive.t

(** [trail t ~frame ?len qs] is the polyline the named frame's origin traces
    over the configurations [qs]. *)
val trail
  :  t
  -> frame:Geom_robot.Robot.frame
  -> Geom_robot.Robot.q array
  -> Geom_viz.Primitive.t

(** [target ?radius pose] is a pose marker: a dashed circle of [radius]
    (default [0.13]) across the approach axis, a short dash along it, and a dot
    at the origin, all carried by [pose]. *)
val target : ?radius:float -> Geom.Motor.t -> Geom_viz.Primitive.t list

(** {1 Scenes} *)

(** [entities t q] are the entities of the robot at [q]: links, joints and the
    base, with the styles and paint layers of [t]. *)
val entities : ?layer:int -> t -> Geom_robot.Robot.q -> Geom_viz.Draw.entity list

(** [animated t ~q frame draw] are the entities of the robot following the
    configurations [q time], with the frame, trail and target markers the
    optional arguments ask for.

    [~frame] adds the named frame's triad; [~trail] adds the path its origin
    takes over the whole configuration array, drawn up to the current frame;
    [~target] adds a pose marker; [~fps] (default [24]) is the frame rate at
    which [time] selects an index into the trail. [~layer] is the base layer
    (default [2] for links, with joints above and markers above those). *)
val animated
  :  ?layer:int
  -> t
  -> q:(float -> Geom_robot.Robot.q)
  -> ?frame:Geom_robot.Robot.frame
  -> ?trail:Geom_robot.Robot.q array
  -> ?target:Geom.Motor.t
  -> ?fps:int
  -> unit
  -> Geom_viz.Draw.entity list

(** {1 Cameras and paths} *)

(** [camera ?eye ?target ?up ?size ()] is a perspective camera looking at the
    robot from [eye] (default [(3., -3.4, 2.1)]) at [target] (default
    [(0., 0., 1.1)]) with a vertical field of view of [pi / 4]. [size] is the
    page size in points and defaults to [(800., 620.)]. *)
val camera
  :  ?eye:Geom_viz.Primitive.p3
  -> ?target:Geom_viz.Primitive.p3
  -> ?up:Geom_viz.Primitive.p3
  -> ?size:float * float
  -> unit
  -> Geom_viz.Camera.t

(** [linear_path from to_ ~n] is the joint-space interpolation from [from] to
    [to_] in [n] steps, as a function of the step index. It is the animation
    path a demo wants when the interesting thing is the pose, not the timing. *)
val linear_path
  :  Geom_robot.Robot.q
  -> Geom_robot.Robot.q
  -> n:int
  -> int
  -> Geom_robot.Robot.q

(** [sample ?fps qs time] is the configuration [qs] holds at [time], with
    [time] converted to an index at [fps] (default [24]) and clamped to the
    array. It is the [q] argument {!animated} wants for a precomputed path. *)
val sample : ?fps:int -> Geom_robot.Robot.q array -> float -> Geom_robot.Robot.q

(** [stage ?radius t] is a ground plane under the robot, of half-width [radius]
    (default [2.4]). *)
val stage : ?radius:float -> t -> Geom_viz.Draw.entity
