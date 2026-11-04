(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The interactive window: V1 of the viewer plan.

    A fixed-step loop that steps a model, draws its scene and lets the mouse
    and the keyboard steer. Everything it draws comes from {!Mjc_viz.Scene},
    so the stills backend and the window show the same picture.

    The key map is the one in VIZ.md §V1: [space] pauses, [.] single-steps, [r]
    resets the state, [c] the camera, [g] the grid, [f] body frames, [n]
    contacts, [x] constraint forces, [i] inertia boxes, [m] centres of mass,
    [j] joint axes, [v] velocities, [e] the energy HUD, [h] the HUD; the digits
    [1]–[9] select a joint, [[] and []] move it, [0] puts it back at its
    reference.

    [log_path] records every drawn frame to a log; [follow_path] replays one
    instead of stepping, with [space] to play or pause, the arrow keys to scrub
    and [home]/[end] to jump. Recording and following are separate: a run that
    does both is writing the frames it reads, which is not a mode anyone wants.
    Right-drag orbits, left-drag pans, the wheel zooms. *)

(** The type for a run: the model, its state, and how to step it. *)
type t

(** [open_ ?width ?height ?eye ?target model] is a viewer for [model], at its
    reference state. The camera defaults to the still backend's view of a unit
    scene; [width] and [height] default to 1280×720. The overlay flags start
    the run with those layers on, which is what a screenshot wants: the same
    switches [mjc-view] takes, so the two backends can be asked for the same
    picture. *)
val open_
  :  ?width:int
  -> ?height:int
  -> ?eye:Mjc_viz.Scene.p3
  -> ?target:Mjc_viz.Scene.p3
  -> ?hud:bool
  -> ?frames:bool
  -> ?contacts:bool
  -> ?forces:bool
  -> ?grid:bool
  -> ?inertia:bool
  -> ?com:bool
  -> ?joint_axes:bool
  -> ?velocity:bool
  -> ?energy:bool
  -> ?log_path:string
  -> ?follow_path:string
  -> Mjc.Model.host
  -> t

(** [run ?steps_per_frame ?frames ?shot ?bench t] is the loop. It returns the
    number of frames drawn, which is zero only if [frames] is zero.

    [steps_per_frame] defaults to [1]. [frames] bounds the loop: unlimited in
    the interactive case, and the headless check and the screenshot use it to
    stop. [shot] is a path to write the last frame to, as a PNG or any other
    extension raylib's image writer knows; it is empty unless [frames] is
    finite, because there is no last frame otherwise. [bench] turns off the
    frame-rate cap and prints the measured frame time, split into scene
    building and everything else. *)
val run : ?steps_per_frame:int -> ?frames:int -> ?shot:string -> ?bench:bool -> t -> int
