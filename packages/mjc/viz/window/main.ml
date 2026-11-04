(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* mjc-window: the interactive viewer. See window.ml and VIZ.md §V1.

   Exit status: 0 for a run that reached its frame count (or the user closing
   the window), 1 for a model that does not load, 2 for a usage error. *)

let usage =
  "mjc-window --model M.mjcb [--steps-per-frame N] [--frames N] [--shot out.png]\n\
  \  [--width W] [--height H] [--eye x,y,z] [--target x,y,z]\n\
  \  [--body-frames] [--contacts|--no-contacts] [--forces] [--no-grid] [--no-hud]\n\
  \  [--inertia] [--com] [--joint-axes] [--velocity] [--energy]\n\
  \  [--log out.mjlog | --follow in.mjlog]\n\
  \  [--bench N] [--check]"

let die fmt =
  Stdlib.Printf.ksprintf
    (fun s ->
       Stdlib.Printf.eprintf "mjc-window: %s\n%s\n" s usage;
       Stdlib.exit 2)
    fmt

let vec s =
  match String.split_on_char ',' s with
  | [ x; y; z ] ->
    (match Float.of_string_opt x, Float.of_string_opt y, Float.of_string_opt z with
     | Some x, Some y, Some z -> Some { Mjc_viz.Scene.x; y; z }
     | _ -> None)
  | _ -> None

let () =
  let model = ref None
  and steps = ref 1
  and frames = ref None
  and check = ref false
  and bench = ref None
  and shot = ref None
  and width = ref 1280
  and height = ref 720
  and eye = ref None
  and target = ref None
  and hud = ref true
  and body_frames = ref false
  and contacts = ref true
  and forces = ref false
  and grid = ref true
  and inertia = ref false
  and com = ref false
  and joint_axes = ref false
  and velocity = ref false
  and energy = ref false
  and log_path = ref None
  and follow_path = ref None in
  let rec go = function
    | [] -> ()
    | "--model" :: v :: r ->
      model := Some v;
      go r
    | "--steps-per-frame" :: v :: r ->
      steps := Stdlib.int_of_string v;
      go r
    | "--frames" :: v :: r ->
      frames := Some (Stdlib.int_of_string v);
      go r
    | "--shot" :: v :: r ->
      shot := Some v;
      go r
    | "--bench" :: v :: r ->
      bench := Some (Stdlib.int_of_string v);
      go r
    | "--width" :: v :: r ->
      width := Stdlib.int_of_string v;
      go r
    | "--height" :: v :: r ->
      height := Stdlib.int_of_string v;
      go r
    | "--eye" :: v :: r ->
      (match vec v with
       | Some p -> eye := Some p
       | None -> die "--eye wants x,y,z");
      go r
    | "--target" :: v :: r ->
      (match vec v with
       | Some p -> target := Some p
       | None -> die "--target wants x,y,z");
      go r
    | "--body-frames" :: r ->
      body_frames := true;
      go r
    | "--contacts" :: r ->
      contacts := true;
      go r
    | "--no-contacts" :: r ->
      contacts := false;
      go r
    | "--forces" :: r ->
      forces := true;
      go r
    | "--inertia" :: r ->
      inertia := true;
      go r
    | "--com" :: r ->
      com := true;
      go r
    | "--joint-axes" :: r ->
      joint_axes := true;
      go r
    | "--velocity" :: r ->
      velocity := true;
      go r
    | "--energy" :: r ->
      energy := true;
      go r
    | "--log" :: v :: r ->
      log_path := Some v;
      go r
    | "--follow" :: v :: r ->
      follow_path := Some v;
      go r
    | "--no-grid" :: r ->
      grid := false;
      go r
    | "--no-hud" :: r ->
      hud := false;
      go r
    | "--check" :: r ->
      check := true;
      go r
    | ("-h" | "--help") :: _ ->
      Stdlib.print_endline usage;
      Stdlib.exit 0
    | a :: _ -> die "unknown argument %s" a
  in
  go (Array.to_list Sys.argv |> List.tl);
  let path =
    match !model with
    | Some p -> p
    | None -> die "--model is required"
  in
  let m =
    match Mjc.Mjcb.read path with
    | Ok m -> m
    | Error e ->
      Stdlib.Printf.eprintf "mjc-window: %s\n" e;
      Stdlib.exit 1
  in
  let view =
    Mjc_viz_window.Window.open_
      ~width:!width
      ~height:!height
      ~hud:!hud
      ~frames:!body_frames
      ~contacts:!contacts
      ~forces:!forces
      ~grid:!grid
      ~inertia:!inertia
      ~com:!com
      ~joint_axes:!joint_axes
      ~velocity:!velocity
      ~energy:!energy
      ?log_path:!log_path
      ?follow_path:!follow_path
      ?eye:!eye
      ?target:!target
      m
  in
  (* The headless check: three frames and a screenshot, then out. A machine
     without a display cannot open a window at all; raylib reports that rather
     than hanging, so the check skips cleanly there. *)
  if !check
  then (
    let shot_path = Filename.temp_file "mjc-check" ".png" in
    match
      Mjc_viz_window.Window.run ~steps_per_frame:!steps ~frames:3 ~shot:shot_path view
      |> ignore
    with
    | () ->
      let png = Stdlib.In_channel.with_open_bin shot_path Stdlib.In_channel.input_all in
      let n = String.length png in
      let magic = n >= 8 && String.sub png 0 8 = "\137PNG\r\n\026\n" in
      Sys.remove shot_path;
      if not magic
      then (
        Stdlib.Printf.printf "check: 3 frames, but the shot is not a PNG (%d bytes)\n" n;
        Stdlib.exit 1);
      Stdlib.Printf.printf "check: 3 frames and a %dx%d PNG ok\n" !width !height
    | exception e ->
      Stdlib.Printf.printf "check: skipped (%s)\n" (Stdlib.Printexc.to_string e);
      Stdlib.exit 0)
  else (
    let frames =
      match !bench with
      | Some n -> Some n
      | None -> !frames
    in
    ignore
      (Mjc_viz_window.Window.run
         ~steps_per_frame:!steps
         ?frames
         ?shot:!shot
         ~bench:(Option.is_some !bench)
         view))
