(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* mjc-view: a still of a model, with the overlays a debugging view needs.

   V0 of the viewer plan (VIZ.md): no window, no GPU, no display. It loads a
   compiled model, steps it, and writes a PNG or an SVG. *)

open Mjc

let usage =
  "mjc-view --model M.mjcb [--steps N] [--shot out.png] [--svg out.svg]\n\
  \  [--width W] [--height H] [--density D]\n\
  \  [--eye x,y,z] [--target x,y,z]\n\
  \  [--frames] [--contacts] [--forces] [--inertia] [--com] [--joint-axes]\n\
  \  [--velocity] [--energy] [--quiet]"

let die fmt =
  Stdlib.Printf.ksprintf
    (fun s ->
       Stdlib.Printf.eprintf "mjc-view: %s\n%s\n" s usage;
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
  and steps = ref 0
  and shot = ref "mjc.png"
  and svg = ref None
  and frames = ref false
  and contacts = ref false
  and forces = ref false
  and inertia = ref false
  and com = ref false
  and joint_axes = ref false
  and velocity = ref false
  and energy = ref false
  and drive = ref false
  and plan = ref None
  and quiet = ref false
  and cam = ref Mjc_viz.Render.default_camera in
  let args =
    Array.to_list Sys.argv
    |> function
    | [] -> []
    | _ :: rest -> rest
  in
  let rec go = function
    | [] -> ()
    | "--model" :: v :: rest ->
      model := Some v;
      go rest
    | "--steps" :: v :: rest ->
      steps := Stdlib.int_of_string v;
      go rest
    | "--shot" :: v :: rest ->
      shot := v;
      go rest
    | "--svg" :: v :: rest ->
      svg := Some v;
      go rest
    | "--width" :: v :: rest ->
      cam
      := Mjc_viz.Render.with_size ~width:(Float.of_string v) ~height:!cam.height !cam ();
      go rest
    | "--height" :: v :: rest ->
      cam
      := Mjc_viz.Render.with_size ~width:!cam.width ~height:(Float.of_string v) !cam ();
      go rest
    | "--eye" :: v :: rest ->
      (match vec v with
       | Some p -> cam := { !cam with eye = Some p }
       | None -> die "--eye wants x,y,z");
      go rest
    | "--target" :: v :: rest ->
      (match vec v with
       | Some p -> cam := { !cam with target = Some p }
       | None -> die "--target wants x,y,z");
      go rest
    | "--frames" :: rest ->
      frames := true;
      go rest
    | "--plan" :: file :: rest ->
      plan := Some file;
      go rest
    | "--plan" :: [] -> die "--plan wants a file"
    | "--drive" :: rest ->
      drive := true;
      go rest
    | "--contacts" :: rest ->
      contacts := true;
      go rest
    | "--forces" :: rest ->
      forces := true;
      go rest
    | "--inertia" :: rest ->
      inertia := true;
      go rest
    | "--com" :: rest ->
      com := true;
      go rest
    | "--joint-axes" :: rest ->
      joint_axes := true;
      go rest
    | "--velocity" :: rest ->
      velocity := true;
      go rest
    | "--energy" :: rest ->
      energy := true;
      go rest
    | "--quiet" :: rest ->
      quiet := true;
      go rest
    | ("-h" | "--help") :: _ ->
      Stdlib.print_endline usage;
      Stdlib.exit 0
    | a :: _ -> die "unknown argument %s" a
  in
  go args;
  let path =
    match !model with
    | Some p -> p
    | None -> die "--model is required"
  in
  let m =
    match Mjcb.read path with
    | Ok m -> m
    | Error e -> die "%s" e
  in
  let d = ref (Data.create m) in
  let nu = m.Mjc.Model.topo.nu in
  (* A plan is one control row per engine step, as `mjc_ilqr` writes it: a
     solved trajectory played back into the engine it was solved on. *)
  let plan =
    Option.map
      (fun file ->
         let rows = Stdlib.In_channel.with_open_text file Stdlib.In_channel.input_lines in
         let parse line =
           let toks = String.split_on_char ' ' (String.trim line) in
           Array.of_list
             (List.filter_map (fun tok -> Stdlib.float_of_string_opt tok) toks)
         in
         Array.of_list (List.map parse rows))
      !plan
  in
  for i = 1 to !steps do
    (* A demo with actuators moves on its own: every step takes the scripted
       control for its own time. *)
    if !drive && nu > 0
    then (
      let t = Nx.item [] !d.Mjc.Data.time in
      let u = Mjc_viz.Drive.ctrl Mjc_viz.Drive.default ~nu ~time:t in
      d := { !d with Data.ctrl = Nx.create Nx.float64 [| nu |] u });
    (match plan with
     | Some rows when Array.length rows > 0 ->
       let row = rows.((i - 1) mod Array.length rows) in
       if Array.length row = nu
       then d := { !d with Data.ctrl = Nx.create Nx.float64 [| nu |] row }
     | _ -> ());
    d := Step.step m !d
  done;
  let d = Forward.forward m !d in
  let scene =
    Mjc_viz.Scene.of_state
      ~frames:!frames
      ~contacts:!contacts
      ~forces:!forces
      ~inertia:!inertia
      ~com:!com
      ~joints:!joint_axes
      ~velocity:!velocity
      ~energy:!energy
      m
      d
  in
  if not !quiet
  then (
    let ncon = Stdlib.Int32.to_int (Nx.item [] d.Data.ncon) in
    let nefc = Stdlib.Int32.to_int (Nx.item [] d.Data.nefc) in
    Stdlib.Printf.printf
      "%s: t=%.3f ncon=%d nefc=%d geoms=%d\n%!"
      path
      (Nx.item [] d.Data.time)
      ncon
      nefc
      (List.length scene.Mjc_viz.Scene.geoms));
  Mjc_viz.Render.save_png ~camera:!cam !shot scene;
  (match !svg with
   | None -> ()
   | Some p ->
     let sc = Mjc_viz.Render.scene ~camera:!cam scene in
     let fr =
       Geom_viz.Frame.create
         ~width:!cam.Mjc_viz.Render.width
         ~height:!cam.Mjc_viz.Render.height
         (Geom_viz.Draw.picture_at sc ~time:0.)
     in
     Geom_viz.Frame.save_svg fr p);
  if not !quiet then Stdlib.Printf.printf "wrote %s\n%!" !shot
