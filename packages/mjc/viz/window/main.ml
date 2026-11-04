(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* mjc-window: the interactive viewer. See window.ml and raylib.md. *)

let usage = "mjc-window --model M.mjcb [--steps-per-frame N] [--frames N] [--check]"

let () =
  let model = ref None
  and steps = ref 1
  and frames = ref None
  and check = ref false in
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
    | "--check" :: r ->
      check := true;
      frames := Some 3;
      go r
    | ("-h" | "--help") :: _ ->
      Stdlib.print_endline usage;
      Stdlib.exit 0
    | a :: _ ->
      Stdlib.Printf.eprintf "mjc-window: unknown argument %s\n%s\n" a usage;
      Stdlib.exit 2
  in
  go (Array.to_list Sys.argv |> List.tl);
  let path =
    match !model with
    | Some p -> p
    | None ->
      Stdlib.Printf.eprintf "mjc-window: --model is required\n%s\n" usage;
      Stdlib.exit 2
  in
  let m =
    match Mjc.Mjcb.read path with
    | Ok m -> m
    | Error e ->
      Stdlib.Printf.eprintf "mjc-window: %s\n" e;
      Stdlib.exit 1
  in
  let view = Mjc_viz.Window.open_ m in
  if !check
  then (
    (* The headless check: three frames, then out. A machine without a display
       cannot open a window, and raylib reports that rather than hanging, so
       this exits cleanly there too. *)
    match Mjc_viz.Window.run ~steps_per_frame:!steps ~frames:3 view with
    | () -> Stdlib.Printf.printf "check: 3 frames ok\n"
    | exception e ->
      Stdlib.Printf.printf "check: skipped (%s)\n" (Stdlib.Printexc.to_string e);
      Stdlib.exit 0)
  else Mjc_viz.Window.run ~steps_per_frame:!steps ?frames:!frames view
