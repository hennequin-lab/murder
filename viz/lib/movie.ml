(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type spec =
  { duration : float
  ; fps : int
  ; scene : Draw.scene
  ; density : float
  }

let frame_count spec = max 1 (int_of_float (spec.duration *. float_of_int spec.fps))

let frame_at spec i =
  let time = float_of_int i /. float_of_int spec.fps in
  let picture = Draw.picture_at spec.scene ~time in
  let camera = Draw.camera spec.scene in
  Frame.create
    ~density:spec.density
    ~width:(Camera.width camera)
    ~height:(Camera.height camera)
    picture

let render_png_sequence spec ~dir =
  if Sys.command ("mkdir -p " ^ Filename.quote dir) <> 0
  then failwith ("Movie: cannot create " ^ dir);
  for i = 0 to frame_count spec - 1 do
    let frame = frame_at spec i in
    let path = Printf.sprintf "%s/frame-%06d.png" dir i in
    Frame.save_png frame path
  done

let rec remove_tree dir =
  if Sys.is_directory dir
  then (
    Array.iter (fun name -> remove_tree (Filename.concat dir name)) (Sys.readdir dir);
    Unix.rmdir dir)
  else Sys.remove dir

let to_mp4 ?(crf = 18) ?frames_dir spec ~path =
  let temporary, dir =
    match frames_dir with
    | Some d -> false, d
    | None -> true, Filename.temp_dir "geom_viz" ""
  in
  render_png_sequence spec ~dir;
  let cmd =
    Printf.sprintf
      "ffmpeg -loglevel error -y -framerate %d -i %s/frame-%%06d.png -c:v libx264 \
       -pix_fmt yuv420p -crf %d %s"
      spec.fps
      (Filename.quote dir)
      crf
      (Filename.quote path)
  in
  let status = Sys.command cmd in
  if temporary then remove_tree dir;
  if status <> 0 then failwith "Movie.to_mp4: ffmpeg failed"
