(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
module Format = Stdlib.Format

let magic_header = "MJLOG\000\000\000"
let version = 1
let header_bytes = 96
let name_bytes = 64

(* {1 Layout} *)

let stride_of ~ngeom ~nconmax = 4 + (12 * ngeom) + (8 * nconmax)

(* Offsets inside one record, in floats. *)
let off_step = 0
let off_time = 1
let off_ncon = 2
let off_nefc = 3
let off_geom = 4
let off_contact_pos ~ngeom = off_geom + (12 * ngeom)
let off_contact_normal ~ngeom ~nconmax = off_contact_pos ~ngeom + (3 * nconmax)
let off_contact_dist ~ngeom ~nconmax = off_contact_normal ~ngeom ~nconmax + (3 * nconmax)
let off_contact_force ~ngeom ~nconmax = off_contact_dist ~ngeom ~nconmax + nconmax

type frame =
  { step : int64
  ; time : float
  ; ncon : int
  ; nefc : int
  ; geom : float array
  ; contact_pos : float array
  ; contact_normal : float array
  ; contact_dist : float array
  ; contact_force : float array
  }

(* {1 Records as bytes}

   One record is [stride] little-endian doubles in a [Bytes.t]: the buffer the
   writer fills, and the one the reader gets back. *)

let put f buf i = Stdlib.Bytes.set_int64_le buf (8 * i) (Stdlib.Int64.bits_of_float f)
let get buf i = Stdlib.Int64.float_of_bits (Stdlib.Bytes.get_int64_le buf (8 * i))

let header ~ngeom ~nconmax ~stride ~name =
  let b = Stdlib.Bytes.make header_bytes '\000' in
  Stdlib.Bytes.blit_string magic_header 0 b 0 8;
  Stdlib.Bytes.set_int32_le b 8 (Stdlib.Int32.of_int version);
  Stdlib.Bytes.set_int32_le b 12 (Stdlib.Int32.of_int ngeom);
  Stdlib.Bytes.set_int32_le b 16 (Stdlib.Int32.of_int nconmax);
  Stdlib.Bytes.set_int32_le b 20 (Stdlib.Int32.of_int stride);
  Stdlib.Bytes.set_int32_le b 24 0l;
  let n = Int.min (String.length name) (name_bytes - 1) in
  Stdlib.Bytes.blit_string (String.prefix name n) 0 b 28 n;
  b

(* {1 Writing} *)

type writer =
  { model : Mjc.Model.host
  ; oc : Stdlib.Out_channel.t
  ; buf : Bytes.t
  ; mutable closed : bool
  }

let create ~path (model : Mjc.Model.host) =
  let t = model.topo in
  let stride = stride_of ~ngeom:t.ngeom ~nconmax:t.nconmax in
  let oc = Stdlib.Out_channel.open_bin path in
  Stdlib.Out_channel.output_bytes
    oc
    (header ~ngeom:t.ngeom ~nconmax:t.nconmax ~stride ~name:t.name);
  { model; oc; buf = Stdlib.Bytes.create (8 * stride); closed = false }

(* The contact force: the sum of the rows that belong to each contact row
   block. Rows are tagged by type and id, so this does not have to assume the
   order the assembler wrote them in. *)
let contact_forces (d : Mjc.Data.host) ~ncon ~nconmax =
  let out = Array.create ~len:nconmax 0. in
  let nefc = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.nefc) in
  if nefc > 0
  then (
    let types = Nx.to_array d.Mjc.Data.efc_type
    and ids = Nx.to_array d.Mjc.Data.efc_id
    and force = Nx.to_array d.Mjc.Data.efc_force in
    for r = 0 to nefc - 1 do
      (* contact_frictionless = 5, contact_pyramidal = 6 *)
      let ty = Stdlib.Int32.to_int types.(r) in
      if ty = 5 || ty = 6
      then (
        let c = Stdlib.Int32.to_int ids.(r) in
        if c >= 0 && c < Stdlib.Array.length out then out.(c) <- out.(c) +. force.(r))
    done);
  Stdlib.Array.sub out 0 ncon

let append w ~step (d : Mjc.Data.host) =
  if w.closed then Stdlib.invalid_arg "Mjc_viz.Log.append: the log is closed";
  let t = w.model.topo in
  let ngeom = t.ngeom
  and nconmax = t.nconmax in
  let ncon = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.ncon)
  and nefc = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.nefc) in
  let geom_pos = Nx.to_array d.Mjc.Data.geom_xpos
  and geom_mat = Nx.to_array d.Mjc.Data.geom_xmat
  and cpos = Nx.to_array d.Mjc.Data.contact_pos
  and cnormal = Nx.to_array d.Mjc.Data.contact_normal
  and cdist = Nx.to_array d.Mjc.Data.contact_dist in
  put (Stdlib.Int64.to_float step) w.buf off_step;
  put (Nx.item [] d.Mjc.Data.time) w.buf off_time;
  put (Float.of_int ncon) w.buf off_ncon;
  put (Float.of_int nefc) w.buf off_nefc;
  for i = 0 to ngeom - 1 do
    for k = 0 to 2 do
      put geom_pos.((3 * i) + k) w.buf (off_geom + (12 * i) + k)
    done;
    for k = 0 to 8 do
      put geom_mat.((9 * i) + k) w.buf (off_geom + (12 * i) + 3 + k)
    done
  done;
  let pos0 = off_contact_pos ~ngeom
  and nrm0 = off_contact_normal ~ngeom ~nconmax
  and dist0 = off_contact_dist ~ngeom ~nconmax
  and force0 = off_contact_force ~ngeom ~nconmax in
  let forces = contact_forces d ~ncon ~nconmax in
  for c = 0 to nconmax - 1 do
    for k = 0 to 2 do
      put cpos.((3 * c) + k) w.buf (pos0 + (3 * c) + k);
      put cnormal.((3 * c) + k) w.buf (nrm0 + (3 * c) + k)
    done;
    put cdist.(c) w.buf (dist0 + c)
  done;
  for c = 0 to ncon - 1 do
    put forces.(c) w.buf (force0 + c)
  done;
  Stdlib.Out_channel.output_bytes w.oc w.buf

let close w =
  if not w.closed
  then (
    w.closed <- true;
    Stdlib.Out_channel.close w.oc)

(* {1 Reading} *)

type t =
  { fd : Unix.file_descr
  ; ngeom : int
  ; nconmax : int
  ; stride : int
  ; name : string
  ; frames : int
  }

let read_exactly fd ~buf ~len =
  let got = ref 0 in
  while !got < len do
    let n = Unix.read fd buf !got (len - !got) in
    if n = 0 then raise Stdlib.End_of_file;
    got := !got + n
  done

let open_ path =
  let fd =
    match Unix.openfile path [ Unix.O_RDONLY ] 0 with
    | fd -> Ok fd
    | exception Unix.Unix_error (e, _, _) ->
      Error (Stdlib.Printf.sprintf "%s: %s" path (Unix.error_message e))
  in
  match fd with
  | Error e -> Error e
  | Ok fd ->
    let fail message =
      Unix.close fd;
      Error message
    in
    let header = Stdlib.Bytes.create header_bytes in
    (match read_exactly fd ~buf:header ~len:header_bytes with
     | exception Stdlib.End_of_file -> fail (Stdlib.Printf.sprintf "%s: no header" path)
     | () ->
       let u32 i = Stdlib.Int32.to_int (Stdlib.Bytes.get_int32_le header (4 * i)) in
       let ngeom = u32 3
       and nconmax = u32 4
       and stride = u32 5 in
       if not (String.equal (Stdlib.Bytes.sub_string header 0 8) magic_header)
       then fail (Stdlib.Printf.sprintf "%s: not a log (bad magic)" path)
       else if u32 2 <> version
       then
         fail
           (Stdlib.Printf.sprintf
              "%s: log version %d, this reader knows %d"
              path
              (u32 2)
              version)
       else if stride <> stride_of ~ngeom ~nconmax
       then
         fail
           (Stdlib.Printf.sprintf
              "%s: stride %d, but %d geoms and %d contact slots need %d"
              path
              stride
              ngeom
              nconmax
              (stride_of ~ngeom ~nconmax))
       else (
         let name =
           let raw = Stdlib.Bytes.sub_string header 28 name_bytes in
           match String.index raw '\000' with
           | Some i -> String.prefix raw i
           | None -> raw
         in
         let size = (Unix.LargeFile.stat path).Unix.LargeFile.st_size in
         let body = Stdlib.Int64.sub size (Stdlib.Int64.of_int header_bytes) in
         let record = Stdlib.Int64.of_int (8 * stride) in
         if not (Stdlib.Int64.equal (Stdlib.Int64.rem body record) Stdlib.Int64.zero)
         then
           fail
             (Stdlib.Printf.sprintf
                "%s: %Ld bytes after the header is not whole %Ld-byte records"
                path
                body
                record)
         else
           Ok
             { fd
             ; ngeom
             ; nconmax
             ; stride
             ; name
             ; frames = Stdlib.Int64.to_int (Stdlib.Int64.div body record)
             }))

let length t = t.frames
let model_shape t = t.ngeom, t.nconmax
let name t = t.name

let frame t i =
  if i < 0 || i >= t.frames
  then
    Stdlib.invalid_arg
      (Stdlib.Printf.sprintf "Mjc_viz.Log.frame: %d is not in [0, %d)" i t.frames);
  let buf = Stdlib.Bytes.create (8 * t.stride) in
  ignore
    (Unix.LargeFile.lseek
       t.fd
       (Stdlib.Int64.of_int (header_bytes + (i * 8 * t.stride)))
       Unix.SEEK_SET);
  read_exactly t.fd ~buf ~len:(8 * t.stride);
  let ncon = Float.to_int (get buf off_ncon)
  and nefc = Float.to_int (get buf off_nefc) in
  let ngeom = t.ngeom
  and nconmax = t.nconmax in
  let geom = Array.init (12 * ngeom) ~f:(fun k -> get buf (off_geom + k)) in
  let pos0 = off_contact_pos ~ngeom
  and nrm0 = off_contact_normal ~ngeom ~nconmax
  and dist0 = off_contact_dist ~ngeom ~nconmax
  and force0 = off_contact_force ~ngeom ~nconmax in
  let contact_pos = Array.init (3 * ncon) ~f:(fun k -> get buf (pos0 + k))
  and contact_normal = Array.init (3 * ncon) ~f:(fun k -> get buf (nrm0 + k))
  and contact_dist = Array.init ncon ~f:(fun k -> get buf (dist0 + k))
  and contact_force = Array.init ncon ~f:(fun k -> get buf (force0 + k)) in
  { step = Stdlib.Int64.of_float (get buf off_step)
  ; time = get buf off_time
  ; ncon
  ; nefc
  ; geom
  ; contact_pos
  ; contact_normal
  ; contact_dist
  ; contact_force
  }

let close_reader t = Unix.close t.fd

let scene (m : Mjc.Model.host) t i =
  let t_model = m.topo in
  if not (Int.equal t.ngeom t_model.ngeom && Int.equal t.nconmax t_model.nconmax)
  then
    Stdlib.invalid_arg
      (Stdlib.Printf.sprintf
         "Mjc_viz.Log.scene: the log has %d geoms and %d slots, the model %d and %d"
         t.ngeom
         t.nconmax
         t_model.ngeom
         t_model.nconmax);
  let f = frame t i in
  let contacts =
    List.init f.ncon ~f:(fun c ->
      Scene.
        { point =
            { x = f.contact_pos.(3 * c)
            ; y = f.contact_pos.((3 * c) + 1)
            ; z = f.contact_pos.((3 * c) + 2)
            }
        ; normal =
            { x = f.contact_normal.(3 * c)
            ; y = f.contact_normal.((3 * c) + 1)
            ; z = f.contact_normal.((3 * c) + 2)
            }
        ; depth = f.contact_dist.(c)
        ; dim = 0
        })
  in
  Scene.of_poses m ~time:f.time ~geom:f.geom ~contacts

let pp ppf t =
  Format.fprintf
    ppf
    "@[<v>log %s@,frames %d@,geoms %d@,contact slots %d@,stride %d@]"
    t.name
    t.frames
    t.ngeom
    t.nconmax
    t.stride
