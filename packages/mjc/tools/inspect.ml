(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* mjc-inspect: what a .mjcb contains, and where its forward kinematics puts
   the frames.

   mjc-inspect MODEL.mjcb
   mjc-inspect MODEL.mjcb --fk          forward kinematics at qpos0
   mjc-inspect MODEL.mjcb --qpos 0.1 -0.2 ...
*)

open Base
module Printf = Stdlib.Printf
module Format = Stdlib.Format

let usage () =
  Printf.printf
    "usage: mjc-inspect MODEL.mjcb [--fk] [--qpos Q...] [--shape-key]\n\n\
     --fk         run forward kinematics and print the frames\n\
     --qpos Q...  set the position coordinates first (default: qpos0)\n\
     --shape-key  print only the compile-time shape key\n";
  Stdlib.exit 2

let print_shapes (m : Mjc.Model.host) =
  let rows =
    [ "body_pos", m.params.body_pos
    ; "body_quat", m.params.body_quat
    ; "body_ipos", m.params.body_ipos
    ; "body_iquat", m.params.body_iquat
    ; "body_mass", m.params.body_mass
    ; "body_inertia", m.params.body_inertia
    ; "jnt_axis", m.params.jnt_axis
    ; "jnt_pos", m.params.jnt_pos
    ; "jnt_range", m.params.jnt_range
    ; "qpos0", m.params.qpos0
    ; "geom_pos", m.params.geom_pos
    ; "geom_quat", m.params.geom_quat
    ; "geom_size", m.params.geom_size
    ; "geom_rgba", m.params.geom_rgba
    ; "site_pos", m.params.site_pos
    ; "gravity", m.params.gravity
    ; "timestep", m.params.timestep
    ]
  in
  List.iter rows ~f:(fun (name, x) ->
    Printf.printf
      "  %-12s [%s]\n"
      name
      (String.concat ~sep:";" (Array.to_list (Array.map (Nx.shape x) ~f:Int.to_string))))

let print_matrix label (x : Mjc.Data.host) getter rows cols =
  let a = Nx.to_array (getter x) in
  Printf.printf "  %s:\n" label;
  for r = 0 to rows - 1 do
    Printf.printf "   ";
    for c = 0 to cols - 1 do
      Printf.printf " %9.5f" a.((r * cols) + c)
    done;
    Printf.printf "\n"
  done

let () =
  let args = Array.to_list Stdlib.Sys.argv |> List.tl_exn in
  match args with
  | [] -> usage ()
  | path :: rest ->
    let m =
      match Mjc.Mjcb.read path with
      | Ok m -> m
      | Error e ->
        Printf.eprintf "%s\n" e;
        Stdlib.exit 1
    in
    (match Mjc.Model.validate m with
     | Ok () -> ()
     | Error e ->
       Printf.eprintf "%s\n" e;
       Stdlib.exit 1);
    let shape_only = List.mem rest "--shape-key" ~equal:String.equal in
    Format.printf "%a@." Mjc.Topo.pp m.topo;
    Format.printf "shape key: %a@." Mjc.Model.pp_shape_key (Mjc.Model.shape_key m);
    if not shape_only
    then (
      Printf.printf "parameters:\n";
      print_shapes m;
      let qpos =
        match List.findi rest ~f:(fun _ s -> String.equal s "--qpos") with
        | None -> Nx.to_array m.params.qpos0
        | Some (i, _) ->
          let values =
            List.drop rest (i + 1)
            |> List.take_while ~f:(fun s -> not (String.is_prefix s ~prefix:"--"))
            |> List.map ~f:Float.of_string
          in
          if List.length values <> m.topo.nq
          then (
            Printf.eprintf
              "--qpos has %d values, the model has %d\n"
              (List.length values)
              m.topo.nq;
            Stdlib.exit 2);
          Array.of_list values
      in
      if List.mem rest "--fk" ~equal:String.equal || Array.length qpos > 0
      then (
        let d = Mjc.Data.create m in
        let d = { d with qpos = Nx.create Nx.float64 [| m.topo.nq |] qpos } in
        let d = Mjc.Kinematics.forward m d in
        Printf.printf "frames at qpos =";
        Array.iter qpos ~f:(fun v -> Printf.printf " %.6g" v);
        Printf.printf "\n";
        List.iteri (Array.to_list m.topo.body_names) ~f:(fun b name ->
          Printf.printf "  body %d %S\n" b name;
          print_matrix "xpos" d (fun d -> Nx.get [ b ] d.xpos) 1 3;
          print_matrix "xquat" d (fun d -> Nx.get [ b ] d.xquat) 1 4)))
