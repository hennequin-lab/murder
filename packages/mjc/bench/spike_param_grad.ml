(* Is the step differentiable with respect to a model parameter? *)

open Base
module Printf = Stdlib.Printf

let f64 = Nx.float64

let load name =
  let candidate d =
    let path = d ^ "/" ^ name ^ ".mjcb" in
    Option.some_if (Stdlib.Sys.file_exists path) path
  in
  let path =
    match
      List.find_map [ "test/fixtures"; "packages/mjc/test/fixtures" ] ~f:candidate
    with
    | Some p -> p
    | None -> failwith (Printf.sprintf "no fixture %s" name)
  in
  match Mjc.Mjcb.read path with
  | Ok m -> m
  | Error e -> failwith e

let () =
  let name = if Array.length Stdlib.Sys.argv > 1 then Stdlib.Sys.argv.(1) else "chain6" in
  let m0 = load name in
  let d0 = Mjc.Data.create m0 in
  (* The compiled closure: its structural reads happen once, at compile time,
     and the program inside is straight-line tensor code. *)
  let loss (m : Mjc.Model.host) =
    let step = Mjc.Step.compile m in
    let d = ref d0 in
    for _ = 1 to 5 do
      d := Mjc.Step.run step !d
    done;
    Nx.sum (Nx.mul !d.qpos !d.qpos)
  in
  match Rune.vjp Mjc.Model.ptree Nx.Ptree.tensor loss m0 with
  | exception exn -> Printf.printf "%s: vjp raised: %s\n%!" name (Exn.to_string exn)
  | value, pullback ->
    let g = pullback (Nx.scalar f64 1.0) in
    let parts =
      [ "body_mass", g.Mjc.Model.params.Mjc.Model.Params.body_mass
      ; "jnt_stiffness", g.Mjc.Model.params.Mjc.Model.Params.jnt_stiffness
      ; "dof_damping", g.Mjc.Model.params.Mjc.Model.Params.dof_damping
      ; "geom_size", g.Mjc.Model.params.Mjc.Model.Params.geom_size
      ]
    in
    Printf.printf
      "%s: loss %.12g, gradient of:\n%!"
      name
      (Nx.item [] (Nx.mul value (Nx.scalar f64 1.0)));
    List.iter parts ~f:(fun (k, x) ->
      let a = Nx.to_array x in
      let mx = Array.fold a ~init:0.0 ~f:(fun acc v -> Float.max acc (Float.abs v)) in
      let nan = Array.count a ~f:Float.is_nan in
      Printf.printf
        "  %-14s %3d values, %d NaN, max |g| %.6g\n%!"
        k
        (Array.length a)
        nan
        mx)
