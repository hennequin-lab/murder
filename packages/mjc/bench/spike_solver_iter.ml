(* How many solver iterations does a model actually need, and does the
   gradient notice the budget? *)

open Base
module Printf = Stdlib.Printf
module Nxi32 = Stdlib.Int32

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

let with_budget m b =
  { m with Mjc.Model.params = { m.Mjc.Model.params with iterations = Nx.scalar f64 b } }

(* A state the engine itself reaches: step from rest with a velocity, so the
   quaternions are the ones the integrator produced rather than ones a spike
   invented. *)
let reached_state m =
  let d0 = Mjc.Data.create m in
  let nv = Array.fold (Nx.shape d0.qvel) ~init:1 ~f:( * ) in
  let qvel =
    Nx.create
      f64
      (Nx.shape d0.qvel)
      (Array.init nv ~f:(fun i -> 0.6 *. Float.sin (Float.of_int (3 * (i + 1)))))
  in
  let d = ref { d0 with qvel } in
  for _ = 1 to 12 do
    d := Mjc.Step.step m !d
  done;
  !d

let () =
  List.iter
    [ "limited"; "looped"; "stiff"; "finger"; "bounce"; "blocks"; "boxes2" ]
    ~f:(fun name ->
      let m = load name in
      let d0 = reached_state m in
      let cap = Float.of_int Mjc.Solver.max_iterations in
      Printf.printf
        "%-8s ncon %3d  cap %g\n%!"
        name
        (Nxi32.to_int (Nx.item [] d0.Mjc.Data.ncon))
        cap;
      (* The reference is the cap itself, so a budget that has not settled by
         then shows up as a difference at [cap - 2], not as one against a
         reference that cannot be reached. *)
      let forces b =
        let m = with_budget m b in
        let d = Mjc.Forward.forward m d0 in
        Nx.to_array d.Mjc.Data.efc_force
      in
      let ref_forces = forces cap in
      let diff a b =
        Array.fold2_exn a b ~init:0.0 ~f:(fun acc x y ->
          Float.max acc (Float.abs (x -. y)))
      in
      List.iter
        [ 1.0; 2.0; 3.0; 4.0; cap -. 2.0; cap -. 1.0 ]
        ~f:(fun b ->
          if Float.( <= ) b cap
          then
            Printf.printf
              "  budget %-5g |f - f(%g)| %.3e\n%!"
              b
              cap
              (diff (forces b) ref_forces)))
