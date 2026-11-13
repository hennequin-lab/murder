(* What does stopping the solver on its KKT condition cost, and what does it
   buy?

   Two questions, one program. The cost is the difference in the solved forces
   against the fixed schedule (`~stop:0.`, the schedule the parity corpus was
   recorded against); the buy is the wall clock of a whole step. Both are per
   model and per state, because the schedule's waste is not uniform: a model
   whose rows settle in one round pays two and a half milliseconds for nothing,
   and a model with a badly conditioned contact state needs the rounds it asks
   for.

   Run from the repository root:

   dune exec packages/mjc/bench/spike_stop.exe

   `doc/mujoco-4.0.md` §4.3 and `doc/perf.md` §3 carry the numbers this prints.
*)

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

(* A state the engine itself reaches: step from rest with a velocity, so the
   quaternions and the contact set are the ones the integrator produced rather
   than ones a spike invented. *)
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

let max_abs_diff a b =
  Array.fold2_exn a b ~init:0.0 ~f:(fun acc x y -> Float.max acc (Float.abs (x -. y)))

(* The forces a step's row solve produces, and how long the forward pass takes,
   interpreted and compiled. The time is the median of [runs] passes from the
   same state, so a cold cache does not decide it. *)
let forces_and_time ?stop m d0 =
  let forward = Mjc.Forward.forward ?stop m in
  let compiled =
    Rune.jit
      (Nx.Ptree.( @-> ) (Mjc.Data.ptree ()) (Nx.Ptree.returns (Mjc.Data.ptree ())))
      forward
  in
  let forces = Nx.to_array (forward d0).Mjc.Data.efc_force in
  let median f =
    let runs = 25 in
    let times =
      Array.init runs ~f:(fun _ ->
        let t0 = Stdlib.Sys.time () in
        ignore (f d0 : _ Mjc.Data.t);
        Stdlib.Sys.time () -. t0)
    in
    Array.sort times ~compare:Float.compare;
    times.(runs / 2)
  in
  forces, median forward, median compiled

let () =
  let stops = [ 0.0; 1e-10; 1e-11; 1e-12; 1e-13; 1e-14 ] in
  List.iter
    [ "limited"
    ; "looped"
    ; "stiff"
    ; "finger"
    ; "bounce"
    ; "blocks"
    ; "boxes2"
    ; "hopper"
    ; "walker"
    ]
    ~f:(fun name ->
      let m = load name in
      let d0 = reached_state m in
      let nefc = Stdlib.Int32.to_int (Nx.item [] d0.Mjc.Data.nefc) in
      let ref_forces, ref_eager, ref_compiled = forces_and_time ~stop:0.0 m d0 in
      Printf.printf
        "%-9s nefc %3d  fixed schedule  eager %6.2f ms  compiled %6.2f ms\n%!"
        name
        nefc
        (ref_eager *. 1e3)
        (ref_compiled *. 1e3);
      List.iter stops ~f:(fun stop ->
        if not (Float.equal stop 0.0)
        then (
          let forces, eager, compiled = forces_and_time ~stop m d0 in
          Printf.printf
            "  stop %-7.0e  |df| %.3e  eager %6.2f ms (%.2fx)  compiled %6.2f ms (%.2fx)\n\
             %!"
            stop
            (max_abs_diff forces ref_forces)
            (eager *. 1e3)
            (ref_eager /. eager)
            (compiled *. 1e3)
            (ref_compiled /. compiled))))
