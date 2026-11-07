(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gate G4's measurement, on the long recording: per-step accuracy from each
   recorded state, and the shadowing horizon of a free run.

   `g4 <model>...` prints, per model, how many of a thousand resynchronised
   steps land within 1e-8, the error distribution, and where a free run leaves
   the trajectory. The fixture is written by
   `tools/record_fixture.py --long`. *)

open Base

let rel_error = G4lib.rel_error

let () =
  let threshold = 1e-8 in
  for i = 1 to Array.length Stdlib.Sys.argv - 1 do
    let name = Stdlib.Sys.argv.(i) in
    let m =
      match Mjc.Mjcb.read (Printf.sprintf "test/fixtures/%s.mjcb" name) with
      | Ok m -> m
      | Error e -> failwith e
    in
    let env name = Option.map (Stdlib.Sys.getenv_opt name) ~f:Stdlib.int_of_string in
    let iters = env "G4_ITERS" in
    let rows = env "G4_ROWS" in
    let step =
      match iters, rows with
      | None, None -> fun d -> Mjc.Step.step m d
      | iters, rows ->
        let c = Mjc.Step.config ?iters ?rows m in
        fun d -> Mjc.Step.step_with c m d
    in
    let path = Printf.sprintf "test/fixtures/%s.g4.mjbf" name in
    let qpos, qvel, qacc, act, ctrl, nefc, ncon_t, (steps, _) = G4lib.fixture path in
    let nefc_ref = Nx.to_array nefc
    and ncon_ref = Nx.to_array ncon_t in
    let state k =
      let d = Mjc.Data.create m in
      { d with
        Mjc.Data.qpos = Nx.slice [ I k; A ] qpos
      ; qvel = Nx.slice [ I k; A ] qvel
      ; act = Nx.slice [ I k; A ] act
      ; ctrl = Nx.slice [ I k; A ] ctrl
      }
    in
    let error_vs_recording k d =
      Float.max
        (rel_error (Nx.slice [ I (k + 1); A ] qpos) d.Mjc.Data.qpos)
        (rel_error (Nx.slice [ I (k + 1); A ] qvel) d.Mjc.Data.qvel)
    in
    (* 1. Row and contact counts, state by state. *)
    let first_rows = ref None
    and first_contacts = ref None
    and row_states = ref 0
    and contact_states = ref 0 in
    for k = 0 to steps do
      let d = Mjc.Forward.forward m (state k) in
      let ours_nefc = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.nefc)
      and ours_ncon = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.ncon) in
      let theirs_nefc = Int.of_float nefc_ref.(k)
      and theirs_ncon = Int.of_float ncon_ref.(k) in
      if not (Int.equal ours_nefc theirs_nefc)
      then (
        Int.incr row_states;
        if Option.is_none !first_rows then first_rows := Some (k, ours_nefc, theirs_nefc));
      if not (Int.equal ours_ncon theirs_ncon)
      then (
        Int.incr contact_states;
        if Option.is_none !first_contacts
        then first_contacts := Some (k, ours_ncon, theirs_ncon))
    done;
    (match !first_rows with
     | None -> Stdlib.Printf.printf "%s: rows match at every state\n%!" name
     | Some (k, ours, theirs) ->
       Stdlib.Printf.printf
         "%s: rows differ at %d of %d states, first at %d (ours %d, MuJoCo %d)\n%!"
         name
         !row_states
         (steps + 1)
         k
         ours
         theirs);
    (match !first_contacts with
     | None -> Stdlib.Printf.printf "%s: contact count matches at every state\n%!" name
     | Some (k, ours, theirs) ->
       Stdlib.Printf.printf
         "%s: contacts differ at %d of %d states, first at %d (ours %d, MuJoCo %d)\n%!"
         name
         !contact_states
         (steps + 1)
         k
         ours
         theirs);
    (* 2. One step from the recording, a thousand times. *)
    let errors =
      Array.init steps ~f:(fun k ->
        let d = state k in
        let e_acc =
          rel_error (Nx.slice [ I k; A ] qacc) (Mjc.Forward.forward m d).Mjc.Data.qacc
        in
        Float.max e_acc (error_vs_recording k (step d)))
    in
    let sorted = Array.sorted_copy errors ~compare:Float.compare in
    let within = Array.count errors ~f:(fun e -> Float.( <= ) e threshold) in
    Stdlib.Printf.printf
      "%s: %d steps, %d within %g (%.1f%%), median %.2e, worst %.2e\n%!"
      name
      steps
      within
      threshold
      (100.0 *. Float.of_int within /. Float.of_int steps)
      sorted.(steps / 2)
      sorted.(steps - 1);
    let worst_idx =
      List.sort
        (List.range 0 (steps - 1))
        ~compare:(fun a b -> Float.compare errors.(b) errors.(a))
      |> Fn.flip List.take 3
    in
    List.iter worst_idx ~f:(fun k ->
      let d = state k in
      let e_acc =
        rel_error (Nx.slice [ I k; A ] qacc) (Mjc.Forward.forward m d).Mjc.Data.qacc
      in
      let e_step = error_vs_recording k (step d) in
      Stdlib.Printf.printf
        "  worst %4d  e=%.2e (forward %.2e, integrate %.2e)\n%!"
        k
        errors.(k)
        e_acc
        e_step);
    (* 3. A free run: nothing from the recording is read back into the engine
       except the controls it has to follow. *)
    let horizon = ref steps in
    let d = ref (state 0) in
    (try
       for k = 0 to steps - 1 do
         d := step !d;
         if Float.( > ) (error_vs_recording k !d) threshold
         then (
           horizon := k;
           raise Stdlib.Exit)
         else d := { !d with Mjc.Data.ctrl = Nx.slice [ I (k + 1); A ] ctrl }
       done
     with
     | Stdlib.Exit -> ());
    Stdlib.Printf.printf
      "%s: shadowing horizon %d steps at %g\n%!"
      name
      !horizon
      threshold
  done
