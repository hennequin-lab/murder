(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* M7's third acceptance: float32 passes the same tests as float64 at 1e-4
   relative, contacts included.

   What is compared, and why it is not "the same trajectory". A float32 run
   cannot agree with the recorded float64 trajectory to 1e-8 — that is the
   float64 gate's job, and the recorded states are MuJoCo's. The claim here is
   narrower and checkable: from the *same state*, a float32 step tracks the
   float64 step to 1e-4 relative, so a training run in float32 is the same
   physics at a documented tolerance.

   The state matters. Starting both dtypes at a model's resting pose makes a
   discrete decision ambiguous: the walker at rest sits within a rounding of the
   contact margin, so float32 finds four contacts where float64 finds none and
   the two runs part ways immediately — a property of a threshold, not of the
   integrator. So the comparison seeds both runs from *recorded mid-trajectory
   states*, where the contact set is unambiguous, and the test also checks that
   the two dtypes agree on [nefc] and [ncon] at every step. If a state is one
   where the active set legitimately flips, the test says so rather than
   reporting a tolerance it did not meet. *)

open Base
open Windtrap
open Mjc

let f32 = Nx.float32
let f64 = Nx.float64

(* The recorded trajectory of a fixture: joint positions and velocities per
   state, as MuJoCo computed them. *)
let trajectory name =
  let f = Support.fixture_traj name in
  let qpos_f = Support.tensor f "traj_qpos"
  and qvel_f = Support.tensor f "traj_qvel" in
  let nstates = (fst qpos_f).(0) in
  ( nstates
  , Array.init nstates ~f:(fun k ->
      Nx.create f64 [| Array.length (Support.row qpos_f k) |] (Support.row qpos_f k))
  , Array.init nstates ~f:(fun k ->
      Nx.create f64 [| Array.length (Support.row qvel_f k) |] (Support.row qvel_f k)) )

let max_abs a b =
  let x = Nx.to_array a
  and y = Nx.to_array b in
  Array.fold2_exn x y ~init:0. ~f:(fun acc p q -> Float.max acc (Float.abs (p -. q)))

(* Relative to the scale of the quantity itself, which is what a tolerance on a
   state means. The floor keeps a quantity that is zero from dividing by it. *)
let scale t =
  Array.fold (Nx.to_array t) ~init:0. ~f:(fun acc x -> Float.max acc (Float.abs x))

let rel a b = max_abs a b /. Float.max 1e-12 (Float.max (scale a) (scale b))
let int_of x = Stdlib.Int32.to_int (Nx.item [] x)

let test_float32_tracks_float64 () =
  List.iter [ "hopper"; "walker"; "humanoid"; "boxes2"; "bounce" ] ~f:(fun name ->
    let m = Support.model name in
    let m32 = Model.cast f32 m in
    let c = Step.config m in
    let nstates, qpos, qvel = trajectory name in
    (* Mid-trajectory states, spread over the recording. *)
    let seeds =
      List.filter_map
        [ nstates / 4; nstates / 2; 3 * nstates / 4 ]
        ~f:(fun k -> Option.some_if (k > 0 && k < nstates) k)
    in
    let worst = ref 0.
    and steps = ref 0 in
    List.iter seeds ~f:(fun k ->
      let seed32 =
        ref
          { (Data.create m32) with
            qpos = Nx.cast f32 qpos.(k)
          ; qvel = Nx.cast f32 qvel.(k)
          }
      and seed64 = ref { (Data.create m) with qpos = qpos.(k); qvel = qvel.(k) } in
      for _ = 1 to 10 do
        seed64 := Step.step_with c m !seed64;
        seed32 := Step.step_with c m32 !seed32;
        Int.incr steps;
        (* The active set is a discrete decision, and disagreement there is a
           different statement from a tolerance miss: name it. *)
        let nefc64 = int_of !seed64.Data.nefc
        and nefc32 = int_of !seed32.Data.nefc
        and ncon64 = int_of !seed64.Data.ncon
        and ncon32 = int_of !seed32.Data.ncon in
        if nefc64 <> nefc32 || ncon64 <> ncon32
        then
          fail
            (Printf.sprintf
               "mjc dtype: %s state %d: float32 and float64 disagree on the active set \
                (nefc %d/%d, ncon %d/%d)"
               name
               k
               nefc64
               nefc32
               ncon64
               ncon32);
        worst := Float.max !worst (rel !seed64.Data.qpos !seed32.Data.qpos);
        worst := Float.max !worst (rel !seed64.Data.qvel !seed32.Data.qvel)
      done);
    Stdlib.Printf.printf
      "  %-9s %d seeds, %d steps, worst relative difference %.2e\n%!"
      name
      (List.length seeds)
      !steps
      !worst;
    if Float.( > ) !worst 1e-4
    then
      fail
        (Printf.sprintf
           "mjc dtype: %s: float32 differs from float64 by %.3e over %d steps, past 1e-4"
           name
           !worst
           !steps))

(* The dtype is a property of the model, and casting keeps everything else: the
   physics has to be the float64 answer at float32 width, not a second engine. *)
let test_cast_keeps_the_model () =
  let m = Support.model "hopper" in
  let m32 = Model.cast f32 m in
  if not (Int.equal (Model.nq m) (Model.nq m32) && Int.equal (Model.nv m) (Model.nv m32))
  then fail "mjc dtype: casting changed the model's counts";
  if not (Model.equal_shape_key (Model.shape_key m) (Model.shape_key m32))
  then fail "mjc dtype: casting changed the model's shape key";
  let c = Step.config m in
  let name_of t = Stdlib.Format.asprintf "%a" Nx.pp_dtype (Nx.dtype t) in
  let d32 = Data.create m32 in
  if not (String.equal (name_of d32.Data.time) "float32")
  then fail "mjc dtype: the created state is not float32";
  (* And it steps, which is the whole point. *)
  let d32 = Step.step_with c m32 d32 in
  if not (String.equal (name_of d32.Data.qpos) "float32")
  then fail "mjc dtype: the stepped state is not float32"

let () =
  Stdlib.exit
    (run
       "mjc dtype"
       [ test "float32 tracks float64 at 1e-4" test_float32_tracks_float64
       ; test "casting keeps the model, and the state" test_cast_keeps_the_model
       ])
