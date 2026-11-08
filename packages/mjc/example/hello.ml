(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The five things a user does, in one file: load a model, step it,
   differentiate the step with respect to the model's parameters, save a
   checkpoint, and read it back.

   dune exec packages/mjc/example/hello.exe *)

open Base
module Mjc = Mjc

let f64 = Nx.float64

let () =
  (* {1 Load} A model is compiled from MJCF by [tools/mjcb.py] and read here. *)
  (* Any [.mjcb] will do; the default is one of the recorded fixtures. *)
  let path =
    match Array.to_list (Sys.get_argv ()) with
    | _ :: path :: _ -> path
    | _ -> "test/fixtures/ball.mjcb"
  in
  let m =
    match Mjc.Mjcb.read path with
    | Ok m -> m
    | Error e -> failwith (path ^ ": " ^ e)
  in
  Stdlib.Printf.printf
    "loaded %s: nq = %d, nv = %d, nbody = %d, timestep = %.4g s\n%!"
    path
    (Mjc.Model.nq m)
    (Mjc.Model.nv m)
    (Mjc.Model.nbody m)
    (Nx.item [] m.params.timestep);
  (* {1 Step} One step is a pure function of state: same state in, same state
     out, which is what makes the next part possible. *)
  let c = Mjc.Step.config m in
  let d = ref (Mjc.Data.create m) in
  for _ = 1 to 100 do
    d := Mjc.Step.step_with c m !d
  done;
  Stdlib.Printf.printf
    "after 100 steps: t = %.3f s, qpos = [%s]\n%!"
    (Nx.item [] !d.Mjc.Data.time)
    (String.concat
       ~sep:", "
       (List.map
          (Array.to_list (Nx.to_array !d.Mjc.Data.qpos))
          ~f:(Printf.sprintf "%.4f")));
  (* {1 Differentiate} The quantity to differentiate is a scalar of the step;
     [Rune.value_and_grad] takes it with respect to the model, which is a tree
     of tensors just like the state. Here the function is the summed coordinates
     after one step from the model's own initial state, so the gradient says how
     each parameter moves the pose: gravity pulls the height down, and the two
     masses trade the fall against each other. *)
  let scalar m =
    let d = Mjc.Step.step_with c m (Mjc.Data.create m) in
    Nx.sum d.Mjc.Data.qpos
  in
  let v, g = Rune.value_and_grad (Mjc.Model.ptree ()) scalar m in
  (* Reading a gradient is reading a model-shaped tree of the same fields: no
     hand-written backward pass appears anywhere in this file. *)
  let show name t =
    Stdlib.Printf.printf
      "  d(sum qpos)/d(%s) = [%s]\n%!"
      name
      (String.concat
         ~sep:", "
         (List.map (Array.to_list (Nx.to_array t)) ~f:(Printf.sprintf "%.6g")))
  in
  show "gravity" g.params.gravity;
  show "dof_damping" g.params.dof_damping;
  show "body_mass" g.params.body_mass;
  Stdlib.Printf.printf
    "  (the step it differentiated reached sum qpos = %.9f)\n%!"
    (Nx.item [] v);
  (* A gradient is worth nothing unverified, so the example carries the check
     rather than only printing the numbers: a central difference of the same
     eager step, against the reverse-mode answer, on gravity's vertical
     component. *)
  (* [1e-3], not [1e-6]: one step from rest is affine in gravity, so the central
     difference is exact at any step size, while a step of [1e-6] puts the
     difference (~6e-13) below the rounding of the sum (1.0) and reports 2e-4 of
     noise as disagreement. The tolerance below is what that buys. *)
  let eps = 1e-3 in
  let with_gravity gz =
    let gravity = Nx.create f64 [| 3 |] [| 0.; 0.; gz |] in
    { m with Mjc.Model.params = { m.Mjc.Model.params with gravity } }
  in
  let gz = Nx.item [ 2 ] m.Mjc.Model.params.gravity in
  let plus = Nx.item [] (scalar (with_gravity (gz +. eps)))
  and minus = Nx.item [] (scalar (with_gravity (gz -. eps))) in
  let fd = (plus -. minus) /. (2. *. eps) in
  let ad = Nx.item [ 2 ] g.Mjc.Model.params.gravity in
  let rel = Float.abs (ad -. fd) /. Float.max (Float.abs fd) 1e-30 in
  Stdlib.Printf.printf
    "  check: d(sum qpos)/d(gravity_z): central difference %.6g, reverse mode %.6g, \
     relative %.2g\n\
     %!"
    fd
    ad
    rel;
  if Float.( > ) rel 1e-6 then failwith "the gradient does not match a finite difference";
  (* {1 Save and reload} A checkpoint is the model: the state is a tree of
     tensors the caller owns, so it needs no serde of its own -- [Nx] already
     reads and writes tensors. The model is what is expensive to compile, so it
     is what [.mjcb] stores, and a round trip through it is bit-exact. *)
  let checkpoint = "/tmp/hello-checkpoint.mjcb" in
  (match Mjc.Mjcb.write checkpoint m with
   | Ok () -> ()
   | Error e -> failwith (checkpoint ^ ": " ^ e));
  let m' =
    match Mjc.Mjcb.read checkpoint with
    | Ok m -> m
    | Error e -> failwith (checkpoint ^ ": " ^ e)
  in
  let same_key =
    Mjc.Model.equal_shape_key (Mjc.Model.shape_key m) (Mjc.Model.shape_key m')
  in
  let same_damping =
    Nx.item [ 0 ] (Nx.equal m.params.dof_damping m'.params.dof_damping)
  in
  Stdlib.Printf.printf
    "checkpoint %s: shape key %s, dof_damping %s\n%!"
    checkpoint
    (if same_key then "matches" else "DIFFERS")
    (if same_damping then "matches" else "DIFFERS");
  ignore f64
