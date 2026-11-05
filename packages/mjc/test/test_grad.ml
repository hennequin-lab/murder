(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* M6: the gradient of the step.

   Two things are checked here, and they are different questions. First, that
   the reverse mode of the step is *finite* — a NaN is never an acceptable
   answer, even at a contact transition, and it is always a bug in a primitive
   (a square root of a negative branch that was masked out, a zero-length
   normal, a [-1e30] filler multiplying zero). Second, where the answer is
   defined and smooth, that it is *right*: it matches central differences, and
   it does not depend on how many solver iterations the step ran.

   The states come from the fixtures, so each one is a state MuJoCo was at and
   our forward pass reproduces; a NaN here is ours. *)

open Base
open Windtrap
open Mjc

let f64 = Nx.float64
let float_of_t x = Nx.item [] x
let int_of_t x = Stdlib.Int32.to_int (Nx.item [] x)

(* {1 Helpers} *)

(* The trajectory a fixture recorded: the joint positions and velocities of
   each state, as MuJoCo computed them. *)
let trajectory name =
  let f = Support.fixture_traj name in
  let width (shape, _) =
    Array.fold (Array.sub shape ~pos:1 ~len:(Array.length shape - 1)) ~init:1 ~f:( * )
  in
  let shape_of (shape, _) = Array.sub shape ~pos:1 ~len:(Array.length shape - 1) in
  let qpos_f = Support.tensor f "traj_qpos"
  and qvel_f = Support.tensor f "traj_qvel" in
  let nq = width qpos_f
  and nv = width qvel_f in
  Array.init f.nstates ~f:(fun s ->
    let slice (shape, data) w =
      let _ = shape in
      Array.sub data ~pos:(s * w) ~len:w
    in
    ( Nx.create f64 (shape_of qpos_f) (slice qpos_f nq)
    , Nx.create f64 (shape_of qvel_f) (slice qvel_f nv) ))

let data_at (m : Model.host) qpos qvel = { (Data.create m) with qpos; qvel }

(* Sum of squares of the position after [n] steps. The smallest scalar that
   still reads the whole trajectory. *)
let loss m ~n (d : Data.host) =
  let d = ref d in
  for _ = 1 to n do
    d := Step.step m !d
  done;
  Nx.sum (Nx.mul !d.qpos !d.qpos)

(* The pullback of [f] at [d], read into a fresh array before anything else
   touches [d]: a value a differentiation read is dead afterwards. *)
let qpos_grad m ~n (d : Data.host) =
  let value = loss m ~n d in
  let _, pullback =
    Rune.vjp
      Nx.Ptree.tensor
      Nx.Ptree.tensor
      (fun qpos -> loss m ~n { d with qpos })
      d.qpos
  in
  let g = pullback (Nx.scalar f64 1.0) in
  float_of_t value, Nx.to_array g

let all_finite a = Array.for_all a ~f:Float.is_finite
let worst a = Array.fold a ~init:0.0 ~f:(fun acc v -> Float.max acc (Float.abs v))

(* Central differences of the same loss, one entry at a time. *)
let central m ~n (d : Data.host) ~h =
  let shape = Nx.shape d.qpos in
  let count = Array.fold shape ~init:1 ~f:( * ) in
  let base = Nx.to_array d.qpos in
  Array.init count ~f:(fun i ->
    let at delta =
      let a = Array.copy base in
      a.(i) <- a.(i) +. delta;
      float_of_t (loss m ~n { d with qpos = Nx.create f64 shape a })
    in
    (at h -. at (-.h)) /. (2.0 *. h))

(* {1 Finite gradients} *)

(* Every recorded state of every model that steps: the gradient of the step is
   a number. The bound is loose on purpose — a contact row's impedance is
   stiff, so a legitimately large number is expected; an infinite or a NaN one
   is the bug this test exists for. *)
let test_finite () =
  let models =
    [ "pendulum"
    ; "chain6"
    ; "floating"
    ; "ball"
    ; "limited"
    ; "looped"
    ; "stiff"
    ; "actuators"
    ; "finger"
    ; "bounce"
    ; "blocks"
    ; "boxes2"
    ; "sensors"
    ]
  in
  (* The largest gradient a model produced over its recorded states. Finite is
     the requirement; how large the largest is says whether a contact
     transition is a usable derivative or only a bounded one, and the gate asks
     for the number. *)
  let worst_of name =
    let m = Support.model name in
    Array.fold (trajectory name) ~init:0.0 ~f:(fun acc (qpos, qvel) ->
      let _, grad = qpos_grad m ~n:1 (data_at m qpos qvel) in
      if not (all_finite grad)
      then (
        let bad =
          Array.to_list (Array.mapi grad ~f:(fun i g -> i, g))
          |> List.filter ~f:(fun (_, g) -> not (Float.is_finite g))
          |> List.map ~f:(fun (i, g) -> Printf.sprintf "%d=%g" i g)
        in
        fail
          (Printf.sprintf
             "mjc grad: %s: not finite at %s"
             name
             (String.concat ~sep:" " bad)));
      Float.max acc (worst grad))
  in
  List.iter models ~f:(fun name ->
    let w = worst_of name in
    Stdlib.Printf.printf "  %-10s worst |dq| %-10.4g\n%!" name w;
    if Float.( > ) w 1e12
    then
      fail
        (Printf.sprintf
           "mjc grad: %s: |grad| = %.3e, past the bound this test states"
           name
           w))

(* {1 Gradients against finite differences} *)

(* Away from contact and from an active limit the step is smooth, and the
   answer is the answer: the reverse mode has to match central differences to
   the accuracy finite differences have. *)
let test_against_finite_differences () =
  List.iter [ "chain6"; "pendulum"; "finger" ] ~f:(fun name ->
    let m = Support.model name in
    let states = trajectory name in
    let qpos, qvel = states.(0) in
    (* Off the reference configuration: at rest, every derivative is zero and
       the comparison proves nothing. *)
    let nq = Array.fold (Nx.shape qpos) ~init:1 ~f:( * ) in
    let qpos =
      Nx.add
        qpos
        (Nx.create
           f64
           (Nx.shape qpos)
           (Array.init nq ~f:(fun i -> 0.25 *. Float.sin (Float.of_int (i + 1)))))
    in
    let d = data_at m qpos qvel in
    let _, grad = qpos_grad m ~n:10 d in
    let fd = central m ~n:10 d ~h:1e-6 in
    Array.iteri grad ~f:(fun i g ->
      let expected = fd.(i) in
      if Float.( > ) (Float.abs expected) 1e-6
      then (
        let rel = Float.abs ((g -. expected) /. expected) in
        if Float.( > ) rel 1e-6
        then
          fail
            (Printf.sprintf
               "mjc grad: %s coordinate %d: reverse mode %.12g, central difference \
                %.12g, relative error %.3e"
               name
               i
               g
               expected
               rel))))
  [@warning "-5"]

(* {1 Model parameters} *)

(* The step differentiates with respect to a mass, a spring stiffness, a geom
   size and an actuator gain. It does so through the compiled dynamics, whose
   [run] reads no parameter with [Nx.item]: a read of a traced parameter raises,
   and a solver budget is structure rather than a derivative anyway. *)
let test_parameters () =
  let start_state name =
    let m = Support.model name in
    let qpos, qvel = (trajectory name).(0) in
    let nq = Array.fold (Nx.shape qpos) ~init:1 ~f:( * ) in
    let qpos =
      Nx.add
        qpos
        (Nx.create
           f64
           (Nx.shape qpos)
           (Array.init nq ~f:(fun i -> 0.3 *. Float.sin (Float.of_int (i + 1)))))
    in
    m, data_at m qpos qvel
  in
  let check name (m, d0) ~label ~field ~set =
    (* Structure from the host model, once; parameters as the argument, so the
       differentiation sees them as tensor leaves. *)
    let dyn = Dynamics.compile m in
    let loss (m : Model.host) =
      let d = Dynamics.horizon dyn m d0 10 in
      Nx.sum (Nx.mul d.qpos d.qpos)
    in
    let _, pullback = Rune.vjp Model.ptree Nx.Ptree.tensor loss m in
    let grad = pullback (Nx.scalar f64 1.0) in
    let base = field m.Model.params in
    let shape = Nx.shape base in
    let base_array = Nx.to_array base in
    let g = Nx.to_array (field grad.Model.params) in
    Array.iteri g ~f:(fun i gi ->
      let at delta =
        let a = Array.copy base_array in
        a.(i) <- a.(i) +. delta;
        float_of_t (loss (set m (Nx.create f64 shape a)))
      in
      (* Richardson-extrapolated central differences: [O(h^4)] truncation, so
         the reference is good to about [1e-11] and a disagreement is the
         reverse mode's, not the difference's. *)
      let central h = (at h -. at (-.h)) /. (2.0 *. h) in
      let h = 1e-5 in
      let fd = ((4.0 *. central (h /. 2.0)) -. central h) /. 3.0 in
      let resolved = Float.max (1e-6 *. Float.abs fd) 1e-12 in
      if Float.( > ) (Float.abs (gi -. fd)) resolved
      then
        fail
          (Printf.sprintf
             "mjc grad: %s %s entry %d: reverse mode %.12g, Richardson difference %.12g, \
              relative error %.3e"
             name
             label
             i
             gi
             fd
             (Float.abs ((gi -. fd) /. fd))))
  in
  let m, d0 = start_state "chain6" in
  check
    "chain6"
    (m, d0)
    ~label:"body_mass"
    ~field:(fun p -> p.body_mass)
    ~set:(fun m x -> { m with Model.params = { m.Model.params with body_mass = x } });
  check
    "chain6"
    (m, d0)
    ~label:"jnt_stiffness"
    ~field:(fun p -> p.jnt_stiffness)
    ~set:(fun m x -> { m with Model.params = { m.Model.params with jnt_stiffness = x } });
  check
    "chain6"
    (m, d0)
    ~label:"geom_size"
    ~field:(fun p -> p.geom_size)
    ~set:(fun m x -> { m with Model.params = { m.Model.params with geom_size = x } });
  let m, d0 = start_state "finger" in
  check
    "finger"
    (m, d0)
    ~label:"actuator_gainprm"
    ~field:(fun p -> p.actuator_gainprm)
    ~set:(fun m x ->
      { m with Model.params = { m.Model.params with actuator_gainprm = x } })

(* {1 The budget} *)

(* {1 The budget} *)

(* An implicit gradient is the derivative of the *solution*, so it does not
   care how many iterations the solver took to get there. An unrolled one
   does. The models here have rows the solver has to resolve. *)
let test_budget_invariance () =
  List.iter [ "looped"; "stiff"; "finger" ] ~f:(fun name ->
    let m = Support.model name in
    let states = trajectory name in
    let qpos, qvel = states.(0) in
    let nq = Array.fold (Nx.shape qpos) ~init:1 ~f:( * ) in
    let qpos =
      Nx.add
        qpos
        (Nx.create
           f64
           (Nx.shape qpos)
           (Array.init nq ~f:(fun i -> 0.2 *. Float.cos (Float.of_int (2 * (i + 1))))))
    in
    let d = data_at m qpos qvel in
    let with_budget b =
      let params = { m.Model.params with iterations = Nx.scalar f64 b } in
      let m = { m with Model.params } in
      snd (qpos_grad m ~n:10 d)
    in
    let g_low = with_budget 3.0 in
    let g_high = with_budget 8.0 in
    Array.iteri g_low ~f:(fun i a ->
      let b = g_high.(i) in
      let rel = if Float.equal b 0.0 then Float.abs a else Float.abs ((a -. b) /. b) in
      if Float.( > ) rel 1e-5
      then
        fail
          (Printf.sprintf
             "mjc grad: %s coordinate %d: budget 3 gives %.12g, budget 8 gives %.12g, \
              relative difference %.3e"
             name
             i
             a
             b
             rel)))
  [@warning "-5"]

let () =
  run
    "mjc grad"
    [ test "the gradient is finite on every recorded state" test_finite
    ; test "gradient matches central differences" test_against_finite_differences
    ; test "gradient does not depend on the iteration budget" test_budget_invariance
    ; test "gradient with respect to model parameters" test_parameters
    ]
  |> ignore
