(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base

let f64 = Nx.float64

(* [column x j] is column [j] of a [nu;k] parameter tensor, as [nu]. *)
let column (x : Nx.host Nx.float64_t) j = Nx.slice [ A; I j ] x

(* A [nu] 0/1 flag tensor, from a [nu] tensor of flags. *)
let flag (x : Nx.host Nx.float64_t) =
  Nx.where (Nx.greater_s x 0.5) (Nx.ones_like x) (Nx.zeros_like x)

let const (m : Model.host) f = Nx.init f64 [| m.topo.nu |] (fun i -> f i.(0))
let clamp x lo hi = Nx.minimum (Nx.maximum x lo) hi

(* [select mask a b] is [a] where the mask is one and [b] elsewhere. *)
let select mask a b = Nx.add (Nx.mul mask a) (Nx.mul (Nx.sub (Nx.ones_like mask) mask) b)
let minval = 1e-15

let check (m : Model.host) =
  let t = m.topo in
  for i = 0 to t.nu - 1 do
    (match t.actuator_trntype.(i) with
     | Topo.Trn_joint ->
       let id = t.actuator_trnid.(2 * i) in
       (match t.jnt_type.(id) with
        | Topo.Hinge | Topo.Slide -> ()
        | Topo.Ball | Topo.Free ->
          raise
            (Invalid_argument
               (Printf.sprintf
                  "actuator %d: joint transmission to a joint with more than one degree \
                   of freedom, which v1 does not have"
                  i)))
     | Topo.Trn_tendon ->
       raise
         (Invalid_argument
            (Printf.sprintf
               "actuator %d: tendon transmission, which v1 does not have yet"
               i)));
    if t.actuator_actearly.(i)
    then
      raise
        (Invalid_argument
           (Printf.sprintf
              "actuator %d: actearly, which v1 does not have: the force would read the \
               next activation"
              i));
    (* the gain, bias and dynamics kinds have only the supported constructors:
       the loader rejects the others by name, so there is nothing to check *)
    ignore t.actuator_gaintype
  done

(* The activation each actuator reads: its state at the address the loader
   recorded, or zero when it has none. A stateful actuator is identified by a
   non-negative address, which is the same thing MuJoCo's [actnum > 0] says. *)
let activation (m : Model.host) (d : Data.host) =
  let t = m.topo in
  if Int.equal t.na 0
  then Nx.zeros f64 [| t.nu |]
  else (
    let idx = Array.init t.nu ~f:(fun i -> max 0 t.actuator_actadr.(i)) in
    let values =
      Nx.take
        ~axis:0
        (Nx.init Nx.int64 [| t.nu |] (fun i -> Int64.of_int idx.(i.(0))))
        d.act
    in
    let has = const m (fun i -> if t.actuator_actadr.(i) >= 0 then 1. else 0.) in
    Nx.mul values has)

type parts =
  { ctrl : Nx.host Nx.float64_t (** [nu], after the control range *)
  ; act : Nx.host Nx.float64_t (** [nu], zero where there is no state *)
  ; length : Nx.host Nx.float64_t
  ; velocity : Nx.host Nx.float64_t
  }

let parts (m : Model.host) (d : Data.host) (tr : Transmission.t) =
  let p = m.params in
  let ctrl =
    select
      (flag p.actuator_ctrllimited)
      (clamp d.ctrl (column p.actuator_ctrlrange 0) (column p.actuator_ctrlrange 1))
      d.ctrl
  in
  { ctrl
  ; act = activation m d
  ; length = tr.Transmission.length
  ; velocity = tr.Transmission.velocity
  }

let affine (s : parts) prm =
  Nx.add
    (column prm 0)
    (Nx.add (Nx.mul (column prm 1) s.length) (Nx.mul (column prm 2) s.velocity))

let force (m : Model.host) (s : parts) =
  let p = m.params in
  let g =
    let k0 = column p.actuator_gainprm 0 in
    if
      Array.for_all m.topo.actuator_gaintype ~f:(function
        | Topo.Gain_fixed -> true
        | Topo.Gain_affine -> false)
    then k0
    else (
      let mask =
        const m (fun i ->
          match m.topo.actuator_gaintype.(i) with
          | Topo.Gain_affine -> 1.
          | Topo.Gain_fixed -> 0.)
      in
      select mask (affine s p.actuator_gainprm) k0)
  in
  let b =
    if
      Array.for_all m.topo.actuator_biastype ~f:(function
        | Topo.Bias_none -> true
        | Topo.Bias_affine -> false)
    then Nx.zeros f64 [| m.topo.nu |]
    else (
      let mask =
        const m (fun i ->
          match m.topo.actuator_biastype.(i) with
          | Topo.Bias_affine -> 1.
          | Topo.Bias_none -> 0.)
      in
      select mask (affine s p.actuator_biasprm) (Nx.zeros f64 [| m.topo.nu |]))
  in
  let stateful = Array.exists m.topo.actuator_actadr ~f:(fun a -> a >= 0) in
  let input =
    if not stateful
    then s.ctrl
    else (
      let mask = const m (fun i -> if m.topo.actuator_actadr.(i) >= 0 then 1. else 0.) in
      select mask s.act s.ctrl)
  in
  let raw = Nx.add (Nx.mul g input) b in
  select
    (flag p.actuator_forcelimited)
    (clamp raw (column p.actuator_forcerange 0) (column p.actuator_forcerange 1))
    raw

let act_dot (m : Model.host) (d : Data.host) (s : parts) =
  let t = m.topo in
  if Int.equal t.na 0
  then d.act_dot
  else (
    let tau = Nx.maximum_s (column m.params.actuator_dynprm 0) minval in
    let filtered = Nx.div (Nx.sub s.ctrl s.act) tau in
    let integrator =
      const m (fun i ->
        match t.actuator_dyntype.(i) with
        | Topo.Dyn_integrator -> 1.
        | Topo.Dyn_none | Topo.Dyn_filter -> 0.)
    and filter =
      const m (fun i ->
        match t.actuator_dyntype.(i) with
        | Topo.Dyn_filter -> 1.
        | Topo.Dyn_none | Topo.Dyn_integrator -> 0.)
    in
    let per_actuator = Nx.add (Nx.mul integrator s.ctrl) (Nx.mul filter filtered) in
    (* a stateless actuator has no slot: it writes to a sink past the end, or
       the last actuator's zero would overwrite a stateful one's derivative *)
    let idx =
      Nx.init Nx.int64 [| t.nu |] (fun i ->
        let a = t.actuator_actadr.(i.(0)) in
        Int64.of_int (if a >= 0 then a else t.na))
    in
    Nx.slice
      [ R (0, t.na) ]
      (Nx.scatter ~combine:Set ~axis:0 idx per_actuator (Nx.zeros f64 [| t.na + 1 |])))

(* The velocity derivative of each actuator's force:
   [biasprm.[2] + gainprm.[2] * input], MuJoCo's [mjd_actuatorVelDeriv] for the
   kinds v1 has. A fixed gain contributes only the bias term, which is what
   makes a velocity servo's [kv] an implicit damping term rather than a force
   the integrator has to chase. *)
let damping (m : Model.host) (d : Data.host) =
  let tr = Transmission.forward m d in
  let s = parts m d tr in
  let p = m.params in
  let bias_vel =
    if
      Array.for_all m.topo.actuator_biastype ~f:(function
        | Topo.Bias_none -> true
        | Topo.Bias_affine -> false)
    then Nx.zeros f64 [| m.topo.nu |]
    else (
      let mask =
        const m (fun i ->
          match m.topo.actuator_biastype.(i) with
          | Topo.Bias_affine -> 1.
          | Topo.Bias_none -> 0.)
      in
      Nx.mul mask (column p.actuator_biasprm 2))
  in
  let gain_vel =
    if
      Array.for_all m.topo.actuator_gaintype ~f:(function
        | Topo.Gain_fixed -> true
        | Topo.Gain_affine -> false)
    then Nx.zeros f64 [| m.topo.nu |]
    else (
      let mask =
        const m (fun i ->
          match m.topo.actuator_gaintype.(i) with
          | Topo.Gain_affine -> 1.
          | Topo.Gain_fixed -> 0.)
      in
      let input =
        if Array.for_all m.topo.actuator_actadr ~f:(fun a -> a < 0)
        then s.ctrl
        else (
          let m_act =
            const m (fun i -> if m.topo.actuator_actadr.(i) >= 0 then 1. else 0.)
          in
          select m_act s.act s.ctrl)
      in
      Nx.mul mask (Nx.mul (column p.actuator_gainprm 2) input))
  in
  let k = Nx.add bias_vel gain_vel in
  (* A saturated actuator's force does not change when its velocity does, at
     least not to first order, so it puts nothing in the matrix: MuJoCo skips
     exactly the actuators whose force sits on a forcerange bound. *)
  let lim = flag p.actuator_forcelimited in
  if
    Stdlib.Array.for_all (fun x -> Float.equal x 0.) (Nx.to_array p.actuator_forcelimited)
  then k
  else (
    let f = force m s in
    let lo = column p.actuator_forcerange 0
    and hi = column p.actuator_forcerange 1 in
    let at_bound =
      Nx.where
        (Nx.logical_or (Nx.less_equal f lo) (Nx.greater_equal f hi))
        (Nx.ones_like f)
        (Nx.zeros_like f)
    in
    let skip = Nx.mul lim at_bound in
    Nx.mul k (Nx.sub (Nx.ones_like skip) skip))

let qfrc_of (m : Model.host) (moment : Nx.host Nx.float64_t) (f : Nx.host Nx.float64_t) =
  Nx.reshape
    [| m.topo.nv |]
    (Nx.matmul (Nx.matrix_transpose moment) (Nx.reshape [| m.topo.nu; 1 |] f))

let qfrc (m : Model.host) (d : Data.host) =
  let tr = Transmission.forward m d in
  qfrc_of m tr.Transmission.moment (force m (parts m d tr))

let forward (m : Model.host) (d : Data.host) =
  check m;
  let tr = Transmission.forward m d in
  let s = parts m d tr in
  let f = force m s in
  { d with
    Data.actuator_length = s.length
  ; actuator_velocity = s.velocity
  ; actuator_force = f
  ; qfrc_actuator = qfrc_of m tr.Transmission.moment f
  ; act_dot = act_dot m d s
  }
