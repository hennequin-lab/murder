(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx

type kind =
  | Hinge
  | Slide
  | Ball
  | Free

type joint =
  { index : int
  ; name : string
  ; kind : kind
  ; qposadr : int
  ; limits : (float * float) option
  }

let kind_of_jnt_type = function
  | Mjc.Topo.Hinge -> Hinge
  | Mjc.Topo.Slide -> Slide
  | Mjc.Topo.Ball -> Ball
  | Mjc.Topo.Free -> Free

let joints (m : Mjc.Model.host) =
  let t = m.Mjc.Model.topo in
  let range = Nx.to_array m.Mjc.Model.params.Mjc.Model.Params.jnt_range in
  Array.init t.njnt ~f:(fun index ->
    { index
    ; name = Mjc.Topo.jnt_name t index
    ; kind = kind_of_jnt_type t.jnt_type.(index)
    ; qposadr = t.jnt_qposadr.(index)
    ; limits =
        (if t.jnt_limited.(index)
         then Some (range.(2 * index), range.((2 * index) + 1))
         else None)
    })

let driven j =
  match j.kind with
  | Hinge | Slide -> true
  | Ball | Free -> false

let step j =
  match j.kind with
  | Hinge -> 0.05 (* about three degrees *)
  | Slide -> 0.01
  | Ball | Free -> 0.

let position d j = Nx.item [ j.qposadr ] d.Mjc.Data.qpos

let label j =
  match j.name with
  | "" -> Stdlib.Printf.sprintf "joint %d" j.index
  | name -> name

(* One line, because it is drawn as one line: the name, the value, the range
   when there is one, and the keys that move it. *)
let describe d j =
  if not (driven j)
  then
    Stdlib.Printf.sprintf
      "%s: %s joint, not driven"
      (label j)
      (match j.kind with
       | Ball -> "ball"
       | Free -> "free"
       | _ -> "")
  else (
    let q = position d j in
    match j.limits with
    | Some (lo, hi) ->
      Stdlib.Printf.sprintf "%s = %.3f  (%.3f .. %.3f)  [ ]" (label j) q lo hi
    | None -> Stdlib.Printf.sprintf "%s = %.3f  [ ]" (label j) q)

let clamp (lo, hi) v = Float.max lo (Float.min hi v)

(* Writing one coordinate and re-running kinematics is the whole of joint
   control: nothing else in [d] is a function of [qpos] alone except the
   frames, and the next step recomputes the rest. *)
let set (m : Mjc.Model.host) (d : Mjc.Data.host) j v =
  if not (driven j)
  then d, position d j
  else (
    let v =
      match j.limits with
      | Some range -> clamp range v
      | None -> v
    in
    let qpos = Nx.set [ I j.qposadr ] (Nx.scalar Nx.float64 v) d.Mjc.Data.qpos in
    Mjc.Kinematics.forward m { d with Mjc.Data.qpos }, v)

let nudge m d j k = set m d j (position d j +. (k *. step j))

let zero (m : Mjc.Model.host) d j =
  set m d j (Nx.item [ j.qposadr ] m.Mjc.Model.params.Mjc.Model.Params.qpos0)
