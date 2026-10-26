(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Geom
module R = Robot
module M = Motor

type t =
  { model : R.t
  ; dim : int
  ; label : string
  ; residual : R.q -> Nx.float32_t
  ; jacobian : R.q -> Nx.float32_t
  }

let model t = t.model
let dim t = t.dim
let label t = t.label
let prod a = Array.fold_left ( * ) 1 a

let batch_shape x =
  let s = Nx.shape x in
  Array.sub s 0 (Array.length s - 1)

(* {2 Objects and blades} *)

(* Objects are grade-pure: the grade is the grade of the largest coefficient.
   Reading it from the value is a host-side decision made once, at construction,
   never inside a residual. *)
let grade_of o =
  let a = Nx.to_array (Nx.abs o) in
  let index = ref 0
  and top = ref (-1.) in
  Array.iteri
    (fun i v ->
       if v > !top
       then (
         top := v;
         index := i))
    a;
  Blade.grade (Blade.of_int !index)

let blades_of_grade grade =
  Array.to_list (Blade.all Cga.n)
  |> List.filter (fun b -> Blade.grade b = grade)
  |> Array.of_list

let mask_of blades =
  Nx.create
    Nx.int64
    [| Array.length blades |]
    (Array.map (fun b -> Int64.of_int (Blade.to_int b)) blades)

let gather mask x = Nx.take ~axis:(-1) ~indices:mask x

(* [swap_last_two x] exchanges the last two axes: the motion Jacobian keeps the
   eight motor coefficients on the second-to-last axis, and [Cga.of_motor] wants
   them last. *)
let swap_last_two x =
  let r = Array.length (Nx.shape x) in
  let axes = List.init r Fun.id in
  let axes =
    List.mapi
      (fun i a -> if i = r - 2 then r - 1 else if i = r - 1 then r - 2 else a)
      axes
  in
  Nx.transpose ~axes x

let normalize_object o =
  if Nx.item [] (Cga.is_point o)
  then Cga.normalize_point o
  else (
    match grade_of o with
    | 1 ->
      if Nx.item [] (Cga.is_flat o) then Cga.normalize_plane o else Cga.normalize_sphere o
    | _ ->
      let n = Cga.norm o in
      if Nx.item [] n > 1e-6 then Cga.normalize o else o)

(* {2 Pose tasks}

   The residual is the logarithm of the pose error and the Jacobian is the body
   Jacobian of the frame. That pair is the standard geometric linearisation: a
   joint step [dq] changes the body twist by [J dq], and the error is the twist
   that cancels the pose error. It is exact at the solution and first-order
   accurate away from it, which makes the iteration Gauss-Newton on the pose;
   the derivative of the logarithm itself would need [d log_E] and is not what
   this returns. *)

let pose model ~frame ~target =
  { model
  ; dim = 6
  ; label = "pose"
  ; residual = (fun q -> M.log (M.mul (M.inverse target) (R.pose model ~frame q)))
  ; jacobian = (fun q -> R.jacobian ~space:`Body model ~frame q)
  }

(* {2 Geometric tasks} *)

let primitive ?(normalize = true) model ~frame ~reference ~target =
  let reference = if normalize then normalize_object reference else reference in
  let target = if normalize then normalize_object target else target in
  let grade_ref = grade_of reference
  and grade_target = grade_of target in
  (* The left contraction [a ⌋ b] is the grade [grade b - grade a] part of [a b],
     and vanishes when [a] is the higher-grade factor; so the lower-grade object
     goes on the left. Only the [moved] slot depends on [q], so the same
     slotting gives the derivative. *)
  let moved_first = grade_ref <= grade_target in
  let blades = blades_of_grade (abs (grade_ref - grade_target)) in
  let dim = Array.length blades in
  let mask = mask_of blades in
  let size = Cga.size in
  { model
  ; dim
  ; label = Printf.sprintf "primitive (%d/%d)" grade_ref grade_target
  ; residual =
      (fun q ->
        let batch = batch_shape q in
        let bsz = prod batch in
        let moved = Cga.apply_motor (R.pose model ~frame q) reference in
        let e = if moved_first then Cga.inner moved target else Cga.inner target moved in
        let rows = gather mask (Nx.reshape [| bsz; size |] e) in
        Nx.reshape (Array.append batch [| dim |]) rows)
  ; jacobian =
      (fun q ->
        let batch = batch_shape q in
        let bsz = prod batch in
        let dof = R.dof model in
        let motion = R.pose model ~frame q in
        let dm = R.motion_jacobian model ~frame q in
        let md = Cga.of_motor motion in
        let rev_md = Cga.reverse md in
        let x3 = Nx.reshape [| 1; 1; size |] reference in
        let dm3 = Nx.reshape [| bsz; dof; size |] (Cga.of_motor (swap_last_two dm)) in
        let md3 = Nx.reshape [| bsz; 1; size |] md in
        let rev3 = Nx.reshape [| bsz; 1; size |] rev_md in
        (* d/dq (M X M~) = dM X M~ + M X dM~: one column per coordinate. *)
        let term1 = Cga.gp (Cga.gp dm3 x3) rev3 in
        let term2 = Cga.gp md3 (Cga.gp x3 (Cga.reverse dm3)) in
        let dy = Nx.add term1 term2 in
        let de = if moved_first then Cga.inner dy target else Cga.inner target dy in
        let rows = gather mask (Nx.reshape [| bsz * dof; size |] de) in
        let rows = Nx.reshape [| bsz; dof; dim |] rows in
        Nx.reshape
          (Array.append batch [| dim; dof |])
          (Nx.transpose ~axes:[ 0; 2; 1 ] rows))
  }

(* {2 Custom tasks} *)

let custom ?(label = "custom") model ~dim ~residual ~jacobian =
  { model; dim; label; residual; jacobian }

(* {2 Combining} *)

let weight t w =
  if Array.length w <> dim t
  then
    invalid_arg
      (Printf.sprintf
         "Task.weight: %d weights for a residual of length %d"
         (Array.length w)
         (dim t));
  let vector = Nx.create Nx.float32 [| Array.length w |] w in
  let rows = Nx.reshape [| Array.length w; 1 |] vector in
  { t with
    residual = (fun q -> Nx.mul (t.residual q) vector)
  ; jacobian = (fun q -> Nx.mul (t.jacobian q) rows)
  }

let scale t s = weight t (Array.make (dim t) s)

let stack ts =
  match ts with
  | [] -> invalid_arg "Task.stack: no tasks"
  | first :: _ ->
    let model = model first in
    List.iter
      (fun t ->
         if not (t.model == model)
         then invalid_arg "Task.stack: the tasks belong to different models")
      ts;
    let dim = List.fold_left (fun n t -> n + dim t) 0 ts in
    { model
    ; dim
    ; label = String.concat " + " (List.map label ts)
    ; residual =
        (fun q -> Nx.concatenate ~axis:(-1) (List.map (fun t -> t.residual q) ts))
    ; jacobian =
        (fun q -> Nx.concatenate ~axis:(-2) (List.map (fun t -> t.jacobian q) ts))
    }

(* {2 Evaluating} *)

let residual t q = t.residual q
let jacobian t q = t.jacobian q

let squared t q =
  Nx.sum ~axes:[ Array.length (Nx.shape q) - 1 ] (Nx.mul (t.residual q) (t.residual q))

let norm t q =
  let r = t.residual q in
  Nx.sqrt (Nx.sum ~axes:[ Array.length (Nx.shape r) - 1 ] (Nx.mul r r))

let pp ppf t =
  Format.fprintf
    ppf
    "@[<hov 1>task %s, %d residual%s, %d coordinates@]"
    t.label
    t.dim
    (if t.dim = 1 then "" else "s")
    (R.dof t.model)
