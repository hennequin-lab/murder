(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Topo

let row width v = Nx.reshape [| 1; width |] v
let flat width v = Nx.reshape [| width |] v

(* [mj_local2Global] for one object: its frame in the world, given its body's
   pose and its own offset in the body frame. The [sameframe] shortcuts MuJoCo
   uses are skipped: an exact zero offset and an identity quaternion give the
   same numbers through the general path. *)
let local2global ~xpos ~xquat ~pos ~quat =
  let bmat = Rigid.quat_to_mat (row 4 xquat) in
  let x = flat 3 (Nx.add (Rigid.mat_vec bmat (row 3 pos)) (row 3 xpos)) in
  let q = Rigid.quat_mul (row 4 xquat) (row 4 quat) in
  let m = flat 9 (Rigid.quat_to_mat q) in
  x, m

let forward (m : Model.host) (d : Data.host) : Data.host =
  let t = m.topo
  and p = m.params in
  let f64 = Nx.float64 in
  (* Constants are taken from the argument's own tensors rather than made on
     the host, so the whole body is traceable by [vmap] and [jit]. *)
  let xpos = ref (Nx.zeros_like d.xpos) in
  let xquat = ref (Nx.zeros_like d.xquat) in
  let xanchor = ref (Nx.zeros_like d.xanchor) in
  let xaxis = ref (Nx.zeros_like d.xaxis) in
  xquat := Nx.set [ I 0 ] (Nx.create f64 [| 4 |] [| 1.0; 0.0; 0.0; 0.0 |]) !xquat;
  for i = 1 to t.nbody - 1 do
    let jntadr = t.body_jntadr.(i)
    and jntnum = t.body_jntnum.(i) in
    let free = jntnum = 1 && Poly.equal t.jnt_type.(jntadr) Free in
    if free
    then (
      let adr = t.jnt_qposadr.(jntadr) in
      let pos = Nx.slice [ R (adr, adr + 3) ] d.qpos in
      let quat =
        flat 4 (Rigid.quat_normalize (row 4 (Nx.slice [ R (adr + 3, adr + 7) ] d.qpos)))
      in
      xpos := Nx.set [ I i ] pos !xpos;
      xquat := Nx.set [ I i ] quat !xquat;
      xanchor := Nx.set [ I jntadr ] pos !xanchor;
      xaxis := Nx.set [ I jntadr ] (Nx.get [ jntadr ] p.jnt_axis) !xaxis)
    else (
      let pid = t.body_parentid.(i) in
      let ppos = Nx.get [ pid ] !xpos
      and pquat = Nx.get [ pid ] !xquat in
      let body_x =
        Rigid.mat_vec (Rigid.quat_to_mat (row 4 pquat)) (row 3 (Nx.get [ i ] p.body_pos))
      in
      let pos = ref (flat 3 (Nx.add body_x (row 3 ppos))) in
      let quat =
        ref (flat 4 (Rigid.quat_mul (row 4 pquat) (row 4 (Nx.get [ i ] p.body_quat))))
      in
      for j = jntadr to jntadr + jntnum - 1 do
        let adr = t.jnt_qposadr.(j) in
        let axis = row 3 (Nx.get [ j ] p.jnt_axis) in
        let jpos = row 3 (Nx.get [ j ] p.jnt_pos) in
        let axis_x = Rigid.quat_rot_vec (row 4 !quat) axis in
        let anchor_x =
          flat 3 (Nx.add (Rigid.quat_rot_vec (row 4 !quat) jpos) (row 3 !pos))
        in
        let rotate qloc =
          quat := flat 4 (Rigid.quat_mul (row 4 !quat) qloc);
          let vec = Rigid.quat_rot_vec (row 4 !quat) jpos in
          pos := flat 3 (Nx.sub (row 3 anchor_x) vec)
        in
        (match t.jnt_type.(j) with
         | Slide ->
           let dq = Nx.sub (Nx.get [ adr ] d.qpos) (Nx.get [ adr ] p.qpos0) in
           pos := flat 3 (Nx.add (row 3 !pos) (Nx.mul axis_x (Nx.reshape [| 1; 1 |] dq)))
         | Ball ->
           rotate (Rigid.quat_normalize (row 4 (Nx.slice [ R (adr, adr + 4) ] d.qpos)))
         | Hinge ->
           let angle = Nx.sub (Nx.get [ adr ] d.qpos) (Nx.get [ adr ] p.qpos0) in
           rotate (Rigid.quat_of_axis_angle ~axis ~angle:(Nx.reshape [| 1 |] angle))
         | Free -> ());
        xanchor := Nx.set [ I j ] anchor_x !xanchor;
        xaxis := Nx.set [ I j ] (flat 3 axis_x) !xaxis
      done;
      quat := flat 4 (Rigid.quat_normalize (row 4 !quat));
      xpos := Nx.set [ I i ] !pos !xpos;
      xquat := Nx.set [ I i ] !quat !xquat)
  done;
  (* Inertial, geom and site frames, from their body's pose. *)
  let identity9 = Nx.reshape [| 9 |] (Nx.eye f64 3) in
  let xipos = ref (Nx.zeros_like d.xipos)
  and ximat = ref (Nx.zeros_like d.ximat) in
  ximat := Nx.set [ I 0 ] identity9 !ximat;
  for i = 1 to t.nbody - 1 do
    let x, mat =
      local2global
        ~xpos:(Nx.get [ i ] !xpos)
        ~xquat:(Nx.get [ i ] !xquat)
        ~pos:(Nx.get [ i ] p.body_ipos)
        ~quat:(Nx.get [ i ] p.body_iquat)
    in
    xipos := Nx.set [ I i ] x !xipos;
    ximat := Nx.set [ I i ] mat !ximat
  done;
  let geom_xpos = ref (Nx.zeros_like d.geom_xpos)
  and geom_xmat = ref (Nx.zeros_like d.geom_xmat) in
  for g = 0 to t.ngeom - 1 do
    let b = t.geom_bodyid.(g) in
    let x, mat =
      local2global
        ~xpos:(Nx.get [ b ] !xpos)
        ~xquat:(Nx.get [ b ] !xquat)
        ~pos:(Nx.get [ g ] p.geom_pos)
        ~quat:(Nx.get [ g ] p.geom_quat)
    in
    geom_xpos := Nx.set [ I g ] x !geom_xpos;
    geom_xmat := Nx.set [ I g ] mat !geom_xmat
  done;
  let site_xpos = ref (Nx.zeros_like d.site_xpos)
  and site_xmat = ref (Nx.zeros_like d.site_xmat) in
  for s = 0 to t.nsite - 1 do
    let b = t.site_bodyid.(s) in
    let x, mat =
      local2global
        ~xpos:(Nx.get [ b ] !xpos)
        ~xquat:(Nx.get [ b ] !xquat)
        ~pos:(Nx.get [ s ] p.site_pos)
        ~quat:(Nx.get [ s ] p.site_quat)
    in
    site_xpos := Nx.set [ I s ] x !site_xpos;
    site_xmat := Nx.set [ I s ] mat !site_xmat
  done;
  { d with
    xpos = !xpos
  ; xquat = !xquat
  ; xanchor = !xanchor
  ; xaxis = !xaxis
  ; xipos = !xipos
  ; ximat = !ximat
  ; geom_xpos = !geom_xpos
  ; geom_xmat = !geom_xmat
  ; site_xpos = !site_xpos
  ; site_xmat = !site_xmat
  }

let check (m : Model.host) : (unit, string) Result.t =
  let t = m.topo in
  let problems =
    List.filter_map (List.range 0 t.njnt) ~f:(fun j ->
      match t.jnt_type.(j) with
      | Free ->
        let b = t.jnt_bodyid.(j) in
        if t.body_parentid.(b) = 0
        then None
        else Some (Printf.sprintf "free joint %d is not on a world child body" j)
      | Hinge | Slide | Ball -> None)
  in
  match problems with
  | [] -> Ok ()
  | problems -> Error (String.concat ~sep:"; " problems)
