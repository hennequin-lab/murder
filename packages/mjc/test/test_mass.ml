(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The position stage's inertia quantities against MuJoCo, plus the properties
   a mass matrix must have whatever the implementation: symmetry, positive
   definiteness, and a solve that inverts the factor. *)

open Base
open Windtrap
open Support
module K = Mjc.Kinematics
module M = Mjc.Mass

let models = [ "pendulum"; "chain6"; "floating"; "ball" ]

let field (d : Mjc.Data.host) name =
  match name with
  | "subtree_com" -> d.subtree_com
  | "cinert" -> d.cinert
  | "cdof" -> d.cdof
  | "mass_matrix" -> d.mass_matrix
  | name -> failwith ("unknown field " ^ name)

let staged f s =
  let m = model f in
  let qpos = tensor (fixture f) "qpos"
  and qvel = tensor (fixture f) "qvel" in
  let d = data_with_qpos m (row qpos s) in
  let d = { d with qvel = vec1 (row qvel s) } in
  M.forward m (K.forward m d)

let test_parity () =
  List.iter models ~f:(fun name ->
    let f = fixture name in
    for s = 0 to f.nstates - 1 do
      let d = staged name s in
      List.iter [ "subtree_com"; "cinert"; "cdof"; "mass_matrix" ] ~f:(fun k ->
        parity
          ~tol:1e-12
          ~name:(Printf.sprintf "%s %s state %d" name k s)
          (row (tensor f k) s)
          (Nx.to_array (field d k)))
    done)

let test_properties () =
  List.iter models ~f:(fun name ->
    List.iter [ 0; 7; 31 ] ~f:(fun s ->
      ignore s;
      let d = staged name s in
      let m = Nx.to_array d.mass_matrix in
      let nv = Mjc.Data.nv (model name) in
      (* symmetry *)
      let worst = ref 0.0 in
      for i = 0 to nv - 1 do
        for j = 0 to nv - 1 do
          let e = Float.abs (m.((i * nv) + j) -. m.((j * nv) + i)) in
          if Float.(e > !worst) then worst := e
        done
      done;
      equal ~msg:(name ^ " symmetry") (float 1e-12) 0.0 !worst;
      (* positive definite: the factor exists and inverts *)
      let l = M.factor (model name) d in
      let b = Nx.ones Nx.float64 [| nv |] in
      let x = M.solve l b in
      let residual = Nx.sub (Nx.matmul d.mass_matrix x) b in
      let norm =
        Nx.to_array residual |> Array.fold ~init:0.0 ~f:(fun a v -> a +. (v *. v))
      in
      equal ~msg:(name ^ " solve residual") bool true Float.(sqrt norm < 1e-12)))

let () =
  Stdlib.exit
    (run
       "mjc mass"
       [ test "matches the C engine" test_parity
       ; test "symmetry and solve" test_properties
       ])
