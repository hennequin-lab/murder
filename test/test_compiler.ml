(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The compiler against brute force: every coefficient is checked against a
   direct sum over the index space, and the compiled closures are checked for
   the two properties that characterize the basis — invariance under the
   group, and the factor round trip through the design solve. *)

open Base
open Windtrap
open Symo

let dims ~left ~right = { Sides.left; right }
let left i = Index.Left i
let right i = Index.Right i
let spec ~left ~right = { Sides.left; right }

let test_partitions () =
  (* Partitions of \[n\] with singleton blocks dropped: one per way of
     choosing a partial partition into blocks of size ≥ 2. *)
  equal int 5 (List.length (Compiler.partitions [ 1; 2; 3 ]));
  equal int 15 (List.length (Compiler.partitions [ 1; 2; 3; 4 ]));
  equal int 52 (List.length (Compiler.partitions [ 1; 2; 3; 4; 5 ]))

let test_ties_one_side () =
  is_true (Compiler.ties_one_side [ left 0; left 1 ]);
  is_true (Compiler.ties_one_side [ right 0; right 1 ]);
  is_false (Compiler.ties_one_side [ left 0; right 0 ]);
  is_false (Compiler.ties_one_side [ left 0; left 1 ] |> not)

let factor_shape ~dims (comp : Component.t) =
  let term =
    match comp with
    | Component.Single t -> t
    | Component.Sum (t :: _) -> t
    | Component.Sum [] -> assert false
  in
  Array.of_list (List.map (term : Term.t).free ~f:(Index.dim_of ~dims))

let random_factors ~key ~dims components =
  List.mapi components ~f:(fun i comp ->
    Nx.Rng.normal (Nx.Rng.fold_in key i) Nx.float32 (factor_shape ~dims comp))

let random_perms ~key ~dims group_axes =
  List.mapi group_axes ~f:(fun i ids ->
    let dim = Index.dim_of ~dims (List.hd_exn ids) in
    Nx.Rng.permutation (Nx.Rng.fold_in key i) dim)

let check_coefficients ~dims components data =
  List.iter components ~f:(fun comp ->
    let expected =
      match comp with
      | Component.Single t -> Support.coefficient_brute ~dims t data
      | Component.Sum ts ->
        List.fold ts ~init:(Nx.scalar Nx.float32 0.0) ~f:(fun acc t ->
          Nx.add acc (Support.coefficient_brute ~dims t data))
    in
    let got = Component.coefficient ~dims comp (`Full data) in
    equal ~msg:"coefficient" (array (float 1e-5)) (Nx.to_array expected) (Nx.to_array got))

let check_basis ?(symmetric = false) ~name ~dims spec =
  let compiled = Compiler.compile ~symmetric ~dims spec in
  let components = compiled.basis.components in
  let data =
    Nx.Rng.normal (Nx.Rng.key 11) Nx.float32 (Array.of_list (dims.left @ dims.right))
  in
  check_coefficients ~dims components data;
  let design = Component.design_matrix ~dims components in
  let design64 = Nx.cast Nx.float64 design in
  is_true
    ~msg:(name ^ ": design matrix is symmetric")
    Float.(
      abs (Nx.item [] (Nx.max (Nx.abs (Nx.sub design (Nx.transpose design))))) < 1e-6);
  let positive_definite =
    match Nx.cholesky design64 with
    | _ -> true
    | exception Nx.Linalg_error _ -> false
  in
  let factors = random_factors ~key:(Nx.Rng.key 12) ~dims components in
  let block = compiled.dense_block ~factors in
  (* Invariance: A S Aᵀ = S for a random group element. *)
  let perms = random_perms ~key:(Nx.Rng.key 13) ~dims compiled.basis.group_axes in
  let transformed = compiled.transform ~perms block in
  equal
    ~msg:(name ^ ": invariance")
    (array (float 1e-4))
    (Nx.to_array block)
    (Nx.to_array transformed);
  (* The factor round trip is exact when the design system is invertible. *)
  if positive_definite
  then (
    let estimated = compiled.estimate_factors (`Full block) in
    List.iter2_exn factors estimated ~f:(fun expected got ->
      equal
        ~msg:(name ^ ": factor round trip")
        (array (float 1e-3))
        (Nx.to_array expected)
        (Nx.to_array got)));
  (* [apply_block] is the action of the dense block on one vector. *)
  let right_size = List.fold dims.right ~init:1 ~f:Int.( * ) in
  let v = Nx.Rng.normal (Nx.Rng.key 14) Nx.float32 [| right_size |] in
  let expected = Nx.matmul block (Nx.reshape [| right_size |] v) |> Nx.to_array in
  let got =
    compiled.apply_block
      ~factors
      (Nx.reshape (Array.append [| 1 |] (Array.of_list dims.right)) v)
    |> Nx.reshape [| -1 |]
    |> Nx.to_array
  in
  equal ~msg:(name ^ ": apply_block") (array (float 1e-4)) expected got

let test_first_order () =
  check_basis
    ~name:"first order"
    ~dims:(dims ~left:[ 2; 3 ] ~right:[])
    (spec ~left:[ Symmetry.Id; Perm 0 ] ~right:[ Symmetry.Absent ])

let test_same_group () =
  check_basis
    ~name:"same group both sides"
    ~dims:(dims ~left:[ 2; 3 ] ~right:[ 2; 3 ])
    (spec ~left:[ Symmetry.Id; Perm 0 ] ~right:[ Symmetry.Id; Symmetry.Perm 0 ])

let test_full_permutation () =
  (* A permutation group acts on axes of equal dimension only. *)
  check_basis
    ~name:"full permutation"
    ~dims:(dims ~left:[ 2; 2 ] ~right:[ 2; 2 ])
    (spec ~left:[ Symmetry.Perm 0; Perm 0 ] ~right:[ Symmetry.Perm 0; Perm 0 ])

let test_two_groups () =
  check_basis
    ~name:"two groups"
    ~dims:(dims ~left:[ 2; 2; 2 ] ~right:[ 2; 2 ])
    (spec
       ~left:[ Symmetry.Id; Perm 0; Perm 1 ]
       ~right:[ Symmetry.Perm 0; Symmetry.Perm 1 ])

let test_symmetric () =
  check_basis
    ~symmetric:true
    ~name:"symmetric merge"
    ~dims:(dims ~left:[ 2; 3 ] ~right:[ 2; 3 ])
    (spec ~left:[ Symmetry.Id; Perm 0 ] ~right:[ Symmetry.Id; Symmetry.Perm 0 ])

let test_compile_manual () =
  let d = dims ~left:[ 2; 3 ] ~right:[ 2; 3 ] in
  let s = spec ~left:[ Symmetry.Id; Perm 0 ] ~right:[ Symmetry.Id; Symmetry.Perm 0 ] in
  let auto = Compiler.compile ~symmetric:true ~dims:d s in
  let manual =
    Compiler.compile_manual
      ~symmetric:true
      ~dims:d
      ~group_axes:auto.basis.group_axes
      auto.basis.components
  in
  let factors = random_factors ~key:(Nx.Rng.key 15) ~dims:d auto.basis.components in
  equal
    ~msg:"manual compilation matches"
    (array (float 1e-6))
    (Nx.to_array (auto.dense_block ~factors))
    (Nx.to_array (manual.dense_block ~factors))

let tests =
  [ test "partitions enumerate set partitions of size >= 2" test_partitions
  ; test "ties one side" test_ties_one_side
  ; test "first order" test_first_order
  ; test "same group both sides" test_same_group
  ; test "full permutation" test_full_permutation
  ; test "two groups" test_two_groups
  ; test "symmetric merge" test_symmetric
  ; test "manual compilation" test_compile_manual
  ]

let () = run "symo compiler" tests
