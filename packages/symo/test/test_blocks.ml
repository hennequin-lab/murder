(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Block coordinates, transport, and the blockwise estimator.

   The commutant algebra of a compiled pair is [⊕_λ M_{m_λ}]; the blocks are
   its size-independent coordinates. The tests check the blocks against the
   dense operator's spectrum (the copy basis is orthonormal, so the blocks'
   eigenvalues are the operator's), that transport preserves them across
   sizes, and that the blockwise estimator agrees with the dense one. *)

open Base
open Windtrap
open Symo

let scalar x = Nx.scalar Nx.float32 x

let random_factors key d =
  List.init d ~f:(fun i ->
    scalar (Nx.item [] (Nx.Rng.normal (Nx.Rng.fold_in key i) Nx.float32 [||])))

let round4 x = Float.round (x *. 1e4) /. 1e4

let distinct m =
  let w, _ = Nx.eigh (Nx.cast Nx.float64 m) in
  Array.to_list (Nx.to_array w)
  |> List.map ~f:round4
  |> List.dedup_and_sort ~compare:Float.compare

let symmetrize m = Nx.mul_s (Nx.add m (Nx.transpose m)) 0.5

let block_eigenvalues matrices =
  Array.to_list matrices
  |> List.concat_map ~f:(fun m ->
    let w, _ = Nx.eigh (Nx.cast Nx.float64 (symmetrize m)) in
    Array.to_list (Nx.to_array w) |> List.map ~f:round4)
  |> List.dedup_and_sort ~compare:Float.compare

let spec_k2 =
  { Sides.left = [ Symmetry.Perm 0; Symmetry.Perm 0 ]
  ; right = [ Symmetry.Perm 0; Symmetry.Perm 0 ]
  }

let compile ~n ~symmetric =
  Compiler.compile ~symmetric ~dims:{ Sides.left = [ n; n ]; right = [ n; n ] } spec_k2

(* The blocks of a [k = 2] operator reproduce its distinct spectrum. *)
let test_blocks_spectrum () =
  let n = 8 in
  let c = compile ~n ~symmetric:true in
  let b = Blocks.build c in
  let d = List.length c.Compiler.basis.components in
  let f = random_factors (Nx.Rng.key 7) d in
  let dense = symmetrize (c.Compiler.dense_block ~factors:f) in
  let matrices = Blocks.block_matrices b (Blocks.to_blocks b f) in
  equal
    ~msg:"block spectrum"
    (list (float 1e-4))
    (distinct dense)
    (block_eigenvalues matrices)

(* The blocks of the asymmetric compilation too (15 components). *)
let test_blocks_spectrum_asym () =
  let n = 8 in
  let c = compile ~n ~symmetric:false in
  let b = Blocks.build c in
  let d = List.length c.Compiler.basis.components in
  let f = random_factors (Nx.Rng.key 11) d in
  let dense = symmetrize (c.Compiler.dense_block ~factors:f) in
  let matrices = Blocks.block_matrices b (Blocks.to_blocks b f) in
  equal
    ~msg:"asymmetric block spectrum"
    (list (float 1e-4))
    (distinct dense)
    (block_eigenvalues matrices)

(* [of_blocks ∘ to_blocks = id]. *)
let test_blocks_round_trip () =
  let n = 6 in
  let c = compile ~n ~symmetric:false in
  let b = Blocks.build c in
  let d = List.length c.Compiler.basis.components in
  let f = random_factors (Nx.Rng.key 13) d in
  let back = Blocks.of_blocks b (Blocks.to_blocks b f) in
  List.iter2_exn f back ~f:(fun a b ->
    equal ~msg:"factor round trip" (float 1e-5) (Nx.item [] a) (Nx.item [] b))

(* Transport preserves the blocks (and hence the spectrum); multiplicities
   change with the size, so only the distinct values are compared. *)
let test_transport () =
  let n = 8
  and m = 12 in
  let cn = compile ~n ~symmetric:true
  and cm = compile ~n:m ~symmetric:true in
  let bn = Blocks.build cn
  and bm = Blocks.build cm in
  let d = List.length cn.Compiler.basis.components in
  let f = random_factors (Nx.Rng.key 17) d in
  let fm = Blocks.transport ~from_:bn ~to_:bm f in
  let fn = Blocks.transport ~from_:bm ~to_:bn fm in
  List.iter2_exn f fn ~f:(fun a b ->
    equal ~msg:"transport round trip" (float 1e-5) (Nx.item [] a) (Nx.item [] b));
  let blocks_n = Blocks.to_blocks bn f
  and blocks_m = Blocks.to_blocks bm fm in
  List.iter2_exn blocks_n blocks_m ~f:(fun a b ->
    equal ~msg:"transported block" (float 1e-5) (Nx.item [] a) (Nx.item [] b));
  equal
    ~msg:"transported spectrum"
    (list (float 1e-4))
    (distinct (symmetrize (cn.Compiler.dense_block ~factors:f)))
    (distinct (symmetrize (cm.Compiler.dense_block ~factors:fm)))

(* Free axes and several groups: the blocks reproduce the dense spectrum. *)
let test_free_axes () =
  let n = 8
  and f = 3 in
  let spec =
    { Sides.left = [ Symmetry.Perm 0; Symmetry.Id ]
    ; right = [ Symmetry.Perm 0; Symmetry.Id ]
    }
  in
  let c =
    Compiler.compile
      ~symmetric:true
      ~dims:{ Sides.left = [ n; f ]; right = [ n; f ] }
      spec
  in
  let b = Blocks.build c in
  let d = List.length c.Compiler.basis.components in
  let factors =
    List.init d ~f:(fun i ->
      Nx.Rng.normal (Nx.Rng.fold_in (Nx.Rng.key 19) i) Nx.float32 [| f; f |])
  in
  let dense = symmetrize (c.Compiler.dense_block ~factors) in
  let matrices = Blocks.block_matrices b (Blocks.to_blocks b factors) in
  equal
    ~msg:"free-axis block spectrum"
    (list (float 1e-4))
    (distinct dense)
    (block_eigenvalues matrices)

let test_two_groups () =
  let n = 8 in
  let spec =
    { Sides.left = [ Symmetry.Perm 0; Symmetry.Perm 1 ]
    ; right = [ Symmetry.Perm 0; Symmetry.Perm 1 ]
    }
  in
  let c =
    Compiler.compile
      ~symmetric:true
      ~dims:{ Sides.left = [ n; n ]; right = [ n; n ] }
      spec
  in
  let b = Blocks.build c in
  let d = List.length c.Compiler.basis.components in
  let f = random_factors (Nx.Rng.key 23) d in
  let dense = symmetrize (c.Compiler.dense_block ~factors:f) in
  let matrices = Blocks.block_matrices b (Blocks.to_blocks b f) in
  equal
    ~msg:"two-group block spectrum"
    (list (float 1e-4))
    (distinct dense)
    (block_eigenvalues matrices)

(* The blockwise estimator agrees with the dense one: the blocks of [H_inv]
   reconstructed through the compiler match the dense functional calculus. *)
let test_blockwise_estimator () =
  let n = 6 in
  let c = compile ~n ~symmetric:true in
  let b = Blocks.build c in
  let d = List.length c.Compiler.basis.components in
  let fw = random_factors (Nx.Rng.key 29) d in
  let fg = random_factors (Nx.Rng.key 31) d in
  let sw = symmetrize (c.Compiler.dense_block ~factors:fw) in
  let sg = symmetrize (c.Compiler.dense_block ~factors:fg) in
  let dense = Solve.hessian_inverse ~damping:1e-3 sw sg in
  let bw = Blocks.block_matrices b (Blocks.to_blocks b fw) in
  let bg = Blocks.block_matrices b (Blocks.to_blocks b fg) in
  let bh =
    Solve.hessian_inverse_blocks ~damping:1e-3 (Array.to_list bw) (Array.to_list bg)
  in
  let factors = Blocks.of_blocks b (Blocks.blocks_of_matrices b (Array.of_list bh)) in
  let reconstructed = c.Compiler.dense_block ~factors in
  equal
    ~msg:"blockwise H_inv"
    (array (float 1e-4))
    (Nx.to_array (Nx.cast Nx.float64 dense))
    (Nx.to_array (Nx.cast Nx.float64 reconstructed))

let tests =
  [ test "blocks reproduce the k=2 spectrum" test_blocks_spectrum
  ; test "blocks reproduce the asymmetric spectrum" test_blocks_spectrum_asym
  ; test "factors round-trip through blocks" test_blocks_round_trip
  ; test "transport preserves blocks and spectrum" test_transport
  ; test "free axes" test_free_axes
  ; test "two groups" test_two_groups
  ; test "blockwise estimator" test_blockwise_estimator
  ]

let () = Stdlib.exit (run "symo blocks" tests)
