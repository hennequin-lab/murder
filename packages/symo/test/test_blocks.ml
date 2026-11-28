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
open Nx
open Windtrap
open Symo

let scalar x = Nx.scalar float32 x

let random_factors key d =
  List.init d ~f:(fun i ->
    scalar (item [] (Nx.Rng.normal ~key:(Nx.Rng.fold_in key i) float32 [||])))

let round4 x = Float.round (x *. 1e4) /. 1e4

let distinct m =
  let w, _ = eigh float64 (cast float64 m) in
  Array.to_list (to_array w)
  |> List.map ~f:round4
  |> List.dedup_and_sort ~compare:Float.compare

let symmetrize m = Infix.((m + transpose m) *$ 0.5)

let block_eigenvalues matrices =
  Array.to_list matrices
  |> List.concat_map ~f:(fun m ->
    let w, _ = eigh float64 (cast float64 (symmetrize m)) in
    Array.to_list (to_array w) |> List.map ~f:round4)
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
    equal ~msg:"factor round trip" (float 1e-5) (item [] a) (item [] b))

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
    equal ~msg:"transport round trip" (float 1e-5) (item [] a) (item [] b));
  let blocks_n = Blocks.to_blocks bn f
  and blocks_m = Blocks.to_blocks bm fm in
  List.iter2_exn blocks_n blocks_m ~f:(fun a b ->
    equal ~msg:"transported block" (float 1e-5) (item [] a) (item [] b));
  equal
    ~msg:"transported spectrum"
    (list (float 1e-4))
    (distinct (symmetrize (cn.Compiler.dense_block ~factors:f)))
    (distinct (symmetrize (cm.Compiler.dense_block ~factors:fm)))

(* Below the stable range the [2] kind is gone, so the operator has one block
   fewer and the factors that belonged to it have no block to reach: the
   spectrum is still the dense one, and transport between the two sizes is
   refused rather than silently dropping a block. *)
let test_below_stable () =
  let cs = compile ~n:3 ~symmetric:true in
  let cl = compile ~n:4 ~symmetric:true in
  let bs = Blocks.build cs
  and bl = Blocks.build cl in
  equal ~msg:"kinds at n = 3" int 3 (List.length bs.Blocks.kinds);
  equal ~msg:"kinds at n = 4" int 4 (List.length bl.Blocks.kinds);
  equal
    ~msg:"n = 3 blocks"
    int
    14
    (List.length
       (Blocks.to_blocks
          bs
          (random_factors (Nx.Rng.key 21) (List.length cs.Compiler.basis.components))));
  (* the spectrum, with the [2] kind's eigenvalue absent at n = 3 *)
  let factors =
    random_factors (Nx.Rng.key 22) (List.length cs.Compiler.basis.components)
  in
  let dense = symmetrize (cs.Compiler.dense_block ~factors) in
  let matrices = Blocks.block_matrices bs (Blocks.to_blocks bs factors) in
  equal
    ~msg:"block spectrum below the stable range"
    (list (float 1e-4))
    (distinct dense)
    (block_eigenvalues matrices);
  (* factors round-trip: the table is a bijection on the operators here *)
  let back = Blocks.of_blocks bs (Blocks.to_blocks bs factors) in
  List.iter2_exn factors back ~f:(fun a b ->
    equal
      ~msg:"factor round trip below the stable range"
      (float 1e-4)
      (item [] a)
      (item [] b));
  raises_match (Exn.invalid_arg ~sub:"do not have the same kinds") (fun () ->
    ignore (Blocks.transport ~from_:bs ~to_:bl factors : _ list))

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
      Nx.Rng.normal ~key:(Nx.Rng.fold_in (Nx.Rng.key 19) i) float32 [| f; f |])
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

(* The blockwise preconditioner agrees with the dense one: the blocks of
   [(S_g^{1/2} + λI)^{-1}] reconstructed through the compiler match the dense
   functional calculus. *)
let test_blockwise_preconditioner () =
  let n = 6 in
  let c = compile ~n ~symmetric:true in
  let b = Blocks.build c in
  let d = List.length c.Compiler.basis.components in
  let fg = random_factors (Nx.Rng.key 31) d in
  let sg = symmetrize (c.Compiler.dense_block ~factors:fg) in
  let blocks = Blocks.block_matrices b (Blocks.to_blocks b fg) in
  let check_form ~msg ~dense ~blocks =
    let factors =
      Blocks.of_blocks b (Blocks.blocks_of_matrices b (Array.of_list blocks))
    in
    let reconstructed = c.Compiler.dense_block ~factors in
    equal
      ~msg
      (array (float 1e-4))
      (to_array (cast float64 dense))
      (to_array (cast float64 reconstructed))
  in
  check_form
    ~msg:"blockwise (S_g^1/2 + lam I)^-1"
    ~dense:(Solve.damped_inverse_root ~damping:0.1 sg)
    ~blocks:(Solve.damped_inverse_root_blocks ~damping:0.1 (Array.to_list blocks));
  check_form
    ~msg:"blockwise (S_g + lam I)^-1"
    ~dense:(Solve.damped_inverse ~damping:0.1 sg)
    ~blocks:(Solve.damped_inverse_blocks ~damping:0.1 (Array.to_list blocks))

(* Two permuted axes with a free axis: the bundled basis is incomplete, so the
   compiler falls back to the full one, and the blocks carry the operator's
   spectrum and functional calculus all the same. *)
let test_two_perms_free () =
  let n = 4
  and f = 3 in
  let dims = { Sides.left = [ n; n; f ]; right = [ n; n; f ] } in
  let spec =
    { Sides.left = [ Symmetry.Perm 0; Symmetry.Perm 0; Symmetry.Id ]
    ; right = [ Symmetry.Perm 0; Symmetry.Perm 0; Symmetry.Id ]
    }
  in
  let c = Compiler.compile ~symmetric:true ~dims spec in
  is_true ~msg:"full basis" (not c.Compiler.basis.symmetric);
  let b = Blocks.build c in
  let d = List.length c.Compiler.basis.components in
  let factors =
    let x =
      c.Compiler.dense_block
        ~factors:
          (List.init d ~f:(fun i ->
             Nx.Rng.normal ~key:(Nx.Rng.fold_in (Nx.Rng.key 39) i) float32 [| f; f |]))
    in
    c.Compiler.estimate_factors (`Full Infix.(x *@ transpose x))
  in
  let dense = symmetrize (c.Compiler.dense_block ~factors) in
  let matrices = Blocks.block_matrices b (Blocks.to_blocks b factors) in
  equal
    ~msg:"multi-perm free-axis block spectrum"
    (list (float 1e-4))
    (distinct dense)
    (block_eigenvalues matrices);
  (* The damped inverse square root is a function of the operator: the blocks'
     calculus and the dense one agree. *)
  let lam = 0.3 in
  let p =
    c.Compiler.dense_block
      ~factors:
        (Blocks.of_blocks
           b
           (Blocks.blocks_of_matrices
              b
              (Array.of_list
                 (Solve.damped_inverse_root_blocks ~damping:lam (Array.to_list matrices)))))
  in
  equal
    ~msg:"blockwise damped inverse square root"
    (array (float 1e-4))
    (to_array (Solve.damped_inverse_root ~damping:lam dense))
    (to_array p)

let tests =
  [ test "blocks reproduce the k=2 spectrum" test_blocks_spectrum
  ; test "blocks reproduce the asymmetric spectrum" test_blocks_spectrum_asym
  ; test "factors round-trip through blocks" test_blocks_round_trip
  ; test "transport preserves blocks and spectrum" test_transport
  ; test "free axes" test_free_axes
  ; test "two groups" test_two_groups
  ; test "below the stable range" test_below_stable
  ; test "two permuted axes with a free axis" test_two_perms_free
  ; test "blockwise preconditioner" test_blockwise_preconditioner
  ]

let () = Stdlib.exit (run "symo blocks" tests)
