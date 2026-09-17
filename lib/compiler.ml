(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* JIT compilation of the commutation-equation basis.

   A symbolic symmetry specification is turned into the set of basis tensors
   of the invariant subspace ({!basis_of_spec}), and each basis element into
   runnable closures: an einsum that applies the corresponding curvature block
   to a batch of vectors, its dense block, and the linear solve that estimates
   the factors from data. [compile] is the entry point.

   Factorization of the small, constant design matrix of a basis is the only
   numeric work done at compile time (Cholesky, with a jittered retry and an
   SVD fallback for degenerate bases). The closures themselves are compositions
   of [Nx] operations — einsums, reshapes, broadcasts and triangular solves —
   so the whole per-step computation is traceable by [Rune.jit]. *)

open Base

(* A compiled basis, instantiated at one set of dimensions. *)
type t =
  { basis : Basis.t
  ; dims : int list Sides.t
  ; apply_block : factors:Nx.float32_t list -> Nx.float32_t -> Nx.float32_t
  ; dense_block : factors:Nx.float32_t list -> Nx.float32_t
  ; estimate_factors :
      [ `Full of Nx.float32_t | `Outer_product of Nx.float32_t * Nx.float32_t ]
      -> Nx.float32_t list
  ; transform : perms:Nx.int32_t list -> Nx.float32_t -> Nx.float32_t
  }

let is_scalar t = Array.length (Nx.shape t) = 0

(* The number of elements on each side of a compilation. *)
let side_counts ~(dims : int list Sides.t) =
  let product list = List.fold list ~init:1 ~f:Int.( * ) in
  product dims.left, product dims.right

(* Compile one term into [fun ~factor v -> ...]: [v] is a batch of vectors
   ([batch × dims.right]) and the result is [batch × dims.left]. Ties are
   contracted against delta operands, free axes travel in the factor, and axes
   involved in neither are broadcast along. *)
let compile_apply_term ~(dims : int list Sides.t) term =
  let dim_of = Index.dim_of ~dims in
  let normalizer = Term.normalization ~dims term in
  let label = Index.to_char in
  let involved = Term.indices_involved term in
  let left_ids = List.mapi dims.left ~f:(fun i _ -> Index.Left i) in
  let right_ids = List.mapi dims.right ~f:(fun i _ -> Index.Right i) in
  let output_entries =
    List.map left_ids ~f:(fun i ->
      if Set.mem involved i then `Char (label i) else `Broadcast)
  in
  let output_labels =
    List.filter_map output_entries ~f:(function
      | `Char c -> Some c
      | `Broadcast -> None)
  in
  let final_output = 'z' :: output_labels in
  let input_labels = 'z' :: List.map right_ids ~f:label in
  let deltas =
    List.map term.ties ~f:(fun group ->
      let dim = dim_of (List.hd_exn group) in
      Delta.tensor ~order:(List.length group) dim, List.map group ~f:label)
  in
  let factor_labels =
    match term.free with
    | [] -> None
    | ids -> Some (List.map ids ~f:label)
  in
  let output_view_shape =
    List.mapi output_entries ~f:(fun i -> function
      | `Char _ -> List.nth_exn dims.left i
      | `Broadcast -> 1)
  in
  let output_shape = Array.of_list dims.left in
  fun ~factor v ->
    if Option.is_none factor_labels then assert (is_scalar factor);
    let t, labels =
      List.fold deltas ~init:(v, input_labels) ~f:(fun (t, labels) (delta, tie_labels) ->
        let out =
          List.filter labels ~f:(fun c -> not (List.mem tie_labels c ~equal:Char.equal))
          @ List.filter tie_labels ~f:(fun c -> not (List.mem labels c ~equal:Char.equal))
        in
        Contract.binary ~labels_a:labels ~labels_b:tie_labels ~out t delta, out)
    in
    let t =
      match factor_labels with
      | None ->
        (* A term with no free axes carries a scalar factor; it must still
           multiply the contraction. [Nx.mul] broadcasts the scalar, and
           unlike [Nx.item] it stays traceable under [Rune.jit]. *)
        assert (is_scalar factor);
        Nx.mul (Contract.permute_sum ~labels ~output:final_output t) factor
      | Some factor_labels ->
        Contract.binary
          ~labels_a:factor_labels
          ~labels_b:labels
          ~out:final_output
          factor
          t
    in
    let batch = (Nx.shape v).(0) in
    let r = Nx.reshape (Array.append [| batch |] (Array.of_list output_view_shape)) t in
    let r = Nx.mul_s r normalizer in
    Nx.broadcast_to (Array.append [| batch |] output_shape) r

let compile_apply_component ~dims = function
  | Component.Single term -> compile_apply_term ~dims term
  | Component.Sum terms ->
    let applies = List.map terms ~f:(compile_apply_term ~dims) in
    fun ~factor v ->
      List.fold applies ~init:(Nx.scalar Nx.float32 0.0) ~f:(fun accu apply ->
        Nx.add accu (apply ~factor v))

(* [apply ~factors v] applies the block described by [factors] to a batch of
   vectors. *)
let compile_apply ~dims components =
  let applies = List.map components ~f:(compile_apply_component ~dims) in
  fun ~factors v ->
    List.fold2_exn
      factors
      applies
      ~init:(Nx.scalar Nx.float32 0.0)
      ~f:(fun acc factor apply -> Nx.add acc (apply ~factor v))

(* The SVD pseudo-inverse, used only as a compile-time fallback for a design
   matrix that is not positive definite. Singular values are cut at a relative
   tolerance, unlike the old implementation's unregularized division. *)
let pseudo_inverse m =
  let n = (Nx.shape m).(0) in
  let u, s, _ = Nx.svd m in
  let inv_s =
    Nx.where
      (Nx.greater s (Nx.scalar Nx.float64 (1e-12 *. Nx.item [ 0 ] s)))
      (Nx.div (Nx.ones Nx.float64 [| n |]) s)
      (Nx.zeros Nx.float64 [| n |])
  in
  Nx.matmul (Nx.mul u (Nx.reshape [| 1; n |] inv_s)) (Nx.transpose u)

(* Factor the constant design matrix once, at compile time: Cholesky when it
   is positive definite (the generic case for a basis of independent
   components), a jittered Cholesky when it is only positive semidefinite, and
   the SVD pseudo-inverse as a last resort. *)
let build_estimator design n =
  let design64 = Nx.cast Nx.float64 design in
  let try_cholesky m =
    match Nx.cholesky m with
    | l -> Some l
    | exception Nx.Linalg_error _ -> None
  in
  match try_cholesky design64 with
  | Some l -> `Cholesky (Nx.cast Nx.float32 l)
  | None ->
    let trace = Nx.item [] (Nx.sum (Nx.diag design64)) in
    let jitter = Nx.mul_s (Nx.eye Nx.float64 n) Float.(1e-8 * trace / of_int n) in
    (match try_cholesky (Nx.add design64 jitter) with
     | Some l -> `Cholesky (Nx.cast Nx.float32 l)
     | None ->
       Stdio.eprintf
         "Symo.Compiler: design matrix is not positive definite (n=%d); falling back to \
          the SVD pseudo-inverse\n\
          %!"
         n;
       `Inverse (Nx.cast Nx.float32 (pseudo_inverse design64)))

(* [estimate_factors data] projects [data] onto the basis and solves the
   constant design system for the factor of each component. With a Cholesky
   factor [L] the solve is two triangular solves; the explicit inverse is only
   the fallback. Both paths are traceable. *)
let compile_estimate_factors ~(dims : int list Sides.t) components =
  let n = List.length components in
  let estimator = build_estimator (Component.design_matrix ~dims components) n in
  fun data ->
    let b =
      List.map components ~f:(fun c ->
        Component.coefficient ~dims c data
        |> fun x -> Nx.reshape (Array.append [| 1 |] (Nx.shape x)) x)
      |> Nx.concatenate ~axis:0
    in
    let factor_shape =
      Array.sub (Nx.shape b) ~pos:1 ~len:(Array.length (Nx.shape b) - 1)
    in
    let b = Nx.reshape [| n; -1 |] b in
    let v =
      match estimator with
      | `Cholesky l ->
        let y = Nx.solve_triangular l b in
        Nx.solve_triangular ~transpose:true l y
      | `Inverse inv -> Nx.matmul inv b
    in
    List.mapi components ~f:(fun i _ -> Nx.reshape factor_shape (Nx.slice [ Nx.I i ] v))

(* Apply one group element to a parameter-shaped tensor. [perms] holds one
   permutation per group id (in [group_axes] order); every axis tied to a
   group is permuted by that group's permutation. *)
let transform ~(dims : int list Sides.t) group_axes =
  let axis_of = Index.axis_of ~dims in
  fun ~perms t ->
    let original_shape = Nx.shape t in
    let t = Nx.reshape (Array.of_list (dims.left @ dims.right)) t in
    let t =
      List.fold2_exn group_axes perms ~init:t ~f:(fun t ids perm ->
        List.fold ids ~init:t ~f:(fun x id -> Nx.take ~axis:(axis_of id) ~indices:perm x))
    in
    Nx.reshape original_shape t

(* All set partitions of [lst], keeping each partition's blocks of size ≥ 2
   (singleton blocks are dropped, so a partition need not cover [lst]). *)
let partitions (lst : 'a list) : 'a list list list =
  let add_element acc x =
    List.concat_map acc ~f:(fun p ->
      let new_block = p @ [ [ x ] ] in
      let add_to_existing =
        List.mapi p ~f:(fun i _ ->
          List.mapi p ~f:(fun j b -> if Int.equal i j then x :: b else b))
      in
      new_block :: add_to_existing)
  in
  List.fold lst ~init:[ [] ] ~f:add_element
  |> List.map
       ~f:
         (List.filter ~f:(function
            | [ _ ] -> false
            | _ -> true))

(* All ways of picking one element from each sublist. *)
let cartesian_product (xss : 'a list list) : 'a list list =
  List.fold_right xss ~init:[ [] ] ~f:(fun xs acc ->
    List.concat_map xs ~f:(fun x -> List.map acc ~f:(fun prod -> x :: prod)))

let rec dedup_map xs ~equal =
  match xs with
  | [] -> []
  | x :: rest ->
    let dups, others = List.partition_tf rest ~f:(fun y -> equal x y) in
    (x :: dups) :: dedup_map others ~equal

(* Merge a term with its transpose into one symmetric component. *)
let bundle_transposes terms =
  dedup_map terms ~equal:(fun x y -> Term.equal x (Term.transpose y))
  |> List.map ~f:(function
    | [ a ] -> Component.Single a
    | [ a; _ ] -> Component.Sum [ a; Term.sort (Term.transpose a) ]
    | _ -> assert false)

let ties_one_side ids =
  let c_left, c_right =
    List.fold ids ~init:(0, 0) ~f:(fun (l, r) -> function
      | Index.Left _ -> l + 1, r
      | Index.Right _ -> l, r + 1)
  in
  c_left > 1 || c_right > 1

(* From a symmetry specification to the symbolic basis.

   - the free axes are the [Id] axes of both sides;
   - each [Perm i] ties its occurrences together, and all partitions of those
     occurrences into blocks of size ≥ 2 enumerate the allowed delta
     structures;
   - the cartesian product across groups gives every term;
   - if [ensure_size_invariance=true], second-order terms that tie two or more
     dimensions on the same side (left or right) are rejected, as empirically
     they appear to break the size invariance of the surrogate;
   - when symmetry is required, a term and its transpose are merged into one
     component. *)
let basis_of_spec_uncached ~ensure_size_invariance ~symmetric (symm : Symmetry.t)
  : Basis.t
  =
  let open Symmetry in
  let is_first_order = Poly.(symm.right = [ Absent ] || symm.left = [ Absent ]) in
  let left i = Index.Left i in
  let right i = Index.Right i in
  let free =
    let discover_free cons init spec =
      List.foldi spec ~init ~f:(fun i acc spec ->
        match spec with
        | Id -> cons i :: acc
        | _ -> acc)
    in
    discover_free right (discover_free left [] symm.left) symm.right
    |> List.sort ~compare:Index.compare
  in
  let unique_perms = Hashtbl.create (module Int) in
  let find_perms cons spec =
    List.iteri spec ~f:(fun i spec ->
      match spec with
      | Perm id -> Hashtbl.add_multi unique_perms ~key:id ~data:(cons i)
      | _ -> ())
  in
  find_perms left symm.left;
  find_perms right symm.right;
  (* Sorted by group id: [transform] takes one permutation per entry of
     [group_axes], and different leaves of the same model must agree on
     which group each entry belongs to, so a deterministic order is part of
     the contract. *)
  let group_ids = Hashtbl.keys unique_perms |> List.sort ~compare:Int.compare in
  let unique_perms =
    Hashtbl.to_alist unique_perms
    |> List.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b)
    |> List.map ~f:(fun (_, ids) -> List.sort ids ~compare:Index.compare)
  in
  let possible_ties = List.map unique_perms ~f:partitions in
  let components =
    cartesian_product possible_ties
    |> List.map ~f:List.concat
    |> List.map ~f:(fun ties -> Term.{ ties; free })
    |> List.map ~f:Term.sort
  in
  let components =
    if is_first_order || not ensure_size_invariance
    then components
    else
      List.filter components ~f:(fun term ->
        List.fold term.Term.ties ~init:true ~f:(fun so_far ids ->
          so_far && not (ties_one_side ids)))
  in
  let components =
    if symmetric
    then bundle_transposes components
    else List.map components ~f:(fun t -> Component.Single t)
  in
  { label = Symmetry.label_of symm
  ; symmetric
  ; components
  ; group_axes = unique_perms
  ; group_ids
  }

(* The basis depends only on the symmetry specification — not on the
   dimensions — so it is memoized: second-order compilation asks for n²
   specifications, and repeated layer shapes revisit the same ones (all
   diagonal pairs, every pair of identical layers, and the surrogate
   compilation of every basis). Only the enumeration is shared; the closures
   and the design matrix are rebuilt per [dims]. *)
let basis_cache : (string, Basis.t) Hashtbl.t = Hashtbl.create (module String)

let basis_cache_key ~symmetric (symm : Symmetry.t) =
  let side specs =
    List.map specs ~f:(function
      | Symmetry.Absent -> "A"
      | Symmetry.Id -> "I"
      | Symmetry.Perm i -> "P" ^ Int.to_string i)
    |> String.concat ~sep:","
  in
  String.concat
    ~sep:"|"
    [ (if symmetric then "sym" else "asym"); side symm.left; side symm.right ]

let basis_of_spec ~ensure_size_invariance ~symmetric spec =
  let key = basis_cache_key ~symmetric spec in
  match Hashtbl.find basis_cache key with
  | Some basis -> basis
  | None ->
    let basis = basis_of_spec_uncached ~ensure_size_invariance ~symmetric spec in
    Hashtbl.set basis_cache ~key ~data:basis;
    basis

let compile_basis ~(dims : int list Sides.t) (basis : Basis.t) =
  let components = basis.components in
  let apply_block = compile_apply ~dims components in
  let dense_block ~factors =
    let _, right_size = side_counts ~dims in
    let v =
      Nx.eye Nx.float32 right_size
      |> Nx.reshape (Array.append [| right_size |] (Array.of_list dims.right))
    in
    apply_block ~factors v
    |> Nx.reshape [| right_size; -1 |]
    |> fun x -> Nx.transpose ~axes:[ 1; 0 ] x
  in
  let estimate_factors = compile_estimate_factors ~dims components in
  let transform = transform ~dims basis.group_axes in
  { basis; dims; apply_block; dense_block; estimate_factors; transform }

let compile ?(symmetric = false) ~ensure_size_invariance ~dims spec =
  compile_basis ~dims (basis_of_spec ~ensure_size_invariance ~symmetric spec)

let compile_manual ?(symmetric = false) ~dims ~group_axes components =
  let group_ids = List.init (List.length group_axes) ~f:Fn.id in
  compile_basis
    ~dims
    { Basis.label = "manual"; symmetric; components; group_axes; group_ids }
