(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Orbit machinery.

   {!Compiler} compiles one symmetry specification at one set of dimensions
   into closures. [First_order] and [Second_order] lift that to a whole
   parameter tree: they instantiate one [Compiler.t] per leaf and expose

   - factor estimation from parameters or from a dense matrix,
   - dense assembly and splitting,
   - the first-order orbit average [R1] and the second-order operator [R2],
   - the *block* coordinates of the second-order operator (see {!Blocks}) and
     transport between sizes.

   Everything here is a composition of tree traversals and [Nx] operations: no
   host read of a tensor value, no dynamic shapes, no RNG, so the eager
   functions are traceable by [Rune.jit] as they stand. The per-leaf compiled
   tables are built when the functor is instantiated; their design matrices and
   block tables are compile-time constants. *)

open Base

module type Model = sig
  include Nx.Ptree.S

  val dims : int list t

  (** One spec per axis of the parameter tensor: [Absent] pads a side with no
      axis, [Id] marks an untouched ("free") axis, and [Perm i] marks an axis
      transformed by group [i]. *)
  val symmetries : Symmetry.spec list t
end

module type S = sig
  type 'a t

  val dims : int list t
  val symmetries : Symmetry.spec list t

  (** [random_transform ~key theta] applies a random element of the global
      symmetry group to the parameter tree [theta]: one uniformly random
      permutation per group id in {!Model.symmetries}, applied to every axis
      tagged [Perm id] (all such axes share the group's dimension); [Id] axes
      are untouched. The element is drawn from [key] with
      {!Nx.Rng.fold_in}, so the function is pure and traces under
      [Rune.jit] — pass the key as an input leaf there, as [Rune.jit] rejects
      a captured key. *)
  val random_transform : key:Nx.Rng.t -> Nx.float32_t t -> Nx.float32_t t

  module First_order : sig
    type factors = Nx.float32_t list t

    (** The per-tensor compilers. *)
    val large : Compiler.t t

    val factors_of_params : Nx.float32_t t -> factors
    val factors_of_dense : Nx.float32_t -> factors
    val dense_of_factors : factors -> Nx.float32_t
    val params_of_dense : Nx.float32_t -> Nx.float32_t t
    val params_of_factors : factors -> Nx.float32_t t

    (** The nearest invariant point, [R1]. *)
    val orbit_average : Nx.float32_t t -> Nx.float32_t t
  end

  module Second_order : sig
    (** One factor list per pair of leaves: row [i] holds the blocks
        [(i, j)] for every leaf [j], in traversal order. *)
    type factors = Nx.float32_t list array t

    val large : ?symmetric:bool -> unit -> Compiler.t array t
    val factors_of_pair : ?symmetric:bool -> Nx.float32_t t -> Nx.float32_t t -> factors
    val factors_of_dense : ?symmetric:bool -> Nx.float32_t -> factors
    val dense_of_factors : ?symmetric:bool -> factors -> Nx.float32_t

    (** [apply ~factors v] applies the second-order operator assembled from
        [factors] to the parameter tree [v]. *)
    val apply : ?symmetric:bool -> factors:factors -> Nx.float32_t t -> Nx.float32_t t

    (** The global kinds, in the canonical order of the assembled blocks. *)
    val kinds : Blocks.kind list

    (** [blocks_of_factors factors] is the operator's blocks: one square
        matrix per global kind, of size [sum_i m_{i,λ} f_i] ([m] the kind's
        multiplicity on leaf [i], [f] the leaf's free size). The copy bases are
        orthonormal, so a symmetric operator's blocks are symmetric. *)
    val blocks_of_factors : ?symmetric:bool -> factors -> Nx.float32_t list

    (** The inverse of {!blocks_of_factors}. *)
    val factors_of_blocks : ?symmetric:bool -> Nx.float32_t list -> factors

    (** The factor lists packed into one flat tensor (a compile-time layout),
        so the optimizer's state and the jit boundary stay tensor-valued. *)
    val factors_size : int

    val pack : factors -> Nx.float32_t
    val unpack : Nx.float32_t -> factors

    (** [transport ~dims factors] re-encodes the operator at new dims (the
        permuted axes' dimensions; free axes are unchanged), preserving its
        blocks, hence its distinct spectrum and any function of it. *)
    val transport : ?symmetric:bool -> dims:int list t -> factors -> factors
  end
end

let product dims = List.fold dims ~init:1 ~f:Int.( * )

(* The offsets at which the leaves start, in traversal order. *)
let offsets sizes =
  let _, acc =
    List.fold sizes ~init:(0, []) ~f:(fun (offset, acc) size ->
      offset + size, offset :: acc)
  in
  Array.of_list (List.rev acc)

(* Split a flat vector at the given boundaries; [starts] has one entry per
   leaf, in traversal order. *)
let split_flat starts dense =
  Nx.array_split ~axis:0 (`Indices (List.tl_exn (Array.to_list starts))) dense
  |> Array.of_list

(* [extend ~groups ~all_groups kind] is [kind], a tuple over [groups], written
   as a tuple over [all_groups] (the empty kind for the groups it omits). *)
let extend ~groups ~all_groups kind =
  List.map all_groups ~f:(fun g ->
    match List.findi groups ~f:(fun _ g' -> Int.equal g g') with
    | Some (i, _) -> List.nth_exn kind i
    | None -> [])

module Make (M : Model) : S with type 'a t = 'a M.t = struct
  module P = Nx.Ptree.Payload

  type 'a t = 'a M.t

  let dims = M.dims
  let symmetries = M.symmetries

  (* The leaves in traversal order. *)
  let to_list t = P.fold (module M) (fun _ x acc -> x :: acc) t [] |> List.rev
  let to_array t = Array.of_list (to_list t)
  let n_leaves = List.length (to_list M.dims)

  (* The traversal index of every leaf. *)
  let leaf_index =
    let next = ref (-1) in
    P.map
      (module M)
      (fun _ _ ->
         Int.incr next;
         !next)
      M.dims

  let specs_arr = to_array symmetries
  let dims_arr = to_array M.dims
  let leaf_size_full = Array.map dims_arr ~f:product

  (* The size of a leaf's free ([Id]) subspace: the product of its free axes'
     dimensions, which is the shape of the factors. *)
  let leaf_free =
    Array.mapi specs_arr ~f:(fun i specs ->
      List.fold2_exn specs dims_arr.(i) ~init:1 ~f:(fun acc spec d ->
        match spec with
        | Symmetry.Id -> acc * d
        | Symmetry.Perm _ | Symmetry.Absent -> acc))

  (* The model's permutation group ids, sorted, and, per leaf and group, the
     number of axes tagged with that group. *)
  let group_ids =
    List.concat_map (Array.to_list specs_arr) ~f:(fun specs ->
      List.filter_map specs ~f:(function
        | Symmetry.Perm id -> Some id
        | _ -> None))
    |> List.dedup_and_sort ~compare:Int.compare

  let leaf_k =
    Array.map specs_arr ~f:(fun specs ->
      List.map group_ids ~f:(fun g ->
        List.count specs ~f:(function
          | Symmetry.Perm id -> Int.equal id g
          | _ -> false)))

  module First_order = struct
    type factors = Nx.float32_t list t

    let compile d =
      P.map2
        (module M)
        (fun _ specs dims ->
           Compiler.compile
             ~dims:{ Sides.left = dims; right = [] }
             { Sides.left = specs; right = [ Symmetry.Absent ] })
        symmetries
        d

    let large = compile dims

    let factors_of_params x =
      P.map2
        (module M)
        (fun _ c x ->
           c.Compiler.estimate_factors (`Outer_product (x, Nx.scalar Nx.float32 1.0)))
        large
        x

    let dense_of_factors factors =
      P.map2 (module M) (fun _ c f -> c.Compiler.dense_block ~factors:f) large factors
      |> to_list
      |> Nx.concatenate ~axis:0

    let factors_of_dense dense =
      let pieces = split_flat (offsets (Array.to_list leaf_size_full)) dense in
      P.map2
        (module M)
        (fun _ c i -> c.Compiler.estimate_factors (`Full pieces.(i)))
        large
        leaf_index

    let params_of_factors factors =
      P.map2
        (module M)
        (fun _ c f ->
           c.Compiler.apply_block ~factors:f (Nx.ones Nx.float32 [| 1 |])
           |> Nx.reshape (Array.of_list c.Compiler.dims.left))
        large
        factors

    let params_of_dense dense = params_of_factors (factors_of_dense dense)
    let orbit_average x = params_of_factors (factors_of_params x)
  end

  (* The global group: every axis tagged [Perm id] anywhere in the tree is
     transformed by the same group element, so its dimension must be the same
     in every leaf. [group_dims] is the ascending list of [(id, dim)] pairs. *)
  let group_dims =
    List.concat_map
      (List.mapi (Array.to_list specs_arr) ~f:(fun i specs ->
         List.zip_exn specs dims_arr.(i)))
      ~f:Fn.id
    |> List.rev_filter_map ~f:(fun (spec, dim) ->
      match spec with
      | Symmetry.Perm id -> Some (id, dim)
      | Symmetry.Id | Symmetry.Absent -> None)
    |> List.fold ~init:[] ~f:(fun acc (id, dim) ->
      match List.Assoc.find acc id ~equal:Int.equal with
      | None -> (id, dim) :: acc
      | Some existing ->
        if Int.equal existing dim
        then acc
        else
          invalid_arg
            (Printf.sprintf
               "Symo.Orbit: group %d acts on axes of different dimensions (%d and %d)"
               id
               existing
               dim))
    |> List.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b)

  (* [Nx.Rng.permutation] builds its sort keys by slicing the two columns of
     an [n × 2] block and reshapes the result; under [Rune.jit] that slice is
     miscompiled (the jitted permutation is not even a permutation), so draw
     the sort keys with one float64 uniform instead. Its 53 random bits make
     the permutation uniform for every practical [n], and the eager and
     jitted draws agree exactly. *)
  let random_permutation ~key n =
    Nx.argsort (Nx.Rng.uniform key Nx.float64 [| n |]) ~axis:0 ~descending:false

  (* Draw the group element and apply it leafwise: each compiler's
     [group_ids] selects the permutations its own axes are tied to, so leaves
     that share a group id are acted on by the same permutation. *)
  let random_transform ~key theta =
    let perms =
      List.map group_dims ~f:(fun (id, dim) ->
        id, random_permutation ~key:(Nx.Rng.fold_in key id) dim)
    in
    P.map2
      (module M)
      (fun _ c x ->
         let perms =
           List.map c.Compiler.basis.group_ids ~f:(fun id ->
             List.Assoc.find_exn perms id ~equal:Int.equal)
         in
         c.Compiler.transform ~perms x)
      First_order.large
      theta

  module Second_order = struct
    type factors = Nx.float32_t list array t

    let compile ~symmetric d =
      let dims_arr = to_array d in
      let symms_arr = to_array symmetries in
      let n = Array.length dims_arr in
      P.map2
        (module M)
        (fun _ _ i ->
           Array.init n ~f:(fun j ->
             Compiler.compile
               ~symmetric:(symmetric && Int.equal i j)
               ~dims:{ Sides.left = dims_arr.(i); right = dims_arr.(j) }
               { Sides.left = symms_arr.(i); right = symms_arr.(j) }))
        d
        leaf_index

    let large_sym = compile ~symmetric:true dims
    let large_asym = compile ~symmetric:false dims
    let large ?(symmetric = true) () = if symmetric then large_sym else large_asym

    (* [pair_blocks sym.(i).(j)] is the [Blocks.t] of the pair [(i, j)]. *)
    let build_blocks cs = Array.map (to_array cs) ~f:(Array.map ~f:Blocks.build)
    let pair_blocks_sym = build_blocks large_sym
    let pair_blocks_asym = build_blocks large_asym

    let pair_blocks ?(symmetric = true) () =
      if symmetric then pair_blocks_sym else pair_blocks_asym

    let factors_of_pair ?(symmetric = true) a b =
      let cs = large ~symmetric () in
      let a_arr = to_array a in
      let b_arr = to_array b in
      P.map2
        (module M)
        (fun _ row i ->
           Array.mapi row ~f:(fun j c ->
             c.Compiler.estimate_factors (`Outer_product (a_arr.(i), b_arr.(j)))))
        cs
        leaf_index

    let dense_of_factors ?(symmetric = true) factors =
      let cs = large ~symmetric () in
      let f_arr = to_array factors in
      let rows =
        P.map2
          (module M)
          (fun _ row i ->
             Array.map2_exn row f_arr.(i) ~f:(fun c f ->
               c.Compiler.dense_block ~factors:f)
             |> Array.to_list
             |> Nx.concatenate ~axis:1)
          cs
          leaf_index
        |> to_list
      in
      let dense = Nx.concatenate ~axis:0 rows in
      if symmetric then Nx.mul_s (Nx.add dense (Nx.transpose dense)) 0.5 else dense

    let factors_of_dense ?(symmetric = true) dense =
      let cs = large ~symmetric () in
      let starts = offsets (Array.to_list leaf_size_full) in
      let row_pieces = split_flat starts dense in
      P.map2
        (module M)
        (fun _ compilers i ->
           let dense_row = row_pieces.(i) in
           let col_pieces =
             Nx.array_split
               ~axis:1
               (`Indices (List.tl_exn (Array.to_list starts)))
               dense_row
             |> Array.of_list
           in
           Array.mapi compilers ~f:(fun j c ->
             c.Compiler.estimate_factors (`Full col_pieces.(j))))
        cs
        leaf_index

    let apply ?(symmetric = true) ~factors v =
      let cs = large ~symmetric () in
      let f_arr = to_array factors in
      let v_arr = to_array v in
      P.map2
        (module M)
        (fun _ row i ->
           let _, acc =
             Array.fold2_exn
               row
               f_arr.(i)
               ~init:(0, Nx.scalar Nx.float32 0.0)
               ~f:(fun (j, acc) c f ->
                 let vj = v_arr.(j) in
                 let vj = Nx.unsqueeze ~axes:[ 0 ] vj in
                 let z = c.Compiler.apply_block ~factors:f vj in
                 j + 1, Nx.add acc (Nx.reshape (Array.of_list c.Compiler.dims.left) z))
           in
           acc)
        cs
        leaf_index

    (* ----------------------------------------------------------------------
       Blocks
       ---------------------------------------------------------------------- *)

    let all_pair_blocks = List.concat_map (Array.to_list pair_blocks_sym) ~f:Array.to_list

    (* The global kinds: every pair's kinds, written over all the model's
       groups (the groups a pair omits have the empty kind there). *)
    let kinds =
      List.concat_map all_pair_blocks ~f:(fun b ->
        List.map b.Blocks.kinds ~f:(fun kind ->
          extend ~groups:b.Blocks.group_ids ~all_groups:group_ids kind))
      |> List.dedup_and_sort ~compare:(fun a b ->
        String.compare (Blocks.kind_to_string a) (Blocks.kind_to_string b))

    let kinds_arr = Array.of_list kinds

    (* Per global kind: the multiplicity on each leaf, the leaf's block size,
       and its offset in the assembled matrix (inactive leaves have size 0). *)
    let leaf_mult =
      Array.map kinds_arr ~f:(fun kind ->
        Array.init n_leaves ~f:(fun i ->
          List.fold2_exn kind leaf_k.(i) ~init:1 ~f:(fun acc k kg ->
            acc * Kind.stable_multiplicity ~k:kg k)))

    let leaf_size =
      Array.map leaf_mult ~f:(fun mult ->
        Array.mapi mult ~f:(fun i m -> m * leaf_free.(i)))

    let offsets_of sizes =
      let _, acc =
        Array.fold sizes ~init:(0, []) ~f:(fun (offset, acc) size ->
          offset + size, (if Int.equal size 0 then 0 else offset) :: acc)
      in
      Array.of_list (List.rev acc)

    let leaf_offset = Array.map leaf_size ~f:offsets_of

    (* For each pair, the global index of each local kind, and the local index
       of each global kind. *)
    let pair_global, pair_local =
      let pairs =
        Array.init n_leaves ~f:(fun i ->
          Array.init n_leaves ~f:(fun j ->
            let b = pair_blocks_sym.(i).(j) in
            Array.of_list
              (List.map b.Blocks.kinds ~f:(fun kind ->
                 let kind =
                   extend ~groups:b.Blocks.group_ids ~all_groups:group_ids kind
                 in
                 Array.findi kinds_arr ~f:(fun _ k -> List.equal Kind.equal k kind)
                 |> Option.value_exn
                 |> fst))))
      in
      let local =
        Array.init n_leaves ~f:(fun i ->
          Array.init n_leaves ~f:(fun j ->
            Array.init (Array.length kinds_arr) ~f:(fun gk ->
              Array.findi pairs.(i).(j) ~f:(fun _ g -> Int.equal g gk)
              |> Option.map ~f:fst)))
      in
      pairs, local

    let blocks_of_factors ?(symmetric = true) factors =
      let pair_blocks = pair_blocks ~symmetric () in
      let f_arr = to_array factors in
      let pair_matrices =
        Array.init n_leaves ~f:(fun i ->
          Array.init n_leaves ~f:(fun j ->
            Blocks.block_matrices
              pair_blocks.(i).(j)
              (Blocks.to_blocks pair_blocks.(i).(j) f_arr.(i).(j))))
      in
      List.mapi kinds ~f:(fun gk _ ->
        let rows =
          List.filter_map (List.range 0 n_leaves) ~f:(fun i ->
            if Int.equal leaf_size.(gk).(i) 0
            then None
            else (
              let cols =
                List.filter_map (List.range 0 n_leaves) ~f:(fun j ->
                  if Int.equal leaf_size.(gk).(j) 0
                  then None
                  else (
                    match pair_local.(i).(j).(gk) with
                    | None -> None
                    | Some local -> Some pair_matrices.(i).(j).(local)))
              in
              Some (Nx.concatenate ~axis:1 cols)))
        in
        Nx.concatenate ~axis:0 rows)

    let factors_of_blocks ?(symmetric = true) blocks =
      let pair_blocks = pair_blocks ~symmetric () in
      let blocks = Array.of_list blocks in
      P.map2
        (module M)
        (fun _ compilers i ->
           Array.mapi compilers ~f:(fun j _ ->
             let b = pair_blocks.(i).(j) in
             let matrices =
               Array.map
                 pair_global.(i).(j)
                 ~f:(fun gk ->
                   let matrix = blocks.(gk) in
                   let rows = leaf_offset.(gk).(i)
                   and cols = leaf_offset.(gk).(j) in
                   let r = leaf_size.(gk).(i)
                   and c = leaf_size.(gk).(j) in
                   if Int.equal r 0 || Int.equal c 0
                   then Nx.zeros Nx.float32 [| r; c |]
                   else Nx.slice [ Nx.R (rows, rows + r); Nx.R (cols, cols + c) ] matrix)
             in
             Blocks.of_blocks b (Blocks.blocks_of_matrices b matrices)))
        large_sym
        leaf_index

    (* ----------------------------------------------------------------------
       Flat packing
       ---------------------------------------------------------------------- *)

    let factors_size =
      Array.fold
        (Array.map pair_blocks_sym ~f:(fun row ->
           Array.fold row ~init:0 ~f:(fun acc b ->
             acc
             + ((Nx.shape b.Blocks.table).(1)
                * fst (Blocks.free_size b)
                * snd (Blocks.free_size b)))))
        ~init:0
        ~f:( + )

    let pack factors =
      let f_arr = to_array factors in
      let pieces =
        List.concat
          (List.mapi (Array.to_list f_arr) ~f:(fun i row ->
             List.mapi (Array.to_list row) ~f:(fun j fs ->
               List.map fs ~f:(fun f -> Nx.reshape [| -1 |] f) |> Nx.concatenate ~axis:0)))
      in
      Nx.concatenate ~axis:0 pieces

    let unpack flat =
      (* Walk the layout in the same order as [pack]. *)
      let chunks = ref [] in
      let offset = ref 0 in
      Array.iteri pair_blocks_sym ~f:(fun _ row ->
        Array.iteri row ~f:(fun _ b ->
          let d = (Nx.shape b.Blocks.table).(1) in
          let f_l, f_r = Blocks.free_size b in
          let shape = Array.append b.Blocks.free_l b.Blocks.free_r in
          let size = f_l * f_r in
          for c = 0 to d - 1 do
            let start = !offset + (c * size) in
            let x = Nx.slice [ Nx.R (start, start + size) ] flat in
            chunks := Nx.reshape shape x :: !chunks
          done;
          offset := !offset + (d * size)));
      let chunks = List.rev !chunks in
      let next = ref chunks in
      let take n =
        let rec go acc n l =
          if Int.equal n 0
          then List.rev acc, l
          else (
            match l with
            | [] -> assert false
            | x :: rest -> go (x :: acc) (n - 1) rest)
        in
        let taken, rest = go [] n !next in
        next := rest;
        taken
      in
      P.map2
        (module M)
        (fun _ _ i ->
           Array.map pair_blocks_sym.(i) ~f:(fun b ->
             let d = (Nx.shape b.Blocks.table).(1) in
             take d))
        large_sym
        leaf_index

    let transport ?(symmetric = true) ~dims:new_dims factors =
      let new_arr = to_array new_dims in
      let f_arr = to_array factors in
      let pair_blocks = pair_blocks ~symmetric () in
      let new_blocks =
        Array.init n_leaves ~f:(fun i ->
          Array.init n_leaves ~f:(fun j ->
            Compiler.compile
              ~symmetric:(symmetric && Int.equal i j)
              ~dims:{ Sides.left = new_arr.(i); right = new_arr.(j) }
              { Sides.left = specs_arr.(i); right = specs_arr.(j) }
            |> Blocks.build))
      in
      P.map2
        (module M)
        (fun _ row i ->
           Array.mapi row ~f:(fun j _ ->
             Blocks.transport
               ~from_:pair_blocks.(i).(j)
               ~to_:new_blocks.(i).(j)
               f_arr.(i).(j)))
        large_sym
        leaf_index
  end
end
