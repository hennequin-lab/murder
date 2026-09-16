(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Orbit machinery.

   {!Compiler} compiles one symmetry specification at one set of dimensions
   into closures. [First_order] and [Second_order] lift that to a whole
   parameter tree: they instantiate one [Compiler.t] per leaf — at full size
   ("large") and at surrogate size ("small", every permuted axis collapsed to
   one) — and expose

   - factor estimation from parameters or from a dense surrogate,
   - dense surrogate assembly and splitting,
   - the first-order orbit average [R1] and the second-order operator [R2].

   Everything here is a composition of tree traversals and [Nx] operations: no
   host read of a tensor value, no dynamic shapes, no RNG, so the eager
   functions are traceable by [Rune.jit] as they stand. The per-leaf compiled
   tables are built when the functor is instantiated; their design matrices are
   compile-time constants. *)

open Base

module type Model = sig
  include Nx.Ptree.Uniform

  val dims : int list t

  (** One spec per axis of the parameter tensor: [Absent] pads a side with no
      axis, [Id] marks an untouched ("free") axis, and [Perm i] marks an axis
      transformed by group [i]. *)
  val symmetries : Symmetry.spec list t

  (** The mock dimension a permuted axis takes at surrogate size: the small
      group the estimator works on (2 by convention). It must be at least 2 so
      the surrogate group is non-trivial. *)
  val surrogate_dim : int
end

module type S = sig
  type 'a t

  val dims : int list t
  val symmetries : Symmetry.spec list t
  val surrogate_dims : int list t

  module First_order : sig
    type factors = Nx.float32_t list t

    (** The per-tensor compilers at full and at surrogate size. *)
    val large : Compiler.t t
    val small : Compiler.t t

    val factors_of_params : Nx.float32_t t -> factors
    val factors_of_dense : Nx.float32_t -> factors
    val dense_of_factors : factors -> Nx.float32_t
    val params_of_dense : Nx.float32_t -> Nx.float32_t t

    (** The nearest invariant point, [R1]. *)
    val orbit_average : Nx.float32_t t -> Nx.float32_t t
  end

  module Second_order : sig
    (** One factor list per pair of leaves: row [i] holds the blocks
        [(i, j)] for every leaf [j], in traversal order. *)
    type factors = Nx.float32_t list array t

    val large : ?symmetric:bool -> unit -> Compiler.t array t
    val small : ?symmetric:bool -> unit -> Compiler.t array t

    val factors_of_pair :
      ?symmetric:bool -> Nx.float32_t t -> Nx.float32_t t -> factors

    val factors_of_dense : ?symmetric:bool -> Nx.float32_t -> factors
    val dense_of_factors : ?symmetric:bool -> factors -> Nx.float32_t

    (** [apply ~factors v] applies the second-order operator assembled from
        [factors] to the parameter tree [v]. *)
    val apply : ?symmetric:bool -> factors:factors -> Nx.float32_t t -> Nx.float32_t t
  end
end

let product dims = List.fold dims ~init:1 ~f:Int.( * )

(* The offsets at which the small leaves start, in traversal order. *)
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

module Make (M : Model) : S with type 'a t = 'a M.t = struct
  type 'a t = 'a M.t

  let dims = M.dims
  let symmetries = M.symmetries

  (* The surrogate keeps [Id] axes at full size and shrinks permuted axes to
     the model's non-trivial mock dimension (see {!Symmetry.surrogate_dims}). *)
  let surrogate_dims =
    M.map2
      (fun specs dims -> Symmetry.surrogate_dims ~surrogate_dim:M.surrogate_dim specs dims)
      M.symmetries
      M.dims

  (* Dense surrogate layout: leaves are concatenated in traversal order; the
     [i]-th leaf occupies [starts.(i) ..] for its own [small_sizes.(i)]
     coordinates. *)
  let small_sizes = M.map product surrogate_dims
  let dense_size = M.fold (fun _ acc size -> acc + size) 0 small_sizes
  let starts = offsets (M.fold (fun _ acc size -> size :: acc) [] small_sizes |> List.rev)

  (* The traversal index of every leaf, threaded through the tree so that
     splits and reconstructions line up with [fold]'s order. *)
  let leaf_index =
    let paths = M.fold (fun path acc _ -> path :: acc) [] surrogate_dims |> List.rev in
    let table = Hashtbl.create (module String) in
    List.iteri paths ~f:(fun i path -> Hashtbl.set table ~key:path ~data:i);
    M.map (fun path -> Hashtbl.find_exn table path) (M.names surrogate_dims)

  (* The leaves in traversal order. *)
  let to_array t = M.fold (fun _ acc x -> x :: acc) [] t |> List.rev |> Array.of_list

  let split_dense dense =
    let pieces = split_flat starts dense in
    M.map (fun i -> pieces.(i)) leaf_index

  module First_order = struct
    type factors = Nx.float32_t list t

    let compile d =
      M.map2
        (fun specs dims ->
          Compiler.compile
            ~dims:{ Sides.left = dims; right = [] }
            { Sides.left = specs; right = [ Symmetry.Absent ] })
        symmetries
        d

    let large = compile dims
    let small = compile surrogate_dims

    let factors_of_params x =
      M.map2
        (fun c x ->
          c.Compiler.estimate_factors (`Outer_product (x, Nx.scalar Nx.float32 1.0)))
        large
        x

    let dense_of_factors factors =
      M.map2 (fun c f -> c.Compiler.dense_block ~factors:f) small factors
      |> fun blocks ->
      M.fold (fun _ acc b -> b :: acc) [] blocks |> List.rev
      |> Nx.concatenate ~axis:0

    let factors_of_dense dense =
      let pieces = split_dense dense in
      M.map2 (fun c piece -> c.Compiler.estimate_factors (`Full piece)) small pieces

    let params_of_dense dense =
      let factors = factors_of_dense dense in
      M.map2
        (fun c f ->
          c.Compiler.apply_block ~factors:f (Nx.ones Nx.float32 [| 1 |])
          |> Nx.reshape (Array.of_list c.Compiler.dims.left))
        large
        factors

    let orbit_average x = params_of_dense (dense_of_factors (factors_of_params x))
  end

  module Second_order = struct
    type factors = Nx.float32_t list array t

    let compile ~symmetric d =
      let dims_arr = to_array d in
      let symms_arr = to_array symmetries in
      let n = Array.length dims_arr in
      M.map2
        (fun _ i ->
          Array.init n ~f:(fun j ->
            Compiler.compile
              ~symmetric:(symmetric && Int.equal i j)
              ~dims:{ Sides.left = dims_arr.(i); right = dims_arr.(j) }
              { Sides.left = symms_arr.(i); right = symms_arr.(j) }))
        d
        leaf_index

    let large_sym = compile ~symmetric:true dims
    let large_asym = compile ~symmetric:false dims
    let small_sym = compile ~symmetric:true surrogate_dims
    let small_asym = compile ~symmetric:false surrogate_dims
    let large ?(symmetric = true) () = if symmetric then large_sym else large_asym
    let small ?(symmetric = true) () = if symmetric then small_sym else small_asym

    let factors_of_pair ?(symmetric = true) a b =
      let cs = large ~symmetric () in
      let a_arr = to_array a in
      let b_arr = to_array b in
      M.map2
        (fun row i ->
          Array.mapi row ~f:(fun j c ->
            c.Compiler.estimate_factors (`Outer_product (a_arr.(i), b_arr.(j)))))
        cs
        leaf_index

    let dense_of_factors ?(symmetric = true) factors =
      let cs = small ~symmetric () in
      let rows =
        M.map2
          (fun row fs ->
            Array.map2_exn row fs ~f:(fun c f -> c.Compiler.dense_block ~factors:f)
            |> Array.to_list |> Nx.concatenate ~axis:1)
          cs
          factors
        |> fun rows -> M.fold (fun _ acc r -> r :: acc) [] rows |> List.rev
      in
      let dense = Nx.concatenate ~axis:0 rows in
      if symmetric then Nx.mul_s (Nx.add dense (Nx.transpose dense)) 0.5 else dense

    let factors_of_dense ?(symmetric = true) dense =
      let cs = small ~symmetric () in
      let row_pieces = split_flat starts dense in
      M.map2
        (fun compilers i ->
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
      M.map2
        (fun row i ->
          let _, acc =
            Array.fold2_exn
              row
              f_arr.(i)
              ~init:(0, Nx.scalar Nx.float32 0.0)
              ~f:(fun (j, acc) c f ->
                let vj = v_arr.(j) in
                let vj = Nx.reshape (Array.append [| 1 |] (Nx.shape vj)) vj in
                let z = c.Compiler.apply_block ~factors:f vj in
                j + 1, Nx.add acc (Nx.reshape (Array.of_list c.Compiler.dims.left) z))
          in
          Nx.contiguous acc)
        cs
        leaf_index
  end
end
