(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Block (Wedderburn) coordinates of the commutant algebra of one compiled
   pair, and transport between sizes.

   For a compilation of [left = L] against [right = R], the commutant algebra
   is [⊕_λ M_{m_{L,λ}, m_{R,λ}}], one block per isotypic kind [λ] (a tuple of
   per-group partitions, see {!Kind}). The diagram basis used by the compiler
   is a size-dependent *encoding* of that algebra: the same factor list means
   a different operator at a different size. The *blocks* are the
   size-independent description. This module builds, at compile time,

   - the canonical test vectors of each side (per group, from {!Kind}), with
     one vector per (kind, copy);
   - the per-group table [P_g[d, λ, a, b] = <v_L, d v_R>] for every per-group
     diagram [d];
   - the full table [M] mapping the compiler's factor list to the block
     entries, and its inverse;
   - the Gram matrices of the copies (per kind), needed to turn inner products
     into the operator's matrix on the multiplicity space.

   [to_blocks] and [of_blocks] convert factors to blocks and back;
   [transport ~from_ ~to_] re-encodes an operator estimated at one size as the
   operator with the same blocks at another. Transport preserves the operator's
   distinct spectrum and any function of it (inverse, powers, square roots);
   only multiplicities — hence trace, determinant and norms — change.

   The dims must be in the stable range per group ([n >= 2k], see
   {!Kind.check_stable}); below it the algebra is not semisimple and the table
   is not invertible. *)

open Base

(* One kind per group, aligned with [Basis.group_ids]. *)
type kind = Kind.t list

let kind_equal = List.equal Kind.equal
let kind_to_string kind = String.concat ~sep:";" (List.map kind ~f:Kind.to_string)

(* Canonical key of a tie pattern, used to match a term's per-group diagram
   against the per-group compiler's components. *)
let tie_key ties =
  ties
  |> List.map ~f:(fun group ->
    List.map group ~f:(fun i -> Sexp.to_string (Index.sexp_of_t i))
    |> List.sort ~compare:String.compare
    |> String.concat ~sep:",")
  |> List.sort ~compare:String.compare
  |> String.concat ~sep:"|"

let cartesian xss =
  List.fold_right xss ~init:[ [] ] ~f:(fun xs acc ->
    List.concat_map xs ~f:(fun x -> List.map acc ~f:(fun prod -> x :: prod)))

(* The mixed-radix index of a tuple of per-group indices. *)
let encode radices tuple =
  List.fold2_exn radices tuple ~init:0 ~f:(fun acc radix i -> (acc * radix) + i)

let decode radices index =
  List.rev radices
  |> List.fold ~init:([], index) ~f:(fun (acc, index) radix ->
    Int.rem index radix :: acc, index / radix)
  |> fst

(* ------------------------------------------------------------------------
   Per-group data
   ------------------------------------------------------------------------ *)

type group =
  { n : int
  ; k_l : int
  ; k_r : int
  ; lpos : int array (* left axis positions in the leaf *)
  ; rpos : int array
  ; common : Kind.t list (* per-group kinds present on both sides *)
  ; table : float array (* [d; b_l; b_r], row-major *)
  ; index_of : (string, int) Hashtbl.t (* diagram key -> component *)
  ; gram_l : Nx.float32_t array (* per common kind *)
  ; gram_r : Nx.float32_t array
  ; d : int
  ; left_kinds : Kind.t array (* per flat test-vector index *)
  ; left_copies : int array
  ; right_kinds : Kind.t array
  ; right_copies : int array
  }

let flat_index kinds copies kind copy =
  let n = Array.length kinds in
  let rec go i =
    if i >= n
    then
      invalid_arg
        (Printf.sprintf
           "Symo.Blocks: no test vector for kind %s copy %d"
           (Kind.to_string kind)
           copy)
    else if Kind.equal kinds.(i) kind && Int.equal copies.(i) copy
    then i
    else go (i + 1)
  in
  go 0

let index_in array value =
  Array.findi array ~f:(fun _ v -> Int.equal v value) |> Option.map ~f:fst

let index_in_list list value =
  List.findi list ~f:(fun _ v -> Poly.equal v value) |> Option.map ~f:fst

let build_group ~n ~lpos ~rpos =
  let k_l = Array.length lpos
  and k_r = Array.length rpos in
  let kinds_l = Kind.stable_kinds ~k:k_l in
  let kinds_r = Kind.stable_kinds ~k:k_r in
  let common = List.filter kinds_l ~f:(fun k -> List.exists kinds_r ~f:(Kind.equal k)) in
  let side_spec k =
    if Int.equal k 0
    then [ Symmetry.Absent ]
    else List.init k ~f:(fun _ -> Symmetry.Perm 0)
  in
  let dims =
    { Sides.left = List.init k_l ~f:(fun _ -> n); right = List.init k_r ~f:(fun _ -> n) }
  in
  let spec = { Sides.left = side_spec k_l; right = side_spec k_r } in
  let compiler = Compiler.compile ~dims spec in
  let components = compiler.Compiler.basis.components in
  let d = List.length components in
  let index_of = Hashtbl.create (module String) in
  List.iteri components ~f:(fun i c ->
    match c with
    | Component.Single t -> Hashtbl.set index_of ~key:(tie_key t.Term.ties) ~data:i
    | Component.Sum _ -> assert false);
  let vectors_l =
    List.filter (Kind.test_vectors ~k:k_l ~n) ~f:(fun (kind, _, _) ->
      List.exists common ~f:(Kind.equal kind))
  in
  let vectors_r =
    List.filter (Kind.test_vectors ~k:k_r ~n) ~f:(fun (kind, _, _) ->
      List.exists common ~f:(Kind.equal kind))
  in
  let left_kinds = Array.of_list (List.map vectors_l ~f:(fun (k, _, _) -> k)) in
  let left_copies = Array.of_list (List.map vectors_l ~f:(fun (_, c, _) -> c)) in
  let right_kinds = Array.of_list (List.map vectors_r ~f:(fun (k, _, _) -> k)) in
  let right_copies = Array.of_list (List.map vectors_r ~f:(fun (_, c, _) -> c)) in
  let b_r = Array.length right_kinds in
  let left_mat =
    Nx.stack (List.map vectors_l ~f:(fun (_, _, v) -> Nx.reshape [| -1 |] v))
  in
  let right_batch = Nx.stack (List.map vectors_r ~f:(fun (_, _, v) -> v)) in
  let gram_of k vectors =
    let vs =
      List.filter_map vectors ~f:(fun (kind, _, v) ->
        if Kind.equal kind k then Some v else None)
      |> Array.of_list
    in
    let m = Array.length vs in
    let entries = Array.create ~len:(m * m) 0.0 in
    Array.iteri vs ~f:(fun i vi ->
      Array.iteri vs ~f:(fun j vj ->
        entries.((i * m) + j) <- Nx.item [] (Nx.sum (Nx.mul vi vj))));
    Nx.create Nx.float32 [| m; m |] entries
  in
  let gram_l = Array.of_list (List.map common ~f:(fun k -> gram_of k vectors_l)) in
  let gram_r = Array.of_list (List.map common ~f:(fun k -> gram_of k vectors_r)) in
  let table =
    List.init d ~f:(fun i ->
      let factors =
        List.init d ~f:(fun j ->
          Nx.scalar Nx.float32 (if Int.equal i j then 1.0 else 0.0))
      in
      compiler.Compiler.apply_block ~factors right_batch
      |> Nx.reshape [| b_r; -1 |]
      |> fun out -> Nx.matmul left_mat (Nx.transpose out))
    |> Nx.stack
    |> Nx.to_array
  in
  { n
  ; k_l
  ; k_r
  ; lpos
  ; rpos
  ; common
  ; table
  ; index_of
  ; gram_l
  ; gram_r
  ; d
  ; left_kinds
  ; left_copies
  ; right_kinds
  ; right_copies
  }

(* ------------------------------------------------------------------------
   The pair
   ------------------------------------------------------------------------ *)

type t =
  { group_ids : int list
  ; kinds : kind list
  ; copies_l : int array
  ; copies_r : int array
  ; free_l : int array (* the leaf's free ([Id]) dims, in axis order *)
  ; free_r : int array
  ; entries : (int * int * int) array (* kind, left copy, right copy *)
  ; table : Nx.float32_t (* [n_entries; n_components] *)
  ; inverse : Nx.float32_t (* [n_components; n_entries] *)
  }

let build (compiler : Compiler.t) =
  let dims = compiler.Compiler.dims in
  let basis = compiler.Compiler.basis in
  let group_ids = basis.group_ids in
  let groups =
    Array.of_list
      (List.mapi basis.group_axes ~f:(fun _ axes ->
         let lpos =
           List.filter_map axes ~f:(function
             | Index.Left i -> Some i
             | _ -> None)
           |> Array.of_list
         in
         let rpos =
           List.filter_map axes ~f:(function
             | Index.Right i -> Some i
             | _ -> None)
           |> Array.of_list
         in
         let n =
           if Array.length lpos > 0
           then List.nth_exn dims.left lpos.(0)
           else List.nth_exn dims.right rpos.(0)
         in
         build_group ~n ~lpos ~rpos))
  in
  let groups_list = Array.to_list groups in
  let kinds = cartesian (List.map groups_list ~f:(fun g -> g.common)) in
  let copies_l =
    Array.of_list
      (List.map kinds ~f:(fun kind ->
         List.fold2_exn kind groups_list ~init:1 ~f:(fun acc k g ->
           acc * Kind.stable_multiplicity ~k:g.k_l k)))
  in
  let copies_r =
    Array.of_list
      (List.map kinds ~f:(fun kind ->
         List.fold2_exn kind groups_list ~init:1 ~f:(fun acc k g ->
           acc * Kind.stable_multiplicity ~k:g.k_r k)))
  in
  let perm_axis side i =
    List.exists basis.group_axes ~f:(fun axes ->
      List.mem
        axes
        (if Poly.(side = `Left) then Index.Left i else Index.Right i)
        ~equal:Index.equal)
  in
  let free_of side dims =
    List.filter_mapi dims ~f:(fun i d -> if perm_axis side i then None else Some d)
    |> Array.of_list
  in
  let free_l = free_of `Left dims.left
  and free_r = free_of `Right dims.right in
  (* The copies are orthonormal, so the Grams are identities; computing them
     keeps the block formula [B = G_L⁻¹ P] independent of that choice. *)
  let gram_of side kind =
    let mats =
      List.map2_exn kind groups_list ~f:(fun k g ->
        let gram = if Poly.(side = `Left) then g.gram_l else g.gram_r in
        let i = Option.value_exn (index_in_list g.common k) in
        gram.(i))
    in
    match mats with
    | [] -> Nx.ones Nx.float32 [| 1; 1 |]
    | m :: rest -> List.fold rest ~init:m ~f:Nx.kron
  in
  let gram_l = Array.of_list (List.map kinds ~f:(gram_of `Left)) in
  let entries =
    List.concat
      (List.mapi kinds ~f:(fun ki _ ->
         List.concat
           (List.init copies_l.(ki) ~f:(fun a ->
              List.init copies_r.(ki) ~f:(fun b -> ki, a, b)))))
    |> Array.of_list
  in
  let n_entries = Array.length entries in
  let components = basis.components in
  let d = List.length components in
  let term_group_indices (term : Term.t) =
    Array.mapi groups ~f:(fun _ g ->
      let ties =
        List.filter term.Term.ties ~f:(fun group ->
          match group with
          | [] -> false
          | Index.Left i :: _ -> Option.is_some (index_in g.lpos i)
          | Index.Right i :: _ -> Option.is_some (index_in g.rpos i))
      in
      let map_index = function
        | Index.Left i -> Index.Left (Option.value_exn (index_in g.lpos i))
        | Index.Right i -> Index.Right (Option.value_exn (index_in g.rpos i))
      in
      let ties = List.map ties ~f:(List.map ~f:map_index) in
      Hashtbl.find_exn g.index_of (tie_key ties))
  in
  let component_entry c (ki, a, b) =
    let kind = List.nth_exn kinds ki in
    let a_parts =
      decode
        (List.map2_exn kind groups_list ~f:(fun k g ->
           Kind.stable_multiplicity ~k:g.k_l k))
        a
    in
    let b_parts =
      decode
        (List.map2_exn kind groups_list ~f:(fun k g ->
           Kind.stable_multiplicity ~k:g.k_r k))
        b
    in
    let product_of terms =
      List.fold terms ~init:1.0 ~f:(fun acc term ->
        let indices = term_group_indices term in
        let factors =
          List.mapi groups_list ~f:(fun j g ->
            let flat_l =
              flat_index
                g.left_kinds
                g.left_copies
                (List.nth_exn kind j)
                (List.nth_exn a_parts j)
            in
            let flat_r =
              flat_index
                g.right_kinds
                g.right_copies
                (List.nth_exn kind j)
                (List.nth_exn b_parts j)
            in
            let b_r = Array.length g.right_kinds in
            let b_l = Array.length g.left_kinds in
            g.table.((((indices.(j) * b_l) + flat_l) * b_r) + flat_r))
        in
        acc *. List.fold factors ~init:1.0 ~f:( *. ))
    in
    match c with
    | Component.Single t -> product_of [ t ]
    | Component.Sum ts -> List.fold ts ~init:0.0 ~f:(fun acc t -> acc +. product_of [ t ])
  in
  (* Raw inner products [P[(λ,a,b), c]] = <v_{L,a}, c v_{R,b}>, then the
     block [B = G_L⁻¹ P] on the left copies of each kind. *)
  let raw =
    Array.init (n_entries * d) ~f:(fun i ->
      component_entry (List.nth_exn components (Int.rem i d)) entries.(i / d))
  in
  let offsets =
    List.folding_map
      (List.mapi kinds ~f:(fun ki _ -> copies_l.(ki) * copies_r.(ki)))
      ~init:0
      ~f:(fun offset count -> offset + count, offset)
  in
  let table = Array.copy raw in
  List.iteri kinds ~f:(fun ki _ ->
    let m_l = copies_l.(ki)
    and m_r = copies_r.(ki) in
    let offset = List.nth_exn offsets ki in
    let ginv = Nx.inv (Nx.cast Nx.float64 gram_l.(ki)) |> Nx.to_array in
    for b = 0 to m_r - 1 do
      for a = 0 to m_l - 1 do
        for c = 0 to d - 1 do
          let acc =
            List.fold (List.range 0 m_l) ~init:0.0 ~f:(fun acc a' ->
              acc +. (ginv.((a * m_l) + a') *. raw.(((offset + (a' * m_r) + b) * d) + c)))
          in
          table.(((offset + (a * m_r) + b) * d) + c) <- acc
        done
      done
    done);
  let table = Nx.create Nx.float32 [| n_entries; d |] table in
  let inverse = Nx.pinv ~rtol:1e-10 (Nx.cast Nx.float64 table) |> Nx.cast Nx.float32 in
  { group_ids; kinds; copies_l; copies_r; free_l; free_r; entries; table; inverse }

(* ------------------------------------------------------------------------
   Factors <-> blocks
   ------------------------------------------------------------------------ *)

let to_blocks t (factors : Nx.float32_t list) =
  let d = List.length factors in
  let n_entries = Array.length t.entries in
  let m = Nx.to_array t.table in
  List.init n_entries ~f:(fun e ->
    List.foldi factors ~init:(Nx.scalar Nx.float32 0.0) ~f:(fun c acc f ->
      Nx.add acc (Nx.mul_s f m.((e * d) + c))))

let of_blocks t (blocks : Nx.float32_t list) =
  let d = (Nx.shape t.table).(1) in
  let n_entries = Array.length t.entries in
  let m = Nx.to_array t.inverse in
  List.init d ~f:(fun c ->
    List.foldi blocks ~init:(Nx.scalar Nx.float32 0.0) ~f:(fun e acc b ->
      Nx.add acc (Nx.mul_s b m.((c * n_entries) + e))))

(* ------------------------------------------------------------------------
   Block matrices
   ------------------------------------------------------------------------ *)

let free_size t =
  Array.fold t.free_l ~init:1 ~f:( * ), Array.fold t.free_r ~init:1 ~f:( * )

(* [block_matrices t blocks] is, per kind, the block as a matrix of shape
   [(m_l * f_l) x (m_r * f_r)]: the rows are (copy, free index) pairs. The
   copy bases are orthonormal, so the matrix is the operator's matrix in an
   orthonormal basis and is symmetric for a symmetric operator. *)
let block_matrices t blocks =
  let f_l, f_r = free_size t in
  let offsets =
    List.folding_map
      (List.mapi t.kinds ~f:(fun ki _ -> t.copies_l.(ki) * t.copies_r.(ki)))
      ~init:0
      ~f:(fun offset count -> offset + count, offset)
  in
  Array.of_list
    (List.mapi t.kinds ~f:(fun ki _ ->
       let m_l = t.copies_l.(ki)
       and m_r = t.copies_r.(ki) in
       let offset = List.nth_exn offsets ki in
       List.sub blocks ~pos:offset ~len:(m_l * m_r)
       |> List.map ~f:(fun e -> Nx.reshape [| f_l; f_r |] e)
       |> Nx.stack
       |> fun m ->
       Nx.reshape [| m_l; m_r; f_l; f_r |] m
       |> fun m ->
       Nx.transpose ~axes:[ 0; 2; 1; 3 ] m
       |> fun m -> Nx.reshape [| m_l * f_l; m_r * f_r |] m))

(* The inverse of {!block_matrices}: entry tensors from per-kind matrices. *)
let blocks_of_matrices t matrices =
  let f_l, f_r = free_size t in
  let free_shape = Array.append t.free_l t.free_r in
  Array.to_list
    (Array.mapi matrices ~f:(fun ki m ->
       let m_l = t.copies_l.(ki)
       and m_r = t.copies_r.(ki) in
       let m = Nx.reshape [| m_l; f_l; m_r; f_r |] m in
       let m = Nx.transpose ~axes:[ 0; 2; 1; 3 ] m in
       let m = Nx.reshape [| m_l * m_r; f_l; f_r |] m in
       List.init (m_l * m_r) ~f:(fun e ->
         Nx.slice [ Nx.I e ] m |> fun x -> Nx.reshape free_shape x)))
  |> List.concat

(* [transport ~from_ ~to_ factors] is the factor list, at [to_]'s dims, of the
   operator whose blocks are [from_]'s blocks of [factors]. *)
let transport ~from_ ~to_ factors =
  if not (List.equal kind_equal from_.kinds to_.kinds)
  then invalid_arg "Symo.Blocks.transport: the two structures do not have the same kinds";
  of_blocks to_ (to_blocks from_ factors)
