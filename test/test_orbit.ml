(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The orbit machinery against exact group enumeration.

   For small specifications the symmetry group is finite and small enough to
   enumerate, so the exact orbit averages R1 and R2 can be computed by
   averaging [Compiler.transform] over every group element. [First_order] and
   [Second_order] must agree with those averages, on a single leaf and across
   a tree: the projection onto the invariant subspace is the group average. *)

open Base
open Windtrap
open Symo

(* ------------------------------------------------------------------------
   Models under test
   ------------------------------------------------------------------------ *)

(* One leaf: a [2; 3] tensor whose second axis is permuted. The free axis is
   the first one, so the surrogate is [2; 1]. *)
module Single = struct
  type 'a t = { w : 'a } [@@deriving ptree]

  let dims : int list t = { w = [ 2; 3 ] }
  let symmetries : Symmetry.spec list t = { w = [ Id; Perm 0 ] }
end

module O = Orbit.Make (Single)

(* One leaf with two independent permutation groups on two axes. *)
module Two_groups = struct
  type 'a t = { w : 'a } [@@deriving ptree]

  let dims : int list t = { w = [ 2; 2; 3 ] }
  let symmetries : Symmetry.spec list t = { w = [ Id; Perm 0; Perm 1 ] }
end

module O2 = Orbit.Make (Two_groups)

(* Two leaves: [w] carries group 0 on its second axis, [v] group 1 on its
   first. The groups are independent, so the global group is a product. *)
module Multi = struct
  type 'a t = { w : 'a; v : 'a } [@@deriving ptree]

  let dims : int list t = { w = [ 2; 3 ]; v = [ 3 ] }
  let symmetries : Symmetry.spec list t = { w = [ Id; Perm 0 ]; v = [ Perm 1 ] }
end

module OM = Orbit.Make (Multi)

(* ------------------------------------------------------------------------
   Group enumeration
   ------------------------------------------------------------------------ *)

(* All permutations of [0 .. n-1], as the [take] index tensors the compiler
   expects. *)
let permutations n =
  let rec go acc = function
    | [] -> [ List.rev acc ]
    | xs ->
      List.concat_map xs ~f:(fun x ->
        go (x :: acc) (List.filter xs ~f:(fun y -> not (Int.equal x y))))
  in
  List.map (go [] (List.range 0 n)) ~f:(fun p ->
    Nx.create Nx.int32 [| n |] (Array.of_list (List.map p ~f:Int32.of_int_trunc)))

let cartesian xss =
  List.fold xss ~init:[ [] ] ~f:(fun acc xs ->
    List.concat_map acc ~f:(fun prefix -> List.map xs ~f:(fun x -> prefix @ [ x ])))

(* One permutation per group of [c], in the compiler's own [group_axes]
   order. *)
let group_elements (c : Compiler.t) =
  let dims =
    List.map c.basis.group_axes ~f:(fun ids ->
      Index.dim_of ~dims:c.dims (List.hd_exn ids))
  in
  cartesian (List.map dims ~f:permutations)

let random_perms (c : Compiler.t) =
  List.mapi c.basis.group_axes ~f:(fun i ids ->
    let dim = Index.dim_of ~dims:c.dims (List.hd_exn ids) in
    Nx.Rng.permutation (Nx.Rng.fold_in (Nx.Rng.key 13) i) dim)

(* The permutations of [c] selected from a global assignment indexed by group
   id. *)
let perms_of_ids (c : Compiler.t) assignment =
  List.map c.basis.group_ids ~f:(fun id -> assignment.(id))

(* ------------------------------------------------------------------------
   Exact references
   ------------------------------------------------------------------------ *)

let vec x = Nx.reshape [| -1 |] (Nx.contiguous x)

let mean tensors n = Nx.mul_s (List.reduce_exn tensors ~f:Nx.add) (1. /. Float.of_int n)

(* [R1(x)] for one compiled leaf, by enumeration. *)
let exact_r1 (c : Compiler.t) x =
  let elements = group_elements c in
  let x = Nx.reshape (Array.of_list c.dims.left) x in
  mean
    (List.map elements ~f:(fun perms -> c.transform ~perms x))
    (List.length elements)

(* The outer product of a flat leaf with itself, laid out as the block of the
   second-order operator: shape [dims.left @ dims.right]. *)
let outer_self (c : Compiler.t) x =
  let left = c.dims.left and right = c.dims.right in
  let x = Nx.reshape (Array.of_list left) x in
  let a_shape = Array.of_list (left @ List.map right ~f:(fun _ -> 1)) in
  let b_shape = Array.of_list (List.map left ~f:(fun _ -> 1) @ right) in
  Nx.mul (Nx.reshape a_shape x) (Nx.reshape b_shape x)

let outer_pair (c : Compiler.t) a b =
  let left = c.dims.left and right = c.dims.right in
  let a = Nx.reshape (Array.of_list left) a in
  let b = Nx.reshape (Array.of_list right) b in
  let a_shape = Array.of_list (left @ List.map right ~f:(fun _ -> 1)) in
  let b_shape = Array.of_list (List.map left ~f:(fun _ -> 1) @ right) in
  Nx.mul (Nx.reshape a_shape a) (Nx.reshape b_shape b)

(* [R2(x, x)] for one compiled pair, by enumeration. *)
let exact_r2 (c : Compiler.t) x =
  let elements = group_elements c in
  let base = outer_self c x in
  mean (List.map elements ~f:(fun perms -> c.transform ~perms base)) (List.length elements)

(* ------------------------------------------------------------------------
   Checks
   ------------------------------------------------------------------------ *)

let check_leaf ~msg ~tol expected got =
  equal ~msg (array (float tol)) (Nx.to_array expected) (Nx.to_array got)

(* Structural index of each leaf, to rebuild trees from flat arrays. *)
let indices tree =
  let paths = Single.fold (fun path acc _ -> path :: acc) [] tree |> List.rev in
  let table = Hashtbl.create (module String) in
  List.iteri paths ~f:(fun i path -> Hashtbl.set table ~key:path ~data:i);
  Single.map (fun path -> Hashtbl.find_exn table path) (Single.names tree)

let check_single ~msg ~tol (expected : Nx.float32_t Single.t) (got : Nx.float32_t Single.t)
  =
  ignore
    (Single.map2
       (fun e g -> check_leaf ~msg ~tol e g)
       expected
       got
      : unit Single.t)

let multi_leaves t = Multi.fold (fun _ acc x -> x :: acc) [] t |> List.rev |> Array.of_list

let multi_indices tree =
  let paths = Multi.fold (fun path acc _ -> path :: acc) [] tree |> List.rev in
  let table = Hashtbl.create (module String) in
  List.iteri paths ~f:(fun i path -> Hashtbl.set table ~key:path ~data:i);
  Multi.map (fun path -> Hashtbl.find_exn table path) (Multi.names tree)

let check_multi ~msg ~tol (expected : Nx.float32_t Multi.t) (got : Nx.float32_t Multi.t) =
  ignore
    (Multi.map2 (fun e g -> check_leaf ~msg ~tol e g) expected got : unit Multi.t)

(* ------------------------------------------------------------------------
   Tests
   ------------------------------------------------------------------------ *)

let test_first_order_exact () =
  let c = O.First_order.large.w in
  let x = Nx.Rng.normal (Nx.Rng.key 11) Nx.float32 [| 2; 3 |] in
  (* R1 is what [orbit_average] returns. The dense surrogate is a compressed
     representation with its own normalization, so only the round trip through
     it is expected to reproduce itself. *)
  let expected = exact_r1 c x in
  let got_orbit = (O.First_order.orbit_average { Single.w = x }).w in
  check_leaf ~msg:"orbit average" ~tol:1e-5 expected got_orbit;
  let dense =
    O.First_order.dense_of_factors (O.First_order.factors_of_params { Single.w = x })
  in
  let round_trip = O.First_order.dense_of_factors (O.First_order.factors_of_dense dense) in
  check_leaf ~msg:"surrogate round trip" ~tol:1e-5 dense round_trip

let test_first_order_equivariant () =
  let c = O.First_order.large.w in
  let x = Nx.Rng.normal (Nx.Rng.key 12) Nx.float32 [| 2; 3 |] in
  let perms = random_perms c in
  let transformed x = c.transform ~perms x in
  let projected x = (O.First_order.orbit_average { Single.w = x }).w in
  check_leaf
    ~msg:"projection commutes with the group"
    ~tol:1e-5
    (transformed (projected x))
    (projected (transformed x))

let test_second_order_exact () =
  let c = (O.Second_order.large ()).w.(0) in
  let x = Nx.Rng.normal (Nx.Rng.key 21) Nx.float32 [| 2; 3 |] in
  let expected = exact_r2 c x in
  let factors = O.Second_order.factors_of_pair { Single.w = x } { Single.w = x } in
  (* Factor estimation is a projection onto the invariant subspace, so the
     assembled block is the exact orbit average. *)
  let assembled = c.dense_block ~factors:factors.w.(0) in
  check_leaf ~msg:"assembled block" ~tol:1e-5 expected assembled;
  (* Invariance of the assembled block under a random group element. *)
  let perms = random_perms c in
  check_leaf ~msg:"block invariance" ~tol:1e-5 assembled (c.transform ~perms assembled);
  (* The dense surrogate round trip. *)
  let dense = O.Second_order.dense_of_factors factors in
  let round_trip =
    O.Second_order.dense_of_factors (O.Second_order.factors_of_dense dense)
  in
  check_leaf ~msg:"surrogate round trip" ~tol:1e-5 dense round_trip;
  (* The action of the assembled operator. *)
  let v = { Single.w = Nx.Rng.normal (Nx.Rng.key 22) Nx.float32 [| 2; 3 |] } in
  let l = 6 and r = 6 in
  let expected_v = Nx.matmul (Nx.reshape [| l; r |] expected) (vec v.w) in
  let got_v = (O.Second_order.apply ~factors v).w in
  check_leaf ~msg:"operator action" ~tol:1e-4 (Nx.reshape [| 2; 3 |] expected_v) got_v

let test_two_groups_exact () =
  let c = O2.First_order.large.w in
  let x = Nx.Rng.normal (Nx.Rng.key 31) Nx.float32 [| 2; 2; 3 |] in
  equal int 12 (List.length (group_elements c));
  let expected = exact_r1 c x in
  let got = (O2.First_order.orbit_average { Two_groups.w = x }).w in
  check_leaf ~msg:"two groups orbit average" ~tol:1e-5 expected got

(* The global group of the two-leaf model: group 0 (dim 3) and group 1
   (dim 3) are independent. *)
let multi_global_elements = cartesian (List.map [ 3; 3 ] ~f:permutations)

let multi_assignments = List.map multi_global_elements ~f:Array.of_list

let exact_multi_r1 (x : Nx.float32_t Multi.t) =
  let n = List.length multi_assignments in
  let acc =
    Multi.map2
      (fun c x ->
        let shape = Array.of_list c.Compiler.dims.left in
        List.fold multi_assignments ~init:(Nx.zeros Nx.float32 shape) ~f:(fun acc assignment ->
          Nx.add acc (c.Compiler.transform ~perms:(perms_of_ids c assignment) x)))
      OM.First_order.large
      x
  in
  Multi.map (fun a -> Nx.mul_s a (1. /. Float.of_int n)) acc

let exact_multi_r2_apply (x : Nx.float32_t Multi.t) (v : Nx.float32_t Multi.t) =
  let rows = OM.Second_order.large () in
  let row_arr =
    Multi.fold (fun _ acc r -> r :: acc) [] rows |> List.rev |> Array.of_list
  in
  let xs = multi_leaves x in
  let vs = multi_leaves v in
  let n = Array.length row_arr in
  let result =
    Array.init n ~f:(fun i ->
      let acc = ref (Nx.zeros Nx.float32 (Array.of_list row_arr.(i).(0).Compiler.dims.left)) in
      for j = 0 to n - 1 do
        let c = row_arr.(i).(j) in
        let base = outer_pair c xs.(i) xs.(j) in
        let block =
          mean
            (List.map multi_assignments ~f:(fun assignment ->
               c.Compiler.transform ~perms:(perms_of_ids c assignment) base))
            (List.length multi_assignments)
        in
        let l = List.fold c.Compiler.dims.left ~init:1 ~f:Int.( * ) in
        let r = List.fold c.Compiler.dims.right ~init:1 ~f:Int.( * ) in
        let z =
          Nx.matmul (Nx.reshape [| l; r |] block) (vec vs.(j))
          |> Nx.reshape (Array.of_list c.Compiler.dims.left)
        in
        acc := Nx.add !acc z
      done;
      !acc)
  in
  let idx = multi_indices (OM.First_order.large) in
  Multi.map2 (fun i _ -> result.(i)) idx OM.First_order.large

let test_multi_leaf () =
  let x =
    { Multi.w = Nx.Rng.normal (Nx.Rng.key 41) Nx.float32 [| 2; 3 |]
    ; v = Nx.Rng.normal (Nx.Rng.key 42) Nx.float32 [| 3 |]
    }
  in
  let expected_r1 = exact_multi_r1 x in
  check_multi ~msg:"multi-leaf orbit average" ~tol:1e-5 expected_r1 (OM.First_order.orbit_average x);
  let v =
    { Multi.w = Nx.Rng.normal (Nx.Rng.key 43) Nx.float32 [| 2; 3 |]
    ; v = Nx.Rng.normal (Nx.Rng.key 44) Nx.float32 [| 3 |]
    }
  in
  let factors = OM.Second_order.factors_of_pair x x in
  let expected = exact_multi_r2_apply x v in
  check_multi ~msg:"multi-leaf operator action" ~tol:1e-4 expected (OM.Second_order.apply ~factors v);
  (* Surrogate round trip across the two blocks. *)
  let dense = OM.Second_order.dense_of_factors factors in
  let round_trip = OM.Second_order.dense_of_factors (OM.Second_order.factors_of_dense dense) in
  check_leaf ~msg:"multi-leaf surrogate round trip" ~tol:1e-5 dense round_trip

let test_multi_leaf_invariance () =
  let x =
    { Multi.w = Nx.Rng.normal (Nx.Rng.key 51) Nx.float32 [| 2; 3 |]
    ; v = Nx.Rng.normal (Nx.Rng.key 52) Nx.float32 [| 3 |]
    }
  in
  let v =
    { Multi.w = Nx.Rng.normal (Nx.Rng.key 53) Nx.float32 [| 2; 3 |]
    ; v = Nx.Rng.normal (Nx.Rng.key 54) Nx.float32 [| 3 |]
    }
  in
  let factors = OM.Second_order.factors_of_pair x x in
  let c = OM.First_order.large in
  let assignment =
    Array.of_list
      (List.map [ 3; 3 ] ~f:(fun dim ->
         Nx.Rng.permutation (Nx.Rng.key 55) dim))
  in
  let transform_tree t =
    Multi.map2 (fun c x -> c.Compiler.transform ~perms:(perms_of_ids c assignment) x) c t
  in
  check_multi
    ~msg:"operator commutes with the group"
    ~tol:1e-4
    (transform_tree (OM.Second_order.apply ~factors v))
    (OM.Second_order.apply ~factors (transform_tree v))

let tests =
  [ test "first order against the group" test_first_order_exact
  ; test "first order commutes with the group" test_first_order_equivariant
  ; test "second order against the group" test_second_order_exact
  ; test "two permutation groups" test_two_groups_exact
  ; test "multi-leaf orbit machinery" test_multi_leaf
  ; test "multi-leaf invariance" test_multi_leaf_invariance
  ]

let () = run "symo orbit" tests
