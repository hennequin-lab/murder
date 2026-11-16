(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The distributed step, and the machinery around it:

   - the step's key is split into one subkey per trial, exactly as the batched
     recognition splits it, so the distributed estimate and a single-process
     one average the same per-trial bounds;
   - a parameter tree crosses the wire as one float array: its float leaves
     packed in walk order. Leaves that cannot carry a gradient — a counter, a
     key — are not sent and keep their value on every rank;
   - the ranks sum their losses and gradients, the root divides by the batch's
     trial count, takes the optimizer step, and broadcasts the new parameters.

   MPI has no collective that both reduces and updates, and the update is the
   caller's code, so the order is fixed: reduce, update on the root,
   broadcast. Every rank runs the collectives in the same order. *)

open Base
open Nx

type range =
  { first : int
  ; count : int
  }

let partition ~ranks ~trials ~rank =
  if ranks <= 0
  then
    invalid_arg
      (Printf.sprintf "Ilqr_vae_mpi.partition: ranks must be positive, got %d" ranks);
  if rank < 0 || rank >= ranks
  then
    invalid_arg
      (Printf.sprintf
         "Ilqr_vae_mpi.partition: rank must be in [0; %d), got %d"
         ranks
         rank);
  if trials < 0
  then
    invalid_arg
      (Printf.sprintf
         "Ilqr_vae_mpi.partition: trials must be non-negative, got %d"
         trials);
  let base = trials / ranks in
  let extra = Int.rem trials ranks in
  { first = (rank * base) + Int.min rank extra
  ; count = (base + if rank < extra then 1 else 0)
  }

(* ── parameter transport ─────────────────────────────────────────────────── *)

(* [float_values x] is [x]'s elements when [x] is a float leaf, and [None] for
   a leaf that cannot carry a gradient. The dtype match refines the payload to
   [float], which is what [Nx.to_array] returns. *)
let float_values : type v s d. (v, s, d) Nx.t -> float array option =
  fun x ->
  match Nx.dtype x with
  | Nx.Dtype.Float64 -> Some (Nx.to_array x)
  | Nx.Dtype.Float32 -> Some (Nx.to_array x)
  | Nx.Dtype.Float16 -> Some (Nx.to_array x)
  | Nx.Dtype.Bfloat16 -> Some (Nx.to_array x)
  | Nx.Dtype.Float8_e4m3fn -> Some (Nx.to_array x)
  | Nx.Dtype.Float8_e5m2 -> Some (Nx.to_array x)
  | Nx.Dtype.Float4_e2m1fn -> Some (Nx.to_array x)
  | Nx.Dtype.Complex128
  | Nx.Dtype.Complex64
  | Nx.Dtype.Int64
  | Nx.Dtype.Uint64
  | Nx.Dtype.Int32
  | Nx.Dtype.Uint32
  | Nx.Dtype.Int16
  | Nx.Dtype.Uint16
  | Nx.Dtype.Int8
  | Nx.Dtype.Uint8
  | Nx.Dtype.Int4
  | Nx.Dtype.Uint4
  | Nx.Dtype.Bool
  | Nx.Dtype.Bit -> None

(* The same split as [float_values], by dtype alone: testing a leaf must not
   copy it, and [packed_size] tests every leaf of the tree at every step. Keep
   the two lists in step. *)
let carries_gradient : type v s d. (v, s, d) Nx.t -> bool =
  fun x ->
  match Nx.dtype x with
  | Nx.Dtype.Float64
  | Nx.Dtype.Float32
  | Nx.Dtype.Float16
  | Nx.Dtype.Bfloat16
  | Nx.Dtype.Float8_e4m3fn
  | Nx.Dtype.Float8_e5m2
  | Nx.Dtype.Float4_e2m1fn -> true
  | Nx.Dtype.Complex128
  | Nx.Dtype.Complex64
  | Nx.Dtype.Int64
  | Nx.Dtype.Uint64
  | Nx.Dtype.Int32
  | Nx.Dtype.Uint32
  | Nx.Dtype.Int16
  | Nx.Dtype.Uint16
  | Nx.Dtype.Int8
  | Nx.Dtype.Uint8
  | Nx.Dtype.Int4
  | Nx.Dtype.Uint4
  | Nx.Dtype.Bool
  | Nx.Dtype.Bit -> false

(* [packed_size structure params] is the number of values [pack] copies. A sum
   over the leaves, so the walk's order does not matter here. *)
let packed_size (type p d) (structure : (p, d) Nx.Ptree.t) (params : p) =
  Nx.Ptree.fold
    structure
    (fun _ leaf total -> if carries_gradient leaf then total + Nx.numel leaf else total)
    params
    0

(* [pack structure params] is one array of the float leaves' elements in walk
   order. It is the wire format of a parameter tree: MPI reduces and
   broadcasts one array, and the leaf order is a property of the structure, so
   every rank packs and unpacks alike. *)
let pack (type p d) (structure : (p, d) Nx.Ptree.t) (params : p) : float array =
  let leaves, _ = Nx.Ptree.flatten structure params in
  let arrays = List.filter_map leaves ~f:(fun (Nx.P leaf) -> float_values leaf) in
  let total =
    List.fold arrays ~init:0 ~f:(fun total values -> total + Array.length values)
  in
  let packed = Array.create ~len:total 0.0 in
  let pos = ref 0 in
  List.iter arrays ~f:(fun values ->
    let n = Array.length values in
    Array.blit ~src:values ~src_pos:0 ~dst:packed ~dst_pos:!pos ~len:n;
    pos := !pos + n);
  packed

(* [accumulate structure ~into grads] adds [grads]' float leaves to [into], in
   walk order. A gradient leaf that does not carry one is the zero Rune made it,
   and contributes nothing. *)
let accumulate
      (type p d)
      (structure : (p, d) Nx.Ptree.t)
      ~(into : float array)
      (grads : p)
  =
  let leaves, _ = Nx.Ptree.flatten structure grads in
  let pos = ref 0 in
  List.iter leaves ~f:(fun (Nx.P leaf) ->
    match float_values leaf with
    | None -> ()
    | Some values ->
      let n = Array.length values in
      for i = 0 to n - 1 do
        into.(!pos + i) <- into.(!pos + i) +. values.(i)
      done;
      pos := !pos + n)

(* [unpack structure ~like ~zero_non_float packed] is [like] with its float
   leaves replaced by [packed], in walk order. A leaf that cannot carry a
   gradient is [like]'s, or its zero for a gradient: the caller says which with
   [zero_non_float].

   Values travel at float64 and are cast back to each leaf's dtype; every float
   value of a narrower dtype is exactly a float64, so the round trip loses
   nothing. The leaves are made on the host, where MPI's buffers live, and
   placed beside [like]'s, so a model whose parameters live on a device is
   rebuilt there. *)
let unpack
      (type p d)
      (structure : (p, d) Nx.Ptree.t)
      ~(like : p)
      ~(zero_non_float : bool)
      (packed : float array)
  : p
  =
  let leaves, _ = Nx.Ptree.flatten structure like in
  let pos = ref 0 in
  let rebuilt =
    List.map leaves ~f:(fun (Nx.P leaf) ->
      if not (carries_gradient leaf)
      then Nx.P (if zero_non_float then Nx.zeros_like leaf else leaf)
      else (
        let n = Nx.numel leaf in
        if !pos + n > Array.length packed
        then
          invalid_arg
            (Printf.sprintf
               "Ilqr_vae_mpi: the parameter tree holds more values than the %d the ranks \
                exchanged"
               (Array.length packed));
        let values = Array.sub packed ~pos:!pos ~len:n in
        pos := !pos + n;
        let host = Nx.create Nx.float64 (Nx.shape leaf) values in
        Nx.P (Nx.place (Nx.beside leaf) (Nx.cast (Nx.dtype leaf) host))))
  in
  if !pos <> Array.length packed
  then
    invalid_arg
      (Printf.sprintf
         "Ilqr_vae_mpi: the parameter tree holds %d values, but the ranks exchanged %d"
         !pos
         (Array.length packed));
  Nx.Ptree.rebuild structure ~like rebuilt

(* ── the driver ──────────────────────────────────────────────────────────── *)

type ('w, 'd, 'l, 'p, 'e, 'r, 's) t =
  { model : ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.t
  ; structure : (('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params, 's) Nx.Ptree.t
  ; comm : Mpi.communicator
  ; rank : int
  ; size : int
  ; trial :
      ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params
      -> ('w, 's) Ilqr_vae.value
      -> ('w, 's) Ilqr_vae.value
      -> 's Nx.Rng.key
      -> ('w, 's) Ilqr_vae.value * ('w, 'd, 'l, 'p, 'e, 'r, 's) Ilqr_vae.Model.params
  }

let make
      (type w d l p e r)
      (model : (w, d, l, p, e, r, 's) Ilqr_vae.Model.t)
      ?(comm = Mpi.comm_world)
      ~iters
      ~n_samples
      ()
  =
  if iters <= 0
  then
    invalid_arg (Printf.sprintf "Ilqr_vae_mpi.make: iters must be positive, got %d" iters);
  if n_samples <= 0
  then
    invalid_arg
      (Printf.sprintf "Ilqr_vae_mpi.make: n_samples must be positive, got %d" n_samples);
  let structure = Ilqr_vae.Model.structure model in
  (* One trial, one key, one differentiation. The batch is not an argument:
     the compiled program's shapes are the trial's, so it serves every batch
     shape and every split. *)
  let trial =
    Rune.jit
      Nx.Ptree.(
        structure
        @-> tensor
        @-> tensor
        @-> Nx.Rng.ptree
        @-> returns (pair tensor structure))
      (fun params beta data key ->
         Rune.value_and_grad
           structure
           (fun p -> Ilqr_vae.Elbo.loss_fixed model p ~beta ~iters ~key ~n_samples data)
           params)
  in
  { model; structure; comm; rank = Mpi.comm_rank comm; size = Mpi.comm_size comm; trial }

let size t = t.size
let rank t = t.rank
let is_root t = Int.equal t.rank 0

let shape_string shape =
  String.concat ~sep:"; " (Array.to_list (Array.map shape ~f:Int.to_string))

(* [batch_size driver data] is the number of trials [data] holds. It checks the
   trailing axes against the model, as [Elbo.compute] does, so that a wrong
   tensor fails here rather than as a shape error in a compiled step or, worse,
   as an MPI count mismatch. *)
let batch_size (type w d l p e r) (t : (w, d, l, p, e, r, 's) t) data : int =
  let t_steps = Ilqr_vae.Model.n_steps t.model
  and n_output = Ilqr_vae.Model.n_output t.model in
  let shape = Nx.shape data in
  let rank = Array.length shape in
  if
    rank < 3
    || (not (Int.equal shape.(rank - 2) t_steps))
    || not (Int.equal shape.(rank - 1) n_output)
  then
    invalid_arg
      (Printf.sprintf
         "Ilqr_vae_mpi.step: data must be [batch…; n_steps; n_output] with n_steps = %d \
          and n_output = %d, got [%s]"
         t_steps
         n_output
         (shape_string shape));
  let batch_dims = Array.sub shape ~pos:0 ~len:(rank - 2) in
  if Array.exists batch_dims ~f:(fun d -> d <= 0)
  then
    invalid_arg
      (Printf.sprintf
         "Ilqr_vae_mpi.step: every trial axis of the data must be positive, got [%s]"
         (shape_string shape));
  Array.fold batch_dims ~init:1 ~f:( * )

(* [local driver ~params ~beta ~keys ~data ~range] is this rank's share of the
   step: the trials' losses summed, and their gradients summed into a packed
   array. A rank that owns no trial still has an array to reduce, all zeros. *)
let local
      (type w d l p e r)
      (t : (w, d, l, p, e, r, 's) t)
      ~(params : (w, d, l, p, e, r, 's) Ilqr_vae.Model.params)
      ~beta
      ~(keys : 's Nx.Rng.key array)
      ~data
      ~(range : range)
  =
  let gradients = Array.create ~len:(packed_size t.structure params) 0.0 in
  let loss = ref 0.0 in
  for j = 0 to range.count - 1 do
    let i = range.first + j in
    let value, grads = t.trial params beta (Nx.slice [ I i ] data) keys.(i) in
    loss := !loss +. Nx.item [] value;
    accumulate t.structure ~into:gradients grads
  done;
  !loss, gradients

let step
      (type w d l p e r)
      (t : (w, d, l, p, e, r, 's) t)
      (params : (w, d, l, p, e, r, 's) Ilqr_vae.Model.params)
      ~(key : 's Nx.Rng.key)
      ?beta
      ~(data : (w, 's) Ilqr_vae.value)
      (update :
        (w, d, l, p, e, r, 's) Ilqr_vae.Model.params
        -> (w, d, l, p, e, r, 's) Ilqr_vae.Model.params
        -> (w, d, l, p, e, r, 's) Ilqr_vae.Model.params)
  =
  let root = 0 in
  let batch = batch_size t data in
  (* One trial axis, whatever leading axes the batch came with. *)
  let flat =
    Nx.reshape
      [| batch; Ilqr_vae.Model.n_steps t.model; Ilqr_vae.Model.n_output t.model |]
      data
  in
  let keys = Nx.Rng.split ~n:batch key in
  let range = partition ~ranks:t.size ~trials:batch ~rank:t.rank in
  let beta =
    Option.value
      beta
      ~default:(Nx.scalar_at (Nx.beside data) (Ilqr_vae.Model.dtype t.model) 1.0)
  in
  let local_loss, local_gradients = local t ~params ~beta ~keys ~data:flat ~range in
  (* A length the ranks disagree on would surface as an MPI truncation error
     inside the reduction; name the real problem before entering it. *)
  let count = Array.length local_gradients in
  let root_count = Mpi.broadcast_int count root t.comm in
  if not (Int.equal count root_count)
  then
    invalid_arg
      (Printf.sprintf
         "Ilqr_vae_mpi.step: the parameters differ between ranks — rank %d packs %d \
          values, the root %d; every rank must pass the parameters of the same model"
         t.rank
         count
         root_count);
  let total_loss = Mpi.allreduce_float local_loss Mpi.Float_sum t.comm in
  let summed = Array.create ~len:count 0.0 in
  Mpi.reduce_float_array local_gradients summed Mpi.Float_sum root t.comm;
  let scale = 1.0 /. Float.of_int batch in
  let next =
    if is_root t
    then (
      Array.iteri summed ~f:(fun i value -> summed.(i) <- value *. scale);
      let grads = unpack t.structure ~like:params ~zero_non_float:true summed in
      update params grads)
    else params
  in
  let packed = if is_root t then pack t.structure next else Array.create ~len:count 0.0 in
  if is_root t && not (Int.equal (Array.length packed) count)
  then
    invalid_arg
      (Printf.sprintf
         "Ilqr_vae_mpi.step: update returned parameters that pack to %d values, not the \
          %d it was given; it must keep the model's structure, dtypes and shapes"
         (Array.length packed)
         count);
  Mpi.broadcast_float_array packed root t.comm;
  let params = unpack t.structure ~like:params ~zero_non_float:false packed in
  params, total_loss *. scale

let run
      (type w d l p e r)
      (t : (w, d, l, p, e, r, 's) t)
      (params : (w, d, l, p, e, r, 's) Ilqr_vae.Model.params)
      ~steps
      ~(key : 's Nx.Rng.key)
      ?beta
      ~(data : int -> (w, 's) Ilqr_vae.value)
      ?report
      (update :
        (w, d, l, p, e, r, 's) Ilqr_vae.Model.params
        -> (w, d, l, p, e, r, 's) Ilqr_vae.Model.params
        -> (w, d, l, p, e, r, 's) Ilqr_vae.Model.params)
  =
  if steps < 0
  then
    invalid_arg
      (Printf.sprintf "Ilqr_vae_mpi.run: steps must be non-negative, got %d" steps);
  let losses = Array.create ~len:steps 0.0 in
  (* Each step's key is a subkey of the run's, so a run is reproducible from
     its first key alone and no two steps share a draw. *)
  let rec go i params =
    if i >= steps
    then params
    else (
      let weight = Option.map beta ~f:(fun at -> at i) in
      let params, loss =
        step t params ~key:(Nx.Rng.fold_in key i) ?beta:weight ~data:(data i) update
      in
      losses.(i) <- loss;
      (match report with
       | Some log when is_root t -> log i loss
       | _ -> ());
      go (i + 1) params)
  in
  go 0 params, losses
