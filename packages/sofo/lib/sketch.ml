(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The sketch driver: run the whole loss under batched forward mode and collect
   the marks' curvature blocks.

   Forward mode is a composition of rune transformations, and it needs nothing
   from rune's internals beyond its public surface: {!Rune.vmap} around
   {!Rune.jvp} pushes the [k] directions through one forward pass — the mapped
   function runs once and its primal operations, whose operands are constants
   of the map, are computed once — while each tensor's tangent becomes the
   whole [k]-lane batch. The map is given the name {!Observe.directions}, the
   name {!Observe.mark}'s rule gathers the lanes of, and the marks' blocks are
   collected by a {!Rune.Total} scope opened inside the map:

   user code  ⊂  jvp  ⊂  collect  ⊂  vmap named Observe.directions  ⊂  driver

   - [jvp] maintains each tensor's tangent; a mark's rule reads the whole
     [k]-lane batch of a prediction's tangent with {!Rune.lanes} and adds the
     block YᵀHY to the total.
   - [vmap] batches the forward pass along the lanes. The scope is inside the
     map, so every lane collects the whole block as a constant of the map, the
     map returns it stacked, and the driver reads lane 0.
   - The tangent of the *returned* total loss is the gradient sketch itself:
     C = Θᵀ∇c, read off the [vmap] result. No instrumentation is involved, so
     the gradient sketch cannot drift from the loss's own arithmetic.

   A mark makes its rule only inside the map ({!Observe.within_sketch}), so
   the same loss also runs unchanged under [Rune.value_and_grad] and under a
   bare [Rune.jvp], where it does nothing.

   The total's scope owns the accumulation: it threads it through a [scan]
   body and a [remat] call itself, drops what reverse mode re-runs, and sums a
   [vmap]'s lanes, so a mark may sit in nested functions, in recurrences,
   inside [Rune.remat] or inside the caller's own maps. *)

open Nx

type ('p, 'd) t =
  { k : int (** The number of sketch directions. *)
  ; loss : 'd Nx.float64_t (** The primal total loss, as a float64 scalar. *)
  ; c : 'd Nx.float64_t
    (** C = Θᵀ∇c, shape [k]: the gradient sketched onto the sampled
        directions. *)
  ; ggn : 'd Nx.float64_t
    (** The sketched generalized Gauss-Newton matrix ΘᵀJᵀHJΘ, shape [k;k],
        accumulated in float64 and symmetrized. *)
  ; dirs : 'p
    (** Θ, the directions the sketch was measured along: one [k]-lane batch per
        float leaf, and zero lanes where no direction can be drawn. The basis
        is the caller's — the update whitens it (see [Optim.coordinates]), so
        nothing downstream has to assume it is orthonormal. *)
  ; apply : 'd Nx.float64_t -> 'p
    (** [apply z] is Θz for z : [k] — the parameter-space direction the
        sketch's coordinates denote, with each leaf in its parameter's
        dtype. An update rule composes [apply] after solving in the [k]
        dimensional subspace. *)
  }

(* [sample structure ?key ~k params] draws the default directions: each float
   leaf gets a standard normal [k]-lane batch from the ambient [Rng] scope, so a
   caller controls reproducibility by wrapping the sketch in
   [Nx.Rng.with_key]. Leaves that are not float — an RNG key or an index
   threaded through the parameters, say — cannot carry a direction and get
   zero lanes; their consumers are untracked operations, so they simply do not
   participate.

   With [key], the [i]-th draw takes [Nx.Rng.fold_in key i], the key a scope
   rooted at [key] gives it, so the draw is the scope's whatever set [key] lies
   on: a scope takes host keys only, and a key a compiled step receives is
   traced. Each batch is made beside its leaf, whole on the leaf's devices. *)
let sample (type p d) (structure : (p, d) Nx.Ptree.t) ?key ~k (params : p) : p =
  let draws = ref 0 in
  Nx.Ptree.map
    structure
    (fun _ leaf ->
       let shape = Array.append [| k |] (Nx.shape leaf) in
       let at = beside leaf in
       if Nx.Dtype.is Nx.Dtype.Float (Nx.dtype leaf)
       then (
         let key = Option.map (fun key -> Nx.Rng.fold_in key !draws) key in
         incr draws;
         cast (Nx.dtype leaf) (Nx.Rng.normal_at at ?key float64 shape))
       else zeros_at at (Nx.dtype leaf) shape)
    params

(* [contract z theta] contracts the lane axis of a parameter leaf's tangent
   batch against [z]: reshape [z] to [k :: 1 ... 1], scale, and reduce. *)
let contract z theta =
  let s = Nx.shape theta in
  let lead = Array.make (Array.length s - 1) 1 in
  let zr = reshape (Array.concat [ [| numel z |]; lead ]) z in
  sum Infix.(cast (Nx.dtype theta) zr * theta) ~axes:[ 0 ]

(* The first tensor leaf of [x], in walk order. *)
let first (type p d) (structure : (p, d) Nx.Ptree.t) (x : p) : d Nx.packed option =
  Nx.Ptree.fold
    structure
    (fun _ leaf acc -> Option.fold ~none:(Some (Nx.P leaf)) ~some:Option.some acc)
    x
    None

(* Where a structure's own constants lie: whole on each device of its first
   tensor leaf, as [Nx.Ptree.dot]'s zero does. [fn] names the caller. *)
let beside_first (type p d) ~fn (structure : (p, d) Nx.Ptree.t) (x : p) : d Nx.Placement.t
  =
  match first structure x with
  | Some (Nx.P leaf) -> beside leaf
  | None -> invalid_arg (fn ^ ": the structure has no tensor leaf")

(* The number of lanes: the leading length of the first tensor leaf of
   [dirs]. *)
let k_of (type p d) (structure : (p, d) Nx.Ptree.t) (dirs : p) : int =
  match first structure dirs with
  | Some (Nx.P leaf) -> dim 0 leaf
  | None -> invalid_arg "Sofo.sketch: the directions have no tensor leaf"

let run
      (type p d)
      (structure : (p, d) Nx.Ptree.t)
      (loss : p -> ('c, 'e, d) Nx.t)
      (params : p)
      (dirs : p)
  : (p, d) t
  =
  let k = k_of structure dirs in
  if k < 1 then invalid_arg (Printf.sprintf "Sofo.sketch: k must be at least 1, got %d" k);
  let (loss_lanes, c), ggn =
    Rune.vmap
      ~axis:Observe.directions
      Nx.Ptree.(structure @-> returns (pair (pair tensor tensor) tensor))
      (fun theta ->
         Observe.within_sketch
         @@ fun () ->
         Rune.Total.collect
           Observe.curvature
           ~init:
             (zeros_at (beside_first ~fn:"Sofo.sketch" structure dirs) float64 [| k; k |])
           (fun () -> Rune.jvp structure Nx.Ptree.tensor loss params theta))
      dirs
  in
  if Nx.shape loss_lanes <> [| k |]
  then
    invalid_arg
      "Sofo.sketch: the loss function must return a scalar (a tensor with one element)";
  let y = reshape [||] (cast float64 (slice [ Nx.I 0 ] loss_lanes)) in
  let ggn = cast float64 (slice [ Nx.I 0 ] ggn) in
  { k
  ; loss = y
  ; c = reshape [| k |] (cast float64 c)
  ; ggn
  ; dirs
  ; apply =
      (fun z ->
        if Nx.shape z <> [| k |]
        then
          invalid_arg
            (Format.asprintf
               "Sofo.sketch: apply takes a k-vector (shape [%d]), got shape %a"
               k
               pp_shape
               (Nx.shape z));
        Nx.Ptree.map structure (fun _ theta -> contract z theta) dirs)
  }
