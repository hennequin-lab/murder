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

   Because a mark is a unit-result {!Rune.custom_jvp}, the same loss also runs
   unchanged under [Rune.value_and_grad] (the mark's tangent map is not applied
   there) and under a bare [Rune.jvp] (the map runs; there is no scope, and the
   additions are dropped).

   The total's scope owns the accumulation: it threads it through a [scan]
   body and a [remat] call itself, drops what reverse mode re-runs, and sums a
   [vmap]'s lanes, so a mark may sit in nested functions, in recurrences,
   inside [Rune.remat] or inside the caller's own maps. *)

type 'p t =
  { k : int (** The number of sketch directions. *)
  ; loss : Nx.float64_t (** The primal total loss, as a float64 scalar. *)
  ; c : Nx.float64_t
    (** C = Θᵀ∇c, shape [k]: the gradient sketched onto the sampled
        directions. *)
  ; ggn : Nx.float64_t
    (** The sketched generalized Gauss-Newton matrix ΘᵀJᵀHJΘ, shape [k;k],
        accumulated in float64 and symmetrized. *)
  ; dirs : 'p
    (** Θ, the directions the sketch was measured along: one [k]-lane batch per
        float leaf, and zero lanes where no direction can be drawn. The basis
        is the caller's — the update whitens it (see [Optim.coordinates]), so
        nothing downstream has to assume it is orthonormal. *)
  ; apply : Nx.float64_t -> 'p
    (** [apply z] is Θz for z : [k] — the parameter-space direction the
        sketch's coordinates denote, with each leaf in its parameter's
        dtype. An update rule composes [apply] after solving in the [k]
        dimensional subspace. *)
  }

(* [sample structure ~k params] draws the default directions: each float leaf
   gets a standard normal [k]-lane batch from the ambient [Rng] scope, so a
   caller controls reproducibility by wrapping the sketch in
   [Nx.Rng.with_key]. Leaves that are not float — an RNG key or an index
   threaded through the parameters, say — cannot carry a direction and get
   zero lanes; their consumers are untracked operations, so they simply do not
   participate. *)
let sample (type p) (structure : p Nx.Ptree.t) ~k (params : p) : p =
  Nx.Ptree.map
    structure
    (fun _ leaf ->
       let shape = Array.append [| k |] (Nx.shape leaf) in
       if Nx_dtype.is_float (Nx.dtype leaf)
       then Nx.cast (Nx.dtype leaf) (Nx.randn Nx.float64 shape)
       else Nx.zeros (Nx.dtype leaf) shape)
    params

(* [contract z theta] contracts the lane axis of a parameter leaf's tangent
   batch against [z]: reshape [z] to [k :: 1 ... 1], scale, and reduce. *)
let contract z theta =
  let s = Nx.shape theta in
  let lead = Array.make (Array.length s - 1) 1 in
  let zr = Nx.reshape (Array.concat [ [| Nx.numel z |]; lead ]) (Nx.contiguous z) in
  Nx.sum (Nx.mul (Nx.cast (Nx.dtype theta) zr) theta) ~axes:[ 0 ]

(* The number of lanes: the leading length of the first tensor leaf of
   [dirs]. *)
let k_of (type p) (structure : p Nx.Ptree.t) (dirs : p) : int =
  match
    Nx.Ptree.fold
      structure
      (fun _ leaf k -> Option.fold ~none:(Some (Nx.dim 0 leaf)) ~some:Option.some k)
      dirs
      None
  with
  | Some k -> k
  | None -> invalid_arg "Sofo.sketch: the directions have no tensor leaf"

let run
      (type p)
      (structure : p Nx.Ptree.t)
      (loss : p -> ('c, 'd) Nx.t)
      (params : p)
      (dirs : p)
  : p t
  =
  let k = k_of structure dirs in
  if k < 1 then invalid_arg (Printf.sprintf "Sofo.sketch: k must be at least 1, got %d" k);
  let (loss_lanes, c), ggn =
    Rune.vmap
      ~axis:Observe.directions
      Nx.Ptree.(structure @-> returns (pair (pair tensor tensor) tensor))
      (fun theta ->
         Rune.Total.collect
           Observe.curvature
           ~zero:(Nx.zeros Nx.float64 [| k; k |])
           (fun () -> Rune.jvp structure Nx.Ptree.tensor loss params theta))
      dirs
  in
  if Nx.shape loss_lanes <> [| k |]
  then
    invalid_arg
      "Sofo.sketch: the loss function must return a scalar (a tensor with one element)";
  let y = Nx.reshape [||] (Nx.cast Nx.float64 (Nx.slice [ Nx.I 0 ] loss_lanes)) in
  let ggn = Nx.cast Nx.float64 (Nx.slice [ Nx.I 0 ] ggn) in
  { k
  ; loss = y
  ; c = Nx.reshape [| k |] (Nx.cast Nx.float64 c)
  ; ggn
  ; dirs
  ; apply =
      (fun z ->
        if Nx.shape z <> [| k |]
        then
          invalid_arg
            (Printf.sprintf
               "Sofo.sketch: apply takes a k-vector (shape [%d]), got shape [%s]"
               k
               (String.concat "," (Array.to_list (Array.map string_of_int (Nx.shape z)))));
        Nx.Ptree.map structure (fun _ theta -> contract z theta) dirs)
  }
