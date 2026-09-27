(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The sketch driver: sample directions, run the loss under batched forward
   mode with a collector installed, and package what an update rule needs.

   Batched forward mode is a composition, and it needs nothing from rune's
   internals beyond its public surface: {!Rune.vmap} around {!Rune.jvp} pushes
   the [k] directions through one forward pass — the mapped function runs once
   and its primal operations, whose operands are constants of the map, are
   computed once — while each tensor's tangent becomes the whole [k]-lane
   batch. The collector is installed *outside* that map:

   user code  ⊂  jvp  ⊂  vmap  ⊂  collector  ⊂  driver

   - [jvp] maintains each tensor's tangent; the user's observations read them
     at the observation site (through the tangent query) and carry them in the
     effect payload.
   - [vmap] batches the tangent arithmetic along the lanes; the collector runs
     beyond its extent, where the payload's lane axis is an ordinary tensor
     axis it can contract (inside the map, [Nx.shape] would report the map's
     virtual, lane-less shape).
   - The tangent of the *returned* total loss is the gradient sketch itself:
     C = Θᵀ∇c, read off the [vmap] result. No instrumentation is involved, so
     the gradient sketch cannot drift from the loss's own arithmetic.

   Because the collector is a plain effect handler and the tangents are read
   through a query, the same user code also runs — unchanged — under
   [Rune.jvp] alone (observations inert: first-order subspace methods) or under
   [Rune.value_and_grad] (observations inert: exact gradients).

   A [Rune.vmap] around the whole sketch is a batch of sketches: each mapped
   parameter tree gets its own lanes, its own directions and its own
   observations, and the mapped result is the batch of their losses. *)

type diagnostics =
  { observed_loss : Nx.float64_t
    (** Σ l over the observed little losses, as a float64 scalar. *)
  ; observed_c : Nx.float64_t
    (** Σ ċ over the observed little losses, shape [k]: the tangent the
        collector saw where it counts. *)
  ; blocks : int (** Observations that contributed a curvature block. *)
  ; skipped : int
    (** Observations whose prediction was a constant of the differentiation:
        no block exists for them, since Y = 0. *)
  }

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
        is the draw's own — the update whitens it (see [Optim.coordinates]),
        so nothing downstream has to assume it is orthonormal. *)
  ; apply : Nx.float64_t -> 'p
    (** [apply z] is Θz for z : [k] — the parameter-space direction the
        sketch's coordinates denote, with each leaf in its parameter's
        dtype. An update rule composes [apply] after solving in the [k]
        dimensional subspace. *)
  ; diagnostics : diagnostics
  }

(* The default sketch: each leaf gets a standard normal [k]-lane batch, drawn
   from the ambient [Nx.Rng] scope (so a caller controls reproducibility by
   wrapping the sketch in [Nx.Rng.with_key]). Leaves that are not float — an
   RNG key or an index threaded through the parameters, say — cannot carry a
   direction and get zero tangents; their consumers are untracked operations,
   so they simply do not participate. *)
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

(* The consistency check. The collector records the value and the tangent of
   every little loss it was shown, so comparing them with the returned total
   detects the mistake that no type can: a loss term the user accumulated but
   never observed (missing from the sketch) or observed but never accumulated
   (missing from the loss). *)
let relative_gap observed total =
  let gap = Nx.item [] (Nx.max (Nx.abs (Nx.sub observed total))) in
  let scale = Nx.item [] (Nx.max (Nx.abs total)) in
  if scale = 0.0 then gap else gap /. scale

(* The cross-check on the numbers rather than on the record: a compiled step
   hands the observed sums back as tensors, and [Sofo.check] must mean the same
   thing there. *)
let check_sums
      ~tol
      ~(loss : Nx.float64_t)
      ~(c : Nx.float64_t)
      ~(observed_loss : Nx.float64_t)
      ~(observed_c : Nx.float64_t)
  : (unit, string) result
  =
  let loss_gap = relative_gap observed_loss loss in
  let c_gap = relative_gap observed_c c in
  if loss_gap <= tol && c_gap <= tol
  then Ok ()
  else
    Error
      (Printf.sprintf
         "the observed little losses do not add up to the loss: their sum differs from \
          it by %.3g (relative) and their tangents differ from the loss's tangent by \
          %.3g. A term accumulated into the loss but never passed to Sofo.observe is \
          missing from the sketch; a term observed but never accumulated is in the \
          sketch but not in the loss"
         loss_gap
         c_gap)

let check (sk : 'p t) : (unit, string) result =
  let tol = 1e-5 in
  check_sums
    ~tol
    ~loss:sk.loss
    ~c:sk.c
    ~observed_loss:sk.diagnostics.observed_loss
    ~observed_c:sk.diagnostics.observed_c

let run
      (type p c d)
      (structure : p Nx.Ptree.t)
      ~k
      ?sketch_sampler
      ?(strict = false)
      (loss : p -> (c, d) Nx.t)
      (params : p)
  : p t
  =
  if k < 1 then invalid_arg (Printf.sprintf "Sofo.sketch: k must be at least 1, got %d" k);
  let thetas =
    match sketch_sampler with
    | Some sampler -> sampler k params
    | None -> sample structure ~k params
  in
  let st = Collector.create ~k in
  (* The map runs the whole forward pass once; the collector, installed
     outside it, handles each observation where the lane axis is physical. *)
  let loss_lanes, dy =
    Effect.Deep.match_with
      (fun () ->
         Rune.vmap
           Nx.Ptree.(structure @-> returns (pair tensor tensor))
           (fun th -> Rune.jvp structure Nx.Ptree.tensor loss params th)
           thetas)
      ()
      (Collector.handler st)
  in
  if Nx.shape loss_lanes <> [| k |]
  then
    invalid_arg
      "Sofo.sketch: the loss function must return a scalar (a tensor with one element)";
  let y = Nx.reshape [||] (Nx.cast Nx.float64 (Nx.slice [ Nx.I 0 ] loss_lanes)) in
  let sk =
    { k
    ; loss = y
    ; c = Nx.reshape [| k |] (Nx.cast Nx.float64 dy)
    ; ggn = Collector.ggn st
    ; dirs = thetas
    ; apply =
        (fun z ->
          if Nx.shape z <> [| k |]
          then
            invalid_arg
              (Printf.sprintf
                 "Sofo.sketch: apply takes a k-vector (shape [%d]), got shape [%s]"
                 k
                 (String.concat
                    ","
                    (Array.to_list (Array.map string_of_int (Nx.shape z)))));
          Nx.Ptree.map structure (fun _ theta -> contract z theta) thetas)
    ; diagnostics =
        { observed_loss = st.Collector.loss
        ; observed_c = st.Collector.c
        ; blocks = st.Collector.blocks
        ; skipped = st.Collector.skipped
        }
    }
  in
  if strict
  then (
    match check sk with
    | Ok () -> ()
    | Error msg -> invalid_arg ("Sofo.sketch: " ^ msg));
  sk
