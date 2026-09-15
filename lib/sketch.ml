(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The sketch driver: sample directions, run the loss under batched forward
   mode with a collector installed, and package what an update rule needs.

   Composition is the whole trick, and it needs nothing from rune's internals
   beyond its public surface: [Rune.jvp_k] installs batched forward mode
   *outside* the collector, the collector is installed *outside* the user's
   loss, and the layer order does the rest.

   user code  ⊂  collector  ⊂  jvp_k  ⊂  driver

   - The user's operations reach the collector, which passes them through to
     [jvp_k]: primals are computed once, tangents in batches of [k].
   - The user's observations reach the collector, which reads the tangents it
     needs through [Rune.tangent] (a tangent is not an Nx operation, so
     [jvp_k] passes the observation through) and accumulates its block.
   - The tangent of the *returned* total loss is the gradient sketch itself:
     C = Θᵀ∇c, read off the [jvp_k] result. No instrumentation is involved, so
     the gradient sketch cannot drift from the loss's own arithmetic.

   Because the collector is a plain effect handler and the tangents are read
   through a query, the same user code also runs — unchanged — under
   [Rune.jvp_k] alone (observations inert: first-order subspace methods) or
   under [Rune.value_and_grad] (observations inert: exact gradients). *)

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
let sample (type p) (module P : Nx.Ptree.S with type t = p) ~k (params : p) : p =
  P.map
    (fun leaf ->
       let shape = Array.append [| k |] (Nx.shape leaf) in
       if Nx_core.Dtype.is_float (Nx.dtype leaf)
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
      (module P : Nx.Ptree.S)
      ~k
      ?sketch_sampler
      ?(strict = false)
      (loss : P.t -> ('c, 'd) Nx.t)
      (params : P.t)
  : P.t t
  =
  if k < 1 then invalid_arg (Printf.sprintf "Sofo.sketch: k must be at least 1, got %d" k);
  let thetas =
    match sketch_sampler with
    | Some sampler -> sampler k params
    | None -> sample (module P) ~k params
  in
  let st = Collector.create ~k in
  let y, dy =
    Rune.jvp_k
      (module P)
      (fun params ->
         Effect.Deep.match_with (fun () -> loss params) () (Collector.handler st))
      params
      thetas
  in
  if Nx.numel y <> 1
  then
    invalid_arg
      "Sofo.sketch: the loss function must return a scalar (a tensor with one element)";
  let sk =
    { k
    ; loss = Nx.reshape [||] (Nx.cast Nx.float64 y)
    ; c = Nx.reshape [| k |] (Nx.cast Nx.float64 dy)
    ; ggn = Collector.ggn st
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
          P.map (fun theta -> contract z theta) thetas)
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
