(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The collector: accumulate what a sketch needs, one little loss at a time.

   {!Sketch.run} installs it around the loss function, *outside* the [Rune.vmap]
   that batches the forward pass; it intercepts [E_observe] and records, per
   little loss l = ℓ(y):

   - its value, into Σ l;
   - its tangent, into Σ ċ — the cross-check against the tangent of the total
     loss, which is what the driver returns as C;
   - the block YᵀHY of the generalized Gauss-Newton matrix, with H the
     curvature description the observation carried and Y the tangent batch of
     y.

   The tangents arrive in the observation payload, read at the observation site
   while the forward mode was in scope. That placement is what makes the lane
   axis ordinary arithmetic here: the payload carries the whole [k]-lane batch,
   and this handler runs where [Nx.shape] reports it (see observe.ml). A
   collector installed *inside* the map would be handed the same tensors with
   the map's virtual, lane-less shapes, and could not contract the lanes.

   The value and its tangent are recorded even when the prediction is a
   constant of the differentiation (its block is then zero by construction,
   since Y = 0), so the cross-check stays a faithful statement about
   *observations* rather than about activity.

   Per observation the work is one application of H over a [k·n] tangent batch
   and one [k×k] contraction — O(k·n + k²·n) — negligible beside the K-batched
   step operations that produced the tangent. The collector's own operations
   run with inactive operands (a tangent is not a primal-with-a-tangent), so
   they execute eagerly without bookkeeping; under an enclosing [jit] they are
   traced like any other operation. Its state is O(k²) plus counters, and it
   keeps no reference to a tensor the user has dropped. *)

type t =
  { k : int
  ; mutable loss : Nx.float64_t (* Σ l, scalar *)
  ; mutable c : Nx.float64_t (* Σ ċ, shape [k] *)
  ; mutable ggn : Nx.float64_t (* Σ YᵀHY, shape [k;k] *)
  ; mutable blocks : int (* observations that contributed a block *)
  ; mutable skipped : int (* observations whose prediction is constant *)
  }

let create ~k =
  { k
  ; loss = Nx.zeros Nx.float64 [||]
  ; c = Nx.zeros Nx.float64 [| k |]
  ; ggn = Nx.zeros Nx.float64 [| k; k |]
  ; blocks = 0
  ; skipped = 0
  }

let shape_string s = String.concat "," (Array.to_list (Array.map string_of_int s))

(* One observation, with the tangents the payload carried. The curvature block
   needs the batched tangent of the prediction; a plain single-tangent [jvp]
   has no lane axis, and saying so is more useful than contracting a
   mismatched shape. *)
let record (type a b c d) (st : t)
    ~(y : (a, b) Nx.t) ~(dy : (a, b) Nx.t option)
    ~(l : (c, d) Nx.t) ~(dl : (c, d) Nx.t option) ~(curv : Curv.t) =
  (* The little loss may itself be packed along an enclosing map's axis — an
     observation performed inside [Rune.vmap], carrying one little loss per
     mapped element. Each of them contributes to the total loss, so the value
     and the tangent are summed over the batch, exactly as the curvature block
     below (a contraction over every element of the prediction) already sums
     over it. The three must agree, and [Sofo.check] compares their sums with
     the loss the driver returned: a loss the user reduced with a mean over the
     map's axis disagrees by the batch factor and is reported. *)
  st.loss <- Nx.add st.loss (Nx.reshape [||] (Nx.cast Nx.float64 (Nx.sum l)));
  (match dl with
   | None -> ()
   | Some dl ->
     let n = Nx.numel dl in
     if n mod st.k <> 0
     then
       invalid_arg
         (Printf.sprintf
            "Sofo: a little loss's tangent has %d elements, not a multiple of the \
             sketch's %d lanes — a tangent of shape [%s] has no lane axis; this sketch \
             reads the batch a Rune.vmap around Rune.jvp produces, not a bare \
             Rune.jvp's single tangent"
            n
            st.k
            (shape_string (Nx.shape dl)));
     let lanes = Nx.reshape [| st.k; -1 |] (Nx.contiguous (Nx.cast Nx.float64 dl)) in
     st.c <- Nx.add st.c (Nx.sum lanes ~axes:[ 1 ]));
  match dy with
  | None -> st.skipped <- st.skipped + 1
  | Some dy ->
    let shape_y = Nx.shape y in
    let shape_dy = Nx.shape dy in
    let expected = Array.append [| st.k |] shape_y in
    if shape_dy <> expected
    then
      invalid_arg
        (Printf.sprintf
           "Sofo: a prediction's tangent has shape [%s], but this sketch contracts the \
            batched form [%s] of a %d-lane forward mode; a tangent of shape [%s] has no \
            lane axis — differentiate under the sketch's vmap, not a bare Rune.jvp"
           (shape_string shape_dy)
           (shape_string expected)
           st.k
           (shape_string shape_y));
    let n = Nx.numel y in
    let row t = Nx.reshape [| st.k; n |] (Nx.contiguous t) in
    let block = Nx.matmul (row dy) (Nx.transpose (row (Curv.apply curv dy))) in
    st.ggn <- Nx.add st.ggn (Nx.cast Nx.float64 block);
    st.blocks <- st.blocks + 1

let rec handler : type r. t -> (r, r) Effect.Deep.handler =
  fun st ->
  let open Effect.Deep in
  (* Bound, and annotated, before it goes into the record: the [type c.] form
     keeps [c] rigid across the GADT match, so the callback generalizes to every
     effect the handler passes through. *)
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
    fun eff ->
    match eff with
    | Observe.E_observe (Observe.Obs (y, dy, l, dl, curv)) ->
      Some
        (fun k ->
          record st ~y ~dy ~l ~dl ~curv;
          continue k ())
    (* Everything else — including a [Rune.scan] inside the loss — falls
       through. The forward mode claims the fold and runs it in its own
       extent, and the little losses its body performs still reach this
       handler: an effect performs outward from the fold's handler, and this
       collector is outermost. *)
    | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }

(* Every block is symmetric by construction — YᵀHY with H symmetric — so the
   accumulator is symmetric up to floating-point asymmetry, which later
   decompositions would rather not see. *)
let ggn st = Nx.mul_s (Nx.add st.ggn (Nx.transpose st.ggn)) 0.5
