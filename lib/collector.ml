(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The collector: accumulate what a sketch needs, one little loss at a time.

   Installed around the user's loss function, inside the batched forward-mode
   scope, it intercepts [E_observe] and records, per little loss l = ℓ(y):

   - its value, into Σ l;
   - its tangent, into Σ ċ — the cross-check against the tangent of the total
     loss, which is what the driver returns as C;
   - the block YᵀHY of the generalized Gauss-Newton matrix, with H the
     curvature description the observation carried and Y the tangent batch of
     y.

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

(* One observation. The curvature block needs the batched tangent of the
   prediction; a single-tangent forward mode has no such thing, and saying so
   is more useful than contracting a mismatched shape. *)
let record (type a b c d) (st : t) ~(y : (a, b) Nx.t) ~(l : (c, d) Nx.t) ~(curv : Curv.t) =
  st.loss <- Nx.add st.loss (Nx.reshape [||] (Nx.cast Nx.float64 l));
  (match Rune.tangent l with
   | None -> ()
   | Some dl -> st.c <- Nx.add st.c (Nx.reshape [| st.k |] (Nx.cast Nx.float64 dl)));
  match Rune.tangent y with
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
            batched form [%s] of a %d-lane forward mode; a single tangent of shape [%s] \
            comes from Rune.jvp, which has no lane axis to contract over — differentiate \
            with Rune.jvp_k"
           (shape_string shape_dy)
           (shape_string expected)
           st.k
           (shape_string shape_y));
    let n = Nx.numel y in
    let row t = Nx.reshape [| st.k; n |] (Nx.contiguous t) in
    let block = Nx.matmul (row dy) (Nx.transpose (row (Curv.apply curv dy))) in
    st.ggn <- Nx.add st.ggn (Nx.cast Nx.float64 block);
    st.blocks <- st.blocks + 1

let handler : type r. t -> (r, r) Effect.Deep.handler =
  fun st ->
  let open Effect.Deep in
  (* Bound, and annotated, before it goes into the record: the [type c.] form
     keeps [c] rigid across the GADT match, so the callback generalizes to every
     effect the handler passes through. *)
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
    fun eff ->
    match eff with
    | Observe.E_observe (Observe.Obs (y, l, curv)) ->
      Some
        (fun k ->
          record st ~y ~l ~curv;
          continue k ())
    | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }

(* Every block is symmetric by construction — YᵀHY with H symmetric — so the
   accumulator is symmetric up to floating-point asymmetry, which later
   decompositions would rather not see. *)
let ggn st = Nx.mul_s (Nx.add st.ggn (Nx.transpose st.ggn)) 0.5
