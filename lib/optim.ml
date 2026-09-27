(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The SOFO update (Algorithm 1, lines 10–12), vega-flavoured: pure functions
   over parameter structures, a state that is nothing but the direction stream
   and an iteration counter, and hyperparameters passed per step.

   Two halves, and the split is forced by the hardware rather than chosen:

   - the *sketching* half is differentiable and compiles, so [Compiled] builds
     the (params, key) → (c, C, G̃, Θ) structure that [Rune.jit] wraps;
   - the *update* half needs the SVD of the sketched GGN, which does not
     compile, so it runs eagerly on the host at O(k³) — negligible against a
     model with P ≫ k parameters, which is the algorithm's whole premise.

   Everything here is a pure function of values. The only state is [state]: the
   key the directions are drawn from, carried as an int32 tensor so it can ride
   a compiled step as an ordinary input leaf, and the iteration counter, which
   stays on the host because it only ever feeds key derivation and schedules.
   An iteration consumes that state and produces its successor, so [step] and
   [Compiled.update] return the two together and a loop threads one state
   rather than remembering to derive the next one. *)

type damping =
  [ `Absolute of float
  | `Relative_from_top of float
  | `Relative_from_bottom of float
  ]

type preconditioner =
  [ `Inverse
  | `Inverse_sqrt
  ]

type state =
  { key : Nx.Rng.t
    (* The stream Θ is drawn from. A tensor, not a host value: it is what a
       compiled sketch step takes as an input, and what makes each replay draw
       fresh directions from one compilation. *)
  ; step : int (* Iterations completed; the counter key derivation folds in. *)
  }

let init ?key () =
  { key =
      (match key with
       | Some k -> k
       | None -> Nx.Rng.next_key ())
  ; step = 0
  }

(* The next state's key is a subkey of this one indexed by the iteration, so a
   run is reproducible from its first key alone and no two steps share a draw. *)
let next st = { key = Nx.Rng.fold_in st.key st.step; step = st.step + 1 }

(* ── directions and their contraction ────────────────────────────────────── *)

let directions (type p) (structure : p Nx.Ptree.t) ?key ~k (params : p) : p =
  let key =
    match key with
    | Some key -> key
    | None -> Nx.Rng.next_key ()
  in
  Nx.Rng.with_key key (fun () -> Sketch.sample structure ~k params)

let apply (type p) (structure : p Nx.Ptree.t) ~k (thetas : p) (z : Nx.float64_t) : p =
  let z = Nx.reshape [| k |] (Nx.contiguous z) in
  Nx.Ptree.map
    structure
    (fun _ theta ->
       (* Leaves that cannot carry a direction — a key, an index, a counter —
          get the zero direction, as they do in the sampler. *)
       if Nx_dtype.is_float (Nx.dtype theta)
       then Sketch.contract z theta
       else
         Nx.zeros
           (Nx.dtype theta)
           (Array.sub (Nx.shape theta) 1 (Array.length (Nx.shape theta) - 1)))
    thetas

(* [params - lr * dw], leafwise, in each leaf's own dtype. Non-float leaves
   are passed through untouched: their directions are zero, and a parameter
   that cannot carry one is not the optimizer's to move (vega's convention for
   the same leaves). *)
let shift (type p) (structure : p Nx.Ptree.t) ~lr (params : p) (dw : p) : p =
  Nx.Ptree.map2
    structure
    (fun _ p d ->
       if Nx_dtype.is_float (Nx.dtype p)
       then Nx.sub p (Nx.mul_s d (Nx_dtype.of_float (Nx.dtype d) lr))
       else p)
    params
    dw

(* ── whitening the sampled basis ─────────────────────────────────────────── *)

(* [gram structure ~k dirs] is Θ Θᵀ: the [k×k] Gram matrix of the tangents,
   accumulated leaf by leaf — each float leaf contributes its own outer product
   rather than a materialized [k×P] matrix. Leaves are promoted to float64
   first, so the matrix does not depend on the model's dtypes, and a leaf that
   cannot carry a direction is zero and contributes nothing. *)
let gram (type p) (structure : p Nx.Ptree.t) ~k (dirs : p) : Nx.float64_t =
  Nx.Ptree.fold
    structure
    (fun _ leaf z ->
       if Nx_dtype.is_float (Nx.dtype leaf)
       then (
         let m =
           Nx.reshape [| k; Nx.numel leaf / k |] (Nx.contiguous (Nx.cast Nx.float64 leaf))
         in
         Nx.add z (Nx.matmul m (Nx.transpose m)))
       else z)
    dirs
    (Nx.zeros Nx.float64 [| k; k |])

(* The whitening transform Q = Z^{-1/2} of a Gram matrix, from its SVD: the
   matrix that turns the tangents a sketch was drawn along into an orthonormal
   basis. [Nx.svd] may return a [vt] that is not [u]'s transpose even though Z
   is symmetric — the singular vectors of a degenerate spectrum are defined
   only up to a rotation within it — so the second factor is [u]'s own
   transpose, and Q is the symmetric inverse square root: [Q G̃ Qᵀ] and the
   step's [Θ (Q z)] are then the same change of basis, which they would not be
   for [u·s^{-1/2}·vt]. [coordinates] makes the same choice for the same
   reason.

   A rank-deficient Gram has no inverse square root and no arithmetic will
   produce one: some drawn lanes span nothing, which is a statement about [k]
   against the number of parameters, so it raises. *)
let inverse_sqrt (z : Nx.float64_t) : Nx.float64_t =
  let k = (Nx.shape z).(0) in
  let u, s, _ = Nx.svd z in
  let smin = Nx.item [ k - 1 ] s
  and smax = Nx.item [ 0 ] s in
  if smin <= 1e-12 *. smax
  then
    invalid_arg
      (Printf.sprintf
         "Sofo.Optim: the drawn directions cannot be whitened — their Gram matrix is \
          rank-deficient (s_min/s_max = %.2g), so %d orthonormal directions do not fit \
          in what they span; whitening needs at least as many parameters that can carry \
          a direction as there are lanes."
         (if smax = 0.0 then 0.0 else smin /. smax)
         k);
  let inv_sqrt = Nx.div (Nx.ones_like s) (Nx.sqrt s) in
  Nx.matmul (Nx.mul u (Nx.reshape [| 1; k |] inv_sqrt)) (Nx.transpose u)

(* ── the update (Alg. 1, lines 10–12) ────────────────────────────────────── *)

(* z = U (S + γ I)^{-p} Vᵀ C, from the SVD of the sketched GGN, with γ set by
   the damping mode and p by the preconditioner (1, Alg. 1's; or 1/2, which
   caps what a poorly resolved direction can contribute). G̃ is symmetric
   positive semi-definite, so V = U up to signs and an eigendecomposition would
   do; the SVD is what the algorithm prescribes and what is used here.

   [gram] whitens before it solves. The sketched GGN is the curvature of the
   sampled subspace expressed in whatever basis the draw happened to produce,
   so it says what the loss does only once that basis is orthonormal:
   G̃ ← Q G̃ Qᵀ and C ← Q C with Q = (ΘΘᵀ)^{-1/2}, and the coordinates come back
   through Qᵀ so that [apply] of them is the step. The whitening belongs here,
   with the Gram handed in, rather than on the directions themselves: the SVD
   that produces Q cannot run inside a jitted step, while this one is host-side
   by construction. *)
let coordinates
      ?(damping : damping = `Relative_from_top 1e-6)
      ?(preconditioner : preconditioner = `Inverse)
      ?gram
      (ggn : Nx.float64_t)
      (c : Nx.float64_t)
  : Nx.float64_t
  =
  let k = (Nx.shape c).(0) in
  let q = Option.map inverse_sqrt gram in
  let ggn =
    match q with
    | None -> ggn
    | Some q -> Nx.matmul q (Nx.matmul ggn (Nx.transpose q))
  in
  let c =
    match q with
    | None -> c
    | Some q -> Nx.matmul q c
  in
  let u, s, _ = Nx.svd ggn in
  let vt = Nx.transpose u in
  let gamma =
    match damping with
    | `Absolute value -> value
    | `Relative_from_top factor -> factor *. Nx.item [ 0 ] s (* singular values descend *)
    | `Relative_from_bottom factor ->
      let smin = Nx.item [ k - 1 ] s in
      if smin <= 0.0
      then
        invalid_arg
          "Sofo.Optim.coordinates: `Relative_from_bottom needs a full-rank sketch, but \
           the smallest singular value of the sketched GGN is 0; damp absolutely or from \
           the top instead";
      factor *. smin
  in
  let damped = Nx.add_s s gamma in
  let scale =
    match preconditioner with
    | `Inverse -> damped
    | `Inverse_sqrt -> Nx.sqrt damped
  in
  let rhs = Nx.matmul vt (Nx.reshape [| k; 1 |] (Nx.contiguous c)) in
  let z = Nx.reshape [| k |] (Nx.matmul u (Nx.div rhs (Nx.reshape [| k; 1 |] scale))) in
  (* Back to the sketch's own coordinates, where [apply] contracts them: the
     step is Θ (Q z). *)
  match q with
  | None -> z
  | Some q -> Nx.matmul (Nx.transpose q) z

let update
      (type p)
      (structure : p Nx.Ptree.t)
      ?(lr = 1.0)
      ?damping
      ?preconditioner
      (sk : p Sketch.t)
      (params : p)
  : p
  =
  let g = gram structure ~k:sk.k sk.dirs in
  let dw = sk.apply (coordinates ?damping ?preconditioner ~gram:g sk.ggn sk.c) in
  shift structure ~lr params dw

(* ── the eager step ──────────────────────────────────────────────────────── *)

let step
      (type p c d)
      (structure : p Nx.Ptree.t)
      ~k
      ~lr
      ?damping
      ?preconditioner
      ?(strict = false)
      (st : state)
      ~(loss : p -> (c, d) Nx.t)
      ~(params : p)
  : p * state * p Sketch.t
  =
  let thetas = directions structure ~key:st.key ~k params in
  let sk =
    Sketch.run structure ~k ~sketch_sampler:(fun _ _ -> thetas) ~strict loss params
  in
  let params = update structure ~lr ?damping ?preconditioner sk params in
  params, next st, sk

(* ── the compiled half ─────────────────────────────────── *)

(* What a jitted step consumes and produces. Both are parameter trees, so a
   training step is [Rune.jit Optim.Compiled.signature (sketch ~k loss)] and
   a loop threads one input record to the next without ever retracing. *)

(* What the [Compiled] functor needs of a value's structure: its type and its
   tree, as {!Nx.Ptree.instantiate} of a [walk] module, or the [ptree] the
   [ptree] deriver writes for a concrete record. *)
module type Structure = sig
  type t

  val ptree : t Nx.Ptree.t
end

(* The trivial auxiliary input: no leaves at all, for a loss that reads the
   parameters and nothing else. *)
module No_aux = struct
  type t = unit

  let ptree = Nx.Ptree.unit
end

module Compiled (P : Structure) (Aux : Structure) = struct
  type in_ =
    { params : P.t
    ; key : Nx.Rng.t
    ; aux : Aux.t
    }

  module In = struct
    type _ t = in_

    let walk c i =
      let open Nx.Ptree.Walk in
      let params = field c "params" (structure P.ptree) i.params in
      let key = field c "key" (structure Nx.Rng.ptree) i.key in
      let aux = field c "aux" (structure Aux.ptree) i.aux in
      { params; key; aux }
  end

  let in_ptree = Nx.Ptree.instantiate (module In)

  type out =
    { loss : Nx.float64_t
    ; c : Nx.float64_t
    ; ggn : Nx.float64_t
    ; dirs : P.t
    ; observed_loss : Nx.float64_t
    ; observed_c : Nx.float64_t
    }

  module Out = struct
    type _ t = out

    let walk cur o =
      let open Nx.Ptree.Walk in
      let loss = field cur "loss" tensor o.loss in
      let c = field cur "c" tensor o.c in
      let ggn = field cur "ggn" tensor o.ggn in
      let dirs = field cur "dirs" (structure P.ptree) o.dirs in
      let observed_loss = field cur "observed_loss" tensor o.observed_loss in
      let observed_c = field cur "observed_c" tensor o.observed_c in
      { loss; c; ggn; dirs; observed_loss; observed_c }
  end

  let out_ptree = Nx.Ptree.instantiate (module Out)

  (* The signature [Rune.jit] takes: an input record of parameters, key and
     auxiliary data, and the output record above. *)
  let signature = Nx.Ptree.(in_ptree @-> returns out_ptree)

  (* The sketching computation, as a pure function of (params, key, aux).
     Compile it with [Rune.jit signature]; run it as it is for an eager step.
     The directions are drawn *inside* it, from the carried key, so one
     compilation serves every iteration, and so is the aux: the loss sees its
     leaves, so a batch that changes between steps is data, not a new trace. *)
  let sketch ~k (loss : P.t -> Aux.t -> ('c, 'd) Nx.t) i =
    let dirs = directions P.ptree ~key:i.key ~k i.params in
    let sk =
      Sketch.run
        P.ptree
        ~k
        ~sketch_sampler:(fun _ _ -> dirs)
        (fun params -> loss params i.aux)
        i.params
    in
    { loss = sk.loss
    ; c = sk.c
    ; ggn = sk.ggn
    ; dirs
    ; observed_loss = sk.diagnostics.observed_loss
    ; observed_c = sk.diagnostics.observed_c
    }

  (* The step, and the state that produced it, advanced together: the caller
     threads one state through the loop, so the key the sketch consumed and the
     key the next sketch is drawn from cannot drift apart. *)
  let update ?(lr = 1.0) ?damping ?preconditioner (st : state) params o =
    let k = (Nx.shape o.c).(0) in
    let g = gram P.ptree ~k o.dirs in
    let dw =
      apply P.ptree ~k o.dirs
        (coordinates ?damping ?preconditioner ~gram:g o.ggn o.c)
    in
    shift P.ptree ~lr params dw, next st

  (* The consistency check, on the numbers the compiled step handed back: the
     same statement as [Sofo.check], which cannot run inside a trace because it
     reads values. *)
  let check o =
    Sketch.check_sums
      ~tol:1e-5
      ~loss:o.loss
      ~c:o.c
      ~observed_loss:o.observed_loss
      ~observed_c:o.observed_c
end
