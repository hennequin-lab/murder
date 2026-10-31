(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The SOFO update (Algorithm 1, lines 10–12), vega-flavoured: pure functions
   over parameter structures, a state that is nothing but the direction stream
   and an iteration counter, and hyperparameters passed per step.

   Two halves, and the split is forced by the hardware rather than chosen:

   - the *sketching* half is differentiable and compiles, so [sketch_jit]
     wraps the (key, aux, params) → (loss, c, C, G̃, Θ, gram) computation in
     [Rune.jit];
   - the *update* half needs the eigendecomposition of the sketched GGN,
     which does not compile, so it runs eagerly on the host at O(k³) —
     negligible against a model with P ≫ k parameters, which is the
     algorithm's whole premise. Only k-sized matrices cross to the host: the
     step on the parameters is compiled again, where they live.

   Everything here is a pure function of values. The only state is [state]: the
   key the directions are drawn from, carried as an int32 tensor so it can ride
   a compiled step as an ordinary input leaf, and the iteration counter, which
   stays on the host because it only ever feeds key derivation and schedules.
   An iteration consumes that state and produces its successor, so [step] and
   [update] return the two together and a loop threads one state rather than
   remembering to derive the next one. *)

open Nx

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
  { key : Nx.host Nx.Rng.key
    (* The stream Θ is drawn from. A tensor, not an OCaml value: it is what a
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

let directions (type p d) (structure : (p, d) Nx.Ptree.t) ?key ~k (params : p) : p =
  match key with
  | Some key -> Sketch.sample structure ~key ~k params
  | None ->
    Nx.Rng.with_key (Nx.Rng.next_key ()) (fun () -> Sketch.sample structure ~k params)

let apply (type p d) (structure : (p, d) Nx.Ptree.t) ~k (thetas : p) (z : d Nx.float64_t)
  : p
  =
  let z = reshape [| k |] z in
  Nx.Ptree.map
    structure
    (fun _ theta ->
       (* Leaves that cannot carry a direction — a key, an index, a counter —
          get the zero direction, as they do in the sampler. *)
       if Nx.Dtype.is Nx.Dtype.Float (Nx.dtype theta)
       then Sketch.contract z theta
       else
         zeros_at
           (beside theta)
           (Nx.dtype theta)
           (Array.sub (shape theta) 1 (Array.length (shape theta) - 1)))
    thetas

(* [params - lr * dw], leafwise, in each leaf's own dtype, for a scalar
   tensor [lr]: a tensor, so a compiled step takes a new rate as data. Non-float
   leaves are passed through untouched: their directions are zero, and a
   parameter that cannot carry one is not the optimizer's to move (vega's
   convention for the same leaves). *)
let shift
      (type p d)
      (structure : (p, d) Nx.Ptree.t)
      ~(lr : d Nx.float64_t)
      (params : p)
      (dw : p)
  : p
  =
  Nx.Ptree.map2
    structure
    (fun _ p d ->
       if Nx.Dtype.is Nx.Dtype.Float (Nx.dtype p)
       then Infix.(p - (d * cast (Nx.dtype d) lr))
       else p)
    params
    dw

(* ── whitening the sampled basis ─────────────────────────────────────────── *)

(* [gram structure ~k dirs] is Θ Θᵀ: the [k×k] Gram matrix of the tangents,
   accumulated leaf by leaf — each float leaf contributes its own outer product
   rather than a materialized [k×P] matrix. Leaves are promoted to float64
   first, so the matrix does not depend on the model's dtypes, and a leaf that
   cannot carry a direction is zero and contributes nothing. *)
let gram (type p d) (structure : (p, d) Nx.Ptree.t) ~k (dirs : p) : d Nx.float64_t =
  Nx.Ptree.fold
    structure
    (fun _ leaf z ->
       if Nx.Dtype.is Nx.Dtype.Float (Nx.dtype leaf)
       then (
         let m = reshape [| k; numel leaf / k |] (cast float64 leaf) in
         Infix.(z + (m *@ transpose m)))
       else z)
    dirs
    (zeros_at
       (Sketch.beside_first ~fn:"Sofo.Optim.gram" structure dirs)
       float64
       [| k; k |])

(* [spectrum z] is [(u, s, smax, smin)] for a symmetric matrix [z]: [z]'s
   singular vectors [u] and values [s], with the largest and smallest of [s].
   A symmetric matrix's singular values are its eigenvalues' magnitudes and its
   singular vectors are its eigenvectors, so [Nx.eigh] gives the factorization
   [Nx.svd] would, [z = u diag(±s) uᵀ], at about half the cost. Taking
   magnitudes keeps the SVD's semantics where rounding leaves an eigenvalue of
   a positive semi-definite matrix slightly negative. *)
let spectrum (z : 'd Nx.float64_t) =
  let w, u = eigh float64 z in
  let s = abs w in
  u, s, item [] (max s), item [] (min s)

(* The whitening transform Q = Z^{-1/2} of a Gram matrix: the matrix that turns
   the tangents a sketch was drawn along into an orthonormal basis. Q is the
   symmetric inverse square root [u s^{-1/2} uᵀ], so [Q G̃ Qᵀ] and the step's
   [Θ (Q z)] are the same change of basis. [coordinates] makes the same choice
   for the same reason.

   A rank-deficient Gram has no inverse square root and no arithmetic will
   produce one: some drawn lanes span nothing, which is a statement about [k]
   against the number of parameters, so it raises. *)
let inverse_sqrt (z : 'd Nx.float64_t) : 'd Nx.float64_t =
  let k = (shape z).(0) in
  let u, s, smax, smin = spectrum z in
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
  let inv_sqrt = rsqrt s in
  Infix.(u * reshape [| 1; k |] inv_sqrt *@ transpose u)

(* ── the update (Alg. 1, lines 10–12) ────────────────────────────────────── *)

(* z = U (S + γ I)^{-p} Uᵀ C, from the spectrum of the sketched GGN, with γ set
   by the damping mode and p by the preconditioner (1, Alg. 1's; or 1/2, which
   caps what a poorly resolved direction can contribute). G̃ is symmetric
   positive semi-definite, so Algorithm 1's SVD is its eigendecomposition
   ([spectrum]).

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
      (ggn : 'd Nx.float64_t)
      (c : 'd Nx.float64_t)
  : 'd Nx.float64_t
  =
  let k = (shape c).(0) in
  let q = Option.map inverse_sqrt gram in
  let ggn =
    match q with
    | None -> ggn
    | Some q -> Infix.(q *@ (ggn *@ transpose q))
  in
  let c =
    match q with
    | None -> c
    | Some q -> Infix.(q *@ c)
  in
  let u, s, smax, smin = spectrum ggn in
  let vt = transpose u in
  let gamma =
    match damping with
    | `Absolute value -> value
    | `Relative_from_top factor -> factor *. smax
    | `Relative_from_bottom factor ->
      if smin <= 0.0
      then
        invalid_arg
          "Sofo.Optim.coordinates: `Relative_from_bottom needs a full-rank sketch, but \
           the smallest singular value of the sketched GGN is 0; damp absolutely or from \
           the top instead";
      factor *. smin
  in
  let damped = Infix.(s +$ gamma) in
  let scale =
    match preconditioner with
    | `Inverse -> damped
    | `Inverse_sqrt -> sqrt damped
  in
  let rhs = Infix.(vt *@ reshape [| k; 1 |] c) in
  let z = reshape [| k |] Infix.(u *@ (rhs / reshape [| k; 1 |] scale)) in
  (* Back to the sketch's own coordinates, where [apply] contracts them: the
     step is Θ (Q z). *)
  match q with
  | None -> z
  | Some q -> Infix.(transpose q *@ z)

let update_sketch
      (type p d)
      (structure : (p, d) Nx.Ptree.t)
      ?(lr = 1.0)
      ?damping
      ?preconditioner
      (sk : (p, d) Sketch.t)
      (params : p)
  : p
  =
  let g = gram structure ~k:sk.k sk.dirs in
  let z = coordinates ?damping ?preconditioner ~gram:g sk.ggn sk.c in
  shift structure ~lr:(scalar_like z lr) params (sk.apply z)

(* ── the eager step ──────────────────────────────────────────────────────── *)

let step
      (type p c e d)
      (structure : (p, d) Nx.Ptree.t)
      ~k
      ~lr
      ?damping
      ?preconditioner
      (st : state)
      ~(loss : p -> (c, e, d) Nx.t)
      ~(params : p)
  : p * state * (p, d) Sketch.t
  =
  let at = Sketch.beside_first ~fn:"Sofo.Optim.step" structure params in
  (* Whole on every device of the set, as [sketch_jit] places it: a leaf on
     another device than the first reads it there. *)
  let key = Nx.Rng.place (Nx.Placement.on (Nx.Placement.devices at)) st.key in
  let thetas = directions structure ~key ~k params in
  let sk = Sketch.run structure loss params thetas in
  let params = update_sketch structure ~lr ?damping ?preconditioner sk params in
  params, next st, sk

(* ── the compiled half ─────────────────────────────────── *)

type ('p, 'd) step_info =
  { loss : 'd Nx.float64_t
  ; c : 'd Nx.float64_t
  ; ggn : 'd Nx.float64_t
  ; gram : 'd Nx.float64_t
  ; dirs : 'p
  }

(* [place_all s p x] is [x] with each array at [p], a placement of its own set. *)
let place_all (type s d) (structure : (s, d) Nx.Ptree.t) (p : d Nx.Placement.t) (x : s) =
  Nx.Ptree.map structure (fun _ a -> place p a) x

let sketch_jit
      ?beam
      ?parallel
      ~k
      (placement : 'd Nx.Placement.t)
      (p : ('p, 'd) Nx.Ptree.t)
      (a : ('aux, 'd) Nx.Ptree.t)
      loss
  =
  let module Out = struct
    type 'd t =
      { loss : 'd Nx.float64_t
      ; c : 'd Nx.float64_t
      ; ggn : 'd Nx.float64_t
      ; gram : 'd Nx.float64_t
      }
    [@@deriving ptree]
  end
  in
  let sketch key aux params =
    let dirs = directions p ~key ~k params in
    let sk = Sketch.run p (fun params -> loss params aux) params dirs in
    Out.{ loss = sk.loss; c = sk.c; ggn = sk.ggn; gram = gram p ~k dirs }, dirs
  in
  let f =
    Rune.jit
      ?beam
      ?parallel
      Nx.Ptree.(v Nx.Rng.walk @-> a @-> p @-> returns (pair (v Out.walk) p))
      sketch
  in
  fun ~key ~aux params ->
    let key = Nx.Rng.place (Nx.Placement.uncut placement) key in
    let aux = place_all a placement aux in
    let params = place_all p placement params in
    let out, dirs = f key aux params in
    { loss = out.loss; c = out.c; ggn = out.ggn; gram = out.gram; dirs }

(* [params - lr * Θ z], compiled, so it runs where the parameters and the
   directions live. [z] and [lr] are host values that join them per call. *)
let shift_by p =
  Rune.jit
    ~beam:2
    ~parallel:8
    Nx.Ptree.(p @-> p @-> tensor @-> tensor @-> returns p)
    (fun params dirs (z : 'd Nx.float64_t) lr ->
       shift p ~lr params (apply p ~k:(shape z).(0) dirs z))

(* The step, and the state that produced it, advanced together: the caller
   threads one state through the loop, so the key the sketch consumed and the
   key the next sketch is drawn from cannot drift apart. The solve runs on
   the host, which copies three k-sized tensors when the sketch ran on a
   GPU. *)
let update
      ?(lr = 1.0)
      ?damping
      ?preconditioner
      (placement : 'd Nx.Placement.t)
      (p : ('p, 'd) Nx.Ptree.t)
      (st : state)
      params
      (o : ('p, 'd) step_info)
  =
  let host = place Nx.Host.on in
  let z =
    coordinates ?damping ?preconditioner ~gram:(host o.gram) (host o.ggn) (host o.c)
  in
  ( shift_by
      p
      (place_all p placement params)
      (place_all p placement o.dirs)
      (place placement z)
      (scalar_at placement float64 lr)
  , next st )
