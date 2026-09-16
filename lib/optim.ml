(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The SOFO update (Algorithm 1, lines 10–12), vega-flavoured: pure functions
   over parameter structures, a state that is nothing but the direction stream
   and an iteration counter, and hyperparameters passed per step.

   Two halves, and the split is forced by the hardware rather than chosen:

   - the *sketching* half is differentiable and compiles, so [Compiled] builds
     the (params, key) → (c, C, G̃, Θ) structure that [Rune.jit2] wraps;
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
  { key : Nx.int32_t
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

let directions (type p) (module P : Nx.Ptree.S with type t = p) ?key ~k (params : p) : p =
  let key =
    match key with
    | Some key -> key
    | None -> Nx.Rng.next_key ()
  in
  Nx.Rng.with_key key (fun () -> Sketch.sample (module P) ~k params)

let apply
      (type p)
      (module P : Nx.Ptree.S with type t = p)
      ~k
      (thetas : p)
      (z : Nx.float64_t)
  : p
  =
  let z = Nx.reshape [| k |] (Nx.contiguous z) in
  P.map
    (fun theta ->
       (* Leaves that cannot carry a direction — a key, an index, a counter —
          get the zero direction, as they do in the sampler. *)
       if Nx_core.Dtype.is_float (Nx.dtype theta)
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
let shift (type p) (module P : Nx.Ptree.S with type t = p) ~lr (params : p) (dw : p) : p =
  P.map2
    (fun p d ->
       if Nx_core.Dtype.is_float (Nx.dtype p)
       then Nx.sub p (Nx.mul_s d (Nx_core.Dtype.of_float (Nx.dtype d) lr))
       else p)
    params
    dw

(* ── the update (Alg. 1, lines 10–12) ────────────────────────────────────── *)

(* z = U (S + γ I)^{-p} Vᵀ C, from the SVD of the sketched GGN, with γ set by
   the damping mode and p by the preconditioner (1, Alg. 1's; or 1/2, which
   caps what a poorly resolved direction can contribute). G̃ is symmetric
   positive semi-definite, so V = U up to signs and an eigendecomposition would
   do; the SVD is what the algorithm prescribes and what is used here. *)
let coordinates
      ?(damping : damping = `Relative_from_top 1e-6)
      ?(preconditioner : preconditioner = `Inverse)
      (ggn : Nx.float64_t)
      (c : Nx.float64_t)
  : Nx.float64_t
  =
  let k = (Nx.shape c).(0) in
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
  Nx.reshape [| k |] (Nx.matmul u (Nx.div rhs (Nx.reshape [| k; 1 |] scale)))

let update
      (type p)
      (module P : Nx.Ptree.S with type t = p)
      ?(lr = 1.0)
      ?damping
      ?preconditioner
      (sk : p Sketch.t)
      (params : p)
  : p
  =
  let dw = sk.apply (coordinates ?damping ?preconditioner sk.ggn sk.c) in
  shift (module P) ~lr params dw

(* ── the eager step ──────────────────────────────────────────────────────── *)

let step
      (type p c d)
      (module P : Nx.Ptree.S with type t = p)
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
  let thetas = directions (module P) ~key:st.key ~k params in
  let sk =
    Sketch.run (module P) ~k ~sketch_sampler:(fun _ _ -> thetas) ~strict loss params
  in
  let params = update (module P) ~lr ?damping ?preconditioner sk params in
  params, next st, sk

(* ── the compiled half ───────────────────────────────────────────────────── *)

(* What a jitted step consumes and produces. Both are parameter trees, so a
   training step is [Rune.jit2 (module In) (module Out) (sketch ~k loss)] and
   a loop threads [in_] to [out] to the next [in_] without ever retracing. *)

(* The trivial auxiliary input: no leaves at all, for a loss that reads the
   parameters and nothing else. *)
module No_aux = struct
  type t = unit

  let map (_ : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) () = ()
  let map2 (_ : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) () () = ()
  let iter (_ : 'a 'b. ('a, 'b) Nx.t -> unit) () = ()
end

module Compiled (P : Nx.Ptree.S) (Aux : Nx.Ptree.S) = struct
  type in_ =
    { params : P.t (* the parameters to sketch at *)
    ; key : Nx.int32_t (* the state's key: the direction stream *)
    ; aux : Aux.t (* whatever else the loss reads; never differentiated *)
    }

  module In = struct
    type t = in_

    let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) i =
      { params = P.map f i.params; key = f i.key; aux = Aux.map f i.aux }

    let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
      { params = P.map2 f a.params b.params
      ; key = f a.key b.key
      ; aux = Aux.map2 f a.aux b.aux
      }

    let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) i =
      P.iter f i.params;
      f i.key;
      Aux.iter f i.aux
  end

  type out =
    { loss : Nx.float64_t (* c, the primal total *)
    ; c : Nx.float64_t (* C = Θᵀ∇c, [k] *)
    ; ggn : Nx.float64_t (* ΘᵀJᵀHJΘ, [k;k] *)
    ; dirs : P.t (* Θ: the directions the sketch was measured along *)
    ; observed_loss : Nx.float64_t (* Σ of the observed little losses *)
    ; observed_c : Nx.float64_t (* Σ of their tangents *)
    }

  module Out = struct
    type t = out

    let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) o =
      { loss = f o.loss
      ; c = f o.c
      ; ggn = f o.ggn
      ; dirs = P.map f o.dirs
      ; observed_loss = f o.observed_loss
      ; observed_c = f o.observed_c
      }

    let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
      { loss = f a.loss b.loss
      ; c = f a.c b.c
      ; ggn = f a.ggn b.ggn
      ; dirs = P.map2 f a.dirs b.dirs
      ; observed_loss = f a.observed_loss b.observed_loss
      ; observed_c = f a.observed_c b.observed_c
      }

    let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) o =
      f o.loss;
      f o.c;
      f o.ggn;
      P.iter f o.dirs;
      f o.observed_loss;
      f o.observed_c
  end

  (* The sketching computation, as a pure function of (params, key, aux).
     Compile it with [Rune.jit2]; run it as it is for an eager step. The
     directions are drawn *inside* it, from the carried key, so one
     compilation serves every iteration, and so is the aux: the loss sees the
     tree's leaves, so a batch that changes between steps is data, not a new
     trace. *)
  let sketch ~k (loss : P.t -> Aux.t -> ('c, 'd) Nx.t) (i : in_) : out =
    let dirs = directions (module P) ~key:i.key ~k i.params in
    let sk =
      Sketch.run
        (module P)
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
  let update ?(lr = 1.0) ?damping ?preconditioner (st : state) (params : P.t) (o : out)
    : P.t * state
    =
    let k = (Nx.shape o.c).(0) in
    let dw =
      apply (module P) ~k o.dirs (coordinates ?damping ?preconditioner o.ggn o.c)
    in
    shift (module P) ~lr params dw, next st

  (* The consistency check, on the numbers the compiled step handed back: the
     same statement as [Sofo.check], which cannot run inside a trace because it
     reads values. *)
  let check (o : out) : (unit, string) result =
    Sketch.check_sums
      ~tol:1e-5
      ~loss:o.loss
      ~c:o.c
      ~observed_loss:o.observed_loss
      ~observed_c:o.observed_c
end
