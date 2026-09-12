# SOFO in OCaml — design plan for an effects-based sketching frontend

**Status:** M1 (batched forward mode, `7d669442`), M2 (the tangent query,
`64534b41`) and M3 (the sofo package) are implemented and committed on the
`batched_jvps` branch — 103 tests across `test/test_jvp_k.ml` and
`test/test_tangent.ml`, 16 in `sofo/test/` — and M4 (composition tests and the
Lorenz example) is implemented: 31 tests in `sofo/test/` and
`sofo/example/lorenz.ml`, on top of two rune fixes it turned up (§14.5). The
optimizer — Algorithm 1's update — now lives in the library too
(`Sofo.Optim`, §14.6, 18 more tests), before M5 rather than as part of it, and
in sofo rather than in vega.
Findings that correct this design are in §13; §14.1, §14.4 and §14.5 record
what M2, M3 and M4 settled. The SOFO *parameter update* itself (Alg. 1, line
10–12: SVD, relative damping, subspace solve) is explicitly **out of scope**
for now; we only design the interface so that a later update module plugs in
cleanly.

Reference: `sofo.pdf` (Yu, Xia, Ma, Lengyel, Hennequin, NeurIPS 2024);
extracted text in `sofo.txt`. Equation/algorithm references below point there.

---

## 1. What SOFO needs from us, mathematically

Training iterates on parameters θ ∈ R^P. Each iteration (Alg. 1):

1. Sample a random sketch Θ ∈ R^{P×K}, columns iid N(0,1) — a fresh random
   K-dimensional subspace of parameter space.
2. Run the model forward under **batched forward-mode AD** seeded with
   tangents Θ, obtaining for every intermediate tensor z a primal z and a
   tangent batch Z ∈ R^{K×shape(z)}.
3. From the outputs y (all of them: every time step, batch element, output
   dim) and their tangent batches Y:
   - **Gradient sketch:** C = Θ^⊤ (∂c/∂θ) ∈ R^K — this is simply the tangent
     batch of the scalar total loss c (line 8: `{c, C} = c({y, Y})`).
   - **GGN sketch:** G̃ = Y^⊤ H Y ∈ R^{K×K} (line 9, Eq. 7), where
     H = ∂²ℓ/∂y² is block-diagonal because the loss is a *sum of little
     losses* ℓ_t, one per (time step, batch element, output) — Eq. 1.
4. (Out of scope) update θ ← θ − η Θ U (S + λ s_max I)^{-1} U^⊤ C.

The crucial structural fact (Alg. 1, note on line 14; §3 discussion): for an
RNN the loss is accumulated over time, so **both C and G̃ can be accumulated
little-loss by little-loss, never storing activations across time**:

```text
c  = Σ_t l_t                    (scalars)
C  = Σ_t ċ_t                    (K-vectors;  ċ_t = tangent batch of l_t)
G̃  = Σ_t Y_t^⊤ H_t Y_t          (K×K blocks; Y_t = tangent batch of y_t,
                                  H_t = Hessian of ℓ_t w.r.t. y_t)
```

with O(1) memory in the time horizon T (paper Fig. 6A, Appendix C).

**Design brief:** give users a way to (a) run their existing compute graph
under batched forward AD, and (b) mark each little loss, such that the
enclosing machinery accumulates C and G̃ automatically — while the *same user
code* also runs under ordinary reverse AD (`Rune.value_and_grad`) or
first-order subspace methods, by just swapping the enclosing handler.

---

## 2. Why algebraic effects

Rune is already an effects library: every Nx op performs an effect
(`Nx_effect.E_add`, …) that transformation handlers (reverse, forward, vmap)
intercept. Three observations drive the design:

1. **User code stays plain OCaml.** The user's RNN is an ordinary function
   over ordinary tensors (and their own typed parameter records via
   `Nx.Ptree.S`). No DSL, no tracing, no monadic plumbing.
2. **A little loss is a semantic marker, and effects are the natural marker.**
   When the user's graph reaches a mini loss, it *performs* an
   `E_observe` effect carrying the prediction tensor y_t, the loss value l_t,
   and a description of the mini loss's curvature. An enclosing SOFO handler
   intercepts it and accumulates the GGN block. Any *other* handler
   (reverse-mode, plain execution, jit) lets it fly past: an unhandled effect
   raises `Effect.Unhandled`, which is catchable (OCaml ≥ 5.2) — the same
   degradation pattern `Rune.scan` already uses. So `observe` is a **no-op
   unless a SOFO handler is installed**: user code is portable across
   optimizers.
3. **The AD machinery can be queried by effect, not by shared state.** The
   SOFO handler needs the tangent batch of y_t. Rather than exposing rune's
   tangent store, it performs a small query effect `E_tangent` that the
   enclosing forward-mode handler answers from its own store. This keeps sofo
   fully decoupled from rune internals and future-proof (a later *compiled*
   forward mode could answer the same protocol).

Rejected alternatives, briefly: threading an explicit accumulator through the
user's loop (invasive, breaks the swap-optimizer story); a monadic loss DSL
(heavy, un-OCaml); callback registration at driver setup (rigid, doesn't
compose with control flow). Effects give all three properties at once.

---

## 3. Architecture: three layers

```text
┌──────────────────────────────────────────────────────────────────────┐
│ L3  sofo (this package)                                              │
│     Curv (curvature descriptions) · observe / mse / ce (E_observe)   │
│     Collector handler (accumulates G̃, per-obs C & c)                │
│     Sofo.sketch driver (samples Θ, composes everything, returns      │
│       {loss; c; ggn; apply})                                         │
├──────────────────────────────────────────────────────────────────────┤
│ L2  rune (small additions)                                           │
│     E_tangent : packed_t -> packed_t option Effect.t                 │
│       ("what is the tangent of this tensor?") — answered by forward  │
│       handlers; Unhandled elsewhere                                  │
├──────────────────────────────────────────────────────────────────────┤
│ L1  rune (the substantive addition)                                  │
│     Forward_k: a *native K-lane* forward-mode handler                │
│       tangents are K-stacked:  tangent(x) : K :: shape(x)            │
│     public API: Rune.jvp_k (+ jvp_k2 / jvp_k_aux / jvp_k')           │
└──────────────────────────────────────────────────────────────────────┘
```

Everything below is specified so it can be implemented and tested
incrementally: L1 is self-contained and independently useful; L2 is tiny; L3
builds on L1+L2 through *public* rune APIs only (no internal exposure).

---

## 4. Layer 1 — native batched forward mode (`Forward_k`, `Rune.jvp_k`)

### 4.1 Semantics

```ocaml
Rune.jvp_k (module P) f params thetas  :  y * dy
```

- `params : P.t` — the user's parameter record (any `Nx.Ptree.S` structure).
- `thetas : P.t` — same structure; each leaf has shape `K :: leaf_shape`
  (the sketch Θ; SOFO samples it iid N(0,1), but any directions are valid).
- Returns `y = f params` (computed **once**) and `dy` with shape
  `K :: shape(y)` — the Jacobian of `f` contracted with all K directions, in
  one pass.
- Degenerate check: `jvp_k` with `K = 1` must agree with `Rune.jvp` up to the
  leading unit axis.

### 4.2 Why not just compose `vmap ∘ jvp` (as in JAX)?

`Rune.vmap' (fun v -> snd (Rune.jvp' f x v)) thetas` already computes the
same mathematical object, and is our **reference implementation for tests**.
But as an execution strategy it is wrong for SOFO:

- The vmap handler sits *outside* the jvp handler, so the jvp handler's
  primal re-performances are batched over K lanes: **the primal computation
  is replicated K times** (compute and memory). For an RNN with S×S recurrent
  matmuls that alone multiplies the dominant cost by K.
- Every op is dispatched through two handlers (double translation).
- The single-tangent store keeps one entry per lane per tensor.

A native K-lane handler computes each primal once and each tangent batch
once: per op, primal FLOPs ×1 + tangent FLOPs ×K — matching the paper's
complexity analysis (Appendix C: O(S²MKT) runtime, O(max(S,M)·S·K) memory)
and its "batched forward-mode AD" implementation.

### 4.3 What changes relative to `forward.ml`

The tangent rules are **formulas are linear in the tangents**, so they are
unchanged mathematically. What changes is shape plumbing:

- The store maps `x` (shape s) → tangent of shape `K :: s`. Inactive
  operands get zeros of shape `K :: s`.
- Every *shape/axis parameter* used in a tangent computation must be
  translated to account for the leading K axis. This is exactly the
  translation problem `vmap.ml` already solves for its batch axis, and the
  same helper logic applies verbatim:
  - `reshape`: target becomes `K :: new_shape` (contiguous first, as vmap
    does);
  - `permute`: axes shift by one (`taxis`: non-negative +1, negative
    unchanged);
  - `expand`, `pad` (prepend `(0,0)` to the config), `shrink` (prepend
    `(0,K)`), `flip`, `sliding_window`, `cat ~axis`, reductions
    (`axes + 1`), `sort`/`argsort`/`gather`/`scatter ~axis`,
    `associative_scan ~axis`, `unfold`/`fold` (pass-through leading dims),
    FFT family (`axes + 1`): all as in vmap;
  - **`matmul` needs no translation**: the backend broadcasts leading batch
    dims (`Nx_backend.matmul` takes `Int.max` over leading dims), so the rule
    `matmul da b + matmul a db` works as-is with `da : K::S1::S2`,
    `b : S2::S3` → `K::S1::S3`. (vmap's matmul case relies on the same
    property.)
  - Elementwise/broadcast rules (add, mul, where-masks, max/min masks, …)
    work as-is: primal-shaped operands broadcast along the leading K axis.
- **Defensive invariant:** on every `set_tangent`, assert
  `shape tangent = K :: shape primal`, raising a descriptive error
  otherwise. This turns miscomposition (e.g. an enclosing vmap
  misinterpreting the K axis, §7) into an immediate, attributable failure
  instead of silent garbage.

Ops excluded initially (raise when active, mirroring existing gaps):
`qr`, `svd`, `eig*` (already unrulable in forward mode), plus
`cholesky`/`triangular_solve` **pending a check that the backend supports
leading batch dims for them** — none of the SOFO target models (RNN/MLP)
need these.

`E_custom_jvp` under K lanes: the user's custom rule takes single tangents;
lift it by mapping the rule over the K axis with the vmap machinery (one
batched translation of the rule body), then stack. Same treatment as the
generic `Curv.hvp` below.

### 4.4 Sharing the rule table (implementation strategy)

`forward.ml` and `forward_k.ml` should not fork ~600 lines of rules that must
then evolve in sync (every new Nx op needs a rule in both). Target design:
parameterize the rule table over a small **tangent-op module** `Tan`:

- `Tan` elementwise ops = plain `Nx` ops in both instances (no translation
  needed — broadcasting covers them);
- `Tan.reshape / sum / pad / cat / gather / …` = identity in the
  single-tangent instance, axis-translated in the K-lane instance (helpers
  shared with/extracted from `vmap.ml`).

Instantiate `Forward` (today's single-tangent handler) and `Forward_k` from
the same functor. A first cut may duplicate forward.ml into forward_k.ml to
deliver quickly, but the tests (§9) pin both to the same reference, and the
functorization should land before the rule table drifts.

### 4.5 Memory: the store must not retain dead steps  ← **the sneaky bit**

`Tensor_map` is a `Hashtbl` keyed by physical tensor identity with **strong
keys**. Under forward mode every active intermediate gets an entry, so for an
RNN unrolled over T steps the store keeps every step's tensors (and their
K-stacked tangents — K× worse) alive: memory grows **O(T)**, destroying
SOFO's defining property (Fig. 6A) *and* the existing single-tangent `jvp`
has the same latent issue (just 1× instead of K×).

Required fix for `Forward_k` (recommended for `Forward` too, as a separate
cleanup): an **ephemeron-keyed** store (`Ephemeron.K1.Make` over the
physical-identity key module). When the user's loop moves past step t and the
step-t primals become unreachable, the entries (and their tangents) become
collectable → O(max live step) memory. Notes:

- keep the existing `stable`/`unwrap` forcing before keying;
- the carried state z_t and its tangent stay live by design (referenced by
  the next step) — that is the true working set;
- `vmap`'s `Ids` marking table has the same strong-key retention for
  long-horizon loops — known rune issue, fix in tandem or recommend the
  explicit-batch-dims style for long horizons (§7);
- expose store statistics (live entry count) so the O(1)-in-T property is
  *testable* (§9).

### 4.6 Layer-1 tests

- `jvp_k` ≡ `vmap ∘ jvp` reference on random functions/structures/dtypes
  (float tolerance), including nested records and mixed dtypes.
- `K = 1` ≡ `jvp` (leading-axis convention).
- Shape-invariant errors on malformed tangent structures.
- Composition: `jvp_k` of `Rune.grad` = K Hessian-vector products in one
  pass (bonus application: batched `hvp`, useful well beyond SOFO).
- Store entry count bounded as T grows (with GC pressure), for a loop model.

---

## 5. Layer 2 — the `E_tangent` query effect

```ocaml
(* rune; small and generally useful *)
type packed_t = Packed : ('a,'b) Nx.t -> packed_t   (* packing as in scan.ml *)
type _ Effect.t +=
  | E_tangent : packed_t -> packed_t option Effect.t
```

"Innermost forward-mode handler: what is the tangent of this tensor?"

- `Forward.handler` answers with the single tangent (shape s) or None;
  `Forward_k.handler` answers with the K-stack (shape K::s) or None. Both
  stay silent when `Gate` disables tracing (consistent with all rules).
- Unhandled anywhere else — callers treat `None`/`Unhandled` as "inactive /
  no forward mode installed".

Uses: sofo's collector (below); debugging forward mode; future effect-based
libraries; a future *compiled* forward mode can answer the same protocol,
making sofo work unchanged on top of it.

---

## 6. Layer 3 — the sofo package

### 6.1 Curvature descriptions (`Curv`)

The GGN block at an observation needs H_t, the Hessian of the mini loss
ℓ_t w.r.t. its argument y_t, **as a linear map** (we only ever use it via
`Y_t^⊤ H_t Y_t`). H is block-diagonal over (batch element, output index), so
the map is cheap and structured for all standard losses:

```ocaml
module Curv : sig
  type t
  val scale     : float -> t                       (* H = s·I       (e.g. MSE)  *)
  val diag      : ('a,'b) Nx.t -> t                (* H = diag(w)  (weighted)   *)
  val softmax_ce: ?scale:float -> ('a,'b) Nx.t -> t(* p = softmax(y);
                                                      H v = p⊙v − p (p·v) per row *)
  val hvp       : (('a,'b) Nx.t -> ('a,'b) Nx.t) -> t  (* generic escape hatch *)
end
```

- Structured cases are applied to the whole K-stack natively (broadcasting
  over the leading K axis — e.g. `diag w`: `HY = w·Y`).
- Generic `hvp` is written for a **single** tangent and is *lifted over the
  K axis with vmap* by the library (one batched translation of the closure;
  same cost as a broadcastable implementation). Contract: the closure is
  linear in its argument (it is a Hessian action) and side-effect free.
- New structured cases worth prepacking because exponential-family NLLs are
  all diagonal-ish: Poisson (H = diag(μ) with μ = exp(y) for log-link),
  Bernoulli/logistic (diag(p(1−p))), etc. Start with `scale`, `diag`,
  `softmax_ce`, `hvp`; add others on demand.
- Every constructor is validated in tests against `Rune.hessian'` of the
  corresponding mini loss as a function of a free y (`Rune.hessian' (fun y ->
  mse y τ)`) — no manual derivations anywhere.

### 6.2 Observing a little loss (`E_observe`)

```ocaml
(* payload (packed existentials as in scan.ml) *)
type obs = {
  y    : packed_t;    (* the prediction tensor entering the mini loss      *)
  l    : packed_t;    (* the scalar mini-loss value  l_t = ℓ_t(y_t)        *)
  curv : Curv.t;      (* H_t as a linear map on shape(y)                   *)
}
type _ Effect.t += E_observe : obs -> unit Effect.t
```

User-facing surface (draft):

```ocaml
(* Generic: user computed l themselves; registers curvature. No-op (returns
   ()) when no SOFO handler runs — incl. under grad, vmap, jit, plain run. *)
val observe : y:('a,'b) Nx.t -> curv:Curv.t -> ('a,'b) Nx.t -> unit

(* Packaged little losses: compute the value with ordinary Nx ops (so its
   tangent ċ_t comes from the standard forward rules), observe with the
   matching curvature, and return the loss tensor for the user to accumulate.
   Scaling of H is derived from the reduction used, so l and curv are
   consistent by construction. *)
val mse        : ?w:('a,'b) Nx.t -> ('a,'b) Nx.t -> ('a,'b) Nx.t -> ('a,'b) Nx.t
val softmax_ce : ('a,'b) Nx.t -> int Nx.t -> ('a,'b) Nx.t
```

Contracts:

- `y` is the tensor whose tangent batch Y_t the collector needs; it must be
  the very tensor flowing into the mini loss (identity-keyed lookup). If its
  tangent is absent (y constant w.r.t. θ), the block is zero and the
  observation is skipped.
- `l` must be a scalar computed from `y` by differentiable Nx ops — then
  ċ_t (the K-vector tangent of l_t) is available in the store for free. The
  collector reads it for a cross-check (below).
- Mini losses over *several* tensors (e.g. output + activity regularizer):
  either make separate observations per tensor (blocks add — correct when the
  mini loss is a sum), or observe a `cat` of the arguments (its tangent is in
  the store) with a joint `hvp`. Both work; document both.
- Convexity of ℓ_t is a *modeling* assumption of the paper; the mechanical
  contract is only "curv is the Hessian action of ℓ_t at y_t" (non-convex
  pieces just contribute their true Hessian block).

Degradation: `observe` performs `E_observe` and catches `Effect.Unhandled`
(→ `()`). Under `Rune.value_and_grad`, `vmap`, `jit`, or plain execution the
marker is inert; user code is identical across optimizers. Precedent:
`Scan.scan`'s own Unhandled fallback.

### 6.3 The collector handler

A deep handler installed **inside** the forward-K scope, around the user's
loss function (how it gets there: §6.4). It matches only `E_observe` (all Nx
op effects pass through to the forward handler). Per observation:

1. `Y = perform (E_tangent y)`; if None → skip (inactive y). Defensive
   `Unhandled` catch → same.
2. `HY = Curv.apply curv Y` (structured: one broadcast op; generic: vmap
   lift).
3. Block: reshape Y, HY to `[K; n]` (n = numel y_t; contiguous by store
   invariant) and `block = Y_r @ HY_r^⊤` — a K×K Gram contraction. For
   `diag`/`scale` curv this specializes to a weighted Gram of Y_r (one op).
4. Accumulate `G += block` (float64 accumulation by default; see numerics),
   `c_obs += l`, `C_obs += ċ_t` (via `E_tangent l`).

At finalize (driver): symmetrize `G ← (G + G^⊤)/2` (H symmetric ⇒ blocks
symmetric; kills float asymmetry before any later eigendecomposition) and
optionally cross-check `c_obs` vs the returned total loss and `C_obs` vs the
returned tangent — a missing observation (a loss term the user accumulated
but forgot to mark) shows up as a mismatch: **warn or error in strict mode**.
This is the debugging story for "did I observe everything I accumulated?".

Collector cost per observation: O(K·n + K²·n) — one extra elementwise pass
over y_t; negligible against the K-batched step matmuls.

Note on handler-callback contexts: while the collector's callback runs, it is
uninstalled but the forward-K handler (outside it) remains installed — so
`E_tangent` is answered, and the collector's own tensor ops flow to forward-K
with *inactive* operands (tangents are not primals-with-tangents), i.e. they
execute eagerly with zero bookkeeping. No gating needed.

### 6.4 The driver: `Sofo.sketch`

```ocaml
type 'p sketch = {
  k    : int;
  loss : ('a,'b) Nx.t;        (* primal total loss (user's return value)    *)
  c    : Nx.float64_t;        (* C = Θ^⊤ ∇c, shape [k]                      *)
  ggn  : Nx.float64_t;        (* G̃, shape [k;k], symmetrized               *)
  apply: Nx.float64_t -> 'p;  (* z:[k] ↦ Θ z  (sketch→parameter direction)  *)
}

val sketch :
  (module Ptree.S with type t = 'p) ->
  k:int ->
  ?sketch_sampler:(int -> 'p -> 'p) ->     (* default: per-leaf randn (K::leaf),
                                              ambient Nx.Rng scope *)
  ('p -> ('a,'b) Nx.t) ->                  (* user's total-loss function     *)
  'p ->                                    (* parameters                     *)
  'p sketch
```

Composition trick — **sofo needs no rune internals**: the driver wraps the
user's function with the collector and passes *that* to `Rune.jvp_k`:

```ocaml
Rune.jvp_k (module P) (fun p ->
    Effect.Deep.match_with (fun () -> loss p) () (Collector.handler acc))
  params thetas
```

`jvp_k` installs forward-K around the wrapped function, giving exactly:

```text
user code  ⊂  collector  ⊂  forward-K  ⊂  (driver / outside world)
```

- user Nx ops → collector passes → forward-K tracks (primal once, K-tangent);
- user `E_observe` → collector handles (forward-K passes it through — it is
  not an Nx effect), queries `E_tangent` (answered: forward-K is outside),
  accumulates;
- `(y, dy)` comes back from `jvp_k` as usual; `C = reshape [k] dy` — **the
  gradient sketch is just the jvp_k tangent of the user's returned total
  loss**, no collector involvement needed (correct by construction; the
  collector's `C_obs` is only a cross-check);
- `G̃` from the collector; `apply z` closes over the sampled Θ leaves
  (`P.map (fun θ_leaf -> reshape leaf_shape (tensordot z θ_leaf ...))`).

Θ sampling: default per-leaf `randn (K :: leaf_shape)` under the ambient RNG
scope (caller wraps the training loop in `Nx.Rng.with_key` for
reproducibility — the nx convention); `~sketch_sampler` for structured
sketches (Rademacher, per-leaf scaling, …). Note the paper's RTRL remark:
Θ = full basis reproduces RTRL — not a target (memory), just a cute
sanity-check thought experiment at small P.

### 6.5 The three modes — the handler-swap payoff

Same user code, three drivers, mirroring the paper's baselines:

| mode | driver | observations do | returns |
| --- | --- | --- | --- |
| SOFO (2nd-order sketch) | `Sofo.sketch ~k` | accumulate G̃ (+checks) | {loss; C; G̃; apply} |
| First-order subspace (FGD / 1st-order SOFO) | `Rune.jvp_k` directly | no-op (Unhandled) | {loss; C} — update ∝ ΘΘ^⊤∇c |
| Exact gradient (Adam-style) | `Rune.value_and_grad` | no-op | {loss; ∇c} |

Plus mode degenerations: `Sofo.sketch` with all-curv-zero observations ≡ FGD;
`K = 1` FGD ≡ forward-gradient descent (paper's FGD baseline). This is the
"swap the enclosing handler, not the model code" property, realized with zero
extra machinery.

---

## 7. Composition matrix (worked out now, so implementation doesn't rediscover it)

| composition | status | notes |
| --- | --- | --- |
| `jvp_k` ∘ plain loop / `Rune.scan` | ✓ core case | scan folds eagerly under the handler (as `Scan` + `Forward` do today); every step's ops tracked, per-step observations fire, old steps become garbage (with the ephemeron store) → **O(1) memory in T** |
| `jvp_k` ∘ `vmap` (vmap innermost) | ✓ | The natural RNN style: write per-trial code, `vmap` over trials, reduce inside. vmap consumes user ops and re-performs them batched; forward-K (outside) tracks the *batched* ops → primals M-batched, tangents K::M-batched. Observations see the physically batched y (M inside), contraction sums over M — correct, one K×K block per observation. User's post-vmap `mean` is tracked → scalar total, C shape [K]. |
| `vmap` ∘ `jvp_k` (vmap outermost) | ✗ must fail loudly | vmap would treat K-stacked tangents as unbatched constants and prepend its own M axis (wrong order + K× duplication). The `set_tangent` shape invariant (§4.3) turns this into an immediate descriptive error. Document: batch dims belong *inside* the tangent axis. |
| `grad` ∘ `jvp_k` (reverse over forward) | ✓ | forward-K's tangent ops are re-performed in the enclosing reverse scope and taped (same principle as today's `jvp` under `grad`). |
| `jvp_k` ∘ `grad` (forward over reverse) | ✓ bonus | = K Hessian-vector products in one pass — batched `hvp`. |
| `jvp_k` ∘ `jvp` / `jvp` ∘ `jvp_k` | ⊙ | should work like today's `jvp∘jvp` (full 2nd-order forward); test later, not a milestone |
| `no_grad`/`detach` inside | ✓ | gate honored (E_tangent also gated — collector treats as inactive; defensive catch) |
| `Rune.jit` inside user code | ⊙ degrades | `jit` sees `Gate.transforming ()` and runs eagerly (`f params`) — correct but uncompiled. Perf note below; staged/compiled forward mode is future work. |
| RNG in the graph (stochastic RNNs) | ✓ | fresh draws are constants w.r.t. θ (zero tangents) — the motor-task multiplicative noise works; for vmap-style batching use lane-decorrelated keys (`E_axis_index` + `fold_in_axis`, already in vmap) or pass noise in as data |
| `custom_jvp` / `custom_vjp` | ✓ / raises | custom_jvp lifted over K via vmap (§4.3); custom_vjp raises under forward mode as today |
| `pmap` over devices | future | K lanes are embarrassingly parallel — a natural later extension |

Also inherited for free: `where`-based masking, `cond`/`while_loop` on host
values (control flow independent of θ differentiates correctly; piecewise
semantics as in all forward mode), complex dtypes through the existing rules.

Measured corrections to this matrix — a compiled forward mode stages correctly
but unrolls scans, and matmul needs a lane-alignment lift — are in §13.5.

---

## 8. Performance & numerics

- **Per-step cost:** primal ops ×1 + tangent ops ×K (batched GEMMs for the
  RNN recurrence). ≈ half the compute of `vmap∘jvp` (which replicates
  primals K×) and one handler dispatch per op instead of two. Paper target:
  1.5–3× Adam wallclock at K ≈ 100 — eager OCaml dispatch was enough there.
- **Eager dispatch overhead:** every op goes OCaml → effect → handler →
  backend. Fine for the first version (the paper's own implementation was
  eager). Future: a *compiled* forward mode (jit-staged tangent ops, K as an
  ordinary batch dim; scan staged as a loop) — the `E_tangent` protocol is
  designed so sofo would not change. Note `jit` currently *unrolls* `scan`
  (README limitation), so staging interacts with that too.
- **Numerics:** contract blocks in leaf dtype (float32 typical); accumulate
  G̃ (and C) in **float64** (K×K and K — negligible memory) to protect the
  later SVD/solve; symmetrize at finalize. G̃ is symmetric PSD (H ⪰ 0 for
  convex mini losses) — a later eigendecomposition (`eigh`) suffices where
  the paper says SVD; flag float32-only mode as an option.
- **Memory:** O(max live step) with the ephemeron store; dominant tensor =
  tangent of the recurrent weight matmul input (S×(M·K)) per paper App. C.

---

## 9. Validation plan (all references computable with *existing* rune)

Status: (1) is in M1's `test_jvp_k.ml` (per-op sweeps against the `vmap∘jvp`
reference), and (7)'s live-entry probe with it — but not its weak-pointer
canary on step-*t* tensors. (2)–(6) and (8) are in M3's `sofo/test/`: every
`Curv` constructor against `Rune.hessian'`, the collector against an explicit
Σ YᵀHY, the end-to-end quadratic model, an exact ΘᵀJᵀHJΘ, C against
`value_and_grad`, and the consistency checker. (9) is M4's
`sofo/test/test_compose.ml` (15 tests: the §7 matrix, plus a jitted sketch and
the memory probe), and the example carries the smoke numbers — §14.5.

1. **`jvp_k` vs `vmap∘jvp`** (L1): random graphs, structures, dtypes; also
   K=1 vs `jvp`.
2. **`Curv` vs `Rune.hessian'`** (L3): each constructor checked against the
   Hessian of the corresponding mini loss as a function of a free y.
3. **Collector vs explicit Gram** (L3): build Y_t per observation via the
   `vmap∘jvp` reference, assemble `G̃_ref = Σ Y_t^⊤ H_t Y_t` manually in the
   test, compare.
4. **End-to-end quadratic model** (the strongest check): for random z ∈ R^K
   and small ε,
   `c(θ + εΘz) ≈ c(θ) + ε ⟨C, z⟩ + (ε²/2) z^⊤ G̃ z` — validates C, G̃, and
   the user's curvature claims jointly, with no reference implementation.
5. **Exact GGN on tiny models**: G = J^⊤HJ with J from `jacfwd'`/`jacrev'`
   of params→(concatenated outputs), H block-diagonal from (2); then
   `G̃ = Θ^⊤GΘ` vs `Sofo.sketch`.
6. **C vs exact**: `C = Θ^⊤ ∇c` against `Rune.value_and_grad` contraction.
7. **Memory-in-T**: store entry count (exposed stat) stays bounded while T
   grows, with GC pressure; a canary weak-pointer test on step-t tensors.
8. **Consistency checker**: user accumulates an unobserved term → strict mode
   raises; observed-but-unaccumulated → mismatch reported.
9. **Composition**: the matrix of §7 has a test per ✓/✓-bonus row, and the
   ✗ row produces the descriptive shape-invariant error.

---

## 10. What we are deliberately not doing (now)

- **The update rule** (Alg. 1 l.10–12: `eigh(G̃)`, γ = λ·s_max,
  `θ ← θ − η Θ U(S+γI)^{-1}U^⊤C` via `apply`) — phase 2, on top of the
  `sketch`/`apply` contract; likely a vega-flavored optimizer later.
- Levenberg–Marquardt-style adaptive damping, line searches.
- Compiled/staged forward mode (jit interplay); `pmap` over K.
- Parameter-space regularizers observed directly into G̃ (the paper absorbs
  them in damping; a `curv-in-θ-space` observation type is a possible later
  extension — the payload shape would be over the P.t tangent structure).
- Complex-dtype GGN; RTRL full-basis mode.

---

## 11. Milestones

- **M1 — rune (done, `7d669442`):** `Forward_k`, ephemeron-keyed
  `Tangent_store`, `Rune.jvp_k` / `jvp_k2` / `jvp_k_aux` / `jvp_k'` /
  `live_tangent_entries`, 94 tests. The `Tan` functorization was not done:
  `forward_k.ml` is a deliberate standalone copy of the rule table with a drift
  warning in both headers (§4.4 allowed this for a first cut). Independently
  useful already — batched `hvp`, fast `jacfwd`, and forward mode that compiles
  under `jit` (§13.5).
- **M2 — rune (done, `64534b41`):** `Rune.tangent`, the tangent query of §5,
  answered by both forward handlers — 9 tests in `test/test_tangent.ml`. Small
  as expected: the store lookup already existed, so M2 is the effect, the
  wrapper, two handler cases and the suite. §14.1 records the three decisions
  it settled.
- **M3 — sofo (done):** `sofo/lib/{curv,observe,collector,sketch,sofo}.ml`
  with `sofo.mli`, and 16 tests in `sofo/test/` (4 curvature, 12 end-to-end).
  §14.2 was the plan; §14.4 records where it changed.
- **M4 — sofo (done):** composition tests §9.9
  (`sofo/test/test_compose.ml`, 15 tests) and the example
  (`sofo/example/lorenz.ml`): one loss function driven by `value_and_grad`,
  `jvp_k` alone and `Sofo.sketch`, with wall-clock and memory smoke numbers
  against a per-lane `vmap∘jvp` Σ YᵀHY reference, and honest jit numbers. Two
  rune fixes were needed (§14.5), and a jitted sketch is pinned by a test.
  §14.5 also records the student-teacher `sofo/example/linear.ml`, which
  exercises the Alg. 1 update outside the library as its bootstrap.
- **M4.5 — the optimizer (done):** `Sofo.Optim` — Algorithm 1's damped solve
  and parameter step, the vega-shaped state it is driven from, the jittable
  sketching half (`Compiled`), and 18 tests (`sofo/test/test_optim.ml`). §14.6
  records the shape and why it is where it is.
- **M5 (phase 2):** the training loop on `Sofo.Optim` — the Lorenz example
  currently steps first-order only — and a damping/schedule policy; revisit
  compiled forward mode (staged forward scan, §13.5, and the collector-state
  question of §14.3). Vega integration is no longer planned (see §10).

Build wiring: `sofo` already sits in the shared dune workspace and declares
deps on `rune`/`nx`; `sofo.txt` is the paper text for reference.

---

## 12. Open questions

1. ~~Ephemeron store mechanics~~ — **answered (M1, §13.4):** the store works
   and needs no cleaning policy (the stdlib ephemeron table drops dead keys
   when it resizes). Still open but out of sofo's path: retrofitting `Forward`
   and `vmap`'s strong-keyed `Ids` (the latter matters only for long-horizon
   *unrolled* per-trial loops, which sofo avoids by keying through `jvp_k`).
2. ~~Batched linalg~~ — **answered (M1, §13.2):** the backend requires exactly
   matching leading batch dimensions, so `cholesky`/`triangular_solve` are
   excluded from `Forward_k` v1 and raise when an input is active.
3. ~~`hvp` lifting~~ — **narrowed (M1):** the mechanism already exists in
   `Forward_k`'s `custom_jvp` case — lift a single-tangent closure over the lane
   axis by marking the K-stacked operands in a `Vmap` state of size `k` and
   running it under `Vmap.handler`. `Curv.hvp` needs the same thing, so M3
   should factor it into one internal helper instead of writing a second copy.
   Whether the generic constructor stays vmap-based (safe) or gains a
   documented K-broadcastable fast path is still open.
4. **Naming** — `jvp_k` / `jvp_k2` / `jvp_k_aux` / `jvp_k'` are settled (M1).
   Still open: `observe` vs `emit` vs `tag_loss`, `Curv` vs `Curvature`.
5. ~~Strictness default~~ — **settled (M3):** the library never prints.
   `Sofo.check` returns `(unit, string) result`, `Sofo.sketch ?strict` raises on
   a mismatch, and the observed sums ride along in `sketch.diagnostics` so a
   caller can report them however it likes.
6. ~~float64 vs float32 accumulation~~ — **settled (M3):** G̃ and C accumulate
   in float64 (every block is cast on entry) and G̃ is symmetrized at finalize,
   as §8 argued.
7. Whether `E_observe`'s payload should also carry a *label* (for logging,
   per-term diagnostics — cheap to add, useful for debugging complex losses).
   Still open: M3 shipped without one.
8. ~~Collector lane validation~~ — **settled (M3):** the collector checks
   `shape Y = k :: shape y` and raises a message naming `Rune.jvp` against
   `Rune.jvp_k`. Through sofo's own driver it is unreachable (§14.4 — the
   driver installs `jvp_k`, and a deeper forward mode is suspended while the
   collector's callback runs), so it guards future drivers rather than user
   error.
9. ~~New (M1): under `jit` the collector's accumulator is a graph-level value,
   so an unrolled horizon grows the trace.~~ **Settled (M4, §14.5):** a jitted
   sketch is correct and pinned by a test, `~strict` raises `Jit_error` inside
   the trace, and the unrolling is reported rather than hidden — the collector
   claims each scan, and compile time grows with the horizon where a plain
   jitted rollout stages it as a loop. Staging the forward scan remains M5's
   question, and it is the collector-state change §14.3 describes.

---

## 13. Phase-1 implementation notes (as built)

M1 shipped as `raven/packages/rune/lib/forward_k.ml` (the batched handler),
`tangent_store.ml` (ephemeron-keyed K-lane store), the public `Rune.jvp_k` /
`jvp_k2` / `jvp_k_aux` / `jvp_k'` plus `live_tangent_entries`, and
`test/test_jvp_k.ml` (88 tests: per-op sweeps against the `vmap ∘ jvp`
reference, K = 1 ≡ `jvp`, lane-structure errors, batched `hvp`, composition,
memory, jit). §4.3's list of translations held, with the corrections below.

### 13.1 matmul *does* need translation (§4.3 corrected)

The backend broadcasts leading batch dimensions *positionally*, so a lane axis
prepended to a tangent lands against the other operand's own leading dimensions
whenever either operand has them: `matmul da b` with `da : k :: [2;3]` and
`b : [2;3;2]` pairs `k` with `b`'s first batch dimension. Only when *both*
operands are plain matrices (`[S;M] × [M;K]`, the hot recurrent path) does the
rule work unchanged. In general the rule views every active operand at the
output's leading layout `k :: lead`: a primal gains a broadcast lane axis and a
tangent grows its own leading dims up to `lead`. Both are no-ops for plain
matrices.

### 13.2 `cholesky` / `triangular_solve` stay excluded (open question 2 answered)

The C backend's batched Cholesky and triangular solve require *exactly matching*
leading batch dimensions (shape equality, no 1-broadcasting), and
`triangular_solve` additionally wants `rank a = rank b`. Including them means
lifting the primal factorizations along the lane axis the way §13.1 lifts matmul
operands; deferred, as the plan allowed.

### 13.3 The `vmap ∘ jvp` reference cannot express mixed-rank matmul

`vmap`'s matmul case passes the operation through untranslated, so the reference
composition hits exactly the alignment problem §13.1 solves — a batched operand
against a constant operand with leading dims of its own. Those two test cases
therefore use a per-lane reference (one single-tangent `jvp` per lane, stacked).
Fixing `vmap`'s matmul the same way is a separate cleanup.

### 13.4 The ephemeron store (§4.5) works and needs no cleaning policy

`Tangent_store` keys bindings on ephemerons over the physical-identity key
module; `live_tangent_entries` reads `stats_alive`'s live-binding count from
inside the differentiated function. A 400-step unrolled recurrence with forced
major collections holds tens of live bindings, not hundreds (test-pinned). The
stdlib ephemeron table already drops dead keys when it resizes, so the store
needs no cleanup of its own. `vmap`'s strong-keyed `Ids` table is untouched
(still the known long-horizon issue for `vmap` itself).

### 13.5 JIT: forward mode stages through the tracer — and unrolls scans

Measured on a toy 8-dimensional `scan` recurrence (CPU device, `JITCACHE=0`);
first call = trace + compile, replay ≈ 1 ms, and every variant agrees with eager
execution:

| horizon | `jit(scan)` | `jit(jvp(scan))` | `jit(jvp_k(scan))` |
| --- | --- | --- | --- |
| 20 | 0.014 s | 0.160 s | 0.183 s |
| 80 | 0.005 s | 0.783 s | 0.885 s |
| 320 | 0.010 s | 6.255 s | 7.256 s |

- **Compiled forward mode works mechanically today.** With `jit` outermost,
  `Forward_k`'s primal re-performances and its tangent operations are traced
  into the same program, exactly as reverse mode's taped operations are, so a
  compiled `jvp_k` replays on device and matches eager results (test-pinned).
  §7's "`jit` inside user code" row is about the other nesting, which still runs
  eagerly.
- **Under forward mode a `scan` unrolls into the trace.** `Forward`/`Forward_k`
  claim `E_scan` and answer the probe `false` — they must, since they track
  every step eagerly — so a staging `jit` never sees the scan and compile time
  grows with the horizon (table), where plain `jit` stages the same recurrence
  as one loop with flat compile time. Reverse mode avoids this with the probe
  dance (§7): it declines the claim and records a staged transpose
  (`E_scan_bwd`) that only a staging jit answers. The forward analogue is a
  staged *forward* scan — trace the body once under the forward handler (primal
  and K-lane tangents together, K as an ordinary batch dim) and emit a single
  loop — i.e. the compiled forward mode of §8/§10. Until it exists, a
  forward-mode recurrence under `jit` compiles per step.

### 13.6 GPU memory: the ephemeron store is the *eager* mechanism

The store's reclamation is OCaml GC over host tensors, so it governs eager
execution only — which is also what `jit` falls back to when it sees a
transformation. In a compiled program the equivalent mechanism is Tolk's memory
planner (`raven/packages/tolk/lib/engine/schedule.ml`): internal buffers are
rewritten into per-device arenas by first-use/last-use liveness, so a buffer
whose last consumer is step *t* is released to the arena and reused by step
*t+1* — the runtime peak is the live working set, not the trace length — and a
staged loop reuses its body's buffers per iteration by construction. So the
O(max live step) property does survive on GPU, through the compiler rather than
through ephemerons; what an unrolled trace costs is host-side trace and schedule
memory plus compile time (§13.5), not device memory. Two caveats: an unrolled
trace's *stacked* per-step outputs (the scan's `ys`) are genuinely O(T) unless
the computation accumulates into the carry — which the §6.3 collector does
anyway — and the §7 row for `jit ∘ jvp_k` should now read "supported, but
unrolls" rather than "degrades".

---

## 14. Phase 2: what M2 and M3 settled

### 14.1 M2 — `Rune.tangent` (done, `64534b41`)

Built as planned in outline, with three decisions settled by the code:

- **The directly typed payload compiled**, so nothing is packed:
  `E_tangent : ('a,'b) Nx_effect.t -> ('a,'b) Nx_effect.t option Effect.t`.
  `Nx_effect.t` is an injective GADT, so the payload's parameters are deducible
  from the result type — the constraint that forced `Nx_effect` to box its own
  `E_to_host` does not apply here. The effect and its payload stay private to
  rune; the only public form is the function.
- **The innermost handler claims the query unconditionally**, answering `None`
  for a tensor its store does not track rather than handing it outward. That is
  what makes the answer's *convention* unambiguous: in a nested scope the inner
  mode answers in its own shape — including for the tensor it seeds, where the
  enclosing mode holds a `k`-batch — and a tensor only the enclosing mode
  tracks is a constant of the inner one. The nesting test pins both halves.
- **The shape is the contract**: `Some dy` with `shape dy = k :: shape y` under
  `jvp_k`, `shape y` under `jvp`. `tangent_query.ml` states it, and §14.2 leans
  on it: the collector checks the leading axis rather than asking which
  transformation is installed, and treats any other shape as an error.

The suite (`test/test_tangent.ml`, 9 tests) covers both conventions, the
degradations (no forward mode, `no_grad`, a constant intermediate, a detached
tensor), nesting, that reads do not perturb a run, and reads inside a `scan`
body — one answer per step, each carrying the lane axis.

### 14.2 M3 — the sofo package (plan; outcome in §14.4)

§6 is the design; these are the refinements M1 turned up. Everything else stands
as written there.

- **The collector reads tangents through `Rune.tangent`** (M2) from inside the
  forward-K scope, and must check the lane shape before using it:
  `Some dy` with `shape dy = k :: shape y`, where `k` is the collector's own
  lane count. A tangent of shape `shape y` means a *single*-tangent forward mode
  is installed (`jvp` rather than `jvp_k`) and the collector should fail loudly
  with a message that says so, rather than contracting mismatched shapes.
  `None` means the observation is skipped — `y` is constant with respect to θ,
  so its GGN block is zero, which is the intended degradation.
- **`Curv.hvp` and `Forward_k`'s `custom_jvp` lift are the same mechanism**
  (mark the K-stacked operands in a `Vmap` state of size `k`, run the
  single-tangent closure under `Vmap.handler`, read the result as `k :: shape`).
  Factor it into one internal helper when M3 lands instead of copying it.
- **`apply`** contracts `z : [k]` against each leaf's lane axis. A formulation
  that needs no new primitive: reshape `z` to `k :: 1 ... 1`, multiply by the
  leaf's `k`-stacked `θ`, and reduce over axis 0 — the result has the leaf's
  shape. Test it against the basis: for `z = e_i`, `apply z` must reproduce the
  `i`-th lane of Θ exactly.
- **Sampling.** Default Θ is `randn (k :: leaf_shape)` per leaf under the
  ambient `Nx.Rng` scope (§6.4); pin reproducibility in a test by wrapping
  `Sofo.sketch` in `Nx.Rng.with_key`, and keep `~sketch_sampler` for structured
  sketches.
- **Numerics and diagnostics** are unchanged from §6.3/§8: accumulate G̃ and C
  in float64, symmetrize at finalize, cross-check the observed `c`/`C` against
  the returned total loss and its tangent. §12.5 settles the strictness of that
  check, §12.7 whether the payload carries a label.
- **Acceptance tests** are §9.2–§9.6 and §9.8: every `Curv` constructor against
  `Rune.hessian'` of its mini loss (no hand-derived formulas anywhere), the
  collector against an explicit `Σ Yᵀ H Y` assembled with the `vmap ∘ jvp`
  reference, the end-to-end quadratic expansion
  `c(θ + εΘz) ≈ c + ε⟨C,z⟩ + (ε²/2) zᵀG̃z` (which validates `C`, G̃ and the
  curvature claims jointly, with no reference implementation), G̃ against an
  exact `JᵀHJ` on a toy model, `C` against `Rune.value_and_grad` contracted with
  Θ, and the consistency checker (a loss term accumulated but never observed →
  strict mode raises).

### 14.3 What a jitted sketch can and cannot do today

M1 measured the interaction (§13.5): with `jit` outermost the batched tangent
operations are traced into the compiled program, so `jvp_k` compiles and — by
the same mechanism — the collector's `E_observe` and its `E_tangent` queries
would be handled at trace time as usual. Two limits follow:

- the accumulator is a *host* value during tracing, so every accumulation is
  another traced operation: a horizon of `T` observations becomes an `O(T)`
  graph rather than a loop; and
- a `scan` inside the forward scope unrolls for the reason §13.5 gives.

So a jitted `Sofo.sketch` should be correct today — the mechanism is
M1-verified (§13.5) and `E_observe` is not an Nx effect, so the tracer falls
past it while the collector's own operations are traced — but M4 should pin
that with a test before relying on it — and its compile time and trace memory
grow with `T`. Staging it needs both halves of §13.5 *plus* one design change: a
collector whose state is a loop carry rather than handler state, i.e. `observe`
would have to hand its accumulator to the staged loop instead of keeping it in
the effect handler. That is a change to §6.3, not a detail, and it is the
substantive thing a compiled forward mode has to solve. Recorded here so M4's
example reports unrolled numbers as unrolled.

### 14.4 M3 as built: four things §6 got wrong

M3 shipped as planned in outline — `Curv`, `observe`/`mse`/`softmax_ce`, the
collector, `Sofo.sketch` — with these corrections, worth reading before M5
builds an update rule on top.

- **`Curv.t` is a variant, and `apply` is a function rather than a record
  field.** §6.1 sketched `t` as `{ apply : 'a 'b. ('a,'b) t -> ('a,'b) t }`. A
  polymorphic record field cannot be built from a closure that mentions the
  prediction's concrete dtype: the equality refinement the `hvp` case needs
  escapes the branch, and generalization fails (`This field value has type …
  which is less general than …`). So `t` is
  `Scale | Diag | Softmax_ce of float * packed | Hvp of closure * dtype`, and
  the dtype adaptation every case needs — a loss can be evaluated in a wider
  type than its model — happens in one polymorphic `apply`, the same shape as
  `Tensor_map.find`: unpack the existential, check the witness, `Type.Equal`
  where it matches.
- **`Curv.hvp` takes `~y`.** With the closure at a concrete dtype and `apply`
  polymorphic, the constructor needs the prediction to know which dtype the
  closure is written at; otherwise callers would have to write dtype-generic
  closures, casting captured weights by hand. Given `~y`, ordinary monomorphic
  code works, and the matching case costs no cast at all.
- **The collector records the value and tangent before it decides about the
  block.** §6.3 ordered "read Y; if absent, skip" first, which would have made
  the `c`/`C` cross-check lie: a little loss whose prediction is constant but
  which depends on θ by another path contributes nothing to the curvature (its
  block is zero by construction, since Y = 0) while still belonging to the
  loss, so skipping it entirely would flag a perfectly correct sketch.
  Recording the value first keeps the check a faithful statement about
  *observations* rather than about activity.
- **`Curv.apply` is public.** §14.2 kept it internal, but §9.2's checks — every
  constructor against `Rune.hessian'` — need to apply the map, and a caller
  inspecting a curvature wants the same thing. The observation effect, the
  collector and the driver stay private.

Smaller notes. `sketch` also reports `diagnostics` (observed sums, blocks,
skipped), which §6.4's draft did not. Non-float parameter leaves get zero
directions rather than an error: they cannot carry one, and their consumers are
untracked operations, so they simply do not participate. The default sampler
draws in float64 and casts to the leaf's dtype, because `Dtype.is_float` is a
predicate rather than a witness and this is the type-safe formulation. `mse` is
a mean over all elements and `softmax_ce` a mean over rows, each curvature
derived from its own reduction, so value and curvature cannot drift apart.

### 14.5 M4 as built: the matrix, two rune fixes, and what a jitted step costs

M4 shipped as `sofo/test/test_compose.ml` — 15 tests, one per ✓/✓-bonus row of
§7 (with the two intractable-in-principle rows, `vmap` outside and the unscaled
in-map observation, checked for their errors instead of their numbers), plus a
jitted sketch and a memory probe — and `sofo/example/lorenz.ml`, the paper's
§4.1 task in miniature. The matrix itself needed no design change. Three other
things did.

- **The transformation gate leaked, and a failing sketch disabled `jit` for the
  rest of the process.** `Rune.jit` asked whether a transformation was
  installed by reading a global `Gate.transform_depth` counter kept balanced by
  `Fun.protect` around each handler installation. An exception raised in an
  enclosing effect handler is re-raised in that handler's fiber, abandoning the
  fiber below it without unwinding it, so the finalizer never ran: one failed
  observation (the collector's reshape error, any user handler that raises) and
  every later `jit` in the process stepped aside silently. Fixed in rune
  (`a02d2364`) by asking the same question as an effect — `Gate.E_transforming`,
  answered by reverse, forward, batched forward, vmap, the jit tracer and the
  debug logger, the way `Scan.E_scan_probe` already asks about staging — and
  pinned by a rune test that raises from an enclosing handler under `grad` and
  under `jvp_k` of `vmap`. This matters to sofo beyond hygiene: the composition
  suite's ✗ rows *must* raise, and a test that raises may not poison the tests
  after it.
- **The scan claim was private, so a collector could not hold the fold.**
  `jvp_k ∘ Rune.scan` — §7's core case, "per-step observations fire" — fired
  *no* observations at all: `Forward_k` claims `E_scan` and folds in its own
  context, which is outside the collector, and sofo had no way to claim the
  scan itself. `Rune.Scan_claim` now exposes the claim (`eager`, the abstract
  `req`/`res`, and the `E_scan`/`E_scan_probe` constructors, rebound so they are
  the same effects `scan` performs); the collector answers the probe `false`
  and folds under a re-installed copy of itself, so the operations still flow
  outward to `jvp_k` as an ordinary unrolled loop (test: `T` observations, GGN
  equal to Σ_t Y_tᵀH_tY_t from per-lane single-tangent jvps). Any handler of
  one's own that performs effects inside a scan body needs the same two cases.
- **A packed observation (vmap innermost) is additive.** `vmap` re-performs the
  mapped body once with physically batched tensors, so an observation inside a
  map fires once carrying one little loss per mapped element. §7 is right that
  the *contraction* sums over M; the value and the tangent follow the same
  reading, so the collector sums them and the three stay consistent. A loss the
  user reduced with a mean over the map's axis is then a factor of M away from
  the observations — `Sofo.check` reports it and `strict` refuses the sketch —
  which is the documented trap and the reason the cross-check exists. Two ways
  out, both tested: fold the factor into the little loss (value and curvature
  scaled by 1/M, the map reduced with a sum outside), or observe the batched
  prediction after the map, which is what the example does. `Curv.hvp`'s
  closure receives the physically batched direction under a map, like every
  other curvature (they broadcast); now documented.
- **An RNN written for `vmap` must keep the activation on the left.** Storing
  the weights transposed (`z·Cᵀ`, `relu(·)·Wᵀ`) is valid eagerly and is the
  form `vmap`'s matmul rule can translate; the textbook `Cz` order is fine
  eagerly but raises `dot: cannot contract [1,3] to [4,3]` under a map, which
  is §13.3's untranslated matmul rule again. Fixing `vmap` there the way §13.1
  fixed `Forward_k` remains the separate cleanup §13.3 flagged.

**§14.3 pinned, and measured.** A jitted sketch is correct: with `jit`
outermost, loss, C and G̃ match eager execution exactly, a replay is identical,
and `~strict:true` raises `Rune.Jit_error` at trace time (the check reads
values, so it refuses rather than freezing them into the program). The cost is
the unrolling, and the example reports it as unrolling rather than as a
disappointment: a single-trial plain rollout stages its scan as a loop
(trace+compile 6.4 ms at T=32, 7.3 ms at T=64, replay 0.8 ms, CPU), while the
same rollout under a sketch claims the scan and unrolls (403 ms → 638 ms). The
staged forward scan stays M5's question (§13.5), together with the
collector-state change §14.3 describes.

**Memory.** The O(1)-in-T claim holds where it was made: a recurrence stepped
in a plain loop holds 7 live tangent bindings at T=32 and 7 at T=128, with a
forced major collection before each reading. Under `Rune.scan` the count grows
with T (37 → 133), because the eager fold stacks every step's output before
returning it and the store follows what is reachable — `scan`'s contract, not
the differentiation accumulating; the activations *inside* a step die with the
step, and the dominant tensor stays the paper's App. C bound, k×M×hidden.

**Smoke numbers** (example defaults: k = 32, 64 trials, T = 32, hidden = 128,
P = 905, K/P = 3.5%, float64 CPU; `--lanes`, `--trials`, `--hidden`, `--runs`
to change them): `value_and_grad` 17 ms, `jvp_k` over 32 lanes 108 ms,
`Sofo.sketch` 125 ms, and the per-lane `vmap∘jvp` plus explicit Σ YᵀHY reference
306 ms — 2.45× the sketch, in line with §8's estimate. The three drivers agree
to 6.5e-16 relative on C = Θᵀ∇c, and the sketch's C is bit-identical to
`jvp_k`'s. At a near-identity initialization the endpoint loss starts at 1.03,
and a first-order subspace smoke run (normalized ΘΘᵀ∇c, η = 0.05, 15 steps)
reaches 0.62 — enough to show one loss function training under the same driver,
not enough to be a result; the update rule of Alg. 1 is M5.

**The update's bootstrap (`sofo/example/linear.ml`).** §10 keeps the update out
of the library, but the interface it will be built on is now exercised
end-to-end: student-teacher linear regression (`y = x·W*`) whose mean-squared
loss is exactly quadratic in the parameters, so the sketch's second-order model
is exact and the example checks it after every step — residuals at 1e-15, which
validates C, G̃, the sampled Θ and the update jointly, with no reference
implementation.

The shape of the example is forced by `Nx.svd` not compiling, which is the
structure M5 will have to live with:

1. `Rune.jit` wraps the sketch — the forward pass, the K tangent batches, the
   collector — and the `(c, C, G̃)` bundle leaves the compiled program as
   ordinary tensors;
2. `Nx.svd` (Alg. 1 line 10; G̃ is symmetric PSD, so `eigh` would give the same
   numbers, but SVD is what the algorithm says) and the damped solve
   `z = U(S + λ s_max I)⁻¹VᵀC` run eagerly on the host, at O(K³);
3. the parameter step `θ ← θ − η·Θz` is applied eagerly.

Directions are drawn *inside* the trace, from an RNG key threaded as an input
leaf of the carried structure — the shapes never change, so one compilation
serves every iteration and each replay draws fresh Θ (the alternative, a
captured key, is a compile-time constant and `jit` refuses it). Measured on CPU
with P = 256, K = 32, 512 samples: trace + compile ≈ 21 ms with a warm kernel
cache (0.95 s cold, `JITCACHE=0`), replay 0.9 ms, host update (SVD of 32×32,
solve, step) 0.2 ms. The loss falls from 2.24 to 0.014 in 40 steps, where the
*same compiled sketch* stepped first-order (`z = C`, normalized) reaches only
1.60 — the GGN's contribution, in ten lines of host code.

One API note for M5: `sketch.apply` is a closure and cannot leave a compiled
program, so the example has the compiled step return the directions Θ it used
and contracts them on the host. Either `sketch` grows an eager
`apply_dirs ~thetas z` (a function of values, not of the closure), or the
compiled-step deployment returns Θ as this example does; the update module
should not have to choose a second time.

### 14.6 The optimizer as built (`Sofo.Optim`)

The update rule landed before M5 and in sofo rather than in vega — the
optimizer is defined against the sketch, and vega's structural optimizers
should not grow a member whose step needs a factorization until the interface
has seen more use. 18 tests in `sofo/test/test_optim.ml`.

`Sofo.Optim` is vega-shaped: a state that is data, pure functions over
parameter structures, hyperparameters passed per step.

- **State** is `{ key : Nx.int32_t; step : int }`. The key is a *tensor*
  because it is what a compiled sketch step takes as an ordinary input leaf;
  the counter is a host `int` because it only ever feeds key derivation
  (`next` folds it into the key) and schedules. The whole run follows from the
  initial key, and no two steps share a subspace.
- **`directions ?key ~k params`** is Θ: one standard-normal `k`-lane batch per
  float leaf, zero lanes for leaves that cannot carry a direction. It is
  `Sketch.sample` under `Nx.Rng.with_key`, so it is a pure function of the key
  — which is exactly what lets a compiled step draw it inside the trace.
- **`coordinates ?damping ggn c`** is Alg. 1's solve, SVD and all. Matrix
  first, then the right-hand side, like `Nx.solve a b`; `eigh` would give the
  same numbers for a symmetric PSD G̃, but the SVD is what the algorithm says.
  The relative damping λ (default 1e-6) is what makes a rank-deficient sketch
  solvable at all.
- **`update`** is the step: solve, contract (`apply`, which takes the
  directions as a value because a compiled step cannot return the record's
  closure), shift by η. Non-float leaves pass through untouched and each float
  leaf keeps its dtype, as in vega; both are pinned by tests.
- **`step`** is the eager one-call iteration — draw Θ from the state, sketch,
  update, advance the state — and returns the sketch alongside, so a loop can
  log the loss and run `Sofo.check` without a second pass.
- **`Compiled (P)`** is the jittable half, and the reason the module is larger
  than `update` alone. `In = { params; key }` and `Out = { loss; c; ggn; dirs;
  observed_loss; observed_c }` are parameter trees, so
  `Rune.jit2 (module O.In) (module O.Out) (O.sketch ~k loss)` traces the whole
  sketching computation once and replays it for the run. `Out` carries the
  observed sums so that `O.check` — the same statement as `Sofo.check`, on the
  numbers — survives a compiled deployment: a trace cannot read values, but
  the host can, on what the trace returned.

`sketch` itself gained nothing for this: the eager update uses the record's
`apply`, the compiled one uses the directions it hands back, and both wrap the
same solve and shift. The split is forced by the hardware rather than chosen —
the sketching half differentiates and compiles, `Nx.svd` does neither.

**Numbers** (`example/linear.ml`, now driven entirely by the library and ~40
lines shorter; P = 256, K = 32, CPU): trace + compile ≈ 21 ms with a warm
kernel cache (0.95 s cold), replay 0.9 ms, eager update (SVD of 32×32, solve,
shift) 0.2 ms; the loss falls 2.24 → 0.014 in 40 steps, against 1.60 for a
first-order control replaying the *same* compiled sketch. The second-order
model residual stays at 1e-15, which is the joint check on C, G̃, Θ and the
update.

Two things to carry into M5:

- **Convergence is k/P per step, not Newton's rate.** A step solves exactly
  inside its random subspace and ignores the rest, so the loss falls by
  whatever share of the gradient the subspace captured — with K/P ≈ 12% that
  is a few percent per step, which is what the paper's curves show too. A
  schedule and a damping policy are the knobs, and both are per-step arguments
  here.
- **What is eager is a policy, not a constraint of the interface.** Everything
  the update consumes is tensors; a factorizing `eigh` that compiles (or a
  host-side step under `pmap`) would change no signature.
