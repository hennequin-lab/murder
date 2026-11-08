# ADR 0014 — Dtype polymorphism, for float32

Status: **accepted and implemented.** float32 instantiates the whole engine and
tracks float64 to 1.47e-07 (hopper), 1.71e-06 (humanoid), 4.38e-06 (walker),
4.36e-07 (bounce) and 6.13e-05 (boxes2, the contact-heavy one) over 30 steps from
recorded mid-trajectory states, with the active set agreeing at every step. M7's
acceptance 3 is met. The float64 gates are unchanged to the digit
(hopper 1.10e-11, walker 8.77e-11, humanoid 4.35e-10 worst over 1000 steps), which
is the evidence that this refactor touched types and not numerics.

Date: 2026-02-16

## Context

M7 asks for float32. Today it cannot be instantiated at all, and the reason is
structural, not stylistic:

- `Model.t` and `Data.t` are already parameterised over their leaf type
  ("`('a, 'd) t`, where `'a` is one tensor). What pins them is the *alias*:
  `type 'd model = ('d Nx.float64_t, 'd) t`.
- `Nx.Placement.t` carries the device brand and **not** the dtype
  (`Nx.Placement : sig type +'d t ... end`), so a placement cannot mint a tensor.
  The dtype has to come from a dtype value, `('v, 's) Nx.dtype`, which is
  obtained from a tensor (`Nx.dtype x`) or held in a structure.
- 166 creation sites in 20 files name `Nx.float64` explicitly; 114 signatures
  name `Nx.float64_t`.

Measured payoff, from `bench/dtype.ml`: 1.5–1.8× per solve on the iGPU, and
**1.40× end-to-end**, because 86% of a step is one Cholesky per solver round and
that is the operation fp32 helps least once it is memory-bound. Accuracy:
2–4e-8 per solve, i.e. 2500× inside the 1e-4 that M7 asks for.

So this is not a performance project. It is a *capability* the milestone asks
for — a training path at half the memory with a documented tolerance — and the
gate above is about not overstating it.

## What the implementation actually had to be

The plan below was written before the code, and two parts of it were wrong in a
useful way. Both are worth recording because they are the shape of this refactor.

**There is no dtype field.** `ppx_ptree` requires the payload parameter of a
derived structure to be a single type parameter — "a tensor's element types
cannot mention a parameter" — and `[@ptree.skip]` may not copy a field that
mentions one. So a `dtype : (float, 's) Nx.dtype` field was impossible. The dtype
is instead *recovered from the model's own tensors*: `Model.dtype m` is
`Nx.dtype m.params.timestep`, and because the aliases substitute, `m.params.timestep`
is a concrete tensor inside any function whose signature names `('s, 'd) Model.m`.
That is what "the dtype recorded in the model" means here: recorded in its
tensors, read through one accessor, not kept beside them.

**The records keep one parameter; the aliases carry the dtype.** `Params.t`,
`Model.t` and `Data.t` are `('a, 'd)` — one payload parameter, so the derivation
is untouched — and the public aliases name the payload:

```ocaml
type ('s, 'd) m = ((float, 's, 'd) Nx.t, 'd) t
type 'd model = (Nx.Dtype.float64_elt, 'd) m       (* unchanged meaning *)
type 'd f32 = (Nx.Dtype.float32_elt, 'd) m
```

This is why the change was additive where it mattered: every existing caller
spelling `'d model` kept compiling, and the parity gates were re-run *unmodified*.

**`Nx.Ptree.map` cannot cast.** Its function argument has to be polymorphic in
the dtype, so `Model.cast` could not be a payload map; it is a 67-field record
walk, generated from the field list rather than typed by hand.

## Decision

**Two type parameters, `('s, 'd)`.** The engine is a float engine: both dtypes of
interest are `float` in OCaml, and the only thing that differs is the Bigarray
kind. Parameterising by `('v, 's, 'd)` would add a third variable to ~300
signatures to buy integer tensors the engine does not have.

```ocaml
(* model.mli *)
type ('s, 'd) t =
  { dtype : ('s, 'd) Dtype.t [@ptree.skip]   (* the dtype, recorded *)
  ; topo : Topo.t [@ptree.skip]
  ; params : ((float, 's, 'd) Nx.t, 'd) Params.t
  }

type 'd model = (Nx.Dtype.float64_elt, 'd) t   (* unchanged meaning *)
type 'd m32 = (Nx.Dtype.float32_elt, 'd) t     (* the float32 instantiation *)
```

`dtype` is skipped by the derived walk for the same reason `topo` is: it is
structure, read on the host (`Nx.item`-shaped reasoning), and a program that
carries it as a leaf would have to trace a value that is not data. It is what
makes "the dtype recorded in the model" true, and it is where every creation site
reads its dtype from.

**Where a creation site's dtype comes from**, in order of preference:

1. the model or the data in scope — `Model.dtype m`, or `Nx.dtype d.qpos`;
2. any tensor in scope — `Nx.dtype x`, which is what the `*_like` constructors
   exist for (`Nx.scalar_like`, `Nx.zeros_like`, `Nx.full_like`);
3. an explicit `('s, 'd) Dtype.t` parameter, for the few helpers that have
   neither a model nor a tensor.

`Dtype.t` is the package's own witness:

```ocaml
module Dtype : sig
  type ('s, 'd) t
  val float64 : ('s, 'd) t -> (Nx.Dtype.float64_elt, 'd) t   (* re-typing *)
  ...
end
```

— concretely, a small module holding `val of_nx : ('v, 's) Nx.dtype -> ('s, 'd) t`
and `val nx : ('s, 'd) t -> ('s, 'd) Nx.dtype`, so the package speaks one
vocabulary and `Nx` another.

**Migrating a model is one function, not a second engine.** `Model.cast dt m` walks
the parameter tree casting every leaf; `Mjcb.read` still returns float64 (the file
format is float64 by design), so a float32 user writes
`Model.cast (Dtype.float32 ()) (Mjcb.read path)`. Nothing in the solver,
collision or constraint code knows which dtype it is running; that is the point,
and it is the same posture as ADR 0012 for devices: one engine, two
instantiations.

**The parity story.** The recorded corpus is float64 (`mjbf`), so the float32 test
compares *a float32 run against the float64 trajectory of the same model*: for
every model in the corpus, step both from the same state and report the relative
error per step. The acceptance is 1e-4 relative, and the test fails if any model
exceeds it — with the measured number printed, so a future regression is visible
rather than merely failing.

## Consequences

- ~300 annotations change shape (`'a Model.model` → `('s, 'a) Model.model`). It is
  mechanical, it is compiler-guided, and the float64 aliases keep every existing
  caller's spelling — `'d model` still means float64, so `bench/` and `test/` do
  not move.
- `Step.config` gains the dtype it must compile for, because the structural reads
  (`Nx.item`) and the compiled program are dtype-specific.
- **fp32 is not a parity path.** MuJoCo's `mj_step` is float64 and our parity
  gates stay float64; a float32 run is a different, documented tolerance, and the
  test says which one in its own failure message.
- Anything that cannot be made dtype-generic cheaply is *refused by name*
  (the same rule as `Collision.unsupported_pairs`) rather than silently cast:
  a caller who asks for float32 on an operation that has nothing but a float64
  implementation gets an `Invalid_argument` naming it.

## What was rejected

- **A third type parameter `'v`.** Integer tensors would be legal and unused.
- **A functor over the whole library** (`Mjc.Make (D : Dtype)`), as MILESTONES
  phrased it. The package is twenty modules with cross-references; a functor
  would either nest twenty functors or reproduce the package inside one, and the
  device work already established that brand parameters beat reproduction
  (ADR 0012).
- **Erasing the dtype at the boundary** (dynamically-typed tensors). It moves the
  checking to run time to avoid a compile-time change, which is the wrong trade in
  an engine whose acceptance is numerical.
- **Leaving it and documenting the miss** (what M7 says today). Rejected by the
  project owner, on the grounds that half the memory is worth having for training
  even at 1.40×, and that the tolerance is what makes it honest.
