# ADR 0012 — Device polymorphism, for GPU acceleration

Status: accepted (decision to do it), implementation not started
Date: 2026-02-14

## Context

The engine works and is measured. It is also, by construction, host-only: every
value in it is branded `Nx.host`, so a `Data.t` cannot be placed on a device and
`Rune.jit` of a step compiles a host program. `doc/perf.md` puts numbers on where
that leaves us — 54x to 13898x MuJoCo's `mj_step` — and identifies what the time
goes into: a fixed 64-round solver schedule (34-55% of a step), pair-wise contact
work, and per-operation overhead in the eager path.

The machine has an AMD Ryzen AI MAX+ 395 with a Radeon 8060S iGPU. `rig` ships
its driver (`rig_amd_amdgpu`, `/dev/kfd` present) and `tolk` compiles kernels for
it, so **jit'd work can target it in this build**; what is missing is only
*eager* GPU kernels in nx, which does not matter for a jit'd step. The engine is
the thing standing in the way, not the toolchain.

Doing this early is the cheap time to do it. The engine is 106 brand mentions
across 12 files, most of them mechanical, and it has 15 host reads — the number
that decides whether this is a type change or a redesign.

## The two obstacles, measured

**1. Brands are pervasive but shallow.** `Nx.host` appears 106 times in
`lib/`, concentrated in `collision_box_tensor` (38), `actuator` (10),
`collision` (6), `transmission` (8), `solver` (7), `dynamics.mli` (4), `tendon`
(6), `model` (3). Nearly all are `Nx.host Nx.float64_t` in a signature that
should read `'d Nx.float64_t`, and `Model.host`/`Data.host` that should be
`'d Model.t`/`'d Data.t`.

**2. Fifteen host reads, and five of them are in the constraint path.** This is
the real work, and it is the *same* constraint that made `Step.config` necessary
(M6): a compiled, differentiated program may not read a value out of the tensors
it is tracing. Today `Constraint.equality` reads `eq_active0`, `eq_obj1id` and
`eq_obj2id` with `Nx.item`, `Constraint.contact` reads the `cone` option, and
`Data`'s sensor stage reads the clock. On the host those are legal; on a device
they are a host round-trip inside a kernel, and in a differentiated program they
are already a latent bug waiting for a caller who differentiates with respect to
`eq_obj1id`. The remaining ten (`step.ml`'s timestep and iteration budget,
`solver.ml`'s default budget, `data.ml`'s time) are in host-only entry points and
already have a `config` for the compiled path.

## Decision

Make the engine polymorphic in its device, bottom up, keeping the host as the
default instantiation so that every existing test keeps checking what it checks
today.

**The shape.** A device is a type parameter, as it is in `Nx`:

```ocaml
(* today *)                          (* after *)
type f = Nx.host Nx.float64_t        type 'd f = 'd Nx.float64_t
type host                            type ('a, 'd) params
type data = { ... : Nx.host ... }    type 'd data = { ... : 'd ... }
Model.host                           'd Model.t
```

`Model.host` and `Data.host` are already *names* for something that is only ever
the host; the change renames them to a type parameter and fixes the 106 sites.
`Step.run`, `Dynamics.run` and the rest then work at any brand, because their
bodies are already written in `Nx` operations that are brand-polymorphic.

**The host reads.** Those that are model *structure* — the equality object ids,
the active flags, the cone option — become fields of `Step.config`, which is the
host side of the structure/parameters split M6 introduced, and are read from
there. Any that resist become tensor masks: `eq_active0` is already a mask in
spirit (`Nx.where` selects the row), and ids that index a table can be gathered.

**What stays host.** Model *loading* (`Mjcb.read`) and the fixtures, which are
files. `Collision`'s box-box host kernel, which is the oracle the tensor port is
checked against, and which is not on the compiled path.

## The gate

This is a performance decision, so it is decided by a number, and the number
comes first:

1. **`bench/gpu.ml` must land and report host against iGPU for the solver's
   round** — a batched Cholesky and triangular solve on `[64; 76; 76]`, which is
   one active-set round at the corpus's row-buffer size. It is written and does
   not type: a `Rune.jit` inside `Bench (Set : Nx.Devices)` keeps its brand as a
   fresh variable that no annotation pins to `Set.d`, and timing through a
   polymorphic helper lets the brand escape. Both errors are in `doc/perf.md`
   §9. The enabler is a `Rune.jit` that takes the device set — a small addition
   to rune's interface, and the same one the engine will need.
2. **If that number does not beat the host clearly, stop.** The refactor is not
   paid for by an iGPU that ties with twelve Zen 5 cores on 76x76 dense
   algebra; it would need the streaming part of the pipeline (kinematics, mass,
   contacts) to win, and that is measurable the same way before touching types.
3. **If it does, proceed, and keep the gate `g4.exe`.** Every step of the
   refactor must leave `g4.exe` at 1000/1000 within 1e-8 and `test_rows_of` at
   1e-12. A device that changes an answer is a stop.

## Order of work

Each step is independently shippable and leaves the suite green.

1. **The enabler and the number** (rune: `jit` on a device set; then
   `bench/gpu.ml` as a functor): host vs iGPU for the solver round, and for the
   contact-free pipeline if the round looks good.
2. **Brand the value types**, bottom-up, no behaviour change: `Rigid`, `Pose`,
   `Kinematics`, `Mass`, then `Data`, `Model.params`. Mechanical, reviewed by
   the compiler; the host instantiation is what the tests use.
3. **Brand the constraint and solver**: `Constraint.f`, `Solver`, the row
   records, `Constraint`'s five host reads into `Step.config` or masks. This is
   the only step with a design question in it, and `test_grad`'s QP oracle plus
   `test_linalg` are the checks.
4. **Brand `Step` and `Dynamics`**, and add `Dynamics.compile_on` (or a `device`
   in the config) so a caller chooses where the program runs. Host stays the
   default.
5. **A device test**: place a fixture's state on the iGPU, step it, and require
   the host's answer to 1e-12 (`test_linalg`'s structure, at the step level).
   Skipped, loudly, where no device is available — the suite must still pass on
   a machine without one.

## Consequences

- **The engine stops being a host library.** Every signature grows a parameter,
  and every caller names one; the default keeps that invisible to the viewer,
  the demos and the benches.
- **Eager device work stays out of reach**, because nx has no eager GPU kernels
  in this build. The viewer, the demos and the fixtures stay host. A device path
  is a *compiled* path — which is what a training loop wants anyway.
- **The schedule and the contact work are still the biggest wins on the host**,
  and this refactor does not touch them. Doing it first would make the GPU work
  land on a faster engine; doing it now risks accelerating the wrong 40%. If
  step 2 of the order above measures badly, that is the signal to go back to
  `doc/perf.md` §6 items 1-4 first.

## Alternatives considered

- **Keep the engine host-only, batch on the host.** `vmap` over lanes already
  works and, per `doc/perf.md` §5, is broken by the solver's batched
  factorisation rather than by the host. This is the status quo and it is what
  the numbers say has more headroom *today*.
- **A second, device-only engine.** Rejected: two implementations of the rows,
  the solver and the integrator is two specifications, and the whole method here
  is that one implementation is checked against MuJoCo.
- **Wrap the hot loop only** (a jit'd solver over device arrays, no engine
  polymorphism). Cheaper, and it is what step 1 measures; rejected as the end
  state because it moves the data across the bus every step.
