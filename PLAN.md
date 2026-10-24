# geom — geometric algebra for robotics on the raven stack

Status: **M1 complete, M2–M4/V1–V3 largely implemented, V4 partial**. The full
`.mli` surface is in `geom/lib/`, and `geom/viz/lib` has complete `.mli`
interfaces too. The core algebra, CGA basics, motors and PGA/E3 basics are
implemented and tested; `geom-viz` renders static Hugin frames and movies with
a PGA camera. The mathematics is in [`doc/notes.typ`](doc/notes.typ) (compiled
to `doc/notes.pdf`); the visualization plan is in [`VIZ.md`](VIZ.md).

## Current implementation state

| Area | Implemented | Still stubbed / next |
|---|---|---|
| `Blade`, `Signature` | full index arithmetic, signs, complements, signatures | — |
| `Algebra` | table-based `gp`/`op`/`inner`, involutions, dual/complement/regressive, norms, inverse, sandwich, adjoint, grades, coefficients | performance pass, more algebraic-law tests |
| `Motor` | packed dual-quaternion product, translate, rotate, `of_rotation`, `of_quaternion`, `exp`, `log`, `slerp`, 6×6 adjoint, inverse, normalize | batched log/adjoint tests |
| `Cga` | points, spheres, planes, centers/radii, incidence, distance, point pairs, lines, circles, line/circle extractors, dilator, transversion, meet, motor bridge, normalization | imaginary-object cases, more batched conformal tests |
| `Pga` | points, planes, lines, incidence, meet/join, line extractors, normalization, motor bridge | more extraction tests |
| `E3` | vectors, rotors, rotor action | more tests |
| `Versor` | `exp_simple`, general `exp` (scaling + Taylor), simple-versor `log`, `slerp` | general even-versor log, accuracy study |
| `geom-viz` | primitives, PGA camera (view motor, eye point, image plane, view rays, meet projection) with a closed-form pinhole fast path, painter's-algorithm drawing, PNG/SVG/PDF frames, `ffmpeg` movie export, CGA/PGA adapters | near-plane clipping, labels, sphere shading |
| Demos | `static`, `movie`, `zoo`, `arm` (6-DOF chain), `conformal` (dilation/transversion) | labels, more robots |
| Tests | algebra, CGA, PGA/E3, motor log/slerp/adjoint, versor exp/log/slerp, viz camera and PGA adapters (`dune runtest`; `dune runtest viz/test`) | golden frames, movie checks |

The camera is a PGA construction: a view motor, an eye point, an image plane
and a line/plane meet for every perspective projection. The closed-form
pinhole evaluation is retained as `Camera.project_pinhole` and is tested
against the meet. The main-library stubs named in the earlier milestones are
implemented; the remaining limitations are the general even-versor (non-simple)
logarithm and viz polish (near clipping, labels, shading).

This document records the decisions, the alternatives that were rejected, and
the order in which the library will be built. It is a working document: update
it when a decision changes, and keep the section numbers in sync with the
notes.

## 1. What we are building

A small, fast, batched Clifford-algebra library for the raven ecosystem, with
three concrete models on top of one generic core:

| Model | Algebra | Objects | Motions |
|---|---|---|---|
| `E3` | Cl(3,0,0) | vectors, bivectors | rotors (quaternions) |
| `Pga` | Cl(3,0,1) | planes (1), lines (2), points (3) | dual quaternions |
| `Cga` | Cl(4,1) | points, spheres, planes, circles, lines, point pairs | motors, dilations, transversions |

The core provides dense multivectors, the geometric/outer/inner products,
involutions, Hodge and complement duals, norms, sandwiching, and the adjoint
of a versor. The model modules add the object constructors, extractors,
incidence predicates and the motor bridges.

The target application is robotics: forward kinematics, task-space
constraints, inverse kinematics, interpolation and geometric queries that mix
rigid motions with spheres, planes, lines and circles. The library should feel
natural in a `Rune` pipeline: batched, differentiable, jittable, no host
round-trips in the hot path.

Non-goals for v1:

- symbolic geometric algebra, arbitrary precision, or general `n` beyond what
  the `2^n` dense representation can hold comfortably (the functor is generic;
  the tested range is `n <= 5`);
- physics/spacetime applications and mesh processing; visualization lives in
  the companion `geom-viz` package (`VIZ.md`), not in the core algebra;
- a typed grade-indexed algebra (GADTs of blades); the flat multivector type is
  deliberate;
- a general-purpose matrix exponential for arbitrary multivectors (see §6.7);
- GPU kernels of our own; all numeric work goes through `Nx`.

## 2. Why CGA, why PGA, and what they cost

CGA is the right default for this project because it puts the objects and the
motions in the same algebra:

- points are null vectors, so distances are inner products
  (`X·Y = -½‖x-y‖²`);
- spheres and planes are vectors (`S = C - ½r²e∞`, `π = n + de∞`), so
  incidence is a contraction (`X·S = 0`);
- lines and circles are 2-blades, intersections are outer products;
- rigid motions are unit even elements (motors) acting by sandwiching, and
  they are exactly dual quaternions in an 8-dimensional subalgebra;
- dilations and transversions complete the 10-dimensional Möbius group.

The cost is 32 coefficients per dense multivector and a 5-dimensional ambient
space. That is small enough for CPU and GPU batches, but it is not the leanest
representation for pure pose algebra. The packed `Motor` type (8 components)
is the answer: kinematics pays dual-quaternion cost, while the object layer
pays the conformal cost only when it needs spheres, planes, lines or circles.

PGA is the natural lower-signature companion:

- Cl(3,0,1) has 16 coefficients;
- planes are vectors, lines are bivectors, points are trivectors;
- the dual is the complement (the pseudoscalar is not invertible), the
  meet/join calculus is exact and polynomial;
- its even subalgebra is the same dual-quaternion algebra as the CGA motor
  subalgebra.

What is shared is the motor algebra, not the object representation. A CGA
point is a null vector, a PGA point is a trivector, and the embeddings into the
two dense algebras differ. Keep the object layers separate; share `Motor`.

Euclidean Cl(3,0,0) is small and is the right first testbed for the generic
core, plus it is useful on its own for orientation-only work.

## 3. Design principles

1. **Nx is the only numeric substrate.** A multivector is an `Nx` array whose
   last axis is the blade axis; leading axes are batch axes. No plain-float
   arrays in the public API, except for small host-side tables and debugging
   helpers.
2. **Everything is traceable.** No `Nx.item`/host reads on values inside an
   algebraic operation, no `if` on tensor contents. Tables are constants
   captured at module initialisation. Branches that depend on values use
   `Nx.where` with safe denominators; small-angle limits use `sinc`-style
   rewrites. This is what makes `Rune.jit` and `Rune.grad` work.
3. **Static signatures, precomputed tables.** A signature is a functor
   parameter. The Cayley structure constants, signs, complements and dual
   permutations are computed once per instantiation, not per call.
4. **Orthonormal storage, null-basis geometry.** CGA coefficients live in the
   orthonormal frame `e1..e5`; `e0` and `e∞` are derived multivectors. This
   keeps the bitmask product rule valid and sparse (notes §8.2).
5. **One dense type, one packed motor type.** `Algebra.t` is the general
   multivector; `Motor.t` is the 8-component dual quaternion. Do not build a
   third representation until measurements demand it.
6. **The `.mli` is the design.** Write the complete interface first, make it
   typecheck with `dune build @check`, then implement bottom-up.
7. **Tests are identities, not examples.** Algebraic laws, CGA identities from
   the notes, and finite-difference checks of the differentiable paths.

## 4. Module graph

```
Blade ──┐
        ├──> Signature ──> Algebra (functor) ──> Versor (functor)
        │                        │
        │                        ├──> E3
        │                        ├──> Pga <── Motor
        │                        └──> Cga <── Motor
        └────────────────────────────> Motor
                                        │
                                        └──> Geom (umbrella)
```

| Module | Role | Depends on |
|---|---|---|
| `Blade` | bitmask blades, grade, reverse/involution signs, complement | — |
| `Signature` | metric signature `(p,q,r)`, orthogonal basis description | — |
| `Algebra` | functor `Make(Signature)`: dense multivectors, products, involutions, duals, norms, sandwich, adjoint | `Blade`, `Signature`, `Nx` |
| `Versor` | functor over an algebra: `exp`/`log` of bivectors/versors, interpolation | `Algebra` |
| `Motor` | packed dual quaternions: product, inverse, exp/log, slerp, 6×6 adjoint | `Nx` |
| `E3` | Cl(3,0,0): vectors, rotors | `Algebra` |
| `Pga` | Cl(3,0,1): planes, lines, points, complement dual, meet/join | `Algebra`, `Motor` |
| `Cga` | Cl(4,1): points, spheres, planes, lines, circles, motors | `Algebra`, `Motor` |
| `Geom` | umbrella re-exports and top-level documentation | all |

`Versor` is separate from `Algebra` so that the core stays a pure product
library and the exponential machinery can be staged. If the split turns out to
be cumbersome in the `.mli` phase, fold `Versor` into `Algebra`; the public
names (`exp`, `log`) should not change.

## 5. Proposed interface (provisional)

This is not the final `.mli`; it is the shape the first pass will take. The
comments indicate the intended shape of the last axis. `Nx.float32_t` is shown
for readability; if the scalar-generic core of §6.5 is adopted, the core
signature gains a `type kind` and the model modules pin it to
`Nx.float32_elt`.

```ocaml
(* blade.mli *)
type t = private int
val grade : t -> int
val reverse_sign : t -> int
val grade_involution_sign : t -> int
val clifford_conjugation_sign : t -> int
val complement : t -> t
val to_string : t -> string

(* signature.mli *)
type t
val make : int array -> t          (* diagonal entries in {-1,0,1} *)
val euclidean : int -> t
val projective : int -> t          (* n Euclidean + one null *)
val conformal3 : t                 (* [|1;1;1;1;-1|] *)
val dim : t -> int
val positives : t -> int
val negatives : t -> int
val nulls : t -> int
val is_degenerate : t -> bool

(* algebra.mli *)
module type ALGEBRA = sig
  type t
  val signature : Signature.t
  val dim : int                    (* 2^n *)

  val scalar : float -> t
  val basis : Blade.t -> t         (* the unit blade *)
  val e : int -> t                 (* e_i, 1-based *)
  val pseudoscalar : t

  val coeff : t -> Blade.t -> Nx.float32_t
  val with_coeff : t -> Blade.t -> Nx.float32_t -> t
  val grade : int -> t -> t
  val even : t -> t
  val odd : t -> t

  val gp : t -> t -> t
  val op : t -> t -> t             (* outer *)
  val inner : t -> t -> t          (* left contraction *)
  val scalar_prod : t -> t -> Nx.float32_t
  val commutator : t -> t -> t

  val reverse : t -> t
  val grade_involution : t -> t
  val conjugate : t -> t
  val dual : t -> t                (* Hodge; requires nondegenerate *)
  val undual : t -> t
  val complement : t -> t          (* always defined *)
  val regressive : t -> t -> t

  val norm2 : t -> Nx.float32_t
  val norm : t -> Nx.float32_t
  val normalize : t -> t

  val sandwich : t -> t -> t       (* V X V^-1 *)
  val adjoint : t -> Nx.float32_t  (* batch @ [dim; dim] *)

  val pp : Format.formatter -> t -> unit
end

module Make (S : SIGNATURE) : ALGEBRA

(* motor.mli — the 8-component dual quaternion *)
type t
val one : t
val mul : t -> t -> t
val compose : t list -> t
val inverse : t -> t
val normalize : t -> t
val translate : Nx.float32_t -> t                 (* ... x 3 *)
val rotate : axis:Nx.float32_t -> angle:Nx.float32_t -> t
val of_rotation : Nx.float32_t -> t               (* 3 x 3 row-major *)
val exp : Nx.float32_t -> t                       (* twist ... x 6 *)
val log : t -> Nx.float32_t                       (* ... x 6 *)
val slerp : t -> t -> Nx.float32_t -> t
val adjoint6 : t -> Nx.float32_t                  (* batch @ [6;6] *)

(* cga.mli *)
include ALGEBRA
val e0 : t
val einf : t
val point : Nx.float32_t -> t                     (* ... x 3 -> ... x 32 *)
val point3 : float -> float -> float -> t
val coords : t -> Nx.float32_t                    (* ... x 32 -> ... x 3 *)
val sphere : center:Nx.float32_t -> radius:Nx.float32_t -> t
val sphere_center : t -> t
val sphere_radius2 : t -> Nx.float32_t
val plane : normal:Nx.float32_t -> offset:Nx.float32_t -> t
val plane_normal : t -> Nx.float32_t
val plane_offset : t -> Nx.float32_t
val line : t -> t -> t
val circle : t -> t -> t -> t
val sphere_through : t -> t -> t -> t -> t
val plane_through : t -> t -> t -> t
val incident : t -> t -> Nx.bool_t
val meet : t -> t -> t
val of_motor : Motor.t -> t
val apply_motor : Motor.t -> t -> t
```

The sketch omits most of the object API (normalised constructors, centre and
radius extractors, the `Pga` equivalents, printing helpers); those belong in
the real `.mli` files. The visualization layer (`VIZ.md`) additionally needs
`Motor.of_rotation`, and the CGA/PGA extractors (`sphere_center`,
`sphere_radius2`, `plane_normal`, `plane_offset`, `line_direction`,
`circle_center`, `circle_radius`, `circle_normal`, `Pga.point_coords`); add
them to the `.mli` pass even though the visualizer is built later.

The first `.mli` pass will settle:

- whether `type t` is the concrete `Nx.float32_t` (recommended, for zero-cost
  interop with `Nx`/`Rune`) or an abstract wrapper with `to_nx`/`of_nx`;
- whether the core is parameterised by a scalar module (float16/32/64) while
  the model modules stay float32;
- the exact labelled-argument conventions for constructors and extractors;
- the final names of the dual operations and the meet/join pair.

## 6. Decisions and alternatives

### 6.1 Dense multivectors (chosen)

`t` is a dense array of `2^n` coefficients, last axis. Alternatives:

- **sparse blade dictionaries**: better for grade-1-heavy work, but dynamic
  shapes, no GPU, no simple batching; rejected for the general layer.
- **grade-indexed types**: e.g. `(k, n) blade`; compile-time safety but a
  combinatorial API, awkward for sums of grades and for the object layer;
  rejected.
- **packed even subalgebra for everything**: motors are even, but points and
  spheres are odd or mixed; you would need several packed types and
  conversions. Packed is used only for `Motor` (§6.4).

Dense is simple, batch-friendly, and the algebra is small (`n <= 5`).

### 6.2 Orthonormal storage basis in CGA (chosen)

The null basis `(e1,e2,e3,e0,e∞)` is not orthogonal (`e0·e∞ = -1`). The
bitmask product rule `e_A e_B = ± e_{A xor B}` assumes orthogonality. Two
options:

- store in the null basis and build a general non-orthogonal product table;
- store in the orthonormal frame and define `e0`, `e∞` as linear
  combinations.

We choose the second. It keeps the product table sparse and the grade
structure uniform, at the cost of a handful of extra nonzero coefficients per
point. PGA's natural basis is already orthogonal, so the issue only arises in
CGA.

### 6.3 Lower-grade working representation of objects (chosen)

CGA has two dual representations of every object. The notes (§4.4) argue for
storing each object in the *lower-grade* of its two representatives:

- points: grade 1 (the point itself);
- spheres, planes: grade 1 (the IPNS vector);
- lines, circles: grade 2 (the IPNS blade);
- point pairs: grade 2 (the direct blade).

Consequences:

- incidence is uniformly `X·O = 0` for a point `X` and an object blade `O`;
- the flat/round test is `e∞⌋O = 0` (notes §3.3);
- constructing from points is "wedge the points (and `e∞` for a flat), then
  dualise";
- `meet` is the outer product **for the constraint blades**. It is exact for
  two grade-one constraints (planes and spheres: a line or a circle). For
  higher-grade constraints the result can be a dual representation (a line and
  a sphere wedge to the grade-three dual of the point pair); the caller
  dualises it. It is not a universal operation: mixing a point (a direct
  object) with a constraint is not the wedge, and incident constraints can
  wedge to zero. Expose `meet` for constraint objects, and named constructors
  for point-based objects.

The alternative — a uniform IPNS or OPNS everywhere — is mathematically
cleaner but makes the common point/object code awkward (a point would be a
4-blade in IPNS) and does not remove the direct/dual mixing in practice. The
working representation is the pragmatic choice.

### 6.4 Packed 8-component `Motor` (chosen)

The rigid-motion subalgebra of Cl(4,1) is generated by `e_ij` and `e_i e∞`; it
is 8-dimensional, with basis

```
{1, e12, e13, e23, e1 e∞, e2 e∞, e3 e∞, e123 e∞},
```

and is isomorphic to the dual quaternions. It is the same algebra as the even
part of Cl(3,0,1). Consequences:

- kinematics uses 8 numbers, not 32;
- the product table is 8×8×8 and is built once, by embedding the basis into
  the dense algebra and projecting back (robust, no hand-derived signs);
- `Cga.of_motor`/`Pga.of_motor` embed into the dense algebra; `apply_motor`
  sandwiches without materialising the dense motor;
- the same `Motor` module serves CGA and PGA.

Alternatives: use the dense even subalgebra (16 components in CGA) for
everything (wasteful), or store motors as 4×4 matrices (loses the GA
composition and interpolation). Rejected.

### 6.5 Functor over signature; scalar policy (chosen)

`Algebra.Make(S)` with `S.signature : Signature.t` is chosen over a runtime
algebra record. Reasons: constants at module init, clean names, no per-call
dispatch. Generic code over algebras can still use a functor over `ALGEBRA`;
first-class modules are not needed for the three models.

Scalar policy is **Option A**: the core `Algebra.Make` is generic over a real
scalar kind (float32/float64), while `E3`/`Pga`/`Cga` are fixed to float32.
Tests can instantiate the core in float64 for tight tolerances. The model APIs
do not expose the scalar parameter; `float32` is the robotics default. The
concrete polynomial is

```ocaml
module type SCALAR = sig
  type kind
  val dtype : (float, kind) Nx.dtype
end

module type ALGEBRA = sig
  type kind
  type t = (float, kind) Nx.t
  ...
end

module Make (S : SIGNATURE) (K : SCALAR) : ALGEBRA with type kind = K.kind
```

### 6.6 Degenerate duals (chosen)

- Non-degenerate algebras: `dual A = A I⁻¹`, `undual A = A I`; for CGA
  `I² = -1`, so `dual` is an involution up to sign and `undual` must be defined
  explicitly (notes §1.6).
- Degenerate algebras: the pseudoscalar is not invertible. `complement` is the
  signed permutation `A ↦ A^c` with `A ∧ A^c = I`; `regressive A B` is the
  meet in the OPNS sense, built from complements.
- Do not hide the distinction behind a single `dual` that silently switches
  behaviour. `Cga` exposes the Hodge `dual`; `Pga` exposes `complement` and
  `regressive` (and may alias `dual` to `complement` with a warning in the
  docs, or not at all).
- Incidence is also different: in CGA it is `X·O = 0`; in PGA it is the wedge
  `p∧π = 0`, and the degenerate metric prevents reducing it to a scalar
  contraction of the dual point and plane. Do not give `Pga` a CGA-shaped
  `incident` without a model-specific meaning.

Tests pin the complement signs; they are a convention, not a derivation.

### 6.7 Exponential and logarithm (staged)

- **Motor exp/log (SE(3))**: closed form via the twist (Rodrigues + the `V`
  matrix), with `atan2` and `sinc`-style small-angle limits. This is the
  primary robotics path and is implemented in `Motor`.
- **Simple bivector exp**: closed form from `B²` (notes §1.7). Covers rotors,
  translators, dilators and transversions.
- **General bivector exp**: a sum of two commuting simple bivectors; implement
  the spectral decomposition later, or use a scaling-and-squaring series as
  the traceable fallback. Not needed for the first robotics milestones.
- **General multivector exp**: out of scope for v1; the matrix exponential of
  a 32×32 adjoint is not the right tool.

Keep `exp`/`log` out of the hot path where a closed form exists; make the
general routine explicit in the API (`Versor.exp`), not implicit in `gp`.

### 6.8 Traceability and autodiff (chosen)

All algebraic operations are pure `Nx`. The structure constants are constants
captured by the closure. The dual/involution/complement maps are signed
permutations of the last axis, implemented with `Nx.take` + `Nx.mul` of a
constant index/sign vector (not a full `gp`). `exp`/`log` use `Nx` primitives
and `Nx.where`, never OCaml `if` on tensor values. `Rune.grad` and `Rune.jit`
are tested on a small IK/pose objective; the library itself does not depend on
`Rune`.

### 6.9 Object typing (chosen for v1: no phantom types)

Use one `t` for all CGA objects, with constructors and extractors named by
object (`point`, `sphere`, `line`, `circle`, `plane`, `point_pair`) and
`incident`/`meet` where the operation is well-defined. Phantom-typed wrappers
(`Point.t`, `Sphere.t`, …) would prevent some mistakes but complicate `meet`
(result type depends on the operands) and every batch/`Nx` interop. Revisit
after the first robotics example; the API can grow typed views without
changing the representation.

### 6.10 Naming and packaging (chosen)

- package/library: `geom` (not taken on opam at the time of writing);
- top module: `Geom`, with `Geom.Cga`, `Geom.Pga`, `Geom.E3`, `Geom.Motor`,
  `Geom.Algebra`, `Geom.Blade`, `Geom.Signature`;
- `Motor` lives at the top level, not under `Cga`, because it is shared with
  PGA;
- `.mli` doc comments reference the notes by section number (`doc/notes.typ`
  §…).

Dependencies: `nx` for the library; `base` only if it earns its keep in the
implementation (the decision can wait until the first module); `windtrap` for
tests; `rune` only in tests/examples, never in the library.

### 6.11 Structures and `ppx_ptree` (chosen)

The numeric types are ordinary `Nx` tensors, so they are already `Nx.Ptree`
leaves; no custom `walk` is written. Record types that appear later (in the
visualizer, or in an optimizer state) derive `ptree` with `ppx_ptree` rather
than hand-writing a traversal. The main library does not currently have such a
record, so `ppx_ptree` is not in `lib/dune` yet; add it when the first deriving
is introduced.

## 7. The `.mli`-first workflow

Dune needs a hint for modules that have an interface but no implementation.
The initial `geom/lib/dune` will be:

```
(library
 (name geom)
 (libraries nx)
 (modules_without_implementation
  blade signature algebra versor motor e3 pga cga geom))
```

and `geom/dune-project`:

```
(lang dune 3.24)
(name geom)
```

Keep the library private (no `public_name`, no package stanza) during the
interface phase: it avoids touching the workspace `dune.lock`, which the
repository's conventions say not to relock without being asked. Packaging and
`public_name` come in M7, once the API is stable.

Then:

1. write all the `.mli` files (including the `module type`s and functor
   signatures);
2. `dune build @check` — this typechecks the interfaces even though no `.ml`
   exists, as long as the module is listed in `modules_without_implementation`;
3. iterate on the interfaces until the API reads well and the dependencies are
   acyclic;
4. implement bottom-up, removing a module from
   `modules_without_implementation` when its `.ml` lands;
5. after each implementation step, `dune build @check && dune runtest`.

The `.mli` order to write:

1. `blade.mli` (no dependencies);
2. `signature.mli` (no dependencies);
3. `algebra.mli` (`module type ALGEBRA`, `Make`);
4. `motor.mli` (depends only on `Nx`);
5. `e3.mli` (small; exercises `include Algebra.ALGEBRA`);
6. `cga.mli` (the largest object API);
7. `pga.mli` (shares `Motor`);
8. `versor.mli` (functor over `ALGEBRA`);
9. `geom.mli` (umbrella aliases and top-level docs).

Expect the first `@check` to expose ordering issues (e.g. `Cga` including
`ALGEBRA` while also adding `Motor` bridges). Fix the interfaces, not the
implementation, at this stage.

## 8. Milestones

**M0 — foundations (this commit).**
`PLAN.md`, `doc/notes.typ` (compiles), decisions recorded, no code.

**M1 — the interface (done).**
`dune-project`, `lib/dune`, all `.mli` files: `blade`, `signature`, `algebra`,
`versor`, `motor`, `e3`, `pga`, `cga`, `geom`. `dune build @check` is green
with every module listed in `modules_without_implementation`. The API has been
read against the notes once; names can still move until the first
implementations land.

**M2 — core algebra.**
`Blade`, `Signature`, `Algebra`: blades, tables, `gp`, `op`, `inner`,
`reverse`, grade projections, `dual`/`complement`, `norm`, `sandwich`,
`adjoint`. Tests: algebraic laws on random multivectors in Cl(3,0,0),
Cl(3,0,1), Cl(4,1); a slow reference implementation; benchmarks of the dense
product (einsum vs. gather/sum vs. grade-sparse).

**M3 — CGA objects.**
Points, spheres, planes, incidence, distances, normalization, lines, circles,
point pairs, meets, the direct/dual constructors. Tests: every identity in
notes §2–§4, including the four-point sphere, the three-point plane, the
three-point circle, the two-point line, and the carrier plane of a circle.

**M4 — motors and kinematics.**
`Motor`: product, inverse, normalization, translate, rotate, exp/log, slerp,
6×6 adjoint; `Cga.of_motor`/`apply_motor`. Tests: dual-quaternion identities,
exp/log round-trip, adjoint vs. sandwich, a 6-DOF forward-kinematics example
against homogeneous matrices.

**M5 — PGA.**
`Pga`: planes, lines, points, complement dual, meet/join, `of_motor`/
`apply_motor`. Tests: incidence, complement involution, meet/join identities,
the shared motor algebra, the CGA/PGA motor consistency on common motions.

**M6 — the rest of the conformal group.**
`Versor.exp/log` for simple bivectors, dilations, transversions, general
bivector decomposition or scaling-and-squaring fallback; interpolation tests;
adjoint consistency.

**M7 — rune, examples, performance.**
A jitted/grad-able IK or pose-estimation example; a batch benchmark; a
`README`; doc comments cross-referencing the notes; a first pass on
`Cga`-based geometric queries (point-to-plane distance, sphere-sphere
intersection, line-sphere intersection).

## 9. Test plan

Algebraic laws (property tests, random multivectors):

- `e_i e_j + e_j e_i = 2 g_ij`;
- associativity of `gp`, distributivity over addition;
- `reverse (A B) = reverse B * reverse A`;
- grade involution is an automorphism, Clifford conjugation is an
  anti-automorphism;
- `dual (dual A) = ± A` with the sign fixed by `I²`;
- `complement (complement A) = ± A` (PGA);
- grade selection of `op`, `inner`, `scalar_prod`;
- `norm2 (reverse A * A)` is scalar; `normalize` is idempotent;
- `sandwich` preserves grades and incidence; `adjoint` agrees with
  `sandwich` on a basis.

CGA identities (from the notes):

- `X² = 0`, `e∞·X = -1`, `X·Y = -½‖x-y‖²`;
- `S² = r²`, `X·S = -½(‖x-c‖² - r²)`, sphere center/radius extraction;
- `π² = 1`, `X·π = n·x - d`, plane normalization;
- four points define a sphere; three points plus `e∞` define a plane; three
  points define a circle; two points plus `e∞` define a line;
- `e∞⌋O` flat/round test;
- motor action preserves nullity and `e∞·X`; translation, rotation, dilation,
  transversion actions against the notes' formulas;
- `exp(log M) = M` on random unit motors; `log(exp ξ) = ξ` on random twists;
- ScLERP endpoints and constant screw for intermediate values;
- `adjoint M` applied to a multivector equals `M X M~`.

PGA identities:

- plane/line/point incidence; complement dual; meet/join;
- motor action on points/planes/lines;
- the shared motor product agrees with CGA on the common subalgebra.

Numerical and integration:

- unbatched vs. batched results agree;
- float32 vs. float64 core agreement on the same inputs (if the scalar
  functor is in place);
- `Rune.grad` finite-difference check on a small pose/IK objective;
- `Rune.jit` produces the same result as eager execution;
- no allocation surprises in the motor hot path (benchmark).
- **Coverage and mutation**: `lib/dune` and `viz/lib/dune` carry the Windtrap
  coverage and mutation backends. Run the suites with
  `--instrument-with ppx_windtrap.coverage` and report with
  `dune exec windtrap -- coverage`; run an individual suite with
  `--instrument-with ppx_windtrap.mutate ... -- --mutate` for mutation testing.

## 10. Open questions

1. **Scalar policy** (resolved): generic core + float32 models; see §6.5.
2. **`type t` concrete vs. abstract** (resolved for now): concrete
   `(float, kind) Nx.t`, documented last-axis convention; revisit only if the
   representation needs to change.
3. **`Versor` split** (resolved for now): `exp`/`log` live in a separate
   `Versor.Make` functor; `Motor` has its own packed exponential.
4. **Constructor convention**: tensor-first with a trailing coordinate axis
   (`point : ...×3 -> ...×32`) vs. three labelled tensor arguments. The former
   is better for batches; the latter reads better for single points. Provide
   both `point` and `point3` if needed.
5. **`meet` semantics**: confirm that `meet` is exposed only for constraint
   blades and that point/point-pair constructions go through named functions
   (§6.3).
6. **OPNS convenience**: should `Cga` also expose `opns`/`ipns` conversion
   helpers with names, or is `dual` sufficient?
7. **Motor basis and signs**: fix the packed basis and the embedding signs in
   tests; make sure `Motor.exp/log` match the dense `exp` of the corresponding
   bivector.
8. **General bivector exp**: decomposition vs. scaling-and-squaring; which is
   accurate enough at float32 for the dilations/transversions we care about?
9. **Phantom-typed objects**: revisit after the first IK example; do not add
   prematurely.
10. **Performance target**: define a concrete budget (e.g. a 6-joint chain and
    128 points per IK iteration under a millisecond on CPU) and measure in M7.
11. **Dependency on `base`**: use it for the implementation or keep the core
    stdlib-only? `Nx` is the only hard dependency either way.
12. **n > 5**: the functor supports it, but the tables grow as `2^{3n}`; decide
    whether to cap `n` in `Signature.make` or document the practical limit.

## 11. Companion visualization: `geom-viz`

The renderer is planned separately in [`VIZ.md`](VIZ.md). In short: it uses
Hugin's `hugin.vg` vector-graphics layer (already in Raven) for PNG/SVG/PDF
frames, a PGA camera for projective projection, CGA objects for the scene, and
`Motor` for animation; movies are PNG frame sequences encoded with the system
`ffmpeg`. Keeping it in a separate `geom-viz` package leaves the core algebra
free of graphics dependencies. The core API must nevertheless grow the
extractors and `Motor.of_rotation` listed in §5, so the two interfaces stay
compatible.

## 12. References

The mathematical references are in `doc/notes.typ` §10. The implementation
references that matter:

- `raven/packages/nx/lib/nx.mli` — the array API, dtypes, `einsum`, `take`,
  `where`, shapes;
- `raven/packages/rune` — `jit`, `grad`, `vmap` for the integration tests;
- `raven/packages/vega` and `sofo`/`symo` — examples of the repository's
  `.mli` documentation style and of how a numerical library is layered on
  `Nx`;
- `raven/packages/hugin/lib/vg` — the `hugin.vg`, `hugin.vg.raster`,
  `hugin.vg.svg` and `hugin.vg.pdf` libraries used by the visualizer
  (`VIZ.md`);
- `raven/AGENTS.md` — build and agent conventions for this workspace.
