# geom — geometric algebra for robotics

`geom` is a batched geometric-algebra library for OCaml, built on
[Raven](https://github.com/hennequin-lab/raven)'s `Nx`. One small core gives you
Clifford algebras. Three models on top give you Euclidean rotations, projective
incidence, and the conformal model of 3D space: points, spheres, planes, lines,
circles and rigid motions in a single algebraic system.

Every value is an ordinary `Nx` tensor. Multivectors batch, broadcast,
differentiate with `Rune.grad`, and compile with `Rune.jit`.

> **Status.** Early. The v1 interface in [`lib/`](lib) is implemented and
> tested; names and conventions can still move. `geom-viz` is experimental.
> The mathematics is in [`doc/notes.typ`](doc/notes.typ); design decisions are
> in [`PLAN.md`](PLAN.md) and [`VIZ.md`](VIZ.md).

## Contents

- [Geometric algebra for engineers](#geometric-algebra-for-engineers)
  - [Let vectors multiply](#let-vectors-multiply)
  - [Blades are subspaces](#blades-are-subspaces)
  - [Rotations are sandwiches](#rotations-are-sandwiches)
  - [Why this is useful](#why-this-is-useful)
  - [Three models](#three-models)
- [A tour of the library](#a-tour-of-the-library)
  - [E3: products and rotors](#e3-products-and-rotors)
  - [CGA: points and distance](#cga-points-and-distance)
  - [CGA: spheres, planes, objects from points](#cga-spheres-planes-objects-from-points)
  - [PGA: meet and join](#pga-meet-and-join)
  - [Motors: poses and kinematics](#motors-poses-and-kinematics)
  - [Interpolation](#interpolation)
  - [Batched, differentiable, compiled](#batched-differentiable-compiled)
- [Conventions](#conventions)
- [API map](#api-map)
- [Build and test](#build-and-test)
- [Limits](#limits)
- [Further reading](#further-reading)

## Geometric algebra for engineers

The snippets below use `open Geom` and a few helpers (`v3`, `sc`, `show`,
`vec`) defined in the [tour](#a-tour-of-the-library). The `show` helper prints
the nonzero blades of a multivector.

Most engineering geometry arrives as a pile of unrelated products: the dot
product, the cross product, quaternions for rotation, homogeneous matrices for
transforms, Plücker coordinates for lines. Geometric algebra replaces the pile
with one product. Everything else is a grade selection of that product.

### Let vectors multiply

Take three orthonormal directions `e1, e2, e3` and *define* a product by

```
e1² = e2² = e3² = 1
e1 e2 = -e2 e1,   e2 e3 = -e3 e2,   e3 e1 = -e1 e3
```

The first line says a vector squares to its squared length. The second says
perpendicular directions anticommute. Extend the product to sums by linearity
and associativity. That is the whole definition.

For two vectors `u` and `v`, the product splits into a symmetric and an
antisymmetric part:

```
u v = u·v + u∧v
```

The scalar part is the ordinary dot product. The other part is a new object,
the **wedge** or **bivector** `u∧v`: the oriented parallelogram spanned by `u`
and `v`. The dot product measures how parallel the vectors are. The wedge
measures how perpendicular they are, and records the plane.

A bivector is to a plane what a vector is to a line. It is not a normal vector:
it is the plane itself, with an orientation and a magnitude. In `geom`:

```ocaml
let e1 = E3.vector3 1. 0. 0.
let e2 = E3.vector3 0. 1. 0.

show ~n:3 (E3.to_array (E3.gp e1 e2))   (* 1 e12 *)
show ~n:3 (E3.to_array (E3.gp e1 e1))   (* 1     *)
```

In three dimensions `e1 e2` squares to `-1`:

```
(e1 e2)(e1 e2) = -e1 e1 e2 e2 = -1
```

So `e1 e2` is a square root of minus one tied to the `e1e2` plane. Together
with `1`, the three bivectors `e2 e3`, `e3 e1`, `e1 e2` span the even
subalgebra, which is a copy of the quaternions. Multiplying a vector in the
plane on the right by `e1 e2` rotates it by 90°:

```
e1 (e1 e2) = e2,       e2 (e1 e2) = -e1
```

and doing it twice gives the minus sign. This is the geometric origin of
`i² = -1`.

A **blade** is a product of distinct basis vectors: `1` (a scalar), `e1` (a
vector), `e1 e2` (a bivector), `e1 e2 e3` (a trivector). Its **grade** is the
number of vectors in it, which is the dimension of the subspace it represents.
A **multivector** is a sum of blades of different grades. In three dimensions
there are `2³ = 8` basis blades, so a multivector is eight numbers:

| grade | blades | geometric meaning |
|---|---|---|
| 0 | `1` | scalar: a number |
| 1 | `e1, e2, e3` | vector: a directed line |
| 2 | `e12, e13, e23` | bivector: an oriented plane |
| 3 | `e123` | trivector: an oriented volume |

In `geom`, a multivector is an `Nx` tensor whose last axis holds the `2^n`
coefficients. The leading axes are ordinary batch axes. A single E3
multivector has shape `[|8|]`; a batch of 128 of them has shape `[|128; 8|]`.
That is the complete data structure.

### Blades are subspaces

A blade is a linear subspace through the origin, with an orientation and a
magnitude. The geometric product builds spans with the wedge, and measures the
metric with the contraction.

```ocaml
let e1 = E3.vector3 1. 0. 0.
let e12 = E3.op e1 (E3.vector3 0. 1. 0.)    (* the e1e2 plane *)

show ~n:3 (E3.to_array (E3.op (E3.vector3 1. 1. 0.) e12))  (* 0      : in the plane *)
show ~n:3 (E3.to_array (E3.op E3.e3 e12))                 (* 1 e123 : out of the plane *)
```

The test `x ∧ B = 0` says that `x` lies in the subspace of `B`. It works in any
dimension and for any grade. It is the incidence test in the projective model.
The conformal model stores each object in the dual representation, so there the
same statement is the contraction `X·O = 0`; see the CGA section.

The three products have one name each:

| function | product | grades |
|---|---|---|
| `gp a b` | geometric product | all |
| `op a b` | outer (wedge) product | `r + s` |
| `inner a b` | left contraction | `s - r` |

`scalar_prod` is the grade-0 part of `gp` after reversing the left operand.

The contraction is the metric part of the product. For a unit bivector, the
contraction of a vector in the plane rotates it by 90° in the plane:

```ocaml
show ~n:3 (E3.to_array (E3.inner e1 e12))   (* 1 e2  *)
show ~n:3 (E3.to_array (E3.inner (E3.vector3 0. 1. 0.) e12))  (* -1 e1 *)
```

`reverse`, `grade`, `even`, `odd`, `dual` and `complement` are the other basic
operations on blades. `dual` maps a blade to its orthogonal complement; that is
how a plane can also be used to describe its normal line, and why the
projective model can put points, lines and planes in one algebra.

### Rotations are sandwiches

Reflecting a vector `x` in the plane perpendicular to a unit vector `n` is

```
x  ↦  -n x n
```

Reflect twice, in planes with unit normals `n` and `m`. The composition is

```
x  ↦  R x R~
```

where `R` is the product of the two unit vectors, and `R~` is the reverse of
`R` (the same blades in reverse order). `R` is **even**: it contains only
grades 0 and 2. It is a **rotor**. A general product of invertible vectors is
a **versor**; the rotors are the even versors.

For an angle `θ` in the plane of a unit bivector `B`, the rotor is an
exponential:

```
R  = exp(-θ B / 2) = cos(θ/2) - sin(θ/2) B
R~ = exp( θ B / 2) = cos(θ/2) + sin(θ/2) B
```

That is exactly a unit quaternion: scalar part `cos(θ/2)`, bivector part
`-sin(θ/2) B`. Rotors compose by multiplication, interpolate by slerp, and
never suffer gimbal lock.

```ocaml
let r = E3.rotor ~axis:(v3 0. 0. 1.) ~angle:(sc (pi /. 2.))
show ~n:3 (E3.to_array r)                      (* 0.7071 - 0.7071 e12 *)
vec (E3.to_vector (E3.apply_rotor r e1))       (* [0, 1, 0]          *)
```

The sandwich is the only action in the library. It preserves grades and
products because it is an algebra automorphism. In the conformal model the
same code rotates points, spheres, planes, lines and circles; there is no
per-object transform function.

![The product e1 e2 is the oriented plane they span; a rotor acts by the
sandwich and rotates e1 by 120° in that plane.](doc/figures/e3-products.svg)

*e1 and e2 span the bivector e1e2; the rotor R = exp(-60° e12) rotates e1 by
120° in that plane.*

### Why this is useful

Three properties make this more than a notation.

1. **Objects and motions live in the same algebra.** A sphere, a plane and a
   rigid motion are all multivectors. The same sandwich transforms all of them,
   and preserves every incidence relation.
2. **Predicates are polynomial.** Distance is an inner product; incidence is a
   contraction; intersection is an outer product. The angle enters once, in the
   rotor; everything downstream is multiplication. There are no cases and no
   coordinate frames in the hot path. Polynomial expressions are what automatic
   differentiation and vectorising hardware like.
3. **The representations are small.** E3 costs 8 coefficients, PGA 16, CGA 32.
   A packed rigid motion costs 8. Those are small enough to batch by the
   thousand.

The cross product is the three-dimensional special case where a plane can be
represented by its normal vector. Quaternions are the even subalgebra of E3.
Homogeneous matrices are one representation of the versor action. Geometric
algebra keeps the underlying geometry instead of one coordinate-dependent
shadow of it.

### Three models

One generic core `Algebra.Make` is instantiated three times:

| module | algebra | coeffs | objects | motions |
|---|---|---|---|---|
| [`E3`](lib/e3.mli) | `Cl(3,0,0)` | 8 | vectors, bivectors | rotors (quaternions) |
| [`Pga`](lib/pga.mli) | `Cl(3,0,1)` | 16 | planes (1), lines (2), points (3) | dual quaternions |
| [`Cga`](lib/cga.mli) | `Cl(4,1)` | 32 | points, spheres, planes, lines, circles, point pairs | motors, dilations, transversions |

`E3` is the rotation algebra. `Pga` is plane-based projective geometry: it adds
one degenerate direction `e0` with `e0² = 0`, and represents a plane by a
vector, a line by a bivector, and a point by a trivector. The same `meet`
operation intersects two planes in a line, and a line with a plane in a point.

`Cga` is the conformal model. It adds two dimensions to `RR³` and represents a
Euclidean point by a null vector

```
X = x + ½‖x‖² e∞ + e0
```

with `e0² = e∞² = 0` and `e0·e∞ = -1`. `e∞` is the point at infinity and `e0`
is the null direction of the origin. The point is null (`X² = 0`) and lies on
the slice `e∞·X = -1`. The point of the construction is the inner product:

```
X·Y = -½ ‖x - y‖²
```

Distance is an inner product. A sphere is a vector `S = C - ½r² e∞`, a plane is
a vector `π = n + d e∞`, and a point `X` lies on either exactly when `X·O = 0`.
Lines and circles are grade-2 blades. Intersections are outer products. This is
the model to use when a task mixes object geometry with kinematics.

The packed [`Motor`](lib/motor.mli) type is shared by `Cga` and `Pga`. It is the
eight-dimensional rigid-motion subalgebra

```
{1, e12, e13, e23, e1e∞, e2e∞, e3e∞, e123e∞}
```

isomorphic to the dual quaternions. Kinematics pays eight numbers, not 32.

## A tour of the library

Every computation below is in [`examples/tour.ml`](examples/tour.ml). Run it
from `geom/`:

```sh
dune exec examples/tour.exe
```

The printed values in the comments are the program's actual output (float32,
rounded). The blocks are toplevel phrases, in order; later blocks reuse names
from earlier ones. These helpers are used throughout:

```ocaml
open Geom

let pi = 4. *. atan 1.
let v3 x y z = Nx.create Nx.float32 [|3|] [|x; y; z|]
let v6 a b c d e f = Nx.create Nx.float32 [|6|] [|a; b; c; d; e; f|]
let sc x = Nx.scalar Nx.float32 x
let item x = Nx.item [] x

let vec v =
  Array.to_list (Nx.to_array v)
  |> List.map (fun x -> Printf.sprintf "%.4g" (if Float.abs x < 1e-6 then 0. else x))
  |> String.concat ", "
  |> Printf.sprintf "[%s]"
```

<details>
<summary>The <code>show</code> printer used in the outputs</summary>

```ocaml
(* Print the nonzero blades of a multivector: "0.7071 - 0.7071 e12". *)
let show ~n a =
  let blades = Blade.all n in
  let terms =
    Array.to_list a
    |> List.mapi (fun i x -> blades.(i), x)
    |> List.filter (fun (_, x) -> Float.abs x > 1e-5)
  in
  let term x b =
    if Blade.is_scalar b then Printf.sprintf "%.4g" x
    else Printf.sprintf "%.4g %s" x (Blade.to_string b)
  in
  match terms with
  | [] -> "0"
  | (b, x) :: rest ->
    let sign x = if x < 0. then " - " else " + " in
    List.fold_left
      (fun acc (b, x) -> acc ^ sign x ^ term (Float.abs x) b)
      (if x < 0. then "-" ^ term (-.x) b else term x b)
      rest
```

The coefficient array is in increasing blade-index order, the order returned by
`Blade.all n`. `Blade.to_string` is what turns index 3 into `e12`.
</details>

### E3: products and rotors

```ocaml
let e1 = E3.vector3 1. 0. 0.
let e2 = E3.vector3 0. 1. 0.

show ~n:3 (E3.to_array (E3.gp e1 e2))    (* 1 e12 *)
show ~n:3 (E3.to_array (E3.gp e1 e1))    (* 1     *)
show ~n:3 (E3.to_array (E3.gp (E3.gp e1 e2) (E3.gp e1 e2)))  (* -1 *)
```

For a pair of unit vectors at 45°, the geometric product shows both parts:

```ocaml
let u = E3.vector3 1. 0. 0.
let v = E3.vector3 (1. /. sqrt 2.) (1. /. sqrt 2.) 0.

item (E3.scalar_prod u v)                 (* 0.7071 *)
show ~n:3 (E3.to_array (E3.op u v))       (* 0.7071 e12 *)
show ~n:3 (E3.to_array (E3.gp u v))       (* 0.7071 + 0.7071 e12 *)
```

`op` and `scalar_prod` are the two grade projections of `gp` for vectors. In
general `gp` mixes several grades; `grade k` extracts one of them.

The rotation plane is a real object, not a normal vector:

```ocaml
let e12 = E3.op e1 e2
show ~n:3 (E3.to_array (E3.op (E3.vector3 1. 1. 0.) e12))  (* 0      *)
show ~n:3 (E3.to_array (E3.op E3.e3 e12))                 (* 1 e123 *)
show ~n:3 (E3.to_array (E3.inner e1 e12))                 (* 1 e2   *)
show ~n:3 (E3.to_array (E3.inner e2 e12))                 (* -1 e1  *)
```

A rotor is an even unit element. `E3.rotor` constructs the one for an axis and
an angle, and `apply_rotor` sandwiches:

```ocaml
let r90 = E3.rotor ~axis:(v3 0. 0. 1.) ~angle:(sc (pi /. 2.))
show ~n:3 (E3.to_array r90)                    (* 0.7071 - 0.7071 e12 *)
vec (E3.to_vector (E3.apply_rotor r90 e1))     (* [0, 1, 0] *)

let r45 = E3.rotor ~axis:(v3 0. 0. 1.) ~angle:(sc (pi /. 4.))
let r135 = E3.gp r90 r45                       (* r45 first, then r90 *)
show ~n:3 (E3.to_array r135)                   (* 0.3827 - 0.9239 e12 *)
vec (E3.to_vector (E3.apply_rotor r135 e1))    (* [-0.7071, 0.7071, 0] *)
```

Composition is multiplication, in the same order as matrices acting on column
vectors. `apply_rotor` uses the reverse because a rotor is unit; the general
`Algebra.sandwich` divides by `~v v` and works for any invertible versor.

### CGA: points and distance

A conformal point is a null vector on the slice `e∞·X = -1`:

```ocaml
let x = Cga.point3 1. 1. 0.
let y = Cga.point3 4. 5. 0.

item (Cga.norm2 x)                       (* 0.   X is null *)
item (Cga.scalar_prod Cga.einf x)        (* -1.  X is on the point slice *)
item (Cga.distance2 x y)                 (* 25.  -2 X·Y *)
item (Cga.distance x y)                  (* 5. *)
```

That is the single most useful identity in the model: `X·Y = -½‖x-y‖²`.
Distance, angle and incidence all follow from it.

`Cga.point` maps a batch of coordinates to a batch of multivectors. Products
broadcast over the leading axes:

```ocaml
let centers =
  Nx.create Nx.float32 [|4; 3|]
    [|0.; 0.; 0.; 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1.|]

let xs = Cga.point centers                       (* shape [|4; 32|] *)
let d2 = Cga.distance2 xs (Cga.point3 0. 0. 0.)  (* shape [|4|]    *)
Nx.to_array d2                                   (* [|0.; 1.; 1.; 1.|] *)
```

### CGA: spheres, planes, objects from points

A sphere is the vector `S = C - ½r² e∞`. A point on the sphere satisfies
`X·S = 0`; the same inner product is `-½(‖x-c‖² - r²)`, a signed squared
distance.

```ocaml
let s = Cga.sphere ~center:(v3 0. 0. 0.) ~radius:(sc 2.)
item (Cga.incident (Cga.point3 2. 0. 0.) s)     (* true  *)
item (Cga.incident (Cga.point3 1. 0. 0.) s)     (* false *)
item (Cga.sphere_radius2 s)                      (* 4.    *)
vec (Cga.coords (Cga.sphere_center s))           (* [0, 0, 0] *)
item (Cga.scalar_prod (Cga.point3 3. 0. 0.) s)   (* -2.5  *)
```

A plane is the vector `π = n + d e∞`:

```ocaml
let pi = Cga.plane ~normal:(v3 0. 0. 1.) ~offset:(sc 1.)
item (Cga.incident (Cga.point3 4. 5. 1.) pi)     (* true *)
item (Cga.scalar_prod (Cga.point3 4. 5. 3.) pi)  (* 2.   signed distance *)
item (Cga.is_flat pi)                             (* true *)
item (Cga.is_round s)                             (* true *)
```

The round/flat test is the contraction with `e∞`. A sphere is round; a plane is
flat. A line is flat, a circle is round, a point is round. One test covers all
of them.

Objects are built from points by wedging and dualising:

```ocaml
let a = Cga.point3 1. 0. 0.
let b = Cga.point3 0. 1. 0.
let c = Cga.point3 (-1.) 0. 0.

let circle = Cga.circle a b c
item (Cga.circle_radius2 circle)                        (* 1. *)
vec (Cga.circle_normal circle)                          (* [0, 0, 1] *)
vec (Cga.coords (Cga.circle_center circle))             (* [0, 0, 0] *)
item (Cga.incident (Cga.point3 0. (-1.) 0.) circle)     (* true  *)
item (Cga.incident (Cga.point3 0. 0. 1.) circle)        (* false *)

let line = Cga.line (Cga.point3 1. 0. 1.) (Cga.point3 (-1.) 0. 1.)
vec (Cga.line_direction line)                           (* [1, 0, 0] *)
vec (Cga.line_moment line)                              (* [0, 1, 0] *)
item (Cga.incident (Cga.point3 0. 0. 1.) line)          (* true  *)
item (Cga.incident (Cga.point3 0. 1. 1.) line)          (* false *)

let pp = Cga.point_pair (Cga.point3 0. 0. 0.) (Cga.point3 2. 0. 0.)
item (Cga.coeff (Cga.gp pp pp) (Blade.of_int 0))        (* 4. = d⁴/4 *)
```

| object | construction | stored grade | incidence test |
|---|---|---|---|
| point | `Cga.point` | 1 | `X·Y = 0` |
| point pair | `Cga.point_pair` | 2 | `X·P = 0` |
| circle | `Cga.circle x y z` | 2 | `X·C = 0` |
| sphere | `Cga.sphere` | 1 | `X·S = 0` |
| line | `Cga.line x y` | 2 | `X·L = 0` |
| plane | `Cga.plane` | 1 | `X·π = 0` |

`Cga.circle` is `(x∧y∧z)*`, `Cga.line` is `(x∧y∧e∞)*`, `Cga.plane_through` is
`(x∧y∧z∧e∞)*`, and `Cga.sphere_through` is `(x∧y∧z∧w)*`. The `meet` operation
on two constraint blades is their outer product: two planes give a line, two
spheres give a circle, a plane and a sphere give a circle. It is not a universal
meet for a point and a constraint; use the named constructors for point-based
objects.

![A conformal object zoo: a point on a sphere, a plane, a line defined by two
points, and a circle defined by three points.](doc/figures/cga-objects.svg)

*Five kinds of Cga.t in one scene: a point (the dot on the sphere), a sphere, a
plane, a line and a circle. The line and the circle are drawn from their
defining points.*

### PGA: meet and join

In plane-based projective geometry, planes are grade 1, lines grade 2 and
points grade 3. The outer product intersects them:

```ocaml
let pi1 = Pga.plane3 1. 0. 0. 0.   (* x = 0 *)
let pi2 = Pga.plane3 0. 1. 0. 0.   (* y = 0 *)
let l = Pga.meet pi1 pi2           (* the z axis *)

vec (Pga.line_direction l)                            (* [0, 0, 1] *)
item (Pga.point_on_line (Pga.point3 0. 0. 2.) l)      (* true  *)
item (Pga.point_on_line (Pga.point3 1. 0. 2.) l)      (* false *)
vec (Pga.point_coords (Pga.point3 0. 3. 1.))          (* [0, 3, 1] *)
item (Pga.point_on_plane (Pga.point3 0. 3. 1.) pi1)   (* true  *)
```

Three planes meet in a point, and two points span a line:

```ocaml
let pi3 = Pga.plane3 0. 0. 1. 2.
vec (Pga.point_coords (Pga.meet (Pga.meet pi1 pi2) pi3))   (* [0, 0, 2] *)

let p = Pga.point3 0. 0. 2. and q = Pga.point3 1. 0. 2.
vec (Pga.line_direction (Pga.join p q))                    (* [-1, 0, 0] *)
```

`meet` is the outer product, `join` is the regressive product. The PGA
pseudoscalar is not invertible (`e0² = 0`), so the Hodge `dual` is unavailable;
`Pga` uses the signed Poincaré `complement` and `regressive` instead, and wraps
them in `meet` and `join`.

![Two planes meet in a line; a third plane cuts the line at a point. A point
lies on a plane exactly when their wedge is zero.](doc/figures/pga-meet.svg)

*Two planes meet in a line, a third cuts it at a point, and q is off the line.
In PGA, incidence is one wedge.*

### Motors: poses and kinematics

A motor is a unit even element of the rigid-motion subalgebra. It is stored as
eight coefficients, in the basis `{1, e12, e13, e23, e1e∞, e2e∞, e3e∞, e123e∞}`:

```ocaml
let rz = Motor.rotate ~axis:(v3 0. 0. 1.) ~angle:(sc (pi /. 2.))
let tz = Motor.translate3 0. 0. 1.
let m = Motor.mul tz rz                    (* rotate first, then translate *)

vec (Nx.reshape [|8|] rz)                  (* [0.7071, -0.7071, 0, 0, 0, 0, 0, 0] *)
vec (Nx.reshape [|8|] tz)                  (* [1, 0, 0, 0, 0, 0, -0.5, 0] *)
vec (Cga.coords (Cga.apply_motor m (Cga.point3 1. 0. 0.)))  (* [0, 1, 1] *)

let back = Cga.apply_motor (Motor.inverse m) (Cga.apply_motor m (Cga.point3 1. 0. 0.))
vec (Cga.coords back)                      (* [1, 0, 0] *)
```

`Motor.mul m1 m2` applies `m2` first, then `m1`, like matrices. `Motor.compose`
folds a list in the same order. `Motor.inverse` is the group inverse,
`Motor.normalize` removes drift from a long product, and `Motor.adjoint` is the
6×6 matrix of the action on twists.

The exponential maps a twist `(ω, v) ∈ RR⁶` to a screw motion:

```ocaml
let twist = v6 0. 0. (pi /. 2.) 0. 0. 1.
let m = Motor.exp twist
vec (Motor.log m)                                        (* [0, 0, 1.571, 0, 0, 1] *)
vec (Cga.coords (Cga.apply_motor m (Cga.point3 1. 0. 0.)))  (* [0, 1, 1] *)
```

Here the axis is `z` and the linear velocity is also along `z`, so the motion
is a quarter turn about `z` while rising by one. `Motor.log` inverts `exp` near
the identity, and `Motor.slerp` uses both.

A kinematic chain is a product of motors. Each joint is a rotation composed
with a fixed translation along its link; the product places every joint in the
frame built by its predecessors:

```ocaml
let lengths = [| 0.65; 0.55; 0.45; 0.35; 0.30; 0.25 |]
let q = [| 0.55; -0.75; 0.95; 0.35; -0.85; 0.65 |]
let axis i = if i land 1 = 0 then v3 0. 0. 1. else v3 0. 1. 0.

let joint i =
  Motor.mul
    (Motor.rotate ~axis:(axis i) ~angle:(sc q.(i)))
    (Motor.translate3 0. 0. lengths.(i))

let rec arm i =
  if i = Array.length lengths
  then Motor.one
  else Motor.mul (joint i) (arm (i + 1))

vec (Cga.coords (Cga.apply_motor (arm 0) (Cga.point3 0. 0. 0.)))
(* [-0.9564, -0.2967, 2.156] *)
```

The same motor type embeds into `Cga` with `Cga.of_motor`, into `Pga` with
`Pga.of_motor`, and acts on every object with `apply_motor`. The action is the
dense sandwich `M X M~`; the packed representation is what kinematics pays for,
and the dense embedding is built when an object is transformed.

![Six motors composed into a kinematic chain, with links, joints and an
end-effector frame.](doc/figures/motor-arm.svg)

*Six motors multiplied into one kinematic chain, drawn with links, joint
spheres and an end-effector frame.*

### Interpolation

`Motor.slerp` is screw linear interpolation:

```
slerp m0 m1 t = m0 exp(t log(m0⁻¹ m1))
```

It is the motor analogue of quaternion slerp. The endpoints are exact up to
rounding and the intermediate poses follow one screw:

```ocaml
let m0 = Motor.one
let m1 =
  Motor.mul
    (Motor.translate3 1.3 0. 0.55)
    (Motor.rotate ~axis:(v3 0. 0. 1.) ~angle:(sc (pi *. 0.85)))

let at t = Motor.slerp m0 m1 (sc t)
let positions =
  List.map
    (fun t -> vec (Cga.coords (Cga.apply_motor (at t) (Cga.point3 0. 0. 0.))))
    [0.; 0.25; 0.5; 0.75; 1.]
(* [0, 0, 0]
   [0.2362, -0.3689, 0.1375]
   [0.65, -0.5124, 0.275]
   [1.064, -0.3689, 0.4125]
   [1.3, 0, 0.55] *)
```

![A sphere and a frame swept along one screw motion by
ScLERP.](doc/figures/slerp-screw.svg)

*ScLERP between two motor poses. The spheres are the same object at even
parameter steps; the frames mark the two endpoints.*

### Batched, differentiable, compiled

The library never reads a tensor value on the host and never branches on one.
Structure constants are computed once per algebra instantiation. That is enough
for the Raven toolchain to work on a geometric residual:

```ocaml
let f p = Cga.distance2 (Cga.point p) (Cga.point3 3. 4. 0.)

let value, grad = Rune.value_and_grad' f (v3 0. 0. 0.)
item value                (* 25. *)
vec grad                  (* [-6, -8, 0] *)

let grad = Rune.grad' f (v3 1. 1. 1.)
vec grad                  (* [-4, -6, 2] *)

let jit = Rune.jit' f
item (jit (v3 1. 1. 1.))  (* 14. *)
```

This is the shape of an inverse-kinematics or pose-estimation loop: write the
geometric residual once, differentiate it, and optimise over the parameters.
`Rune` is an example/test dependency; the `geom` library depends only on `nx`.

## Conventions

The formulas the code implements, gathered in one place. `x` is a Euclidean
coordinate vector, `X` its conformal point.

| quantity | formula | constructor or function |
|---|---|---|
| point | `X = x + ½‖x‖² e∞ + e0` | `Cga.point`, `Cga.point3` |
| point tests | `X² = 0`, `e∞·X = -1` | `Cga.norm2`, `Cga.scalar_prod` |
| distance | `X·Y = -½‖x-y‖²` | `Cga.distance2`, `Cga.distance` |
| sphere | `S = C - ½r² e∞` | `Cga.sphere` |
| plane | `π = n + d e∞` | `Cga.plane` |
| sphere test | `X·S = -½(‖x-c‖²-r²)` | `Cga.incident`, `Cga.scalar_prod` |
| plane test | `X·π = n·x-d` | `Cga.incident`, `Cga.scalar_prod` |
| round/flat | `e∞⌟O = 0` for flat | `Cga.is_flat`, `Cga.is_round` |
| rotor | `R = exp(-θB/2)` | `E3.rotor` |
| rotor action | `x ↦ R x R~` | `E3.apply_rotor` |
| motor | `M = T R`, 8 coefficients | `Motor.translate`, `Motor.rotate` |
| motor action | `O ↦ M O M~` | `Cga.apply_motor`, `Pga.apply_motor` |
| interpolation | `m0 exp(t log(m0⁻¹m1))` | `Motor.slerp` |

- **Storage.** Coefficients are in the orthonormal basis `e1, …, en`, in
  increasing blade-index order. The last axis of a multivector has length
  `2^n`; `Blade.all n` gives the blades in that order and `Blade.to_string`
  names them. In CGA the null vectors `e0` and `e∞` are derived multivectors,
  not basis vectors. This keeps the product table sparse and the grade
  structure uniform.
- **Shapes.** A multivector has shape `batch_shape @ [|2^n|]`. Every operation
  broadcasts over the leading axes. A motor has shape `batch_shape @ [|8|]`.
- **Rotations.** `E3.rotor ~axis ~angle` follows the right-hand rule.
  `R = exp(-(angle/2) B)` where `B` is the unit bivector normal to `axis`;
  the sandwich with the reverse rotates by `+angle`. `Motor.rotate` uses the
  same convention.
- **Normalisation.** `Algebra.normalize` is the metric normalisation
  `a / ‖a‖`. The model normalisations are projective: `normalize_point`,
  `normalize_sphere` and `normalize_plane` divide by the scale carried by the
  representative, not by its Clifford magnitude. In PGA, `normalize_point` and
  `normalize_plane` do the same for the projective representatives.
- **Duality.** `dual` is the Hodge dual and requires a non-degenerate
  signature; `Cga` has one, `Pga` does not. In PGA use `complement`,
  `regressive`, `meet` and `join`. The constructors in `Cga` use the dual
  internally; the objects are stored in the lower-grade representation.
- **Scalars.** The model modules are fixed at `Nx.float32`. The core
  `Algebra.Make` is a functor over a scalar kind, so float64 experiments are
  possible without changing the models.

## API map

| module | contents |
|---|---|
| [`Blade`](lib/blade.mli) | bitmask blades, grades, signs, complements |
| [`Signature`](lib/signature.mli) | metric signatures `(p,q,r)`, `euclidean`, `projective`, `conformal` |
| [`Algebra`](lib/algebra.mli) | dense multivectors, `gp`/`op`/`inner`/`scalar_prod`, involutions, duals, norms, `sandwich`, `adjoint`, the `Make` functor |
| [`Versor`](lib/versor.mli) | exponential and logarithm of versors, `exp_simple`, `slerp` |
| [`Motor`](lib/motor.mli) | packed dual quaternions: `mul`, `translate`, `rotate`, `of_rotation`, `of_quaternion`, `exp`, `log`, `slerp`, 6×6 `adjoint`, `translation`, `rotation` |
| [`E3`](lib/e3.mli) | `Cl(3,0,0)`: vectors, bivectors, `rotor`, `apply_rotor` |
| [`Pga`](lib/pga.mli) | `Cl(3,0,1)`: points, planes, lines, `meet`/`join`, incidence |
| [`Cga`](lib/cga.mli) | `Cl(4,1)`: points, spheres, planes, lines, circles, point pairs, conformal generators, motors |
| [`Geom`](lib/geom.mli) | the umbrella module |
| [`Robot`](robot/lib/robot.mli) | `geom_robot`: robot models, batched forward kinematics, geometric and analytic Jacobians, a `custom_jvp` rule for the pose |
| [`Task`](robot/lib/task.mli) | `geom_robot`: task residuals — pose errors, and geometric tasks (point/line/plane/circle/sphere/point pair) with compact residuals and Jacobians, weights and stacking |
| [`Ik`](robot/lib/ik.mli) | `geom_robot`: batched damped Gauss-Newton with a trust region, joint limits and per-sample convergence |
| [`Geom_robot_viz`](robot/viz/lib/geom_robot_viz.mli) | `geom_robot_viz`: stick-figure robots, frames, pose markers and trails as `geom-viz` entities, with animated and still demos in [`robot/viz/demo`](robot/viz/demo) |

Every module has a complete `.mli`; the doc comments state the conventions and
the shape of every argument. The mathematical source of truth is
[`doc/notes.typ`](doc/notes.typ).

## Build and test

`geom` builds with dune (lang 3.24) and depends on `nx`. The visualization
library additionally depends on Hugin's vector-graphics packages. The project is
developed inside the raven workspace; outside it, those packages must be
available to dune.

From `geom/`:

```sh
dune build
dune runtest                     # core, geom-viz and geom_robot suites
dune runtest viz/test            # only the camera and primitive suites
dune runtest robot/test          # only the robot model and kinematics

dune exec examples/tour.exe      # the numerical tour above
dune exec examples/figures.exe   # regenerates doc/figures/*.svg
dune exec robot/demo/reach.exe   # damped Gauss-Newton on a 6R arm
dune exec robot/demo/tour.exe         # what the robot layer can do today
dune exec robot/viz/demo/arm.exe      # the arm and its reach (PNG, SVG, MP4)
dune exec robot/viz/demo/workspace.exe   # workspace cloud coloured by sigma_min
dune exec robot/viz/demo/swarm.exe       # ten IK starts converging, animated
dune exec robot/viz/demo/tasks.exe       # the tool point driven onto a circle
```

The suites are Windtrap executables. `lib/dune` and `viz/lib/dune` carry
coverage and mutation instrumentation, so the same tests can report what they
miss:

```sh
dune runtest --instrument-with ppx_windtrap.coverage
dune exec windtrap -- coverage -u          # -u shows uncovered source

dune exec --instrument-with ppx_windtrap.mutate ./test/test_motor.exe -- --mutate
```

The figures in this README are generated by
[`examples/figures.ml`](examples/figures.ml) with
[`geom-viz`](viz/README.md). The visualizer also has demos:

```sh
dune exec viz/demo/static.exe
dune exec viz/demo/movie.exe
dune exec viz/demo/zoo.exe
dune exec viz/demo/arm.exe
dune exec viz/demo/conformal.exe
```

See [`viz/README.md`](viz/README.md) for the rendering model and the demo
outputs.

## Limits

- **General even-versor logarithm.** `Versor.exp` handles a general bivector
  by scaling and squaring, and `Versor.exp_simple` is the exact closed form for
  one simple bivector. `Versor.log` is exact for a simple versor; the logarithm
  of a general even versor with several independent bivector components is
  still a refinement. `Motor.exp` and `Motor.log` are the full SE(3)
  exponential and logarithm.
- **Dense representation.** A multivector has `2^n` coefficients and the
  structure tables grow as `2^{3n}`. The tested range is `n ≤ 5`. Larger
  algebras are mathematically supported by the functor but not the target.
- **No typed grades.** There is one `t` per algebra, with constructors and
  extractors named by object. Phantom-typed points, spheres and lines are
  deliberately absent; they complicate `meet`, batching and `Nx` interop.
- **`meet` is not universal.** It is the outer product for constraint blades.
  Points are direct objects; mixing them with constraints needs the named
  constructors. See the note in `Cga.meet`.
- **Visualisation.** `geom-viz` has complete `.mli` interfaces but still uses
  a painter's algorithm and has no near-plane clipping, shading or labels.
  Its camera is a PGA construction (view motor, image plane, view rays and a
  line/plane meet) with an equivalent closed-form pinhole fast path. See its
  README.
- **Not on opam.** The package is developed in-tree; the interface is not
  frozen.

## Further reading

- [`doc/notes.typ`](doc/notes.typ) — the mathematical notes: conventions,
  derivations, the CGA identities, the motor algebra, and the references.
  Compile with `typst compile doc/notes.typ`.
- [`PLAN.md`](PLAN.md) — design decisions, alternatives, milestones.
- [`ROBOTICS.md`](ROBOTICS.md) — the proposed robotics layer: robot models,
  kinematics, tasks, dynamics, optimal control and the research roadmap.
- [`VIZ.md`](VIZ.md) — the rendering stack and the scene model.
- [`viz/README.md`](viz/README.md) — using `geom-viz`.
- L. Dorst, D. Fontijne, S. Mann, *Geometric Algebra for Computer Science*.
- C. Doran, A. Lasenby, *Geometric Algebra for Physicists*.
- A. Macdonald, *Linear and Geometric Algebra*.
