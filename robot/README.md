# geom-robot — robot models, tasks and inverse kinematics

`geom-robot` is the robotics layer of [`geom`](../README.md): a small, batched,
differentiable stack that turns a kinematic model into the values a controller,
a planner or an optimizer consumes — poses, Jacobians, task residuals.

It is two libraries and four modules:

| library | module | what it is |
|---|---|---|
| `geom_robot` | [`Robot`](lib/robot.mli) | links, joints and frames; forward kinematics; geometric and analytic Jacobians |
| `geom_robot` | [`Task`](lib/task.mli), [`Ik`](lib/ik.mli) | task residuals with compact Jacobians, and a batched Gauss–Newton solver |
| `geom_robot_viz` | [`Geom_robot_viz`](viz/lib/geom_robot_viz.mli) | robots as [`geom-viz`](../viz/README.md) entities, with stills and movies |

They sit on `geom` (motors, the conformal model) and the raven stack (`nx` for
tensors, `rune` for differentiation). `geom_robot_viz` adds `geom_viz`. Nothing
here depends on a file format, a physics engine or a robotics middleware.

> **Status.** Research code, complete `.mli` interfaces, tested. The API can
> still move. `ROBOTICS.md` in the package root is the design note: what is
> built (§4.0), what was decided and why (§5.2, §5.4, §5.7), and what is open
> (§10).

## Contents

- [Quick start](#quick-start)
- [The model](#the-model)
- [Forward kinematics](#forward-kinematics)
- [Conventions](#conventions)
- [Differential kinematics](#differential-kinematics)
- [Tasks](#tasks)
- [Inverse kinematics](#inverse-kinematics)
- [Differentiation](#differentiation)
- [Visualization](#visualization)
- [Design notes](#design-notes)
- [API map](#api-map)
- [Tests](#tests)
- [Limits](#limits)
- [Further reading](#further-reading)

## Quick start

The package is developed inside the raven workspace. From `geom/`:

```sh
dune build
dune runtest robot/test            # the model and task suites
dune exec robot/demo/tour.exe      # a tour of everything the layer does
dune exec robot/demo/reach.exe     # pose IK with the shipped solver
```

The rendering demos write PNG, SVG and MP4 into `robot/viz/demo/out/`:

```sh
dune exec robot/viz/demo/arm.exe        # the arm reaching a pose, animated
dune exec robot/viz/demo/workspace.exe  # tool positions coloured by sigma_min
dune exec robot/viz/demo/swarm.exe      # ten IK starts converging to one target
dune exec robot/viz/demo/tasks.exe      # the tool point driven onto a circle
```

MP4 export shells out to `ffmpeg`; everything else is pure OCaml.

A complete program, in the order the rest of this document explains it:

```ocaml
open Nx
module R = Geom_robot.Robot
module T = Geom_robot.Task
module Ik = Geom_robot.Ik

let () =
  let model = R.six_dof () in
  let tool = R.frame model "ee" in
  (* A target pose, taken from a configuration so that it is reachable. *)
  let q_goal = Nx.create float32 [| 6 |] [| 0.5; -0.4; 0.6; -0.8; 0.3; 0.7 |] in
  let target = R.pose model ~frame:tool q_goal in
  (* The task: a residual vector and its Jacobian. *)
  let task = T.pose model ~frame:tool ~target in
  (* The solver: batched damped Gauss-Newton with a trust region. *)
  let result = Ik.solve task ~q:(Nx.create float32 [| 6 |] [| 0.; 0.; 0.; 0.; 0.; 0. |]) in
  Format.printf
    "reached in %d iterations, residual %.2e, converged %b@."
    result.iterations
    (Nx.item [] result.residual)
    (Nx.item [] result.converged)
```

## The model

A model is a tree of links, driven by joints, with named frames hanging off the
links. You describe it with records and `Robot.create` validates it:

```ocaml
module M = Geom.Motor

let model =
  let open Geom_robot.Robot in
  match
    create
      ~links:[ "base"; "turret"; "slider"; "tool" ]
      ~joints:
        [ { name = "yaw"
          ; kind = Revolute (0., 0., 1.)
          ; parent = "base"
          ; child = "turret"
          ; origin = M.one
          ; limits = Some (-1.5, 1.5)
          }
        ; { name = "lift"
          ; kind = Prismatic (0., 0., 1.)
          ; parent = "turret"
          ; child = "slider"
          ; origin = M.translate3 0. 0. 1.
          ; limits = None
          }
        ; { name = "mount"
          ; kind = Fixed
          ; parent = "slider"
          ; child = "tool"
          ; origin = M.translate3 0.5 0. 0.
          ; limits = None
          }
        ]
      ~frames:
        [ { name = "ee"; link = "tool"; pose = M.one }
        ; { name = "eye"; link = "turret"; pose = M.translate3 0. 0. 2. }
        ]
      ()
  with
  | Ok m -> m
  | Error msg -> failwith msg
```

The pieces:

- **Links** are names. The head of the list is the root — the fixed base.
- **Joints** carry the geometry: `origin` is the fixed motion from the parent
  link's frame to the joint's frame, and `kind` is `Fixed`, `Revolute (x, y, z)`
  or `Prismatic (x, y, z)` with the axis in the joint frame (normalised for you).
  A continuous joint is a `Revolute` without `limits`. A joint's axis does not
  have to pass through the joint origin: put the origin where the axis is.
- **Frames** are named rigid transforms on a link. End effector, camera, tool,
  sensor: all frames, no special cases, and `relative` compares any two.

Construction is where validation happens, and it returns a `Result` rather than
raising, with the first fault named: unknown or duplicate names, a root that is
driven, two joints driving one link, joints out of topological order, an
unreachable link, a zero axis, reversed limits.

Inspection: `dof`, `link_count`, `joint_count`, `link_names`, `joint_names`,
`kind_of_joint`, `limits`, `frame_names`, `frame model "ee"` (raising, with the
list of frames it does have), and `pp` for a readable dump.

Two models ship as builders, for tests and demos: `planar2 ?l1 ?l2 ()` and
`six_dof ?lengths ()`.

## Forward kinematics

`q` is a tensor with one column per moving joint, in the order the joints were
given, fixed joints skipped, and any number of leading batch axes:

```ocaml
let ee = R.frame model "ee"                      (* the frames of the model above *)
let eye = R.frame model "eye"

let q = Nx.create float32 [| 2; 2 |] ...         (* a batch of two configurations *)
let poses = R.link_motors model q                (* [| 2; 4; 8 |]: every link      *)
let frames = R.frame_motors model q              (* [| 2; 2; 8 |]: every frame     *)
let one = R.pose model ~frame:ee q               (* [| 2; 8 |]                     *)
let seen = R.relative model ~from:eye ~to_:ee q  (* the tool seen from the eye     *)
```

| function | shape |
|---|---|
| `q` | `batch_shape @ [dof]` |
| `joint_motion model j q` | `batch_shape @ [8]` |
| `link_motors model q` | `batch_shape @ [link_count; 8]` |
| `frame_motors model q` | `batch_shape @ [frame_count; 8]` |
| `pose model ~frame q` | `batch_shape @ [8]` |
| `relative model ~from ~to_ q` | `batch_shape @ [8]` |

The batch axes are tensor axes, not a loop: a configuration, a trajectory and a
sweep of parameters differ only in the shape of `q`. There is no branch on a
coordinate's value anywhere, so the whole thing traces.

## Conventions

Every robotics bug is a convention bug, so these are stated once and tested:
`test_robot.ml` pins each one against an independent construction.

- **A motion maps coordinates.** `M_ab` takes coordinates in frame `b` to frame
  `a`: `p_a = M_ab p_b M_ab~`. Composition is `M_ac = M_ab * M_bc`, so the motion
  applied first is the right operand, as with matrices acting on column vectors.
- **A joint's local motion** is `M_fixed * exp (q * g)` in the parent link's
  frame, where `M_fixed` is the joint's `origin` and `g` its unit screw. Hence

  ```
  M_world_child = M_world_parent * M_origin * exp (q * g)
  ```

  and the joint frame coincides with the child link's frame at `q = 0`.
- **A screw is a twist `(omega, v)`**, angular part first, exactly as
  `Motor.exp` and `Motor.log` take and return it. `v` is the velocity of the
  point at the origin of the frame the twist is expressed in, so a body point
  `p` moves with `v + omega x p`.
- **A frame's world pose** is `M_world_link * M_pose`.
- **A pose error** is `log (M_target^-1 M_current)` — six numbers in the
  current frame's body coordinates — and pairs with the **body** Jacobian.
  `log (M_current M_target^-1)` is a spatial twist referred to the *world
  origin*, a third convention, and pairing it with the body Jacobian is how a
  solver ends up creeping.

## Differential kinematics

Two derivatives, with different jobs:

```ocaml
let jm = R.motion_jacobian model ~frame:tool q   (* d M / d q, [8; dof]       *)
let jw = R.jacobian model ~frame:tool q          (* [6; dof], world axes      *)
let jb = R.jacobian ~space:`Body model ~frame:tool q
```

- **`motion_jacobian`** is the derivative of the eight motor coefficients. It is
  what a residual that is polynomial in the motion needs, and what the chain
  rule into a task Jacobian goes through.
- **`jacobian`** is the geometric Jacobian: column `i` is the twist that
  coordinate `i` contributes to the frame's velocity, so `J * qdot` is that
  velocity. Its linear part is the velocity of the point at the *frame's*
  origin, and `?space` says in which axes:

  | `?space` | rows |
  |---|---|
  | `` `World`` (default) | `(omega_world, dp/dt)` — world axes, frame origin |
  | `` `Body`` | the same physical velocity in the frame's own axes; the classical body Jacobian |

  Both have one column per coordinate, in `q`'s order, with zeros for joints off
  the frame's chain, so `J * qdot` works for a full-length `qdot`.

The body Jacobian is defined by

```
J_body qdot = d/dh log (M^-1 M(q + h qdot)) |_{h=0}
```

and that identity is the test. [`robot/demo/tour.ml`](demo/tour.ml) checks the
world one against a finite difference of the pose and prints the agreement.

One subtlety worth knowing before it bites: the derivative of
`log (M_target^-1 M(q))` is *not* the body Jacobian — they agree to first order
and coincide at the solution, because the differential of the logarithm is the
identity only at the identity. `Task.pose` pairs them anyway, which is what every
operational-space controller does and what makes Gauss–Newton on a pose converge
quadratically; the exact `d log` is listed as an open refinement in
`ROBOTICS.md` §10.

## Tasks

A task is a map from a configuration to a residual vector and its Jacobian.
Nothing reduces it to a scalar, and no solver owns it.

```ocaml
(* A pose task: six residuals, body coordinates. *)
let pose_task = T.pose model ~frame:tool ~target

(* A geometric task: the tool's origin must lie on a sphere. *)
let sphere =
  Geom.Cga.sphere
    ~center:(Nx.create float32 [| 3 |] [| 0.; 0.; 1.2 |])
    ~radius:(Nx.scalar float32 0.4)

let point_task =
  T.primitive model ~frame:tool ~reference:(Geom.Cga.point3 0. 0. 0.) ~target:sphere
```

A geometric task is one contraction. `geom` stores objects in their lower-grade
representative, where incidence is `X dot O = 0`, so

```
E(q) = X_d dot (M(q) X M(q)~)
```

with the lower-grade factor on the left, compacted to the blades that can be
nonzero. The same line covers every object pair, and the residual dimension falls
out of the grades:

| reference | target | residual | reads as |
|---|---|---|---|
| point | point | 1 | coincidence, as `-d^2 / 2` |
| point | plane | 1 | signed distance to the plane |
| point | sphere | 1 | power with respect to the sphere |
| point | line | 5 | the point's moment about the line |
| point | circle | 5 | the point's moment about the circle |
| point | point pair | 5 | either of two targets, without a branch |
| line | point | 5 | the tool axis passes through the point |

Objects are normalised at construction — points projectively, planes and spheres
by their own normalisers, lines and circles by Clifford magnitude — so a residual
does not scale with the representative someone happened to write.

The rest of the interface:

```ocaml
let r = T.residual pose_task q     (* [batch; dim]      *)
let j = T.jacobian pose_task q     (* [batch; dim; dof] *)
let s = T.squared pose_task q      (* [batch]           *)
let n = T.norm pose_task q         (* [batch]           *)
let d = T.dim pose_task            (* int               *)

(* Weighted, then stacked with another task: one least-squares problem. *)
let weighted = T.weight pose_task (Array.make d 1.0)
let both = T.stack [ pose_task; weighted ]

(* Or your own residual, with the Jacobian you supply. *)
let mine = T.custom model ~dim:2 ~residual:my_residual ~jacobian:my_jacobian
```

Weights are row scales on the residual *and* its Jacobian, which is what a
geometric task needs since its components have different units. `stack`
concatenates residuals and Jacobians, so a solver sees one least-squares problem
made of several tasks.

Residuals are values rather than a loss on purpose, and §5.7 of the design note
says why: a sketched second-order method (the `sofo` package in this workspace)
consumes exactly this shape — mark each squared-error block with its curvature
and it accumulates a Gauss–Newton matrix on a random subspace, no backpropagation
involved.

## Inverse kinematics

`Ik` is batched Levenberg–Marquardt over any task, with per-sample damping, a
trust region on the step, and optional joint limits:

```ocaml
let result =
  Ik.solve
    ~max_iter:100
    ~tolerance:1e-6
    ~step_limit:0.3
    ~lower:(Nx.full float32 [| 6 |] (-.Float.pi))
    ~upper:(Nx.full float32 [| 6 |] Float.pi)
    pose_task
    ~q:q_start
```

(A stacked task is the same call: `Ik.solve both ~q:q_start`.)

and the result carries what a caller needs to decide what to do next:

| field | shape | |
|---|---|---|
| `result.q` | `[batch; dof]` | the answer |
| `result.residual` | `[batch]` | its residual norm |
| `result.iterations` | `int` | steps taken for the whole batch |
| `result.converged` | `[batch] bool` | per sample, against the tolerance |
| `result.lambda` | `[batch]` | the damping it ended with |

Every configuration of a batch is its own solver: a step that lowers the residual
for one sample is kept there and its damping relaxes, a step that does not is
dropped for that sample and its damping stiffens — all with `Nx.where`, so a
batch of targets costs one tensor computation per iteration, not a loop.

`Ik.iterate` is the same step, and the hook a demo or an experiment uses to drive
the loop itself:

```ocaml
let () =
  let state = ref (Ik.init ~q:q_start ()) in
  for _ = 1 to 20 do
    state := Ik.iterate ~step_limit:0.2 pose_task !state;
    Format.printf "residual %.2e@." (Nx.item [] (T.norm pose_task (Ik.q !state)))
  done
```

Two practical points:

- **The trust region matters.** A pose inside the workspace usually has a
  solution that winds a joint through several turns. It is a correct answer and a
  useless trajectory, and undamped Gauss–Newton finds it happily.
- **This is a local solver.** Pose IK has local minima of the pose error: on the
  six-joint arm, starts within about a radian of the goal all converge, and
  around 1.7 rad roughly a third settle into one. Restarts are the caller's
  decision, not a hidden policy.

There is no nullspace projection, no hierarchy, no line search and no restart.
Those are the research questions, and `Task` being a value is what makes them
addable without touching the model.

## Differentiation

AD composes; closed forms are the leaves. The kinematics is pure `Nx`, so
`Rune.jit`, `Rune.grad`, `Rune.jacfwd` and `Rune.vmap` work on it as it stands:

```ocaml
let compiled = Rune.jit' (fun q -> R.pose model ~frame:tool q)
let jacobian = Rune.jacfwd' (fun q -> R.pose model ~frame:tool q) q   (* [8; dof] *)
```

Where a closed form is cheaper and branch-free, it is attached with
`Rune.custom_jvp` instead of being a second implementation to keep in sync:

```ocaml
let value = R.pose model ~frame:tool q        (* AD differentiates through exp *)
let same = R.pose_diff model ~frame:tool q    (* same values, analytic tangent *)
```

`Robot.pose_diff` is `Robot.pose` with `Robot.pose_tangent` attached as its rule,
so forward *and* reverse mode use the closed form and higher orders differentiate
the rule. The test suite compares the two paths — `Rune.jacfwd` through `exp`, an
independent implementation of the same derivative, against the analytic Jacobian
— and compares `grad` through the rule with `grad` through the branches it
replaces. In the six-joint arm they agree to float32: 4e-8 on the Jacobian.

## Visualization

`geom_robot_viz` turns a model and a configuration into [`geom-viz`](../viz/README.md)
entities. It reads positions from `Robot.link_motors` — one batched call per
configuration — and hands the renderer nothing but `Primitive.t` values.

```ocaml
module V = Geom_robot_viz
let drawn = V.make model                     (* styles and marker sizes       *)
let camera = V.camera ~eye:(Geom_viz.Primitive.p3 3. (-3.4) 2.1) ()
let q = Nx.create float32 [| 6 |] [| 0.4; -0.3; 0.8; 0.2; -0.5; 0.9 |]

let scene =
  Geom_viz.Draw.create camera (V.stage drawn :: V.entities drawn q)
```

For a picture of one configuration, that is all of it. For an animation, give
`animated` a configuration per time, plus what you want around the robot: a
named frame's triad, the path its origin has taken, a pose marker.

```ocaml
let path = Array.init 48 (fun k -> V.linear_path q_start q_end ~n:48 k)

let entities =
  V.animated
    drawn
    ~q:(V.sample ~fps:24 path)
    ~frame:tool
    ~trail:path
    ~target:(R.pose model ~frame:tool q_end)
    ()
```

Links are thick segments between consecutive joint origins, joints are spheres,
the base is a disc, frames are triads and a pose target is a dashed circle across
its approach axis. There are no meshes and no capsules: a stick figure shows a
Jacobian's geometry better than a surface does, and the primitives are what
`geom-viz` already draws. Styles are per part (`link`, `joint`, `base`, `tool`,
`target`, `trail`) and meant to be replaced — the swarm demo builds ten
differently coloured robots from one model that way.

Two conventions the demos follow and that are worth copying: seed the RNG scope,
or the picture and any number it prints move between runs; and animate the path
from start to answer rather than the solver's iterates, which converge
geometrically and would put the whole approach in the first frames.

## Design notes

**The model layer is a functor.** `Robot.Make (Mo : MOTION)` builds everything
over an abstract motion algebra, and `Geom_robot.Robot` is `Make (Float32)`. A
float64 instance is a second implementation of `MOTION` — product, inverse,
`exp`, `log`, adjoint, `translation`, `rotation` — and a second application, with
no change to this layer. That is the seam the precision decision needs; the
algebra modules themselves are still float32, which is why `Task` (which needs
`Cga`) is not functorised yet.

**Tasks are values, not losses.** A residual vector and its Jacobian can be
squared by a solver, marked as a little loss by a sketched method, put in a QP,
or differentiated for learning. A library that returns "the objective" can only
serve its own optimizer.

**The solver is a policy.** Damping, trust region, acceptance, convergence are
decisions, and research users disagree about them. `Ik` exposes `iterate` and a
structured result for that reason, and does not own the task.

**Batch first.** Model size is static; the batch, the time axis and the sweep are
tensor axes. No loop over configurations anywhere, which is also what makes the
whole layer traceable.

## API map

| module | contents |
|---|---|
| [`Robot`](lib/robot.mli) | `create`, `dof`, `frame`, `link_motors`, `frame_motors`, `pose`, `relative`, `joint_motion`, `motion_jacobian`, `jacobian`, `pose_tangent`, `pose_diff`, `planar2`, `six_dof`, `pp` |
| [`Task`](lib/task.mli) | `pose`, `primitive`, `custom`, `weight`, `scale`, `stack`, `residual`, `jacobian`, `squared`, `norm`, `dim`, `label`, `pp` |
| [`Ik`](lib/ik.mli) | `init`, `iterate`, `solve`, `q`, `lambda`, `iterations`, and the `result` record |
| [`Geom_robot_viz`](viz/lib/geom_robot_viz.mli) | `make`, `links`, `joints`, `base`, `tool`, `trail`, `target`, `entities`, `animated`, `camera`, `stage`, `sample`, `linear_path`, `default_style` |

Every module has a complete `.mli`; the doc comments state the conventions and
the shape of every argument. The design note is
[`ROBOTICS.md`](../ROBOTICS.md); the mathematics of the algebra underneath is
[`doc/notes.typ`](../doc/notes.typ).

## Tests

```sh
dune runtest robot/test                                    # both suites
dune runtest robot/test --force                            # and again
dune exec robot/test/test_robot.exe                        # one suite, verbosely
dune runtest robot/test --instrument-with ppx_windtrap.coverage
dune exec windtrap -- coverage -u                          # uncovered lines
dune exec --instrument-with ppx_windtrap.mutate ./robot/test/test_robot.exe -- --mutate
```

`test_robot.ml` pins conventions rather than numbers: the planar arm against its
hand-derived closed form, the six-joint arm against an independent 4×4 matrix
chain, the world Jacobian against a finite difference of the pose, the body
Jacobian against `log (M^-1 M(q + h qdot)) / h`, the motor Jacobian against
differences of its eight coefficients, the tangent map against the Jacobian, the
rule against `Rune.jacfwd` through `exp` and `grad` through the branches, plus
validation, dof bookkeeping, and the fixed-chain and root-frame edge cases.

`test_task.ml` checks that a residual means what it says and that its Jacobian is
the derivative of that residual, by central differences: the pose error against
its log, a point against a plane, a sphere, a line and through a line, weights
and stacking, batching, and the solver on limits, step limits and batches.

## Limits

Not implemented, in rough dependency order:

- `Jdot`, manipulability and velocity propagation;
- dynamics: `Spatial` types, RNEA, mass matrix, forward dynamics — these need the
  float64 motion module first, which is why the model layer is a functor;
- model I/O and a robot zoo (URDF/YAML, Panda, UR5) — the builders here are a 2R
  arm and a 6R chain;
- collision, distance queries and constraints;
- nullspace and hierarchical task resolution, restarts, and a non-local IK;
- floating bases and multi-DOF joints: the model stores every joint as a unit
  screw, so a helical joint is a small addition, a free joint is not;
- float64: everything is float32 today, `Task` because `Cga` is.

The exact differential of the `SE(3)` logarithm is the one refinement this
document promises and does not deliver; it is open question 6 in
[`ROBOTICS.md`](../ROBOTICS.md) §10.

## Further reading

- [`ROBOTICS.md`](../ROBOTICS.md) — the design note: scope, decisions, milestones,
  open questions, and the research directions this unlocks.
- [`PLAN.md`](../PLAN.md) — the algebra package this rides on.
- [`../viz/README.md`](../viz/README.md) — the renderer.
- [`../dilqr`](../../dilqr) — differentiable iLQR on the same `Nx`/`Rune` stack,
  the natural consumer of `Task` costs for control.
- [`../sofo`](../../sofo) — sketched second-order optimization; §5.7 of the design
  note explains how its Gauss–Newton sketch applies to large-parameter versions
  of these problems.
- GAFRO, the C++ reference for geometric-algebra robotics:
  [website](https://geometric-algebra.tobiloew.ch/gafro/),
  [paper](https://arxiv.org/abs/2310.19090),
  [optimal control](https://arxiv.org/abs/2212.07237).
