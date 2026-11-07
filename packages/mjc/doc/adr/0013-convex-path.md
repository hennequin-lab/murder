# 0013 — The convex contact path

**Status:** in progress. `Convex.support` and `Gjk.intersect` are in and
tested; EPA, the assembly and the heightfield are not.

## What MuJoCo actually does

Checked against 3.3.7 rather than assumed, because this decides the size of the
work:

- `engine_collision_primitive.c` has **eight** functions: plane×{sphere,
  capsule, cylinder, box}, sphere×{sphere, capsule, cylinder}, capsule×capsule.
  Everything else is convex.
- The convex path is **not libccd** by default. `mjc_CCDIteration` calls
  `mjc_ccd`, a native implementation in `engine_collision_gjk.c` (2365 lines):
  GJK for the contact test, EPA for the depth and the witness points, a polytope
  with face adjacency for the rest, and up to `max_contacts` witness pairs per
  pair of geoms. libccd's `ccdMPRPenetration` is the fallback when
  `mjDSBL_NATIVECCD` is set.
- Two conventions matter. Support points are pushed out by **half the margin**
  along the direction, each object contributing its own half — so a caller passing
  the pair margin to both gets the pair's threshold. And with a zero distance
  cutoff GJK never reports a distance, only whether the origin is inside the
  inflated Minkowski difference: a separated pair is the *caller's* business,
  which for this engine means `dist` beyond `includemargin` like any primitive.
- Contact assembly is `dist = margin + signed_distance`, `frame = normalize(x1 −
  x2)`, `pos = (x1 + x2)/2`, `frame+3` zeroed, then `mjc_fixNormal` for the pairs
  whose normal convention differs.

## Order of work

1. **Support functions** (`lib/convex.ml`) — done, tested against the defining
   property and against their batched form.
2. **GJK contact test** (`lib/gjk.ml`) — done, on float arrays, tested against
   four closed forms and against a brute-force distance subalgorithm.
3. **EPA** — the depth and the witness pair. **Started and reverted.** An
   expanding-polytope search seeded by GJK's containing tetrahedron was written
   and did not work: it never captured a tetrahedron for two spheres (the
   containing simplex is not always four points) and converged to the wrong face
   for two boxes (depth 0.63 against a closed form's 0.21). Shipping a depth
   function that is quietly wrong is worse than shipping none, so it came out and
   the tests with it. Two things to do differently next time: validate that the
   seed polytope actually contains the origin before expanding, and consider
   Minkowski Portal Refinement — which needs no containing polytope at all, and is
   what MuJoCo itself keeps as the libccd fallback.
4. **The multi-contact extraction** (`polytope2/3/4`, `mjc_isDistinctContact`,
   `maxContacts`) and `mjc_fixNormal`.
5. **The heightfield**: `mjc_ConvexHField` iterates the cells a geom's bound
   covers, makes a prism per cell, and runs the convex path on each.
6. **Batching**: every step above is data-dependent iteration, so the compiled
   path needs it as a fixed-length masked loop, exactly as the solver's rounds
   are a `Rune.scan`. That is the last step and it is what makes the quadruped
   run on the device rather than the host.

## The differentiability stance

Contact geometry through GJK and EPA is not differentiated, and this should be
stated rather than left implied: the iteration's output is not a smooth function
of its inputs at the points where the simplex changes. MuJoCo and MJX both treat
contacts as non-differentiable and differentiate the *solve* instead, which is
what the KKT rule in `Solver` already does. So the convex pairs will contribute a
value and a zero derivative, and the implicit rule keeps the trajectory's
derivative meaningful on the rows that exist.

## Why not approximate

A textbook GJK would agree on distances and disagree on contact count, ordering
and normals. This project's claim is the second kind of agreement, which is why
the plan above is a port with its own tests at each step rather than one
implementation that "works".
