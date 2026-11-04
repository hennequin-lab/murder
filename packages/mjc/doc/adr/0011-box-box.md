# 0011 — box-box: the kernel first, the pipeline second

Status: **the geometry kernel is in and matches the C engine; the step does not
use it yet.** M4's contact work is otherwise done, and this records why the port
is split in two and what the second half has to do.

## Why it is split

Every other pair in `Collision` is tensor code: one expression over the whole
candidate list, with `where` choosing between the cases so that `Rune.jit` and
`vmap` see a straight line. MuJoCo's `mjc_BoxBox` is not shaped like that. It
has two stages with data-dependent control flow — a separating-axis search over
fifteen candidate axes that returns early on the first axis that separates, then
two *manifestly different* contact generators, one returning a single point from
an edge-edge closest-approach and the other returning up to twelve points from a
Sutherland-Hodgman clip of one face against another's side planes.

Writing that directly as traced tensor code would mean computing all fifteen
axes, both generators, and masking — which is a different program from the one
the C engine ran, and therefore one that cannot be checked against it line by
line. So `lib/collision_box.ml` is a faithful host port, statement for
statement, and it is checked against the engine as an oracle:

    test/test_boxbox.ml: 151 pairs match at 1e-12, 314 pairs correctly make no contact

over the 32 recorded states of `boxes2` (a slab, a squarely stacked box, a box
turned 15°, a box tipped 45° and a box overhanging the slab's edge and turned
30°, which is the case whose clip yields more than four vertices).

## What the kernel is

`Collision_box.manifold ~pos1 ~mat1 ~size1 ~pos2 ~mat2 ~size2 ~margin` returns
an array of `{dist; pos; normal}`, host floats, in the C engine's order. It takes
the layout the data already has (`geom_xpos`, `geom_xmat`, `geom_size`), so a
caller slices and passes.

The constants are MuJoCo 3.17's, and each is a tie-break that decides whether a
stack sits still, so they are copied rather than tuned:

| Constant | Value | What it is for |
| --- | --- | --- |
| `sep_eps` | 1e-13 | slack on the separation tests, times the summed half-extents: without it a pair overlapping by less than its own support's rounding error is declared separated |
| `par_eps` | 1e-16 | `sin²` below which an edge-cross axis is rounding noise |
| `sgn_eps` | 1e-9 | axis component below which a supporting corner is ambiguous (both signs are then enumerated) |
| `dupe_eps` | 1e-14 | squared relative radius for dropping clip vertices the clipping duplicated at a corner |
| `edge_bias` | 1e-6 | relative penalty an edge axis pays on a near-tie with a face axis, because a face gives a multi-point manifold |
| `max_verts` | 12 | clip capacity, and the most contacts one pair can make |

Two rules beyond the test are worth naming, because a naive port omits them and
they are invisible until a stack creeps:

- **Face substitution.** A winning edge axis within ~8° of the best face axis
  duplicates it, and the C engine substitutes the face unless the edge is better
  by five percent of the face's depth. Without it, resting stacks flip between
  the two codes on rounding noise.
- **Every accepted clip vertex is a contact.** MuJoCo's comment says reducing
  the patch to four points costs two to three orders of magnitude in residual
  motion at rest, because the support polygon shrinks and its vertex subset
  changes from step to step.

## What the second half has to do

1. **Slots.** `Collision.slots` sizes each pair's row block; box-box needs 12.
   That inflates `nconmax`/`nefcmax` for any model with box pairs, so the budget
   function in `tools/mjcb.py` and `Topo`'s row accounting have to agree — a
   model whose budget is computed by the converter and by the reader
   differently is a bug that shows up as a truncated contact list.
2. **Tensor form.** Stage 1 vectorises cleanly: fifteen candidate separations,
   a `where`-min over them, and the face-substitution rule as a masked update.
   Stage 2's face path needs a fixed-size polygon: four clip passes over twelve
   slots with masks, which is exact but wants care with the "untouched when all
   inside" shortcut and with vertex counts.
3. **The oracle.** Step 2 is where the host kernel pays off: a property test
   that drops boxes at random poses, computes both the host kernel's manifold
   and the tensor path's, and requires them equal to 1e-12. Without that, a
   tensor port can be right on the corpus and wrong on the next pose.
4. **The parity suite.** The corpus and the test are already there; step 1 of
   this list is what makes `boxes2` a normal fixture that `test_contact.ml`
   checks like every other pair.

## What is honest to say now

- The kernel is correct against MuJoCo on 465 pair checks. That is a strong
  claim and it is measured.
- The engine still **refuses** box-box in a model that needs it: `Collision`
  raises `collision between geom types 6 and 6 is not implemented`. Nothing
  silently approximates.
- The `two-box` acceptance in `VIZ.md` is still unmet, for that reason.
