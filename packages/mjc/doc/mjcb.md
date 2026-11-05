# The `.mjcb` model format

Version 7. Little-endian throughout. Written by `tools/mjcb.py` from a MuJoCo
`mjModel` that has been through `mj_setConst`, read by `Mjc.Mjcb`.

The format exists so that the runtime never links MuJoCo: the compiler's
work — defaults inheritance, inertia inference, `qpos0`, the reference
configuration — happens once, in the converter, and the result is a flat file
of the arrays a step reads. It is the v1 answer to `PLAN.md` §13 Q3; the
native MJCF reader is M10.

## Layout

```
"MJCB"          4 bytes, magic
version         u16
flags           u16, zero; reserved
model name      string
MuJoCo version  string, as the converter reported it ("3.17.0")
counts          u32 count, then that many i32
int arrays      u32 count, then per array: u32 length, i32 elements
name tables     four tables, each u32 count, then per name: u32 length, UTF-8
tensors         u32 count, then per tensor: string name, u32 rank,
                u32 dims, f64 payload, row-major
"MJCE"          4 bytes, trailer
```

A `string` is `u32 length` then that many bytes: the same encoding as a tensor
name, so a reader has one string function.

Every count, array and tensor is named. A reader checks for the names it
needs and reports the first one missing, rather than reading by position, so a
field added by a later milestone is an error in an older reader and not a
misaligned payload.

## Counts, in order

`nbody ngeom njnt nsite ntendon nu na nsensor nmesh nq nv nconmax nefcmax
max_geom_pairs`

The first eleven come from `mjModel`. The last three are compile-time budgets
the converter chooses from the model size (`nconmax = max(16, 2*ngeom)`,
`nefcmax = max(64, 4*ngeom)`, `max_geom_pairs = max(16, 2*ngeom)`); a caller
can override them when converting. They are budgets, not measurements: a step
sized to them and a model whose contacts exceed them is an `Nx.check`
failure, never a silent drop.

## Integer arrays, in order

| Name | Shape | Source |
| --- | --- | --- |
| `body_parentid` | `[nbody]` | `mjModel.body_parentid` |
| `body_jntadr` | `[nbody]` | `mjModel.body_jntadr` |
| `body_jntnum` | `[nbody]` | `mjModel.body_jntnum` |
| `body_geomadr` | `[nbody]` | `mjModel.body_geomadr` |
| `body_geomnum` | `[nbody]` | `mjModel.body_geomnum` |
| `body_rootid` | `[nbody]` | `mjModel.body_rootid` |
| `jnt_type` | `[njnt]` | see the codes below |
| `jnt_bodyid` | `[njnt]` | `mjModel.jnt_bodyid` |
| `jnt_qposadr` | `[njnt]` | `mjModel.jnt_qposadr` |
| `jnt_dofadr` | `[njnt]` | `mjModel.jnt_dofadr` |
| `dof_bodyid` | `[nv]` | `mjModel.dof_bodyid` |
| `dof_jntid` | `[nv]` | `mjModel.dof_jntid` |
| `dof_parentid` | `[nv]` | `mjModel.dof_parentid` |
| `geom_bodyid` | `[ngeom]` | `mjModel.geom_bodyid` |
| `geom_type` | `[ngeom]` | see the codes below |
| `geom_contype` | `[ngeom]` | `mjModel.geom_contype` |
| `geom_conaffinity` | `[ngeom]` | `mjModel.geom_conaffinity` |
| `geom_priority` | `[ngeom]` | `mjModel.geom_priority` |
| `site_bodyid` | `[nsite]` | `mjModel.site_bodyid` |

The file uses its own joint codes, not MuJoCo's (MuJoCo numbers free, ball,
slide, hinge in the opposite order to the way they read in a model file):

| Code | Joint |
| --- | --- |
| 0 | hinge |
| 1 | slide |
| 2 | ball |
| 3 | free |

Geom codes are MuJoCo's: 0 plane, 1 hfield, 2 sphere, 3 capsule, 4 ellipsoid,
5 cylinder, 6 box, 7 mesh. A reader accepts all eight; v1's kernels implement
plane, sphere, capsule, cylinder and box, and report the others when reached.

The world body is body 0, and its `body_parentid` is 0: it is its own parent,
as in `mjModel`. Every other body's parent index is smaller than its own.

## Name tables, in order

`body_names jnt_names geom_names site_names`

Each has one entry per object; an unnamed object carries an empty string.
Names are not read by a kernel. They are read by `tools/inspect.exe`, by the
viewer, and by error messages.

## Tensors, in order

All `float64`, row-major. `nq` is the number of position coordinates; `nv`
the number of velocity coordinates, which is also the number of degrees of
freedom.

| Name | Shape | Reads |
| --- | --- | --- |
| `body_pos` | `[nbody;3]` | FK |
| `body_quat` | `[nbody;4]` | FK |
| `body_ipos` | `[nbody;3]` | FK |
| `body_iquat` | `[nbody;4]` | FK |
| `body_mass` | `[nbody]` | M2 |
| `body_inertia` | `[nbody;3]` | M2 |
| `body_subtreemass` | `[nbody]` | M2 |
| `jnt_axis` | `[njnt;3]` | FK |
| `jnt_pos` | `[njnt;3]` | FK |
| `jnt_range` | `[njnt;2]` | M3 |
| `jnt_limited` | `[njnt]` | M3 |
| `jnt_stiffness` | `[njnt]` | M2 |
| `dof_armature` | `[nv]` | M3 |
| `dof_damping` | `[nv]` | M2 |
| `dof_frictionloss` | `[nv]` | M3 |
| `qpos0` | `[nq]` | FK |
| `qpos_spring` | `[nq]` | M2 |
| `geom_pos` | `[ngeom;3]` | FK |
| `geom_quat` | `[ngeom;4]` | FK |
| `geom_size` | `[ngeom;3]` | M4 |
| `geom_rgba` | `[ngeom;4]` | viewer |
| `geom_friction` | `[ngeom;3]` | M4 |
| `geom_margin` | `[ngeom]` | M4 |
| `geom_gap` | `[ngeom]` | M4 |
| `geom_solmix` | `[ngeom]` | M4 |
| `geom_solref` | `[ngeom;2]` | M4 |
| `geom_solimp` | `[ngeom;5]` | M4 |
| `site_pos` | `[nsite;3]` | FK |
| `site_quat` | `[nsite;4]` | FK |
| `site_size` | `[nsite;3]` | M5 |
| `gravity` | `[3]` | M2 |
| `timestep` | `[]` | M2, M3 |

`jnt_limited` is written as `0.` or `1.` because every consumer multiplies or
selects with it; MuJoCo stores it as a byte. `timestep` is a 0-d tensor so
that it is one leaf of the parameter structure, like the rest.

## Version policy

A change that adds fields bumps `version`. A reader accepts only its own
version and says so, naming both numbers. The reader is strict about the
order of named fields in v1, because the writer is ours; if a second writer
appears, the order check becomes a set check.

## Regenerating the corpus

The fixtures under `test/fixtures/` are generated, never edited:

```sh
python3 -m venv /tmp/mjcenv
/tmp/mjcenv/bin/pip install mujoco==3.17.0 numpy
/tmp/mjcenv/bin/python tools/record_fixture.py test/fixtures
```

This writes, per model, the source `.xml` (so the fixture can be regenerated
from the repository alone), the `.mjcb` model and the `.mjbf` array fixture,
plus `manifest.json` with the MuJoCo version, the seed and the model sizes.
The OCaml tests read the same files. A MuJoCo version bump regenerates the
corpus in one change and updates the manifest; a test that disagrees with the
corpus is fixed by regenerating, not by editing a value.
