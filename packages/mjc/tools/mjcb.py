"""Write the mjc model and fixture formats.

The formats are documented in packages/mjc/doc/mjcb.md. This module is the
development-time side of the interchange: it is not installed, not linked, and
never runs in CI. It exists so that a model and the numbers MuJoCo computes
for it travel from the C engine to OCaml without a shared library.

Both formats are little-endian, name-keyed and self-describing: a reader
checks for the fields it needs rather than their position.
"""

import struct

import numpy as np

MJCB_MAGIC = b"MJCB"
MJCB_END = b"MJCE"
MJBF_MAGIC = b"MJBF"
MJBF_END = b"MJFE"
VERSION = 1

# Topology arrays, in the order the format writes them.
INT_ARRAYS = [
    "body_parentid", "body_jntadr", "body_jntnum", "body_geomadr", "body_geomnum",
    "body_rootid", "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr",
    "dof_bodyid", "dof_jntid", "dof_parentid", "geom_bodyid", "geom_type",
    "geom_contype", "geom_conaffinity", "geom_priority", "site_bodyid",
]

# MuJoCo joint codes to ours: free 0, ball 1, slide 2, hinge 3 becomes
# hinge 0, slide 1, ball 2, free 3.
JNT_CODES = {0: 3, 1: 2, 2: 1, 3: 0}

# Parameter tensors, in the order the format writes them, with their shape.
def fields(m):
    nb, nj, nv, ng, ns, nq = m.nbody, m.njnt, m.nv, m.ngeom, m.nsite, m.nq
    return [
        ("body_pos", m.body_pos), ("body_quat", m.body_quat),
        ("body_ipos", m.body_ipos), ("body_iquat", m.body_iquat),
        ("body_mass", m.body_mass), ("body_inertia", m.body_inertia),
        ("body_subtreemass", m.body_subtreemass),
        ("jnt_axis", m.jnt_axis), ("jnt_pos", m.jnt_pos),
        ("jnt_range", m.jnt_range), ("jnt_limited", m.jnt_limited),
        ("jnt_stiffness", m.jnt_stiffness),
        ("dof_armature", m.dof_armature), ("dof_damping", m.dof_damping),
        ("dof_frictionloss", m.dof_frictionloss),
        ("qpos0", m.qpos0), ("qpos_spring", m.qpos_spring),
        ("geom_pos", m.geom_pos), ("geom_quat", m.geom_quat),
        ("geom_size", m.geom_size), ("geom_rgba", m.geom_rgba),
        ("geom_friction", m.geom_friction), ("geom_margin", m.geom_margin),
        ("geom_gap", m.geom_gap), ("geom_solmix", m.geom_solmix),
        ("geom_solref", m.geom_solref), ("geom_solimp", m.geom_solimp),
        ("site_pos", m.site_pos), ("site_quat", m.site_quat),
        ("site_size", m.site_size),
        ("gravity", m.opt.gravity), ("timestep", np.asarray(m.opt.timestep)),
    ]


def _put_i32(buf, v):
    buf += struct.pack("<i", int(v))


def _put_u16(buf, v):
    buf += struct.pack("<H", int(v))


def _put_f64(buf, v):
    buf += struct.pack("<d", float(v))


def _put_string(buf, s):
    raw = s.encode("utf-8")
    _put_i32(buf, len(raw))
    buf += raw


def _put_ints(buf, xs):
    xs = np.asarray(xs, dtype=np.int32).ravel()
    _put_i32(buf, xs.size)
    buf += xs.astype("<i4").tobytes()


def _put_names(buf, xs):
    _put_i32(buf, len(xs))
    for s in xs:
        _put_string(buf, s)


def _put_tensor(buf, name, x):
    # np.ascontiguousarray promotes a 0-d array to 1-d, which would change the
    # shape a reader reconstructs; flatten for the bytes, keep the shape.
    a = np.asarray(x, dtype="<f8")
    _put_string(buf, name)
    _put_i32(buf, a.ndim)
    for d in a.shape:
        _put_i32(buf, d)
    buf += np.ascontiguousarray(a.reshape(-1)).tobytes()


def _names(m, objtype, n):
    import mujoco

    return [mujoco.mj_id2name(m, objtype, i) or "" for i in range(n)]


def budgets(m):
    """Compile-time budgets for a model, from its size."""
    ng = m.ngeom
    return {
        "nconmax": max(16, 2 * ng),
        "nefcmax": max(64, 4 * ng),
        "max_geom_pairs": max(16, 2 * ng),
    }


def model_bytes(m, nconmax=None, nefcmax=None, max_geom_pairs=None, name=None):
    import mujoco

    b = budgets(m)
    nconmax = b["nconmax"] if nconmax is None else nconmax
    nefcmax = b["nefcmax"] if nefcmax is None else nefcmax
    max_geom_pairs = b["max_geom_pairs"] if max_geom_pairs is None else max_geom_pairs
    buf = bytearray()
    buf += MJCB_MAGIC
    _put_u16(buf, VERSION)
    _put_u16(buf, 0)
    _put_string(buf, name or "model")
    _put_string(buf, mujoco.__version__)
    _put_i32(buf, 14)
    for v in (m.nbody, m.ngeom, m.njnt, m.nsite, m.ntendon, m.nu, m.na,
              m.nsensor, m.nmesh, m.nq, m.nv, nconmax, nefcmax, max_geom_pairs):
        _put_i32(buf, v)
    _put_i32(buf, len(INT_ARRAYS))
    for name in INT_ARRAYS:
        xs = np.asarray(getattr(m, name))
        if name == "jnt_type":
            xs = np.array([JNT_CODES[int(c)] for c in xs], dtype=np.int32)
        _put_ints(buf, xs)
    _put_i32(buf, 4)
    _put_names(buf, _names(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody))
    _put_names(buf, _names(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt))
    _put_names(buf, _names(m, mujoco.mjtObj.mjOBJ_GEOM, m.ngeom))
    _put_names(buf, _names(m, mujoco.mjtObj.mjOBJ_SITE, m.nsite))
    fs = fields(m)
    _put_i32(buf, len(fs))
    for name, x in fs:
        _put_tensor(buf, name, x)
    buf += MJCB_END
    return bytes(buf)


def write_model(m, path, **kwargs):
    with open(path, "wb") as f:
        f.write(model_bytes(m, **kwargs))
    return path


def write_fixture(path, model_name, tensors):
    """Write named tensors, each with a leading state axis."""
    buf = bytearray()
    buf += MJBF_MAGIC
    _put_u16(buf, VERSION)
    _put_u16(buf, 0)
    _put_string(buf, model_name)
    names = list(tensors)
    nstates = np.asarray(tensors[names[0]]).shape[0]
    _put_i32(buf, nstates)
    _put_i32(buf, len(names))
    for name in names:
        a = np.asarray(tensors[name])
        assert a.shape[0] == nstates, f"{name}: {a.shape[0]} states, expected {nstates}"
        _put_tensor(buf, name, a)
    buf += MJBF_END
    with open(path, "wb") as f:
        f.write(buf)
    return path
