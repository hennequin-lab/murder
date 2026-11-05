"""Write the mjc model and fixture formats.

The formats are documented in packages/mjc/doc/mjcb.md. This module is the
development-time side of the interchange: it is not installed, not linked, and
never runs in CI. It exists so that a model and the numbers MuJoCo computes
for it travel from the C engine to OCaml without a shared library.

Both formats are little-endian, name-keyed and self-describing: a reader
checks for the fields it needs rather than their position.
"""

import struct

import mujoco


import numpy as np

MJCB_MAGIC = b"MJCB"
MJCB_END = b"MJCE"
MJBF_MAGIC = b"MJBF"
MJBF_END = b"MJFE"
VERSION = 13

# Topology arrays, in the order the format writes them.
INT_ARRAYS = [
    "body_parentid", "body_jntadr", "body_jntnum", "body_geomadr", "body_geomnum",
    "body_rootid", "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr",
    "dof_bodyid", "dof_jntid", "dof_parentid", "geom_bodyid", "geom_type",
    "geom_contype", "geom_conaffinity", "geom_priority", "site_bodyid",
    "eq_type", "eq_objtype",
    # v1 actuator surface: what each actuator drives and how, plus the codes
    # for the three gain/bias/dynamics kinds it supports.
    "actuator_trntype", "actuator_trnid", "actuator_gaintype",
    "actuator_biastype", "actuator_dyntype", "actuator_actadr",
    "actuator_actearly",
    # fixed tendons: a path of wrapped joints, and the limit flags
    "sensor_type", "sensor_objtype", "sensor_objid", "sensor_adr", "sensor_dim", "sensor_refid",
    "tendon_adr", "tendon_num", "wrap_type", "wrap_objid",
    # MuJoCo keeps the tendon limit flag as a byte; ours goes in the same
    # integer block as the other flags.
    "ten_limited",
    # `<exclude>` pairs. MuJoCo stores only a packed signature, so the two ids
    # are written out separately: (hi << 16) + lo.
    "exclude_body1", "exclude_body2",
]

# MuJoCo joint codes to ours: free 0, ball 1, slide 2, hinge 3 becomes
# hinge 0, slide 1, ball 2, free 3.
JNT_CODES = {0: 3, 1: 2, 2: 1, 3: 0}

# MuJoCo's transmission codes to ours: joint 0, tendon 3 becomes joint 0,
# tendon 1. The rest are not v1, and a model that asks for one is refused
# rather than written with a code the engine would misread.
TRN_CODES = {0: 0, 3: 1}

# MuJoCo's wrap codes: none 0, joint 1, pulley 2, site 3, sphere 4,
# cylinder 5. Ours are joint 0, site 1.
WRAP_CODES = {1: 0, 3: 1}

# Sensor kinds v1 has, by MuJoCo's name so the table cannot drift with the
# release. A kind that is not here is written as 99 and refused by the reader.
SENSOR_KINDS = [
    "mjSENS_JOINTPOS", "mjSENS_JOINTVEL", "mjSENS_FRAMEPOS", "mjSENS_FRAMEQUAT",
    "mjSENS_FRAMELINVEL", "mjSENS_FRAMEANGVEL", "mjSENS_ACTUATORFRC",
    "mjSENS_TENDONPOS", "mjSENS_TENDONVEL", "mjSENS_SUBTREECOM",
    "mjSENS_SUBTREELINVEL", "mjSENS_SUBTREEANGMOM",
]
# The object a sensor measures, by MuJoCo's name.
OBJ_CODES = {
    int(mujoco.mjtObj.mjOBJ_BODY): 0,      # the inertial frame: xipos, ximat
    int(mujoco.mjtObj.mjOBJ_XBODY): 1,     # the body frame: xpos, xquat
    int(mujoco.mjtObj.mjOBJ_SITE): 2,
    int(mujoco.mjtObj.mjOBJ_JOINT): 3,
    int(mujoco.mjtObj.mjOBJ_ACTUATOR): 4,
    int(mujoco.mjtObj.mjOBJ_TENDON): 5,
}
SENSOR_CODES = {int(getattr(mujoco.mjtSensor, k)): i for i, k in enumerate(SENSOR_KINDS)}

# Parameter tensors, in the order the format writes them, with their shape.
def fields(m):
    nb, nj, nv, ng, ns, nq = m.nbody, m.njnt, m.nv, m.ngeom, m.nsite, m.nq
    return [
        ("body_pos", m.body_pos), ("body_quat", m.body_quat),
        ("body_ipos", m.body_ipos), ("body_iquat", m.body_iquat),
        ("body_mass", m.body_mass), ("body_inertia", m.body_inertia),
        ("body_subtreemass", m.body_subtreemass),
        ("body_invweight0", m.body_invweight0),
        ("jnt_axis", m.jnt_axis), ("jnt_pos", m.jnt_pos),
        ("jnt_range", m.jnt_range), ("jnt_limited", m.jnt_limited),
        ("jnt_stiffness", m.jnt_stiffness),
        ("jnt_margin", m.jnt_margin), ("jnt_solref", m.jnt_solref),
        ("jnt_solimp", m.jnt_solimp),
        ("dof_armature", m.dof_armature), ("dof_damping", m.dof_damping),
        ("dof_frictionloss", m.dof_frictionloss),
        ("dof_solref", m.dof_solref), ("dof_solimp", m.dof_solimp),
        ("dof_invweight0", m.dof_invweight0),
        ("qpos0", m.qpos0), ("qpos_spring", m.qpos_spring),
        ("geom_pos", m.geom_pos), ("geom_quat", m.geom_quat),
        ("geom_size", m.geom_size),
        ("geom_condim", m.geom_condim), ("geom_rbound", m.geom_rbound),
        ("geom_rgba", m.geom_rgba),
        ("geom_friction", m.geom_friction), ("geom_margin", m.geom_margin),
        ("geom_gap", m.geom_gap), ("geom_solmix", m.geom_solmix),
        ("geom_solref", m.geom_solref), ("geom_solimp", m.geom_solimp),
        ("site_pos", m.site_pos), ("site_quat", m.site_quat),
        ("site_size", m.site_size),
        ("eq_obj1id", m.eq_obj1id), ("eq_obj2id", m.eq_obj2id),
        ("eq_data", m.eq_data), ("eq_solref", m.eq_solref),
        ("eq_solimp", m.eq_solimp),
        ("eq_active0", np.asarray(m.eq_active0, dtype=np.float64)),
        # An empty table comes back one-dimensional, so the shape is stated
        # rather than inferred: the reader checks it.
        ("wrap_prm", np.asarray(m.wrap_prm).reshape(m.nwrap)),
        ("ten_range", np.asarray(m.tendon_range).reshape(m.ntendon, 2)),
        ("ten_margin", np.asarray(m.tendon_margin).reshape(m.ntendon)),
        ("ten_invweight0", np.asarray(m.tendon_invweight0).reshape(m.ntendon)),
        ("ten_solref", np.asarray(m.tendon_solref_lim).reshape(m.ntendon, 2)),
        ("ten_solimp", np.asarray(m.tendon_solimp_lim).reshape(m.ntendon, 5)),
        ("actuator_gear", m.actuator_gear),
        # MuJoCo 3.17 keeps ten coefficients per actuator for DC motors and
        # LuGre friction; v1's gain, bias and dynamics polynomials are the
        # first three of each, which is the part that has been stable across
        # releases and the part this engine reads.
        ("actuator_gainprm", np.asarray(m.actuator_gainprm)[:, :3]),
        ("actuator_biasprm", np.asarray(m.actuator_biasprm)[:, :3]),
        ("actuator_dynprm", np.asarray(m.actuator_dynprm)[:, :3]),
        ("actuator_ctrlrange", m.actuator_ctrlrange),
        ("actuator_forcerange", m.actuator_forcerange),
        ("actuator_ctrllimited", np.asarray(m.actuator_ctrllimited, dtype=np.float64)),
        ("actuator_forcelimited", np.asarray(m.actuator_forcelimited, dtype=np.float64)),
        ("gravity", m.opt.gravity), ("timestep", np.asarray(m.opt.timestep)),
        ("integrator", np.asarray(m.opt.integrator, dtype=np.float64)),
        # MuJoCo's friction cone: 0 pyramidal, 1 elliptic. Recorded so that a
        # model asking for the one v1 does not implement is refused by name
        # instead of being silently run with the other.
        ("cone", np.asarray(m.opt.cone, dtype=np.float64)),
        ("disableflags", np.asarray(m.opt.disableflags, dtype=np.float64)),
        ("iterations", np.asarray(m.opt.iterations)),
        ("tolerance", np.asarray(m.opt.tolerance)),
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


# Rows an equality constraint contributes, by MuJoCo's eq_type code.
EQ_ROWS = {0: 3, 1: 6, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0}


def budgets(m, nconmax=None):
    """Compile-time budgets for a model, from its size.

    The row budget covers what v1 can instantiate: two sides for every hinge
    or slide limit, the equality rows, and one friction row per dof. The
    contact budget is a guess from the geom count, which is wrong as soon as a
    box pair makes a manifold — a recorder that has seen the model's contacts
    passes the count it measured here.
    """
    ng = m.ngeom
    limited = sum(1 for t, on in zip(m.jnt_type, m.jnt_limited)
                  if int(t) in (2, 3) and on)
    eq = sum(EQ_ROWS.get(int(t), 0) for t in m.eq_type)
    friction = sum(1 for d in m.dof_frictionloss if d > 0)
    nconmax = max(8, 2 * ng) if nconmax is None else nconmax
    # A model that disables contacts, or has nothing to collide with, pays no
    # row budget for them: the dense factor is quadratic in this number.
    disabled = bool(m.opt.disableflags & (1 << 4))
    condim = int(max(m.geom_condim)) if m.ngeom else 1
    rows_per_contact = 1 if condim <= 1 else 2 * (condim - 1)
    contact_rows = 0 if disabled or ng < 2 else nconmax * rows_per_contact
    return {
        "nconmax": nconmax,
        "nefcmax": max(4, 2 * limited + eq + friction + contact_rows),
        "max_geom_pairs": max(16, 2 * ng),
    }


def model_bytes(m, nconmax=None, nefcmax=None, max_geom_pairs=None, name=None):
    # an explicit contact budget has to drive the row budget too

    b = budgets(m, nconmax)
    nconmax = b["nconmax"] if nconmax is None else nconmax
    nefcmax = b["nefcmax"] if nefcmax is None else nefcmax
    max_geom_pairs = b["max_geom_pairs"] if max_geom_pairs is None else max_geom_pairs
    buf = bytearray()
    buf += MJCB_MAGIC
    _put_u16(buf, VERSION)
    _put_u16(buf, 0)
    _put_string(buf, name or "model")
    _put_string(buf, mujoco.__version__)
    _put_i32(buf, 17)
    for v in (m.nbody, m.ngeom, m.njnt, m.neq, m.nsite, m.ntendon, m.nu, m.na,
              m.nsensor, m.nmesh, m.nq, m.nv, nconmax, nefcmax, max_geom_pairs,
              m.nwrap, m.nsensordata):
        _put_i32(buf, v)
    _put_i32(buf, len(INT_ARRAYS))
    # The names this format uses for MuJoCo's tables, where they differ.
    SOURCE = {"ten_limited": "tendon_limited"}
    # The exclusion pairs unpack from one signature each.
    exclude1 = [int(sig) >> 16 for sig in m.exclude_signature]
    exclude2 = [int(sig) & 0xFFFF for sig in m.exclude_signature]
    for name in INT_ARRAYS:
        if name == "exclude_body1":
            xs = np.asarray(exclude1, dtype=np.int32)
        elif name == "exclude_body2":
            xs = np.asarray(exclude2, dtype=np.int32)
        else:
            xs = np.asarray(getattr(m, SOURCE.get(name, name)))
        if name == "jnt_type":
            xs = np.array([JNT_CODES[int(c)] for c in xs], dtype=np.int32)
        elif name == "actuator_trntype":
            xs = np.array([TRN_CODES[int(c)] for c in xs], dtype=np.int32)
        elif name == "wrap_type":
            xs = np.array([WRAP_CODES.get(int(c), 99) for c in xs], dtype=np.int32)
        elif name == "sensor_type":
            xs = np.array([SENSOR_CODES.get(int(c), 99) for c in xs], dtype=np.int32)
        elif name == "sensor_objtype":
            xs = np.array([OBJ_CODES.get(int(c), 99) for c in xs], dtype=np.int32)
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
