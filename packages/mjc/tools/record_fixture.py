"""Record the mjc parity corpus from MuJoCo.

Run with the development MuJoCo install:

    /tmp/mjcenv/bin/python tools/record_fixture.py test/fixtures

For each model in MODELS it writes

    <name>.xml    the source, committed so a fixture can be regenerated
    <name>.mjcb   the compiled model the OCaml reader consumes
    <name>.mjbf   the arrays MuJoCo computes, for the parity tests

and a manifest recording the MuJoCo version and the seed. Regenerating the
corpus is the only way to change it: the fixture files are generated, never
edited.
"""

import hashlib
import json
import os
import sys

import numpy as np

import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mjcb  # noqa: E402

SEED = 20261103
NSTATES = 32
DT = 1e-3

MODELS = {
    # One hinge about y, an offset body, a rotated geom and a site.
    "pendulum": """
<mujoco model="pendulum">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="arm" pos="0 0 0.5" quat="0.9238795 0.3826834 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0" pos="0 0 0.1" stiffness="2" damping="0.3"/>
      <geom name="arm" type="capsule" size="0.02 0.3" pos="0 0 -0.3"
            quat="0.7071068 0 0.7071068 0" rgba="0.8 0.2 0.2 1"/>
      <site name="tip" pos="0 0 -0.6"/>
    </body>
  </worldbody>
</mujoco>""",
    # Six hinges on one chain, alternating axes, to exercise the topological
    # order and the composed rotations.
    "chain6": """
<mujoco model="chain6">
  <option timestep="0.001"/>
  <worldbody>
    <body name="l0" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom name="g0" type="capsule" size="0.03 0.15" pos="0 0 -0.15"/>
      <body name="l1" pos="0 0 -0.3" quat="0.9659258 0.258819 0 0">
        <joint name="j1" type="hinge" axis="1 0 0"/>
        <geom name="g1" type="capsule" size="0.03 0.15" pos="0 0 -0.15"/>
        <body name="l2" pos="0 0 -0.3">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom name="g2" type="capsule" size="0.03 0.15" pos="0 0 -0.15"/>
          <body name="l3" pos="0 0 -0.3">
            <joint name="j3" type="slide" axis="0 0 1"/>
            <geom name="g3" type="capsule" size="0.03 0.15" pos="0 0 -0.15"/>
            <body name="l4" pos="0 0 -0.3" quat="0.9238795 0 0.3826834 0">
              <joint name="j4" type="hinge" axis="0 0 1"/>
              <geom name="g4" type="capsule" size="0.03 0.15" pos="0 0 -0.15"/>
              <body name="l5" pos="0 0 -0.3">
                <joint name="j5" type="hinge" axis="1 0 0"/>
                <geom name="g5" type="capsule" size="0.03 0.15" pos="0 0 -0.15"/>
                <site name="tool" pos="0 0 -0.3"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>""",
    # A free joint: the body's pose comes straight from qpos, including the
    # normalization of a quaternion that has drifted off the unit sphere.
    "floating": """
<mujoco model="floating">
  <option timestep="0.001"/>
  <worldbody>
    <body name="box">
      <freejoint name="root"/>
      <geom name="box" type="box" size="0.1 0.2 0.3" pos="0 0 0.1"
            quat="0.9238795 0.3826834 0 0" rgba="0.2 0.6 0.8 1"/>
      <site name="mark" pos="0.1 0 0.3"/>
    </body>
  </worldbody>
</mujoco>""",
    # A ball joint with an off-centre anchor: the off-centre correction is
    # where a sign error shows.
    "ball": """
<mujoco model="ball">
  <option timestep="0.001"/>
  <worldbody>
    <body name="ball" pos="0.1 0.2 0.3" quat="0.9659258 0 0 0.258819">
      <joint name="hinge3" type="ball" pos="0.05 0 -0.05"/>
      <geom name="sphere" type="sphere" size="0.12" pos="0 0 0.2" rgba="0.6 0.8 0.2 1"/>
      <body name="child" pos="0.2 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1"/>
        <geom name="bar" type="capsule" size="0.02 0.1" pos="0 0 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>""",
    # Joint limits, both sides, on a hinge and a slide; the sampled states
    # leave the ranges so that the limit rows are exercised.
    # The same pendulum under implicitfast: damping is inside the velocity
    # solve whether or not any dof is damped, so the trajectory is the
    # integrator's, not Euler's, whenever passive forces depend on velocity.
    "fast": """
<mujoco model="fast">
  <option timestep="0.001" integrator="implicitfast"/>
  <worldbody>
    <body name="arm" pos="0 0 0.5" quat="0.9238795 0.3826834 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0" pos="0 0 0.1" stiffness="2"
             damping="0.3"/>
      <geom name="arm" type="capsule" size="0.02 0.3" pos="0 0 -0.3"/>
    </body>
    <body name="wheel" pos="0.4 0 0.3">
      <joint name="spin" type="hinge" axis="0 1 0" damping="4"/>
      <geom name="wheel" type="capsule" fromto="0.1 0 0 0.3 0 0" size="0.05"/>
    </body>
  </worldbody>
</mujoco>""",
    # Frictionless contacts: one row each, which is what the solver handles
    # until the friction cone is in.
    "slide1": """
<mujoco model="slide1">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="1"/>
    <body name="ball" pos="0 0 0.1">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.12" condim="1"/>
    </body>
    <body name="pebble" pos="0.3 0.1 0.07">
      <freejoint/>
      <geom name="pebble" type="sphere" size="0.09" condim="1"/>
    </body>
    <body name="rod" pos="-0.35 0 0.16" quat="0.9238795 0.3826834 0 0">
      <freejoint/>
      <geom name="rod" type="capsule" fromto="0 0 -0.18 0 0 0.18" size="0.05"
            condim="1"/>
    </body>
  </worldbody>
</mujoco>""",
    # Contacts of every pair v1 computes: two spheres against a plane, a
    # capsule against a plane (two endpoint contacts) and the spheres against
    # each other and against the capsule.
    "bounce": """
<mujoco model="bounce">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 0.5">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.1"/>
    </body>
    <body name="tip" pos="0.4 0 0.5">
      <freejoint/>
      <geom name="tip" type="sphere" size="0.08"/>
    </body>
    <body name="rod" pos="-0.4 0 0.5" quat="0.9238795 0.3826834 0 0">
      <freejoint/>
      <geom name="rod" type="capsule" fromto="0 0 -0.2 0 0 0.2" size="0.05"/>
    </body>
  </worldbody>
</mujoco>""",
    # One dof, one friction row, and a loss large enough that the row stays in
    # the quadratic branch: the complement of the saturated case above.
    "drag": """
<mujoco model="drag">
  <option timestep="0.002"/>
  <worldbody>
    <body name="blade" pos="0 0 0.5">
      <joint name="spin" type="hinge" axis="0 1 0" frictionloss="50"/>
      <geom name="blade" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02"/>
    </body>
  </worldbody>
</mujoco>""",
    "limited": """
<mujoco model="limited">
  <compiler angle="radian"/>
  <!-- Contacts are M4's corpus; disabling them keeps the row count a function
       of the friction and limit rows this model measures. -->
  <option timestep="0.002">
    <flag contact="disable"/>
  </option>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-0.4 0.4"
             frictionloss="0.4"/>
      <geom name="arm" type="capsule" size="0.02 0.3" pos="0 0 -0.3"/>
      <body name="slider" pos="0 0 -0.6">
        <joint name="extend" type="slide" axis="1 0 0" range="-0.1 0.15"
               frictionloss="5"/>
        <geom name="hand" type="sphere" size="0.05"/>
      </body>
    </body>

  </worldbody>
</mujoco>""",
    # Equality rows of all three shapes, each in its own subtree, plus one
    # hinge whose spring drives it past its range so limit rows share the
    # buffer. See the model's own comment.
    "looped": open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "test", "fixtures", "looped.xml")).read(),
    # The same corpus with a friction cone: every geom asks for condim 3, so
    # each contact becomes four one-sided rows, the two opposing edges of the
    # pyramid per friction axis.
    # A sphere resting against a box's side: the sphere-box narrowphase,
    # alongside plane-box and plane-sphere in the same pose.
    "blocks": """
<mujoco model="blocks">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="1"/>
    <body name="crate" pos="0 0 0.09">
      <freejoint/>
      <geom name="crate" type="box" size="0.2 0.2 0.1" condim="1"/>
    </body>
    <body name="ball" pos="0.24 0 0.16">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.09" condim="1"/>
    </body>
  </worldbody>
</mujoco>""",
    "friction3": """
<mujoco model="friction3">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="3"/>
    <body name="ball" pos="0 0 0.1">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.12" condim="3" friction="0.7 0.005 0.0001"/>
    </body>
    <body name="rod" pos="-0.5 0.1 0.16" quat="0.9238795 0.3826834 0 0">
      <freejoint/>
      <geom name="rod" type="capsule" fromto="0 0 -0.18 0 0 0.18" size="0.05"
            condim="3" friction="0.4 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>""",    # A box on a plane: up to four contacts from one pair, kept in corner
    # order, which is what the per-pair cap has to reproduce.
    # Box on box: a slab under two boxes, one stacked squarely (a face manifold
    # whose clip keeps four corners), one turned 15 degrees (whose clip makes
    # six or eight), and one dropped on to a corner so the trajectory passes
    # through edge-edge contacts. Every contact here is box-box.
    "boxes2": """
<mujoco model="boxes2">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="box" size="2 2 0.1" pos="0 0 -0.1" condim="3"/>
    <body name="base" pos="0 0 0.25">
      <freejoint/>
      <geom name="base" type="box" size="0.25 0.25 0.15" condim="3"/>
    </body>
    <body name="stack" pos="0.02 0.01 0.5">
      <freejoint/>
      <geom name="stack" type="box" size="0.2 0.2 0.1" condim="3"/>
    </body>
    <body name="turned" pos="0.75 0 0.2" quat="0.9914449 0 0 0.1305262">
      <freejoint/>
      <geom name="turned" type="box" size="0.15 0.15 0.06" condim="3"/>
    </body>
    <body name="tipped" pos="-0.7 0.1 0.55" quat="0.9238795 0.3826834 0 0">
      <freejoint/>
      <geom name="tipped" type="box" size="0.12 0.12 0.12" condim="3"/>
    </body>
    <!-- overhangs the slab's edge and turned 30 degrees: the clip of its face
         against the slab's side planes makes more than four vertices -->
    <body name="overhang" pos="1.9 0 0.16" quat="0.9659258 0 0 0.258819">
      <freejoint/>
      <geom name="overhang" type="box" size="0.2 0.2 0.06" condim="3"/>
    </body>
  </worldbody>
</mujoco>""",
    "boxes": """
<mujoco model="boxes">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" condim="1"/>
    <body name="crate" pos="0 0 0.09" quat="0.9238795 0.3826834 0 0">
      <freejoint/>
      <geom name="crate" type="box" size="0.2 0.15 0.1" condim="1"/>
    </body>
  </worldbody>
</mujoco>""",
}


def random_qpos(m, rng):
    qpos = np.array(m.qpos0, dtype=np.float64)
    for j in range(m.njnt):
        adr = m.jnt_qposadr[j]
        kind = int(m.jnt_type[j])  # mjJNT_FREE 0, BALL 1, SLIDE 2, HINGE 3
        limited = bool(m.jnt_limited[j])
        lo, hi = (m.jnt_range[j] if limited else (-1.0, 1.0))
        # A limited joint is sampled past its range half the time, so that the
        # constraint rows are exercised rather than always satisfied.
        pad = 0.6 * (hi - lo) if limited else 0.0
        if kind == 3:  # hinge
            qpos[adr] = rng.uniform(lo - pad, hi + pad)
        elif kind == 2:  # slide
            if limited:
                qpos[adr] = rng.uniform(lo - (0.6 * (hi - lo)), hi + (0.6 * (hi - lo)))
            else:
                qpos[adr] = rng.uniform(-0.2, 0.2)
        elif kind == 1:  # ball
            q = rng.normal(size=4)
            qpos[adr:adr + 4] = q / np.linalg.norm(q)
        elif kind == 0:  # free
            qpos[adr:adr + 3] = rng.uniform(-0.5, 0.5, size=3)
            q = rng.normal(size=4)
            qpos[adr + 3:adr + 7] = q / np.linalg.norm(q)
    return qpos


POSE_FIELDS = [
    "xpos", "xquat", "xanchor", "xaxis", "xipos", "ximat",
    "geom_xpos", "geom_xmat", "site_xpos", "site_xmat",
]

# Constraint rows, which the C engine sizes at run time; the fixture pads them
# to the widest state and records the count separately.
EFC_FIELDS = ["efc_type", "efc_id", "efc_pos", "efc_margin", "efc_J",
              "efc_vel", "efc_R", "efc_D", "efc_aref", "efc_b", "efc_force"]

# Contacts, padded to the widest state like the rows are. The normal is the
# first three components of the contact frame.
CONTACT_FIELDS = {
    "contact_dist": 1, "contact_pos": 3, "contact_normal": 3,
    "contact_frame": 9,
    "contact_geom": 2, "contact_body": 2, "contact_dim": 1,
    "contact_includemargin": 1, "contact_friction": 5,
    "contact_solref": 2, "contact_solimp": 5,
}


def contact_state(m, d):
    """The contact arrays of one state, as a dict of rows.

    The binding hands back one MjContact object per contact, so the fields are
    gathered one at a time.
    """
    cs = [d.contact[i] for i in range(d.ncon)]
    def gather(field, width):
        if not cs:
            return np.zeros((0, width))
        rows = [np.asarray(getattr(c, field), dtype=float).reshape(-1) for c in cs]
        return np.array([r[:width] for r in rows])

    geom = gather("geom", 2)
    body = m.geom_bodyid[geom.astype(int)] if len(cs) else np.zeros((0, 2))
    return {
        "contact_dist": gather("dist", 1),
        "contact_pos": gather("pos", 3),
        "contact_normal": gather("frame", 3),
        "contact_frame": gather("frame", 9),
        "contact_geom": geom,
        "contact_body": np.array(body, dtype=float).reshape(-1, 2),
        "contact_dim": gather("dim", 1),
        "contact_includemargin": gather("includemargin", 1),
        "contact_friction": gather("friction", 5),
        "contact_solref": gather("solref", 2),
        "contact_solimp": gather("solimp", 5),
    }


# Written by mj_forward after the pose: the position stage's centre of mass,
# centre-of-mass inertia and motion subspaces, then the velocity stage's
# spatial velocities and the forces it leaves behind.
# MuJoCo has no separate global centre of mass in 3.17: subtree_com[0] is it.
STAGE_FIELDS = [
    "subtree_com", "cinert", "cdof", "cvel", "cdof_dot",
    "qfrc_bias", "qfrc_passive", "qfrc_smooth",
]

TRAJ_STEPS = 10


def record(name, xml, outdir, rng):
    m = mujoco.MjModel.from_xml_string(xml)
    # The port is dense; record the rows in the layout the port uses, whatever
    # the model's own auto choice would be at this size.
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    d = mujoco.MjData(m)
    xml_path = os.path.join(outdir, f"{name}.xml")
    with open(xml_path, "w") as f:
        f.write(xml.lstrip() + "\n")

    states = []
    qvels = []
    cons = []
    for _ in range(NSTATES):
        qpos = random_qpos(m, rng)
        qvel = rng.uniform(-2.0, 2.0, size=m.nv)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        mujoco.mj_forward(m, d)
        full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, full)
        fields = {f: np.array(getattr(d, f)) for f in POSE_FIELDS + STAGE_FIELDS}
        fields["mass_matrix"] = full
        nefc = int(d.nefc)
        efc = {"nefc": nefc}
        for f in EFC_FIELDS:
            if f == "efc_J":
                v = np.array(d.efc_J)[: nefc * m.nv].reshape(nefc, m.nv)
            else:
                v = np.array(getattr(d, f))[:nefc]
            efc[f] = v
        states.append((qpos, fields, efc))
        cons.append(contact_state(m, d))
        qvels.append(qvel)

    tensors = {"qpos": np.stack([s[0] for s in states])}
    tensors["qvel"] = np.stack(qvels)
    for f in POSE_FIELDS + STAGE_FIELDS + ["mass_matrix"]:
        tensors[f] = np.stack([s[1][f] for s in states])
    width = max(1, max(s[2]["nefc"] for s in states))
    tensors["nefc"] = np.array([s[2]["nefc"] for s in states], dtype=float)
    for f in EFC_FIELDS:
        rows = [np.asarray(s[2][f], dtype=float) for s in states]
        shape = (width,) if not f == "efc_J" else (width, m.nv)
        padded = np.zeros((len(states),) + shape)
        for i, r in enumerate(rows):
            if f == "efc_J":
                padded[i, : r.shape[0], :] = r
            else:
                padded[i, : r.shape[0]] = r
        tensors[f] = padded

    ncon_width = max(1, max(len(c["contact_dist"]) for c in cons))
    tensors["ncon"] = np.array([len(c["contact_dist"]) for c in cons], dtype=float)
    for field, width in CONTACT_FIELDS.items():
        padded = np.zeros((len(cons), ncon_width, width))
        for i, c in enumerate(cons):
            r = np.asarray(c[field], dtype=float).reshape(-1, width)
            padded[i, : r.shape[0], :] = r
        # A one-column field is a vector of rows, not a matrix of one column.
        tensors[field] = padded[:, :, 0] if width == 1 else padded

    # Position integration, from the same states and velocities.
    qpos_next = []
    for qpos, qvel in zip([s[0] for s in states], qvels):
        nxt = np.array(qpos)
        mujoco.mj_integratePos(m, nxt, qvel, DT)
        qpos_next.append(nxt)
    tensors["integ_qpos"] = tensors["qpos"]
    tensors["integ_qvel"] = tensors["qvel"]
    tensors["integ_qpos_next"] = np.stack(qpos_next)

    # A short trajectory through mj_step, from the first state, with no
    # controls: this is the end-to-end test of forward plus the Euler
    # integrator.
    d.qpos[:] = states[0][0]
    d.qvel[:] = qvels[0]
    mujoco.mj_forward(m, d)
    traj_qpos = [np.array(d.qpos)]
    traj_qvel = [np.array(d.qvel)]
    traj_qacc = []
    for _ in range(TRAJ_STEPS):
        # mj_step leaves qacc as it was before the step, so recompute it at
        # the state it belongs to.
        mujoco.mj_forward(m, d)
        traj_qacc.append(np.array(d.qacc))
        mujoco.mj_step(m, d)
        traj_qpos.append(np.array(d.qpos))
        traj_qvel.append(np.array(d.qvel))
    traj_qacc.append(np.array(d.qacc))
    mjcb.write_fixture(
        os.path.join(outdir, f"{name}.traj.mjbf"),
        name,
        {
            "traj_qpos": np.stack(traj_qpos),
            "traj_qvel": np.stack(traj_qvel),
            "traj_qacc": np.stack(traj_qacc),
        },
    )

    # The engine's contact budget is static, and a box pair fills a manifold
    # rather than a slot: the contacts this model actually made are the only
    # honest source for it. Twice the maximum is slack for a longer run.
    recorded = int(max(tensors["ncon"])) if len(tensors["ncon"]) else 0
    budget = max(mjcb.budgets(m)["nconmax"], 2 * recorded)
    mjcb.write_model(
        m, os.path.join(outdir, f"{name}.mjcb"), name=name, nconmax=budget
    )
    mjcb.write_fixture(os.path.join(outdir, f"{name}.mjbf"), name, tensors)
    return {
        "name": name,
        "nbody": int(m.nbody),
        "ngeom": int(m.ngeom),
        "njnt": int(m.njnt),
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nstates": NSTATES,
        "xml_sha256": hashlib.sha256(xml.encode()).hexdigest()[:16],
    }


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "test/fixtures"
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    entries = [record(name, xml, outdir, rng) for name, xml in MODELS.items()]
    manifest = {
        "generator": "tools/record_fixture.py",
        "mujoco": mujoco.__version__,
        "seed": SEED,
        "dt": DT,
        "models": entries,
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest["models"], indent=2))


if __name__ == "__main__":
    main()
