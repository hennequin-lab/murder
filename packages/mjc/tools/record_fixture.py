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
      <joint name="shoulder" type="hinge" axis="0 1 0" pos="0 0 0.1"/>
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
}


def random_qpos(m, rng):
    qpos = np.array(m.qpos0, dtype=np.float64)
    for j in range(m.njnt):
        adr = m.jnt_qposadr[j]
        kind = int(m.jnt_type[j])  # mjJNT_FREE 0, BALL 1, SLIDE 2, HINGE 3
        limited = bool(m.jnt_limited[j])
        lo, hi = (m.jnt_range[j] if limited else (-1.0, 1.0))
        if kind == 3:  # hinge
            qpos[adr] = rng.uniform(lo, hi)
        elif kind == 2:  # slide
            qpos[adr] = rng.uniform(lo / 4 if limited else -0.2,
                                    hi / 4 if limited else 0.2)
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

INTEGRATION_FIELDS = ["integ_qpos", "integ_qvel", "integ_qpos_next"]


def record(name, xml, outdir, rng):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    xml_path = os.path.join(outdir, f"{name}.xml")
    with open(xml_path, "w") as f:
        f.write(xml.lstrip() + "\n")
    mjcb.write_model(m, os.path.join(outdir, f"{name}.mjcb"), name=name)

    states = []
    for _ in range(NSTATES):
        qpos = random_qpos(m, rng)
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        states.append((qpos, {f: np.array(getattr(d, f)) for f in POSE_FIELDS}))

    tensors = {"qpos": np.stack([s[0] for s in states])}
    for f in POSE_FIELDS:
        tensors[f] = np.stack([s[1][f] for s in states])

    # Position integration, from the same states.
    qpos_all, qvel_all, qpos_next = [], [], []
    for qpos, _ in states:
        qvel = rng.uniform(-2.0, 2.0, size=m.nv)
        nxt = np.array(qpos)
        mujoco.mj_integratePos(m, nxt, qvel, DT)
        qpos_all.append(qpos)
        qvel_all.append(qvel)
        qpos_next.append(nxt)
    # The pose fixture is keyed on qpos; integration reuses the same ones.
    tensors["integ_qpos"] = np.stack(qpos_all)
    tensors["integ_qvel"] = np.stack(qvel_all)
    tensors["integ_qpos_next"] = np.stack(qpos_next)

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
