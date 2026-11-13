"""Compare two versions of MuJoCo on the fixture corpus.

This is the script behind `doc/mujoco-4.0.md` section 2: step every model in
`test/fixtures` under whatever MuJoCo is importable, twice, in two
interpreters, and diff the trajectories. It is the answer to "does the release
we just moved to compute the same numbers?" and it is the first thing to run
before re-recording the corpus.

Two modes, one interpreter each:

    # with the engine the corpus was recorded from, and with the candidate
    /tmp/mjcenv/bin/python   tools/cmp_engines.py record test/fixtures > /tmp/a.json
    /tmp/mjcenv40/bin/python tools/cmp_engines.py record test/fixtures > /tmp/b.json

    # any interpreter: diff needs no numpy and no mujoco
    python3 tools/cmp_engines.py diff /tmp/a.json /tmp/b.json

`record` writes the fields the comparison needs, not everything: the sum of
`qpos`, the sum of `qvel`, and the largest `|qacc|` and `|efc_force|` of each
step. The last one is the diagnostic that matters. A model whose constraint
Jacobian is rank deficient gets `efc_D` at rounding level, the row solve
divides by it, and the forces reach 1e16: such a state is not reproducible
between builds and must not be read as a semantic difference. `cyl` is the
corpus's example, and `doc/mujoco-4.0.md` section 2.2 is about it.

This script is development-time only. It is not installed, not linked, and
never runs in CI: `record` needs a MuJoCo install and two of them to be useful.
"""

import glob
import json
import os
import sys

# Fixed so two interpreters generate the same states: every draw is
# `standard_normal`, whose stream is stable across numpy versions, and every
# scale is a literal here rather than a default that could change.
SEEDS = range(3)
STEPS = 200
QPOS_SPREAD = 0.05
QVEL_SPREAD = 0.1
CTRL_SPREAD = 0.2

# Above this, a constraint force is cancellation noise rather than a force.
DEGENERATE_FORCE = 1e12


def run(path):
    """Step one model from SEEDS random states; return the recorded columns."""
    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_setConst(m, d)
    runs = []
    for seed in SEEDS:
        mujoco.mj_resetData(m, d)
        rng = np.random.default_rng(seed)
        d.qpos[:] = np.array(m.qpos0, dtype=np.float64)
        if m.nq:
            d.qpos[:] += QPOS_SPREAD * rng.standard_normal(m.nq)
            # Free joints carry a quaternion; keep it a rotation.
            for j in range(m.njnt):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    a = m.jnt_qposadr[j]
                    q = np.array([1.0, 0.0, 0.0, 0.0]) + QPOS_SPREAD * rng.standard_normal(4)
                    d.qpos[a + 3 : a + 7] = q / np.linalg.norm(q)
        d.qvel[:] = QVEL_SPREAD * rng.standard_normal(m.nv)
        mujoco.mj_forward(m, d)
        steps = []
        for _ in range(STEPS):
            if m.nu:
                d.ctrl[:] = CTRL_SPREAD * rng.standard_normal(m.nu)
            mujoco.mj_step(m, d)
            steps.append(
                [
                    float(np.sum(d.qpos)),
                    float(np.sum(d.qvel)),
                    float(np.max(np.abs(d.qacc))) if m.nv else 0.0,
                    float(np.max(np.abs(d.efc_force))) if d.nefc else 0.0,
                ]
            )
        runs.append(steps)
    return {"nq": m.nq, "nv": m.nv, "runs": runs}


def record(fixtures):
    import mujoco

    models = {}
    for path in sorted(glob.glob(os.path.join(fixtures, "*.xml"))):
        name = os.path.basename(path)[:-4]
        try:
            models[name] = run(path)
        except Exception as exc:  # noqa: BLE001 - a model a version refuses is a result
            models[name] = {"error": f"{type(exc).__name__}: {exc}"}
    json.dump({"mujoco": mujoco.__version__, "models": models}, sys.stdout, indent=1)


def _worst(runs, column):
    """The largest absolute value of one column, over every run and step."""
    return max(
        (abs(step[column]) for run in runs for step in run),
        default=0.0,
    )


def _worst_delta(x, y, column):
    """The largest absolute difference of one column between two recordings."""
    worst = 0.0
    for run_x, run_y in zip(x, y):
        for step_x, step_y in zip(run_x, run_y):
            worst = max(worst, abs(step_x[column] - step_y[column]))
    return worst


def diff(old, new):
    """Print the comparison table. Returns 0, or 1 if a model could not run."""
    a, b = old["models"], new["models"]
    print(f"{old['mujoco']} -> {new['mujoco']}, {STEPS} steps x {len(SEEDS)} seeds")
    print()
    header = (
        f"{'model':12s} {'max|dqacc|':>11s} {'rel d qpos':>11s} {'max|efc_force|':>15s}"
    )
    print(header)
    print("-" * len(header))
    failed = False
    degenerate = []
    for name in sorted(set(a) | set(b)):
        reason = a.get(name, {}).get("error") or b.get(name, {}).get("error")
        if reason:
            print(f"{name:12s} {reason}")
            failed = True
            continue
        x, y = a[name]["runs"], b[name]["runs"]
        if len(x) != len(y) or any(len(p) != len(q) for p, q in zip(x, y)):
            print(f"{name:12s} recorded lengths differ, {STEPS} steps expected")
            failed = True
            continue
        scale = max(1.0, _worst(x, 0))
        force = _worst(x, 3)
        print(
            f"{name:12s} {_worst_delta(x, y, 2):11.3e}"
            f" {_worst_delta(x, y, 0) / scale:11.3e} {force:15.3e}"
        )
        if force > DEGENERATE_FORCE:
            degenerate.append(name)
    print()
    if degenerate:
        print(
            "degenerate, not comparable: " + ", ".join(degenerate) + "\n"
            "the constraint Jacobian is rank deficient there, so the row solve divides by a\n"
            "rounding-level impedance and the forces reach 1e16. Compare contacts, not forces."
        )
    return 1 if failed else 0


def main(argv):
    if len(argv) == 2 and argv[0] == "record":
        record(argv[1])
        return 0
    if len(argv) == 3 and argv[0] == "diff":
        with open(argv[1]) as f:
            old = json.load(f)
        with open(argv[2]) as f:
            new = json.load(f)
        return diff(old, new)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
