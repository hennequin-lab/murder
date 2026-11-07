"""How much the solver's working set changes from one step to the next.

The free block is what the solver actually factors, and it is 3-4 rows against a
21-34 row buffer, so compacting to it is the biggest arithmetic win left -- if it
can be sized in advance. The only thing in advance is the warm start, which is
the previous step's force.

This steps MuJoCo along a recorded trajectory and compares the free set (rows
whose force is strictly inside its bounds) between consecutive steps. Run from
packages/mjc:

    /tmp/mjcenv/bin/python tools/measure_churn.py
"""

import sys

import numpy as np

sys.path.insert(0, "/tmp")
import mujoco  # noqa: E402
import readfix  # noqa: E402


def free_set(d, tol_scale=1e-9):
    if d.nefc == 0:
        return 0, None
    lam = d.efc_force[: d.nefc]
    scale = max(1e-12, float(np.abs(lam).max()))
    return d.nefc, np.abs(lam) > tol_scale * scale


def main(names=("hopper", "walker", "humanoid")):
    for name in names:
        f = readfix.read(f"test/fixtures/{name}.g4.mjbf")
        ctrl, qpos = f["traj_ctrl"], f["traj_qpos"]
        steps = f["traj_nefc"].shape[0] - 1
        m = mujoco.MjModel.from_xml_path(f"test/fixtures/{name}.xml")
        d = mujoco.MjData(m)
        d.qpos[:] = qpos[0]
        prev, inn, out, same, rows = None, [], [], 0, []
        for k in range(steps):
            d.ctrl[:] = ctrl[k]
            mujoco.mj_step(m, d)
            n, free = free_set(d)
            if free is None:
                prev = None
                continue
            rows.append(n)
            if prev is not None and prev[0] == n:
                together = int(np.sum(free & ~prev[1]))
                apart = int(np.sum(~free & prev[1]))
                inn.append(together)
                out.append(apart)
                same += int(together == 0 and apart == 0)
            prev = (n, free)
        inn, out = np.array(inn), np.array(out)
        print(
            f"{name:9s} steps {len(inn):4d} rows med {int(np.median(rows)):2d} "
            f"max {max(rows):2d} | unchanged {100 * same / len(inn):5.1f}% | "
            f"in mean {inn.mean():.2f} max {inn.max():2d} | "
            f"out mean {out.mean():.2f} max {out.max():2d}"
        )


if __name__ == "__main__":
    main()
