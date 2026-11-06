"""Wall time per mj_step, for the engines' shared models.

The mjc counterpart is `bench/perf.ml`, which times the OCaml engine's step on
the same .mjcb models from the same state: this model's reference pose, no
velocity, no control, warm-up, then a timed loop, one thread. Not shared:
the solver's schedule (MuJoCo's Newton exits on tolerance, theirs runs a fixed
budget) and the machine serialisation details, both documented in `doc/perf.md`.

    /tmp/mjcenv/bin/python packages/mjc/tools/perf_mujoco.py \
      packages/mjc/test/fixtures hopper walker humanoid ...

Outputs a markdown table one row per model.
"""

import sys
import time

import mujoco


def main():
    fixture_dir = sys.argv[1] if len(sys.argv) > 1 else "packages/mjc/test/fixtures"
    models = sys.argv[2:] or ["pendulum", "chain6", "bounce", "boxes2",
                              "hopper", "walker", "humanoid"]
    steps = 500
    warmup = 50
    print("| model | nv | nefc | dt | mj_step |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name in models:
        m = mujoco.MjModel.from_xml_path(f"{fixture_dir}/{name}.xml")
        d = mujoco.MjData(m)
        # The reference pose, at rest, same state the OCaml bench starts from.
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        if m.na:
            d.act[:] = 0.0
        for _ in range(warmup):
            mujoco.mj_step(m, d)
        nefc = int(d.nefc)
        t0 = time.perf_counter()
        for _ in range(steps):
            mujoco.mj_step(m, d)
        per = (time.perf_counter() - t0) / steps
        print(f"| {name} | {m.nv} | {nefc} | {m.opt.timestep:.3g} | {per * 1e6:.0f} us |")
    print(f"The solver budgets the models ask for: "
          f"iterations={m.opt.iterations} tolerance={m.opt.tolerance} "
          f"integrator={m.opt.integrator}, cone={m.opt.cone} (last model)")


if __name__ == "__main__":
    main()
