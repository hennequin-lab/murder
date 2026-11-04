# A demo model

Something to look at: a five-link chain that swings, three spheres dropped from
different heights, and a box resting on the floor. Ten geoms, 29 degrees of
freedom, contacts on.

## The window

    cd /home/hennequi/code/modern/murder-wt/mjc
    dune exec packages/mjc/viz/window/main.exe -- \
      --model packages/mjc/viz/demo/demo.mjcb --steps-per-frame 8 \
      --eye 2.0,-2.8,1.7 --target 0,0,1.0

`--steps-per-frame 8` runs the simulation at about real time (8 × 2 ms per
frame at 60 fps). Leave it off to watch in slow motion, which is the useful
setting while debugging a contact. Without `--eye`/`--target` the window frames
the whole model the way the still backend does, which for a 2.2 m chain leaves
it small; the two above are a closer view.

Keys: `space` pause, `.` single step, `r` reset, `c` camera, `g` grid,
`f` body frames, `n` contacts, `x` constraint forces, `i` inertia boxes,
`m` centres of mass, `j` joint axes, `v` velocities, `e` energy, `h` HUD,
`1`–`9` select a joint and `[`/`]` move it, `0` zeroes it, `esc` quits.
Right-drag orbits, left-drag pans, the wheel zooms.

Joint control is the forward-kinematics tool: press `1`, then `]` a few times
and the first link of the chain turns while the rest follows.

## A still, no window

    dune exec packages/mjc/viz/bin/mjc_view.exe -- \
      --model packages/mjc/viz/demo/demo.mjcb --steps 120 \
      --frames --contacts --inertia --com --shot demo.png

## What the model asks of the engine

The touching pairs are plane-box, plane-sphere, sphere-sphere and sphere-box —
all implemented. The chain carries `contype="0" conaffinity="0"` so it takes
part in no contacts at all: capsule-box is not implemented yet, and the
engine's unimplemented-pair guard fires on any pair the masks allow, whether or
not the two geoms are anywhere near each other (see `doc/log.md`).

`demo.mjcb` is generated from `demo.xml` by `tools/record_fixture.py`'s writer:

    /tmp/mjcenv/bin/python -c "import sys, mujoco; sys.path.insert(0,'tools'); \
      import mjcb; m = mujoco.MjModel.from_xml_path('viz/demo/demo.xml'); \
      mjcb.write_model(m, 'viz/demo/demo.mjcb', name='demo')"

(run from `packages/mjc`).
