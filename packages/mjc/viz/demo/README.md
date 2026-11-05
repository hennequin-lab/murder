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

## A tendon demo, driving itself

`gripper.xml` is the M5 demo: two mirrored fingers, each two links, each pulled
by one *fixed tendon* over its own two joints, with one motor per tendon. The
joint springs open the hand and the tendon closes it, so every motion in the
picture comes from the actuator path: moment arm, length, force, `qfrc_actuator`.

It has no keyboard to lean on, so the window can drive it: `--drive` feeds each
actuator a sine (amplitude 1.2, period 0.9 s, from `Mjc_viz.Drive`), and the
fingers close and open on their own. The fingers touch nothing — their joint
ranges are the stopper, sized so the tips just meet when both joints reach the
limit, which is what keeps the closing symmetric instead of one finger shoving
the other through. The limits are written in the *direct* `solref` form
(a negative stiffness) because these links are light; see `doc/log.md` for the
engine bug the demo found on the way.

    dune exec packages/mjc/viz/window/main.exe -- \
      --model packages/mjc/viz/demo/gripper.mjcb --drive \
      --steps-per-frame 4 --eye 0,-0.62,0.88 --target 0,0,0.88

The HUD carries an M5 line under the joint line:

    tendons 0.28 0.28 | force 1.18 1.18 | sensors 0.28 0.02 0.13 1.18 -0.03 0.00 0.93

which is `ten_length` (growing as the fingers close), `actuator_force` (the
sine's current value) and the model's sensors — two tendon readings, a joint
angle, an actuator force and a subtree centre of mass. The same command with
`--shot` writes a PNG; `doc/figures/viewer-tendon.png` is one.

A still needs no window and no keyboard either:

    dune exec packages/mjc/viz/bin/mjc_view.exe -- \
      --model packages/mjc/viz/demo/gripper.mjcb --steps 100 --drive \
      --eye 0,-0.55,0.88 --target 0,0,0.88 --shot gripper.png

`gripper.mjcb` is generated from `gripper.xml` the way `demo.mjcb` is:

    /tmp/mjcenv/bin/python -c "import sys, mujoco; sys.path.insert(0,'tools'); \
      import mjcb; m = mujoco.MjModel.from_xml_path('viz/demo/gripper.xml'); \
      mjcb.write_model(m, 'viz/demo/gripper.mjcb', name='gripper')"

## Record a run, then replay it

    # 200 frames of a 0.002 s step, recorded
    dune exec packages/mjc/viz/window/main.exe -- \
      --model packages/mjc/viz/demo/demo.mjcb --frames 200 --log /tmp/run.mjlog

    # replay it: space plays or pauses, left/right scrub, home/end jump
    dune exec packages/mjc/viz/window/main.exe -- \
      --model packages/mjc/viz/demo/demo.mjcb --follow /tmp/run.mjlog

A replay frame costs the draw and nothing else, so scrubbing a long run is
instant; the log stores geom poses, so the replay does not need the engine's
kinematics to agree with it.

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
