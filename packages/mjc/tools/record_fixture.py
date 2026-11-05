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

def _chain(depth):
    """A chain of [depth] hinge bodies, nested the way MuJoCo wants them."""
    inner = ""
    for k in range(depth, 0, -1):
        inner = (
            f'<body name="b{k}" pos="0 0 {"-0.08" if k == 1 else "-0.06"}">'
            f'<joint name="j{k}" type="hinge" axis="0 1 0" damping="0.004"/>'
            f'<geom name="g{k}" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>'
            f'<site name="s{k}" pos="0 0 -0.06"/>'
            f"{inner}</body>"
        )
    return inner


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
    # The actuator surface: one joint per actuator kind v1 has, so that a
    # single model exercises a fixed gain, an affine bias through a position
    # and a velocity servo, an integrated setpoint, a damper and a general
    # gain/bias polynomial, plus the control and force ranges.
    "actuators": """
<mujoco model="actuators">
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 0.6">
      <body name="l1" pos="0 0 0">
        <joint name="j_motor" type="hinge" axis="0 0 1"/>
        <geom name="g1" type="capsule" size="0.02 0.15" pos="0.15 0 0"
              quat="0.7071068 0 0.7071068 0"/>
        <body name="l2" pos="0.3 0 0">
          <joint name="j_position" type="hinge" axis="0 0 1" armature="0.01"/>
          <geom name="g2" type="capsule" size="0.02 0.15" pos="0.15 0 0"
                quat="0.7071068 0 0.7071068 0"/>
          <body name="l3" pos="0.3 0 0">
            <joint name="j_velocity" type="hinge" axis="0 0 1" damping="0.02"/>
            <geom name="g3" type="capsule" size="0.02 0.15" pos="0.15 0 0"
                  quat="0.7071068 0 0 0.7071068"/>
            <body name="l4" pos="0.3 0 0">
              <joint name="j_intvelocity" type="slide" axis="1 0 0" armature="0.02"/>
              <geom name="g4" type="sphere" size="0.05"/>
              <body name="l5" pos="0 0 -0.15">
                <joint name="j_damper" type="hinge" axis="0 1 0"/>
                <geom name="g5" type="capsule" size="0.015 0.1" pos="0 0 -0.1"/>
                <body name="l6" pos="0 0 -0.22">
                  <joint name="j_general" type="hinge" axis="0 1 0"/>
                  <geom name="g6" type="sphere" size="0.04"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="a_motor" joint="j_motor" gear="1.5"/>
    <position name="a_position" joint="j_position" kp="40" kv="1.5" ctrlrange="-0.8 0.8"/>
    <velocity name="a_velocity" joint="j_velocity" kv="3" forcerange="-2 2" forcelimited="true"/>
    <intvelocity name="a_intvelocity" joint="j_intvelocity" kp="60" kv="2" ctrlrange="-0.5 0.5"/>
    <general name="a_damper" joint="j_damper" gainprm="0" biasprm="0 0 -0.8" ctrlrange="-1 1"/>
    <general name="a_general" joint="j_general" gainprm="2.5 -1 -0.4"
             biasprm="0.3 -0.5 -0.2" ctrlrange="-1 1" forcerange="-1.5 1.5"
             forcelimited="true"/>
  </actuator>
</mujoco>
""",

    # A two-link finger driven through a fixed tendon over both joints: the
    # tendon length is a linear combination of the two coordinates, so one
    # actuator moves both, and the tendon Jacobian is what says how.
    "finger": """
<mujoco model="finger">
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="prox" pos="0 0 0.3">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.005"/>
      <geom name="g1" type="capsule" size="0.012 0.05" pos="0 0 0.05"/>
      <body name="dist" pos="0 0 0.1">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.008"/>
        <geom name="g2" type="capsule" size="0.012 0.04" pos="0 0 0.04"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="1"/>
      <joint joint="j2" coef="0.5"/>
    </fixed>
  </tendon>
  <actuator>
    <motor name="a_tendon" tendon="t1" gear="2"/>
  </actuator>
</mujoco>
""",

    # Sensors on a seventeen-body chain: the state-projection family, the
    # subtree reductions (which is what needs a body deep enough to have a
    # subtree), and a tendon over two joints for the tendon sensors.
    "sensors": """
<mujoco model="sensors">
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b0" pos="0 0 1.0">
      <body name="b1" pos="0 0 -0.08">
        <joint name="j1" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g1" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s1" pos="0 0 -0.06"/>
      <body name="b2" pos="0 0 -0.06">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g2" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s2" pos="0 0 -0.06"/>
      <body name="b3" pos="0 0 -0.06">
        <joint name="j3" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g3" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s3" pos="0 0 -0.06"/>
      <body name="b4" pos="0 0 -0.06">
        <joint name="j4" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g4" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s4" pos="0 0 -0.06"/>
      <body name="b5" pos="0 0 -0.06">
        <joint name="j5" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g5" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s5" pos="0 0 -0.06"/>
      <body name="b6" pos="0 0 -0.06">
        <joint name="j6" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g6" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s6" pos="0 0 -0.06"/>
      <body name="b7" pos="0 0 -0.06">
        <joint name="j7" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g7" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s7" pos="0 0 -0.06"/>
      <body name="b8" pos="0 0 -0.06">
        <joint name="j8" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g8" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s8" pos="0 0 -0.06"/>
      <body name="b9" pos="0 0 -0.06">
        <joint name="j9" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g9" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s9" pos="0 0 -0.06"/>
      <body name="b10" pos="0 0 -0.06">
        <joint name="j10" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g10" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s10" pos="0 0 -0.06"/>
      <body name="b11" pos="0 0 -0.06">
        <joint name="j11" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g11" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s11" pos="0 0 -0.06"/>
      <body name="b12" pos="0 0 -0.06">
        <joint name="j12" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g12" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s12" pos="0 0 -0.06"/>
      <body name="b13" pos="0 0 -0.06">
        <joint name="j13" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g13" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s13" pos="0 0 -0.06"/>
      <body name="b14" pos="0 0 -0.06">
        <joint name="j14" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g14" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s14" pos="0 0 -0.06"/>
      <body name="b15" pos="0 0 -0.06">
        <joint name="j15" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g15" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s15" pos="0 0 -0.06"/>
      <body name="b16" pos="0 0 -0.06">
        <joint name="j16" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g16" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s16" pos="0 0 -0.06"/>
      <body name="b17" pos="0 0 -0.06">
        <joint name="j17" type="hinge" axis="0 1 0" damping="0.004"/>
        <geom name="g17" type="capsule" size="0.01 0.03" pos="0 0 -0.03"/>
        <site name="s17" pos="0 0 -0.06"/>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="1"/>
      <joint joint="j2" coef="0.5"/>
    </fixed>
  </tendon>
  <actuator>
    <motor name="a1" joint="j1" gear="1"/>
    <motor name="a2" tendon="t1" gear="1.5"/>
  </actuator>
  <sensor>
    <jointpos joint="j1"/>
    <jointvel joint="j2"/>
    <framepos objtype="site" objname="s3"/>
    <framequat objtype="site" objname="s4"/>
    <framepos objtype="body" objname="b5"/>
    <framequat objtype="body" objname="b6"/>
    <framelinvel objtype="body" objname="b7"/>
    <frameangvel objtype="body" objname="b8"/>
    <actuatorfrc actuator="a1"/>
    <tendonpos tendon="t1"/>
    <tendonvel tendon="t1"/>
    <subtreecom body="b3"/>
    <subtreelinvel body="b9"/>
    <subtreeangmom body="b11"/>
  </sensor>
</mujoco>
""",

    "sensors": f"""
<mujoco model="sensors">
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b0" pos="0 0 1.0">{_chain(17)}</body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="1"/>
      <joint joint="j2" coef="0.5"/>
    </fixed>
  </tendon>
  <actuator>
    <motor name="a1" joint="j1" gear="1"/>
    <motor name="a2" tendon="t1" gear="1.5"/>
  </actuator>
  <sensor>
    <jointpos joint="j1"/>
    <jointvel joint="j2"/>
    <framepos objtype="site" objname="s3"/>
    <framequat objtype="site" objname="s4"/>
    <framepos objtype="body" objname="b5"/>
    <framequat objtype="body" objname="b6"/>
    <framelinvel objtype="body" objname="b7"/>
    <frameangvel objtype="body" objname="b8"/>
    <actuatorfrc actuator="a1"/>
    <tendonpos tendon="t1"/>
    <tendonvel tendon="t1"/>
    <subtreecom body="b3"/>
    <subtreelinvel body="b9"/>
    <subtreeangmom body="b11"/>
  </sensor>
</mujoco>
""",

    # A stiff limit: the direct solref form, which is a negative stiffness, not
    # a time constant. A joint pushed against it by a spring settles a
    # millimetre past its range, not a hundred, and the difference is the
    # refsafe clamp being applied where it does not belong.
    "stiff": """
<mujoco model="stiff">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" range="-0.20 0.20"
             stiffness="40" damping="0.2" springref="0.5" solreflimit="-1e5 -400"/>
      <geom name="g_arm" type="capsule" size="0.02 0.2" pos="0 0 -0.2"/>
      <body name="tip" pos="0 0 -0.4">
        <joint name="wrist" type="hinge" axis="0 1 0" range="-0.05 0.05"
               stiffness="8" damping="0.1" springref="0.3" solreflimit="-2e5 -800"/>
        <geom name="g_tip" type="sphere" size="0.04"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",

    # MuJoCo's humanoid, the G4 gate's hardest model: a free joint, 21 motors,
    # two limited tendons, and every geom colliding with every other one except
    # the two excluded pairs. It lives in test/fixtures/humanoid.xml because it
    # is long, and it is that copy, with the cone and solver set to the ones v1
    # implements, that is recorded.
    "humanoid": "@test/fixtures/humanoid.xml",
    # A two-link arm: two hinges in the xz plane, one motor each, gravity, no
    # contacts and no limits. It is the model the iLQR demo swings to a target,
    # and it is here rather than in the demo so that the dynamics it uses are
    # the ones the corpus checks against MuJoCo.
    "arm2": """
<mujoco model="arm2">
  <compiler angle="radian"/>
  <option timestep="0.01" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="upper" pos="0 0 0.6">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom name="g_upper" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03"/>
      <body name="lower" pos="0 0 -0.3">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom name="g_lower" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025"/>
        <body name="hand" pos="0 0 -0.25">
          <geom name="g_hand" type="sphere" size="0.04"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m_shoulder" joint="shoulder" gear="1"/>
    <motor name="m_elbow" joint="elbow" gear="1"/>
  </actuator>
</mujoco>
""",

    # A tendon with a range: two hinges joined by one tendon whose length is
    # limited on both sides, so a single row type is exercised with both signs.
    # The tendon is short enough that the limit is reachable by the joints the
    # motors drive.
    "tendonlim": """
<mujoco model="tendonlim">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="a" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
      <geom name="g_a" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03"/>
      <body name="b" pos="0 0 -0.25">
        <joint name="j2" type="hinge" axis="0 1 0" range="-1.0 1.0"/>
        <geom name="g_b" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.025"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t" limited="true" range="0.05 0.30" margin="0.004"
           solreflimit="0.004 1" solimplimit="0.9 0.95 0.002">
      <joint joint="j1" coef="1"/>
      <joint joint="j2" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <motor name="m1" joint="j1" gear="1"/>
    <motor name="m2" joint="j2" gear="1"/>
  </actuator>
</mujoco>
""",

    # Tendon equalities: one tendon with an offset, and a coupling between two
    # tendon lengths through the quartic MuJoCo defines. The four cases that
    # matter are all here — a single tendon (no polynomial at all), a pair with
    # a linear term, and a pair whose polynomial reaches the cubic and quartic
    # terms, which is where the derivative in the Jacobian stops being constant.
    # Contacts are off: the rows are the equalities, and nothing else, so a
    # mismatch cannot be blamed on the collision stage.
    "tendoneq": """
<mujoco model="tendoneq">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="a" pos="0 0 0.6">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"
            contype="0" conaffinity="0"/>
      <body name="b" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.2"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.025"
              contype="0" conaffinity="0"/>
        <body name="c" pos="0.25 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" damping="0.3"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"
                contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="1"/>
    </fixed>
    <fixed name="t2">
      <joint joint="j2" coef="1"/>
      <joint joint="j3" coef="-2"/>
    </fixed>
  </tendon>
  <equality>
    <tendon name="e_off" tendon1="t1" polycoef="0.15 0 0 0 0"/>
    <tendon name="e_linear" tendon1="t1" tendon2="t2" polycoef="0 1 0 0 0"/>
    <tendon name="e_quartic" tendon1="t2" tendon2="t1"
            polycoef="0.02 0.5 -0.3 0.1 -0.05"/>
  </equality>
  <actuator>
    <motor name="m1" joint="j1" gear="0.4"/>
    <motor name="m2" joint="j2" gear="0.3"/>
    <motor name="m3" joint="j3" gear="0.2"/>
  </actuator>
</mujoco>
""",

    # Two limbs that overlap and are told not to touch. Without the `<exclude>`
    # their spheres collide at every state; with it, the third body's sphere —
    # which is not excluded — is the only contact. The model exists so that a
    # filter cannot pass by filtering everything.
    "exclude": """
<mujoco model="exclude">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="torso" pos="0 0 0.6">
      <freejoint/>
      <geom name="g_torso" type="sphere" size="0.05" mass="1"/>
      <body name="left" pos="0.04 0 0">
        <joint name="jl" type="hinge" axis="0 1 0"/>
        <geom name="g_left" type="sphere" size="0.06" mass="0.2"/>
      </body>
      <body name="right" pos="-0.04 0 0">
        <joint name="jr" type="hinge" axis="0 1 0"/>
        <geom name="g_right" type="sphere" size="0.06" mass="0.2"/>
      </body>
    </body>
    <body name="pebble" pos="0 0 0.62">
      <freejoint/>
      <geom name="g_pebble" type="sphere" size="0.03" mass="0.1"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="left" body2="right"/>
  </contact>
</mujoco>
""",

}


def source_of(name, text):
    """The MJCF text for a model: the string itself, or a file it names."""
    if text.startswith("@"):
        with open(text[1:], encoding="utf-8") as f:
            return f.read()
    return text


# Models whose random poses are not physical. A free-jointed robot sampled
# uniformly in position is buried in the floor; its states are drawn near the
# reference pose instead, so that a trajectory is something a controller could
# have produced and the gate is a statement about physics rather than about how
# far a solver can be pushed in a pose no controller would command.
PHYSICAL = {"humanoid", "hopper", "walker", "quadruped"}

MODELS.update(
    {
        # dm_control's locomotion suite: the models gate G4 names. All three are
        # planes and capsules with the engine's own default options.
        "hopper": "@test/fixtures/hopper.xml",
        "walker": "@test/fixtures/walker.xml",
        "quadruped": "@test/fixtures/quadruped.xml",
    }
)


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
EFC_FIELDS = ["efc_type", "efc_id", "efc_pos", "efc_margin", "efc_frictionloss",
              "efc_J", "efc_vel", "efc_R", "efc_D", "efc_aref", "efc_b",
              "efc_force"]

# Contacts, padded to the widest state like the rows are. The normal is the
# first three components of the contact frame.
CONTACT_FIELDS = {
    "contact_dist": 1, "contact_pos": 3, "contact_normal": 3,
    "contact_frame": 9,
    "contact_geom": 2, "contact_body": 2, "contact_dim": 1,
    "contact_includemargin": 1, "contact_friction": 5,
    "contact_solref": 2, "contact_solimp": 5,


}


def dense_ten_J(m, d):
    """The tendon Jacobian, which the engine keeps sparse, as a dense matrix."""
    J = np.zeros((m.ntendon, m.nv))
    for i in range(m.ntendon):
        adr = int(m.ten_J_rowadr[i])
        nnz = int(m.ten_J_rownnz[i])
        J[i, np.asarray(m.ten_J_colind[adr:adr + nnz], dtype=int)] = np.asarray(
            d.ten_J[adr:adr + nnz]
        )
    return J


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
    "subtree_com", "subtree_linvel", "subtree_angmom",
    "cinert", "cdof", "cvel", "cdof_dot",
    "qfrc_bias", "qfrc_passive", "qfrc_smooth",
    # the actuator surface: what the transmissions measure and what the
    # actuators make of it
    "qfrc_actuator", "actuator_length", "actuator_velocity", "actuator_force",
    "act_dot", "ten_length", "ten_velocity", "sensordata",
]

TRAJ_STEPS = 10

# Gate G4's criterion is about a thousand steps, not ten: a ten-step fixture
# says the engine takes the same steps, and a thousand says whether it stays on
# the trajectory. `--long` records the same run at this length for the physical
# models, in a separate fixture so the ordinary one stays small.
G4_STEPS = 1000


def control_range(m):
    """A control box per actuator: the declared range, or [-1, 1]."""
    lo = np.where(m.actuator_ctrllimited, m.actuator_ctrlrange[:, 0], -1.0)
    hi = np.where(m.actuator_ctrllimited, m.actuator_ctrlrange[:, 1], 1.0)
    return lo, hi


def trajectory(m, d, qpos0, qvel0, crng, lo, hi, steps):
    """[steps] of [mj_step] from [qpos0], [qvel0], with fresh controls.

    The control of a step is chosen before both halves of it: [mj_step] is a
    forward pass, which reads [ctrl] for the activation derivative, and an
    advance, which uses it. Choosing it in between would make the recorded
    force describe a control the step did not use. [mj_step] also leaves qacc
    as it was before the step, so the acceleration is recomputed at the state it
    belongs to.
    """
    d.qpos[:] = qpos0
    d.qvel[:] = qvel0
    d.act[:] = 0.0
    if m.nu:
        d.ctrl[:] = crng.uniform(lo, hi)
    mujoco.mj_forward(m, d)
    traj_qpos = [np.array(d.qpos)]
    traj_qvel = [np.array(d.qvel)]
    traj_qacc = []
    traj_act = [np.array(d.act)]
    traj_ctrl = []
    traj_nefc = []
    traj_contacts = []
    for _ in range(steps):
        if m.nu:
            d.ctrl[:] = crng.uniform(lo, hi)
        traj_ctrl.append(np.array(d.ctrl))
        mujoco.mj_forward(m, d)
        traj_qacc.append(np.array(d.qacc))
        traj_nefc.append(int(d.nefc))
        traj_contacts.append(contact_state(m, d))
        mujoco.mj_step(m, d)
        traj_act.append(np.array(d.act))
        traj_qpos.append(np.array(d.qpos))
        traj_qvel.append(np.array(d.qvel))
    # The last state needs its own forward pass: the loop leaves [qacc] as it
    # was before the last step, and the row and contact counts of the final
    # state are otherwise never computed. Aligning the per-state lists with
    # [traj_qpos] is what lets a diagnostic ask "what did MuJoCo see here?".
    mujoco.mj_forward(m, d)
    traj_qacc.append(np.array(d.qacc))
    traj_nefc.append(int(d.nefc))
    traj_contacts.append(contact_state(m, d))
    # The fixture's tensors share a state axis, so the schedule carries the
    # control that would drive one more step: index k is the control of step k,
    # and the last is the one no step used.
    traj_ctrl.append(np.array(d.ctrl) if m.nu else np.zeros(0))
    # How many contacts and rows each state had, plus the contact pairs and
    # their distances. Without these a trajectory that diverges says only *that*
    # it diverged; with them the first state whose contact set was modelled
    # differently is a lookup. A wide-enough pair table comes from the run
    # itself, so it is padded to its own maximum.
    width = max(1, max(len(c["contact_dist"]) for c in traj_contacts))
    traj_ncon = np.array([len(c["contact_dist"]) for c in traj_contacts], dtype=float)
    traj_pair = np.zeros((len(traj_contacts), width, 2))
    traj_dist = np.zeros((len(traj_contacts), width))
    for i, c in enumerate(traj_contacts):
        n = len(c["contact_dist"])
        traj_pair[i, :n] = c["contact_geom"]
        traj_dist[i, :n] = np.asarray(c["contact_dist"]).reshape(-1)
    return {
        "traj_qpos": np.stack(traj_qpos),
        "traj_qvel": np.stack(traj_qvel),
        "traj_qacc": np.stack(traj_qacc),
        "traj_act": np.stack(traj_act),
        "traj_ctrl": np.stack(traj_ctrl),
        "traj_nefc": np.array(traj_nefc, dtype=float),
        "traj_ncon": traj_ncon,
        "traj_pair": traj_pair,
        "traj_dist": traj_dist,
    }


def record(name, xml, outdir, rng):
    # Read the source once: a file-sourced model would otherwise be truncated by
    # its own rewrite before it is copied to the corpus.
    xml_text = source_of(name, xml)
    m = (
        mujoco.MjModel.from_xml_path(xml[1:])
        if xml.startswith("@")
        else mujoco.MjModel.from_xml_string(xml_text)
    )
    # Actuator controls come from their own stream: adding them must not
    # renumber the poses and velocities the rest of the corpus is built from.
    crng = np.random.default_rng(SEED + 1 + len(MODELS))
    lo, hi = control_range(m)
    # The port is dense; record the rows in the layout the port uses, whatever
    # the model's own auto choice would be at this size.
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    d = mujoco.MjData(m)
    xml_path = os.path.join(outdir, f"{name}.xml")
    if not xml.startswith("@"):
        with open(xml_path, "w") as f:
            f.write(xml_text.lstrip() + "\n")

    states = []
    qvels = []
    cons = []
    for _ in range(NSTATES):
        qpos = (
            np.array(m.qpos0) + rng.uniform(-0.05, 0.05, size=m.nq)
            if name in PHYSICAL
            else random_qpos(m, rng)
        )
        qvel = rng.uniform(-2.0, 2.0, size=m.nv)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        if m.nu:
            d.ctrl[:] = crng.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        # The subtree velocities are computed lazily, inside the sensor stage,
        # by the engine; the fixture asks for them so a test can compare them.
        mujoco.mj_subtreeVel(m, d)
        full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, full)
        fields = {f: np.array(getattr(d, f)) for f in POSE_FIELDS + STAGE_FIELDS}
        fields["ctrl"] = np.array(d.ctrl)
        fields["act"] = np.array(d.act)
        fields["ten_J"] = dense_ten_J(m, d)
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
    for f in POSE_FIELDS + STAGE_FIELDS + ["mass_matrix", "ctrl", "act", "ten_J"]:
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

    # A trajectory through mj_step, from the first state: this is the
    # end-to-end test of forward plus the integrator, and for a stateful
    # actuator of the activation dynamics as well. Gate G4's criterion is about
    # a thousand steps rather than ten, so the physical models are recorded
    # once at each length.
    traj = trajectory(m, d, states[0][0], qvels[0], crng, lo, hi, TRAJ_STEPS)
    mjcb.write_fixture(os.path.join(outdir, f"{name}.traj.mjbf"), name, traj)
    if long and name in PHYSICAL:
        mjcb.write_fixture(
            os.path.join(outdir, f"{name}.g4.mjbf"),
            name,
            trajectory(m, d, states[0][0], qvels[0], crng, lo, hi, G4_STEPS),
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


long = False


def main():
    global long
    argv = [a for a in sys.argv[1:] if a != "--long"]
    long = "--long" in sys.argv[1:]
    outdir = argv[0] if argv else "test/fixtures"
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
