(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let forward (m : Model.host) (d : Data.host) : Data.host =
  let d = Kinematics.forward m d in
  let d = Mass.forward m d in
  let d = Passive.forward m d in
  let d = Bias.forward m d in
  let qfrc_smooth =
    Nx.add (Nx.sub (Nx.add d.qfrc_passive d.qfrc_applied) d.qfrc_bias) d.qfrc_actuator
  in
  let qacc = Mass.solve (Mass.factor d.mass_matrix) qfrc_smooth in
  { d with qfrc_smooth; qacc }
