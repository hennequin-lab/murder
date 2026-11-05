(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base

type t =
  { amplitude : float
  ; period : float
  }

let default = { amplitude = 1.6; period = 2.4 }

let ctrl (t : t) ~nu ~time =
  let w = 2. *. Float.pi /. t.period in
  let u = t.amplitude *. Float.sin (w *. time) in
  Array.init nu ~f:(fun _ -> u)
