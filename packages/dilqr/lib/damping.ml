(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Dilqr authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Off
  | Fixed of float
  | Adaptive of Regularisation.t

let default = Adaptive Regularisation.init
