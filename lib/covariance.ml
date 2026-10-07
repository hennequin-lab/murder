(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Types

type 'a params =
  { d : 'a
  ; t : 'a
  }
[@@deriving ptree]

let structure () =
  Nx.Ptree.instantiate
    (module struct
      type 'a t = 'a params

      let walk = walk_params
    end)

let make ?(sigma2 = 1.0) ~n dtype =
  if n <= 0 then invalid_arg "Covariance.make: n must be positive";
  if not (Float.( > ) sigma2 0.0)
  then invalid_arg "Covariance.make: sigma2 must be positive";
  { d = Nx.full dtype [| n |] (Float.sqrt sigma2); t = Nx.zeros dtype [| n; n |] }

let factor { d; t } = Nx.add (Nx.triu ~k:1 t) (Nx.diag d)
let log_det { d; _ } = Nx.mul_s (Nx.sum (Nx.log d)) 2.0
