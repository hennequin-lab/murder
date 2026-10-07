(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The log-gamma function, needed by the Student-t prior and the Poisson
   likelihood. Lanczos' approximation with g = 7 and nine coefficients gives
   about fifteen correct digits for positive arguments, which is all this
   library asks of it. Written with tensor operations so that it traces and
   differentiates like everything else. *)

open Base

let coefficients =
  [| 0.99999999999980993
   ; 676.5203681218851
   ; -1259.1392167224028
   ; 771.32342877765313
   ; -176.61502916214059
   ; 12.507343278686905
   ; -0.13857109526572012
   ; 9.9843695780195716e-6
   ; 1.5056327351493116e-7
  |]

let g = Float.of_int 7

(* [lgamma x] is log Γ(x) for x > 0, elementwise. *)
let lgamma (x : (float, 'w) Nx.t) : (float, 'w) Nx.t =
  let z = Nx.sub_s x 1.0 in
  let a = ref (Nx.full_like x coefficients.(0)) in
  for i = 1 to Array.length coefficients - 1 do
    a
    := Nx.add !a (Nx.div (Nx.full_like x coefficients.(i)) (Nx.add_s z (Float.of_int i)))
  done;
  let t = Nx.add_s z (g +. 0.5) in
  let half_log_2pi = 0.5 *. Float.log (2.0 *. Float.pi) in
  Nx.add
    (Nx.add (Nx.mul (Nx.add_s z 0.5) (Nx.log t)) (Nx.add (Nx.neg t) (Nx.log !a)))
    (Nx.add_s (Nx.zeros_like x) half_log_2pi)
