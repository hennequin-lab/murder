(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Kronecker deltas as dense tensors (identity tensors).

   A term's ties are compiled into einsum contractions against a delta tensor
   rather than into repeated labels within one operand. Nx's einsum loses track
   of which axis a repeated label names once it has extracted a diagonal (the
   diagonal moves to the end of the shape while the label list does not), so a
   term with more than one tie would contract the wrong axes. A separate delta
   operand is equivalent and exercises only ordinary cross-operand
   contractions. *)

open Base

(* [tensor ~order dim] is [δ_{i₁…iₙ} = 1 iff all indices equal], built as a
   product of consecutive Kronecker deltas. *)
let tensor ~order dim =
  if order < 2 then invalid_arg (Printf.sprintf "Delta.tensor: order %d is < 2" order);
  let eye = Nx.eye Nx.float32 dim in
  if Int.equal order 2
  then eye
  else (
    let shape = Array.create ~len:order dim in
    let t = ref (Nx.ones Nx.float32 shape) in
    for k = 0 to order - 2 do
      let pair_shape = Array.create ~len:order 1 in
      pair_shape.(k) <- dim;
      pair_shape.(k + 1) <- dim;
      t := Nx.mul !t (Nx.reshape pair_shape eye)
    done;
    !t)
