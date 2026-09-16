(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Mini losses, and the marker that hands one to a collector.

   An observation is a semantic marker, and effects are the natural way to
   express one: the user's graph performs [E_observe] where a little loss is
   known, and whichever handler encloses the graph decides what to do with it.
   A collector accumulating sketch quantities intercepts it; every other
   handler — plain execution, {!Rune.grad}, {!Rune.vmap}, {!Rune.jit} — lets it
   pass, and the observation is inert. So user code is identical under every
   optimizer: [observe] is a no-op unless a collector is installed.

   The payload packs the prediction and the loss value together with the
   curvature description. Their dtypes are independent (a loss evaluated in a
   wider type than its prediction is perfectly reasonable), which is why the
   constructor carries them as two existential pairs rather than one shared
   type; the collector is polymorphic in both. *)

type obs = Obs : ('a, 'b) Nx.t * ('c, 'd) Nx.t * Curv.t -> obs
type _ Effect.t += E_observe : obs -> unit Effect.t

(* [observe ~y ~curv l] marks [l] as a little loss in [y], with curvature
   [curv]: [l] must be a scalar computed from [y] by differentiable operations.
   Its tangent is available to the collector because [l]'s operations were
   tracked, so the user's own arithmetic — not this call — defines both the
   value and its tangent. *)
let observe ~y ~curv l =
  if Nx.numel l <> 1
  then
    invalid_arg
      (Printf.sprintf
         "Sofo.observe: the mini loss must be a scalar (one element), got shape [%s]"
         (String.concat "," (Array.to_list (Array.map string_of_int (Nx.shape l)))));
  match Effect.perform (E_observe (Obs (y, l, curv))) with
  | () -> ()
  | exception Effect.Unhandled _ -> ()

(* Packaged little losses: the value is computed with ordinary operations, so
   its tangent comes from the standard rules, and the curvature is derived from
   the same reduction, so the two cannot drift apart. *)

(* [mse ?w y target] is the mean weighted squared error
   [mean (w * (y - target)²)], whose Hessian with respect to [y] is
   [diag (2 w / numel y)] — [Curv.scale (2 / numel y)] without weights. *)
let mse ?w y target =
  let shape_y = Nx.shape y in
  if shape_y <> Nx.shape target
  then invalid_arg "Sofo.mse: prediction and target must have the same shape";
  let n = float_of_int (Nx.numel y) in
  let d = Nx.sub y target in
  let sq = Nx.mul d d in
  let loss, curv =
    match w with
    | None -> Nx.mean sq, Curv.scale (2.0 /. n)
    | Some w ->
      let w = Nx.cast (Nx.dtype y) w in
      let two_over_n = Nx_core.Dtype.of_float (Nx.dtype y) (2.0 /. n) in
      Nx.mean (Nx.mul w sq), Curv.diag (Nx.mul_s w two_over_n)
  in
  observe ~y ~curv loss;
  loss

(* [softmax_ce y labels] is the cross entropy of the last axis of [y] against
   [labels], averaged over the leading (row) axes: [labels] has [y]'s shape
   without its last axis, and holds class indices in [0, num_classes). Its
   Hessian with respect to [y] is [1/rows] times [diag p − p pᵀ] per row, with
   [p = softmax y]. *)
let softmax_ce y labels =
  let shape_y = Nx.shape y in
  let rank = Array.length shape_y in
  if rank = 0 then invalid_arg "Sofo.softmax_ce: the prediction must have a class axis";
  let labels_shape = Array.sub shape_y 0 (rank - 1) in
  if Nx.shape labels <> labels_shape
  then
    invalid_arg
      "Sofo.softmax_ce: labels must have the prediction's shape without its class axis";
  let rows = Array.fold_left ( * ) 1 labels_shape in
  let p = Nx.softmax y in
  let idx = Nx.reshape (Array.append labels_shape [| 1 |]) labels in
  let picked = Nx.take_along_axis ~axis:(-1) ~indices:idx (Nx.log_softmax y) in
  let loss = Nx.neg (Nx.mean picked) in
  let curv = Curv.softmax_ce ~scale:(1.0 /. float_of_int rows) p in
  observe ~y ~curv loss;
  loss
