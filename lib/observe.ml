(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Mini losses, and the mark that hands a prediction's curvature to a sketch.

   A mark is a {!Rune.custom_jvp} with a unit result whose rule gathers the
   sketch's direction lanes and adds the little loss's Gauss-Newton block
   YᵀHY to a total the sketch driver collects. Nothing else reads it: plain
   execution and reverse mode never apply its tangent map — a unit result has
   nothing to differentiate — so a marked loss is an ordinary loss under every
   other optimizer, and the same model trains with {!Rune.value_and_grad} and
   under a sketch.

   The rule reads the tangent of the *prediction*, not of the loss: the block
   is YᵀHY with Y the tangent batch of y and H the curvature description the
   caller supplied, and the value and tangent of the loss are the caller's own
   arithmetic, returned to the sketch as the objective itself. A mean or a
   scale the caller applies to a marked little loss therefore reaches the
   returned loss and C but not the block: the block enters with weight one
   (law 6 of RFC 0009), and a mean over trials goes inside each little loss. *)

(* The map the sketch's directions live in, and the total its blocks are added
   to. The mark and the driver are the only users: a total collects the
   innermost open scope, so one library-wide total serves every sketch. *)
let directions = Rune.axis ()
let curvature : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

let mark ~curv y =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.unit (fun _ -> (), fun dy ->
    let ys = Rune.lanes directions dy in
    let rows t = Nx.reshape [| Nx.dim 0 t; -1 |] t in
    let block =
      Nx.cast Nx.float64 (Nx.matmul (rows ys) (Nx.transpose (rows (Curv.apply curv ys))))
    in
    (* Every block is symmetric by construction — YᵀHY with H symmetric — so
       the accumulator is symmetrized up to floating-point asymmetry, which
       later decompositions would rather not see. *)
    let block = Nx.mul_s (Nx.add block (Nx.transpose block)) 0.5 in
    Rune.Total.add curvature block) y

(* [observe ~y ~curv l] marks [l] as a little loss in the prediction [y], whose
   Hessian with respect to [y] is [curv]: under a sketch the block joins the
   Gauss-Newton matrix, under any other driver the mark is inert. [l] must be a
   scalar computed from [y] by differentiable operations — the caller's own
   arithmetic defines both the value and the tangent of the loss the sketch
   returns. *)
let observe ~y ~curv l =
  if Nx.numel l <> 1
  then
    invalid_arg
      (Format.asprintf
         "Sofo.observe: the mini loss must be a scalar (one element), got shape %a"
         Nx.pp_shape
         (Nx.shape l));
  mark ~curv y

(* Packaged little losses: the value is computed with ordinary operations, so
   its tangent comes from the standard rules, and the curvature is derived from
   the same reduction, so the two cannot drift apart. *)

(* [mse ?w y target] is the mean weighted squared error
   [mean (w * (y - target)²)], marked with its exact curvature
   [diag (2 w / numel y)] — [Curv.scale (2 / numel y)] without weights. *)
let mse ?w ~target y =
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
      let two_over_n = Nx_dtype.of_float (Nx.dtype y) (2.0 /. n) in
      Nx.mean (Nx.mul w sq), Curv.diag (Nx.mul_s w two_over_n)
  in
  observe ~y ~curv loss;
  loss

let sse ?w ~target y =
  let shape_y = Nx.shape y in
  if shape_y <> Nx.shape target
  then invalid_arg "Sofo.mse: prediction and target must have the same shape";
  let d = Nx.sub y target in
  let sq = Nx.mul d d in
  let loss, curv =
    match w with
    | None -> Nx.sum sq, Curv.scale 2.0
    | Some w ->
      let w = Nx.cast (Nx.dtype y) w in
      let two_over_n = Nx_dtype.of_float (Nx.dtype y) 2.0 in
      Nx.sum (Nx.mul w sq), Curv.diag (Nx.mul_s w two_over_n)
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
