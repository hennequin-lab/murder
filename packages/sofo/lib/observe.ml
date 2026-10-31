(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Mini losses, and the mark that hands a prediction's curvature to a sketch.

   A mark is a {!Rune.custom_jvp} with a unit result whose rule gathers the
   sketch's direction lanes and adds the little loss's Gauss-Newton block
   YᵀHY to a total the sketch driver collects. Outside a sketch a mark makes
   no rule, so a marked loss is an ordinary loss under every other optimizer,
   and the same model trains with {!Rune.value_and_grad} and under a sketch.

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
open Nx

let directions = Rune.axis ()
let curvature : (float, Nx.float64_elt, 'd) Rune.Total.t = Rune.Total.make ()

(* Whether the sketch's map is around the calling code. Rune traces a rule's
   tangent map wherever it traces the rule, as inside a [Rune.scan] under
   [Rune.grad], applied or not, and [Rune.lanes] raises outside the map it
   names: so a mark outside a sketch makes no rule. The sketch's extent is a
   handler of [Sketching], as a [Nx.Rng] scope is one of its key requests, so
   it is the extent of the code the map runs, whatever thread or domain runs
   other code. *)
type _ Effect.t += Sketching : unit Effect.t

(* Re-raised with the backtrace of its raise, which a plain [raise] would
   restart at the handler. *)
let reraise e = Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* [within_sketch f] runs [f] as the sketch's map around it. *)
let within_sketch f =
  let effc (type a) (e : a Effect.t) =
    match e with
    | Sketching ->
      Some (fun (kont : (a, _) Effect.Deep.continuation) -> Effect.Deep.continue kont ())
    | _ -> None
  in
  Effect.Deep.match_with f () { retc = Fun.id; exnc = reraise; effc }

(* [sketching ()] is [true] iff a sketch's map is around the call. *)
let sketching () =
  match Effect.perform Sketching with
  | () -> true
  | exception Effect.Unhandled Sketching -> false

let mark ~curv y =
  Rune.custom_jvp
    Nx.Ptree.tensor
    Nx.Ptree.unit
    (fun _ ->
       ( ()
       , fun dy ->
           let ys = Rune.lanes directions dy in
           let rows t = reshape [| dim 0 t; -1 |] t in
           let block =
             cast float64 Infix.(rows ys *@ transpose (rows (Curv.apply curv ys)))
           in
           (* Every block is symmetric by construction — YᵀHY with H symmetric — so
              the accumulator is symmetrized up to floating-point asymmetry, which
              later decompositions would rather not see. *)
           let block = Infix.((block + transpose block) *$ 0.5) in
           Rune.Total.add curvature block ))
    y

(* [observe ~y ~curv l] marks [l] as a little loss in the prediction [y], whose
   Hessian with respect to [y] is [curv]: under a sketch the block joins the
   Gauss-Newton matrix, under any other driver the mark is inert. [l] must be a
   scalar computed from [y] by differentiable operations — the caller's own
   arithmetic defines both the value and the tangent of the loss the sketch
   returns. *)
let observe ~y ~curv l =
  if numel l <> 1
  then
    invalid_arg
      (Format.asprintf
         "Sofo.observe: the mini loss must be a scalar (one element), got shape %a"
         pp_shape
         (shape l));
  if sketching () then mark ~curv y

(* Packaged little losses: the value is computed with ordinary operations, so
   its tangent comes from the standard rules, and the curvature is derived from
   the same reduction, so the two cannot drift apart. *)

(* [mse ?w y target] is the mean weighted squared error
   [mean (w * (y - target)²)], marked with its exact curvature
   [diag (2 w / numel y)] — [Curv.scale (2 / numel y)] without weights. *)
let mse ?w ~target y =
  let shape_y = shape y in
  if shape_y <> shape target
  then invalid_arg "Sofo.mse: prediction and target must have the same shape";
  let n = float_of_int (numel y) in
  let d = Infix.(y - target) in
  let sq = Infix.(d * d) in
  let loss, curv =
    match w with
    | None -> mean sq, Curv.scale (2.0 /. n)
    | Some w ->
      let w = cast (Nx.dtype y) w in
      let two_over_n = Nx.Dtype.of_float (Nx.dtype y) (2.0 /. n) in
      mean Infix.(w * sq), Curv.diag Infix.(w *$ two_over_n)
  in
  observe ~y ~curv loss;
  loss

let sse ?w ~target y =
  let shape_y = shape y in
  if shape_y <> shape target
  then invalid_arg "Sofo.mse: prediction and target must have the same shape";
  let d = Infix.(y - target) in
  let sq = Infix.(d * d) in
  let loss, curv =
    match w with
    | None -> sum sq, Curv.scale 2.0
    | Some w ->
      let w = cast (Nx.dtype y) w in
      let two_over_n = Nx.Dtype.of_float (Nx.dtype y) 2.0 in
      sum Infix.(w * sq), Curv.diag Infix.(w *$ two_over_n)
  in
  observe ~y ~curv loss;
  loss

(* [softmax_ce y labels] is the cross entropy of the last axis of [y] against
   [labels], averaged over the leading (row) axes: [labels] has [y]'s shape
   without its last axis, and holds class indices in [0, num_classes). Its
   Hessian with respect to [y] is [1/rows] times [diag p − p pᵀ] per row, with
   [p = softmax y]. *)
let softmax_ce y labels =
  let shape_y = shape y in
  let rank = Array.length shape_y in
  if rank = 0 then invalid_arg "Sofo.softmax_ce: the prediction must have a class axis";
  let labels_shape = Array.sub shape_y 0 (rank - 1) in
  if shape labels <> labels_shape
  then
    invalid_arg
      "Sofo.softmax_ce: labels must have the prediction's shape without its class axis";
  let rows = Array.fold_left ( * ) 1 labels_shape in
  let p = softmax y in
  let idx = reshape (Array.append labels_shape [| 1 |]) labels in
  let picked = take_along_axis ~axis:(-1) idx (log_softmax y) in
  let loss = Infix.(-mean picked) in
  let curv = Curv.softmax_ce ~scale:(1.0 /. float_of_int rows) (Rune.detach p) in
  observe ~y ~curv loss;
  loss
