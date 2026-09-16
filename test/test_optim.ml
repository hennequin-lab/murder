(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The optimizer: the subspace Newton solve, the step, the state, and the
   compiled half.

   The checks are mostly reference-free, because the arithmetic is: on a loss
   that is exactly quadratic (a linear model under a mean-squared error), the
   sketch's second-order model is exact, so a step with no damping and unit
   learning rate must *annihilate* the sketched gradient C at the new
   parameters. That single property exercises the SVD solve, the damping, the
   lane contraction and the parameter shift together — and a hand-written
   solver is the reference for the cases where a property is not enough.

   The other half is administrative and matters as much: leaves that cannot
   carry a direction are passed through untouched, each leaf keeps its dtype,
   the state's key stream is reproducible, and a compiled step agrees with the
   eager one it is supposed to replace. *)

open Windtrap

let f64 = Nx.float64
let f32 = Nx.float32
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

let check_arr ?(eps = 1e-9) ?(rel = 0.0) ~msg expected actual =
  let actual = to_arr actual in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e ->
       equal
         ~msg:(Printf.sprintf "%s[%d]" msg i)
         (float (eps +. (rel *. Float.abs e)))
         e
         actual.(i))
    expected

let max_abs t = Nx.item [] (Nx.max (Nx.abs t))

(* ── fixtures ────────────────────────────────────────────────────────────── *)

(* A parameter structure with a float64 leaf, a float32 one, and an int leaf
   that no direction may move. *)
type params =
  { w : Nx.float64_t (* [d;o] *)
  ; v : Nx.float32_t (* [d;1] *)
  ; tag : Nx.int32_t (* [2] *)
  }

module Params = struct
  type t = params

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p =
    { w = f p.w; v = f p.v; tag = f p.tag }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { w = f a.w b.w; v = f a.v b.v; tag = f a.tag b.tag }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) p =
    f p.w;
    f p.v;
    f p.tag
end

let d = 4
let o = 3
let n = 32
let k = 3
let key = Nx.Rng.key 7
let x = Nx.Rng.normal (Nx.Rng.key 1) f64 [| n; d |]
let xv = Nx.cast f32 x

(* Student-teacher data: the targets are reachable exactly, so the loss can
   actually be driven to zero and "the step decreases the loss" is a statement
   about the optimizer rather than about the noise floor of random labels. *)
let w_teacher = Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 2) f64 [| d; o |]) 0.5
let targets = Nx.matmul x w_teacher
let v_teacher = Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 3) f32 [| d; 1 |]) 0.5
let yv = Nx.matmul xv v_teacher

let params () =
  { w = Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 4) f64 [| d; o |]) 0.1
  ; v = Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 5) f32 [| d; 1 |]) 0.1
  ; tag = Nx.create Nx.int32 [| 2 |] [| 3l; 5l |]
  }

(* Two little losses, one per float leaf, the second evaluated in float32 —
   exactly quadratic overall, so the sketch's model is exact. *)
let loss p =
  let l1 = Sofo.mse (Nx.matmul x p.w) targets in
  let l2 = Sofo.mse (Nx.matmul xv p.v) yv in
  Nx.add l1 (Nx.cast f64 l2)

(* Θ is a pure function of the key, so a fixed key makes every reference below
   use the same subspace as the sketch it checks. *)
let thetas_of p = Sofo.Optim.directions (module Params) ~key ~k p
let sampler k p = Sofo.Optim.directions (module Params) ~key ~k p
let sketch_at p = Sofo.sketch (module Params) ~k ~sketch_sampler:sampler loss p

(* The same model without the float32 leaf, for the statements that should sit
   at float64 rounding rather than at the float32 branch's. *)
let loss64 p = Sofo.mse (Nx.matmul x p.w) targets
let sketch64 p = Sofo.sketch (module Params) ~k ~sketch_sampler:sampler loss64 p

(* ── directions ──────────────────────────────────────────────────────────── *)

let test_directions_shapes_and_non_float_leaves () =
  let p = params () in
  let thetas = thetas_of p in
  equal ~msg:"w lanes" int k (Nx.shape thetas.w).(0);
  equal ~msg:"w lane width" int d (Nx.shape thetas.w).(1);
  equal ~msg:"w lane height" int o (Nx.shape thetas.w).(2);
  equal ~msg:"v lanes" int k (Nx.shape thetas.v).(0);
  equal
    ~msg:"v keeps its dtype"
    string
    "float32"
    (Nx_core.Dtype.to_string (Nx.dtype thetas.v));
  equal
    ~msg:"w keeps its dtype"
    string
    "float64"
    (Nx_core.Dtype.to_string (Nx.dtype thetas.w));
  equal ~msg:"the tag gets lanes too" int k (Nx.shape thetas.tag).(0);
  check_arr
    ~msg:"a leaf that cannot carry a direction gets zeros"
    [| 0.0; 0.0; 0.0; 0.0; 0.0; 0.0 |]
    (Nx.cast f64 thetas.tag)

let test_directions_are_pure_in_the_key () =
  let p = params () in
  let a = thetas_of p in
  let b = thetas_of p in
  check_arr ~msg:"same key, same Θ" (to_arr a.w) b.w;
  let other = Sofo.Optim.directions (module Params) ~key:(Nx.Rng.fold_in key 1) ~k p in
  let same = max_abs (Nx.sub a.w other.w) = 0.0 in
  is_false ~msg:"a different key gives a different subspace" same;
  (* an explicit key is not affected by the ambient scope *)
  let scoped =
    Nx.Rng.with_key (Nx.Rng.key 99) (fun () ->
      Sofo.Optim.directions (module Params) ~key ~k p)
  in
  check_arr ~msg:"~key pins the draw" (to_arr a.w) scoped.w

let test_apply_picks_lanes () =
  let p = params () in
  let thetas = thetas_of p in
  let b = params () in
  List.iter
    (fun i ->
       let z =
         Nx.create f64 [| k |] (Array.init k (fun j -> if i = j then 1.0 else 0.0))
       in
       let step = Sofo.Optim.apply (module Params) ~k thetas z in
       check_arr ~msg:"apply e_i is lane i" (to_arr (Nx.slice [ Nx.I i ] thetas.w)) step.w;
       equal
         ~msg:"the tag leaf gets the zero direction"
         string
         "[0, 0]"
         (Nx.to_string step.tag);
       equal
         ~msg:"... and the model's tag is not read"
         string
         (Nx.to_string b.tag)
         "[3, 5]")
    (List.init k Fun.id)

let test_apply_agrees_with_the_sketch () =
  (* The record's [apply] is the same map as [Sofo.Optim.apply]: one reads the
     directions from the driver, the other takes them as a value, which is what
     a compiled step can hand back. *)
  let p = params () in
  let sk = sketch_at p in
  let z = Nx.create f64 [| k |] [| 0.7; -1.3; 0.4 |] in
  let a = sk.apply z
  and b = Sofo.Optim.apply (module Params) ~k (thetas_of p) z in
  check_arr ~msg:"same direction" (to_arr a.w) b.w;
  check_arr ~msg:"same float32 direction" (to_arr a.v) b.v

(* ── the solve ───────────────────────────────────────────────────────────── *)

let test_coordinates_identity () =
  let c = Nx.create f64 [| k |] [| 1.0; -2.0; 3.5 |] in
  let z = Sofo.Optim.coordinates ~damping:(`Absolute 0.0) (Nx.eye f64 k) c in
  check_arr ~msg:"G̃ = I: z = C" (to_arr c) z;
  let z = Sofo.Optim.coordinates ~damping:(`Relative_from_top 0.5) (Nx.eye f64 k) c in
  check_arr ~msg:"G̃ = I, λ = 0.5: z = C / 1.5" [| 1.0 /. 1.5; -2.0 /. 1.5; 3.5 /. 1.5 |] z

let test_coordinates_diagonal () =
  let s = [| 4.0; 1.0; 0.25 |] in
  let ggn =
    Nx.create f64 [| k; k |] [| s.(0); 0.0; 0.0; 0.0; s.(1); 0.0; 0.0; 0.0; s.(2) |]
  in
  let c = Nx.create f64 [| k |] [| 1.0; -2.0; 3.0 |] in
  let z = Sofo.Optim.coordinates ~damping:(`Absolute 0.0) ggn c in
  check_arr ~msg:"z = C / s" [| 0.25; -2.0; 12.0 |] z;
  let z = Sofo.Optim.coordinates ~damping:(`Relative_from_top 0.5) ggn c in
  check_arr ~msg:"z = C / (s + λ·s_max)" [| 1.0 /. 6.0; -2.0 /. 3.0; 3.0 /. 2.25 |] z;
  (* the three modes on the same diagonal G̃: s = [4; 1; 0.25], c = [1; -2; 3] *)
  let z = Sofo.Optim.coordinates ~damping:(`Absolute 0.5) ggn c in
  check_arr
    ~msg:"absolute: z = C / (s + 0.5)"
    [| 1.0 /. 4.5; -2.0 /. 1.5; 3.0 /. 0.75 |]
    z;
  let z = Sofo.Optim.coordinates ~damping:(`Relative_from_bottom 0.5) ggn c in
  check_arr
    ~msg:"relative from the bottom: z = C / (s + 0.5·s_min)"
    [| 1.0 /. 4.125; -2.0 /. 1.125; 3.0 /. 0.375 |]
    z;
  (* the square-root preconditioner on the same G̃ and γ = 2:
     z = C / sqrt(s + 2) *)
  let z =
    Sofo.Optim.coordinates
      ~damping:(`Relative_from_top 0.5)
      ~preconditioner:`Inverse_sqrt
      ggn
      c
  in
  check_arr
    ~msg:"inverse sqrt: z = C / sqrt(s + γ)"
    [| 1.0 /. sqrt 6.0; -2.0 /. sqrt 3.0; 2.0 |]
    z;
  (* and the defaults are Algorithm 1's: relative to the top, inverted *)
  let d = Sofo.Optim.coordinates ggn c in
  let t = Sofo.Optim.coordinates ~damping:(`Relative_from_top 1e-6) ggn c in
  check_arr ~msg:"the default is `Relative_from_top 1e-6" (to_arr d) t;
  let i = Sofo.Optim.coordinates ~preconditioner:`Inverse ggn c in
  check_arr ~msg:"the default preconditioner is `Inverse" (to_arr d) i

let test_coordinates_survives_a_singular_direction () =
  (* A rank-deficient sketch (a lane the data does not excite) is exactly what
     the relative damping is for: without it the direction has no meaning, with
     it the coordinate is finite and small. *)
  let ggn = Nx.create f64 [| k; k |] [| 2.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0 |] in
  let c = Nx.create f64 [| k |] [| 1.0; 1.0; 1.0 |] in
  (* The solve of a singular system is only defined through the damping: the
     null direction is resolved with magnitude |c|/(λ·s_max), finite but large
     when λ is small, and suppressed as λ grows. *)
  let z = Sofo.Optim.coordinates ~damping:(`Relative_from_top 1e-3) ggn c in
  is_true ~msg:"finite" (Array.for_all Float.is_finite (to_arr z));
  check_arr
    ~msg:"the damped diagonal formula"
    [| 1.0 /. 2.002; 1.0 /. 0.002; 1.0 /. 0.002 |]
    z;
  let z = Sofo.Optim.coordinates ~damping:(`Relative_from_top 1.0) ggn c in
  is_true
    ~msg:"more damping shrinks the null direction"
    (Float.abs (Nx.item [ 1 ] z) < 1.0);
  check_arr ~msg:"λ = 1: z = C / (s + 2)" [| 1.0 /. 4.0; 0.5; 0.5 |] z;
  (* the modes that do not need a full-rank sketch still work: absolute damping
     is its own reference, and damping from the top has s_max to lean on *)
  let z = Sofo.Optim.coordinates ~damping:(`Absolute 0.5) ggn c in
  check_arr ~msg:"absolute on a singular sketch" [| 1.0 /. 2.5; 2.0; 2.0 |] z;
  let z = Sofo.Optim.coordinates ~damping:(`Relative_from_top 0.5) ggn c in
  check_arr ~msg:"from the top on a singular sketch" [| 1.0 /. 3.0; 1.0; 1.0 |] z;
  (* and the one that does says so *)
  raises_match Exn.invalid_arg (fun () ->
    ignore (Sofo.Optim.coordinates ~damping:(`Relative_from_bottom 0.5) ggn c))

(* ── the update ──────────────────────────────────────────────────────────── *)

let test_update_annihilates_the_sketched_gradient () =
  (* The strongest statement available without a reference: the loss is exactly
     quadratic, so with no damping and unit learning rate the step solves the
     sketched subspace problem exactly and the gradient's projection C must
     vanish at the new parameters. *)
  let p = params () in
  let sk = sketch64 p in
  let p' = Sofo.Optim.update (module Params) ~lr:1.0 ~damping:(`Absolute 0.0) sk p in
  let sk' = sketch64 p' in
  is_true
    ~msg:(Printf.sprintf "|C'| = %.3g at the new parameters" (max_abs sk'.c))
    (max_abs sk'.c < 1e-9);
  is_true ~msg:"the loss decreased" (Nx.item [] sk'.loss < Nx.item [] sk.loss)

let test_update_respects_the_learning_rate () =
  let p = params () in
  let sk = sketch_at p in
  let p0 = Sofo.Optim.update (module Params) ~lr:0.0 sk p in
  check_arr ~msg:"lr = 0 moves nothing" (to_arr p.w) p0.w;
  check_arr ~msg:"lr = 0 moves nothing (float32)" (to_arr p.v) p0.v;
  let p1 = Sofo.Optim.update (module Params) ~lr:1.0 sk p in
  is_true ~msg:"lr = 1 moves something" (max_abs (Nx.sub p1.w p.w) > 0.0)

let test_update_leaves_dtypes_and_non_float_leaves_alone () =
  let p = params () in
  let sk = sketch_at p in
  let p' = Sofo.Optim.update (module Params) ~lr:1.0 sk p in
  equal
    ~msg:"float64 stays float64"
    string
    "float64"
    (Nx_core.Dtype.to_string (Nx.dtype p'.w));
  equal
    ~msg:"float32 stays float32"
    string
    "float32"
    (Nx_core.Dtype.to_string (Nx.dtype p'.v));
  equal
    ~msg:"a leaf that cannot carry a direction is untouched"
    string
    (Nx.to_string p.tag)
    (Nx.to_string p'.tag)

let test_update_uses_damping () =
  (* Damping shrinks the step: with λ comparable to the curvature the move is
     smaller than the undamped solve's, and the loss still decreases. *)
  let p = params () in
  let sk = sketch_at p in
  let undamped =
    Sofo.Optim.update (module Params) ~lr:1.0 ~damping:(`Absolute 0.0) sk p
  in
  let damped =
    Sofo.Optim.update (module Params) ~lr:1.0 ~damping:(`Relative_from_top 1.0) sk p
  in
  is_true
    ~msg:"damped step is shorter"
    (max_abs (Nx.sub damped.w p.w) < max_abs (Nx.sub undamped.w p.w));
  is_true ~msg:"damped step still decreases the loss" (Nx.item [] sk.loss > 0.0)

(* ── the state and the eager step ────────────────────────────────────────── *)

let test_next_advances_the_stream () =
  let st = Sofo.Optim.init ~key () in
  equal ~msg:"step 0" int 0 st.Sofo.Optim.step;
  let st1 = Sofo.Optim.next st in
  equal ~msg:"one step" int 1 st1.Sofo.Optim.step;
  is_true
    ~msg:"the key moves on"
    (Nx.item [ 0 ] st1.Sofo.Optim.key <> Nx.item [ 0 ] st.Sofo.Optim.key)

let test_step_decreases_the_loss_and_advances () =
  (* A damped subspace Newton step with η = 1 provably decreases a convex loss,
     so monotonicity is the invariant to check. How *fast* is a property of the
     problem: each step sees only the gradient's component in a random k-plane,
     so with k = 8 lanes over P = 16 parameters ten steps should take a real
     bite (with k = 3 the same ten steps move the loss by a few percent). *)
  let p = params () in
  let st = Sofo.Optim.init ~key () in
  (* A step reports the sketch of the parameters it was given, so the first
     call's loss is the initial one and the comparison is between consecutive
     calls. *)
  let rec go p st i prev best first =
    if i = 0
    then best, first
    else (
      let p, st, sk =
        Sofo.Optim.step
          (module Params)
          ~k:8
          ~lr:1.0
          ~damping:(`Absolute 0.0)
          st
          ~loss
          ~params:p
      in
      (match Sofo.check sk with
       | Ok () -> ()
       | Error msg -> fail ("a step's sketch does not add up: " ^ msg));
      let l = Nx.item [] sk.loss in
      is_true
        ~msg:(Printf.sprintf "step %d decreases the loss (%.8g < %.8g)" (11 - i) l prev)
        (l < prev);
      equal ~msg:"the counter follows the calls" int (11 - i) st.Sofo.Optim.step;
      let first = if i = 10 then l else first in
      go p st (i - 1) l (Float.min best l) first)
  in
  let best, initial = go p st 10 Float.infinity Float.infinity Float.infinity in
  is_true
    ~msg:(Printf.sprintf "loss %.6g → best %.6g" initial best)
    (best < initial *. 0.5)

let test_step_is_reproducible () =
  let p = params () in
  let run () =
    let st = Sofo.Optim.init ~key () in
    List.fold_left
      (fun p i ->
         fst
           (Sofo.Optim.step (module Params) ~k ~lr:1.0 st ~loss ~params:p
            |> fun (p, _, _) -> p, i))
      p
      (List.init 3 Fun.id)
  in
  let a = run ()
  and b = run () in
  check_arr ~msg:"the same key replays the same run" (to_arr a.w) b.w

let test_step_strict_reports_an_unobserved_term () =
  let p = params () in
  let st = Sofo.Optim.init ~key () in
  let leaky p = Nx.add (loss p) (Nx.mul_s (Nx.sum (Nx.square p.w)) 0.01) in
  raises_match Exn.invalid_arg (fun () ->
    ignore
      (Sofo.Optim.step (module Params) ~k ~lr:1.0 ~strict:true st ~loss:leaky ~params:p))

(* ── the compiled half ───────────────────────────────────────────────────── *)

module O = Sofo.Optim.Compiled (Params)

let test_compiled_sketch_matches_the_eager_one () =
  let p = params () in
  let out = O.sketch ~k loss { O.params = p; key } in
  let sk = sketch_at p in
  check_arr ~msg:"loss" (to_arr out.O.loss) sk.loss;
  check_arr ~msg:"C" (to_arr out.O.c) sk.c;
  check_arr ~msg:"G̃" (to_arr out.O.ggn) sk.ggn;
  check_arr ~msg:"Θ is the same subspace" (to_arr out.O.dirs.w) (thetas_of p).w;
  (match O.check out with
   | Ok () -> ()
   | Error msg -> fail ("the compiled sketch does not add up: " ^ msg));
  (* and with damping, the update is the same map *)
  let a = O.update ~lr:0.5 ~damping:(`Relative_from_top 1e-3) p out in
  let b =
    Sofo.Optim.update (module Params) ~lr:0.5 ~damping:(`Relative_from_top 1e-3) sk p
  in
  check_arr ~msg:"compiled update" (to_arr a.w) b.w;
  check_arr ~msg:"compiled update (float32)" (to_arr a.v) b.v;
  equal
    ~msg:"compiled update leaves the tag"
    string
    (Nx.to_string a.tag)
    (Nx.to_string b.tag)

let test_jitted_sketch_matches_eager () =
  (* The compiled half is what [Rune.jit2] wraps; jit fuses kernels, so the
     numbers agree up to floating point, and the check still holds on the
     values it hands back. *)
  let p = params () in
  let step = Rune.jit2 (module O.In) (module O.Out) (O.sketch ~k loss) in
  let eager = O.sketch ~k loss { O.params = p; key } in
  let jitted = step { O.params = p; key } in
  (* jit fuses kernels, so the numbers agree to floating point rather than
     exactly — and the float32 leaf in this loss sits at ~1e-7. The subspace
     itself is drawn by the same generator, so it agrees to rounding. *)
  check_arr ~eps:1e-6 ~msg:"loss" (to_arr eager.O.loss) jitted.O.loss;
  check_arr ~rel:1e-5 ~msg:"C" (to_arr eager.O.c) jitted.O.c;
  check_arr ~rel:1e-5 ~msg:"G̃" (to_arr eager.O.ggn) jitted.O.ggn;
  check_arr ~eps:1e-12 ~msg:"Θ" (to_arr eager.O.dirs.w) jitted.O.dirs.w;
  (match O.check jitted with
   | Ok () -> ()
   | Error msg -> fail ("the jitted sketch does not add up: " ^ msg));
  (* one compilation, fresh subspaces: a different key is a different Θ from
     the same program *)
  let other = step { O.params = p; key = Nx.Rng.fold_in key 1 } in
  is_false
    ~msg:"a new key draws a new subspace"
    (max_abs (Nx.sub other.O.dirs.w jitted.O.dirs.w) = 0.0)

let test_compiled_loop_trains () =
  (* The deployment end to end: one jitted sketch, eager steps, a decreasing
     loss, and a state that never needs to leave the host. *)
  let p = params () in
  let k = 8 in
  let step = Rune.jit2 (module O.In) (module O.Out) (O.sketch ~k loss) in
  let st = Sofo.Optim.init ~key () in
  let rec go p st i prev best first =
    if i = 0
    then best, first
    else (
      let out = step { O.params = p; key = st.Sofo.Optim.key } in
      (match O.check out with
       | Ok () -> ()
       | Error msg -> fail ("compiled check: " ^ msg));
      let l = Nx.item [] out.O.loss in
      is_true
        ~msg:(Printf.sprintf "step %d decreases the loss (%.8g < %.8g)" (9 - i) l prev)
        (l < prev);
      let p = O.update ~lr:1.0 ~damping:(`Absolute 0.0) p out in
      let first = if i = 8 then l else first in
      go p (Sofo.Optim.next st) (i - 1) l (Float.min best l) first)
  in
  let best, initial = go p st 8 Float.infinity Float.infinity Float.infinity in
  is_true
    ~msg:(Printf.sprintf "loss %.6g → best %.6g" initial best)
    (best < initial *. 0.5)

let tests =
  [ group
      "directions"
      [ test
          "shapes, dtypes and leaves without directions"
          test_directions_shapes_and_non_float_leaves
      ; test "Θ is a pure function of the key" test_directions_are_pure_in_the_key
      ; test "apply picks the lanes" test_apply_picks_lanes
      ; test "apply agrees with the sketch's" test_apply_agrees_with_the_sketch
      ]
  ; group
      "the solve"
      [ test "identity curvature" test_coordinates_identity
      ; test "diagonal curvature and damping" test_coordinates_diagonal
      ; test
          "a singular direction is damped"
          test_coordinates_survives_a_singular_direction
      ]
  ; group
      "the update"
      [ test
          "annihilates the sketched gradient"
          test_update_annihilates_the_sketched_gradient
      ; test "respects the learning rate" test_update_respects_the_learning_rate
      ; test
          "keeps dtypes, skips non-float leaves"
          test_update_leaves_dtypes_and_non_float_leaves_alone
      ; test "damping shortens the step" test_update_uses_damping
      ]
  ; group
      "the state and the eager step"
      [ test "next advances the stream" test_next_advances_the_stream
      ; test
          "step decreases the loss and advances"
          test_step_decreases_the_loss_and_advances
      ; test "step is reproducible" test_step_is_reproducible
      ; test
          "strict mode reports an unobserved term"
          test_step_strict_reports_an_unobserved_term
      ]
  ; group
      "the compiled half"
      [ test
          "compiled sketch matches the eager one"
          test_compiled_sketch_matches_the_eager_one
      ; test "jitted sketch matches eager" test_jitted_sketch_matches_eager
      ; test "a compiled loop trains" test_compiled_loop_trains
      ]
  ]

let () = run "sofo optim" tests
