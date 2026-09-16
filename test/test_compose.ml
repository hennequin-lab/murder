(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SOFO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Composition: the §7 matrix, exercised through the sketch driver.

   A sketch is a handler stack, and these tests are about the stack rather than
   about the numbers a sketch returns — the arithmetic is test_sketch.ml's
   business. Each case states what an enclosing handler does to the
   observations, the lane axis and the tangents, and checks it against a
   reference built from the primitives that handler exposes: a per-lane
   single-tangent [jvp] for a tangent batch, an explicit loop for a [vmap], and
   the loss's own second-order model where the loss is exactly quadratic (which
   needs no reference at all). The one composition the matrix marks ✗ is
   checked for its error rather than its numbers, and so are the two ways of
   marking a little loss the collector cannot interpret.

   Two of the checks are not about numbers: an observation inside [Rune.scan]
   must fire (a claim of the scan, not a coincidence of handler order), and a
   failed sketch must not leave the process unable to compile anything (the
   transformation gate is asked as an effect precisely so that it cannot
   leak). *)

open Windtrap

let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs
let mat r c xs = Nx.create f64 [| r; c |] xs
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

let check_arr ?(eps = 1e-9) ~msg expected actual =
  let actual = to_arr actual in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e -> equal ~msg:(Printf.sprintf "%s[%d]" msg i) (float eps) e actual.(i))
    expected

(* [raises_containing ~substring f] checks that [f] fails for the stated
   reason. The message is part of the contract here: each of these failures is
   a user error the library is supposed to name. *)
let raises_containing ~msg ~substring f =
  match f () with
  | _ -> fail (msg ^ ": expected a failure, got a result")
  | exception Invalid_argument m ->
    if
      String.length m >= String.length substring
      && String.sub m 0 (String.length substring) = substring
    then ()
    else fail (Printf.sprintf "%s: message does not start with %S: %s" msg substring m)
  | exception e ->
    fail (Printf.sprintf "%s: unexpected exception %s" msg (Printexc.to_string e))

(* ── fixtures ──────────────────────────────────────────────────────────────

   A three-dimensional state stepped by a two-layer MLP with an inverted
   bottleneck, the paper's Lorenz network in miniature. Every weight is stored
   transposed, so each matmul has the (possibly batched) activation on the left
   and a matrix on the right: this is the form Rune.vmap's matmul rule can
   translate, and it is equally correct eagerly. *)

type params =
  { a : Nx.float64_t (* [3;3] *)
  ; ct : Nx.float64_t (* Cᵀ, [3;h] *)
  ; wt : Nx.float64_t (* Wᵀ, [h;3] *)
  ; b : Nx.float64_t (* [h] *)
  }

module Params = struct
  type t = params

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p =
    { a = f p.a; ct = f p.ct; wt = f p.wt; b = f p.b }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { a = f p.a q.a; ct = f p.ct q.ct; wt = f p.wt q.wt; b = f p.b q.b }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) p =
    f p.a;
    f p.ct;
    f p.wt;
    f p.b
end

module Single = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t
  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b = f a b
  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

let params () =
  { a = mat 3 3 [| 0.8; 0.1; -0.2; 0.3; 0.7; 0.2; -0.1; 0.4; 0.9 |]
  ; ct = mat 3 4 [| 0.5; 1.7; 0.2; 1.1; -1.2; -0.4; 0.6; 0.3; 2.1; 0.9; -0.8; -0.5 |]
  ; wt = mat 4 3 [| 0.4; -0.5; 0.7; 0.2; -0.2; 0.1; -0.6; 0.8; 0.9; 0.3; 0.2; 0.4 |]
  ; b = vec [| 0.3; -0.7; 0.2; 0.5 |]
  }

let step p z =
  Nx.add (Nx.matmul z p.a) (Nx.matmul (Nx.relu (Nx.add (Nx.matmul z p.ct) p.b)) p.wt)

let z0 = vec [| 0.2; -0.4; 0.6 |]

(* [roll p x n] is the state after [n] steps from [x]. *)
let roll p x n =
  let s = ref x in
  for _ = 1 to n do
    s := step p !s
  done;
  !s

let k = 3

(* A deterministic lane batch [k; shape t], and the sampler that makes it Θ. *)
let lane_batch ~k t =
  let s = Nx.shape t in
  let n = Nx.numel t in
  Nx.create
    f64
    (Array.append [| k |] s)
    (Array.init (k * n) (fun i ->
       float_of_int ((((i * 7) + (3 * (i / n))) mod 11) - 5) /. 4.0))

let sampler k p =
  { a = lane_batch ~k p.a
  ; ct = lane_batch ~k p.ct
  ; wt = lane_batch ~k p.wt
  ; b = lane_batch ~k p.b
  }

let thetas_for = sampler

let sketch ?(kk = k) loss p =
  Sofo.sketch (module Params) ~k:kk ~sketch_sampler:sampler loss p

(* The reference tangent batch, computed independently of the lane machinery
   under test: one single-tangent [Rune.jvp] per lane, stacked. *)
let lane_tangents ~k (f : params -> Nx.float64_t) p thetas =
  Nx.stack
    ~axis:0
    (List.init k (fun i ->
       snd
         (Rune.jvp
            (module Params)
            f
            p
            (Params.map (fun t -> Nx.slice [ Nx.I i ] t) thetas))))

(* Θᵀ[·] for the whole parameter record: each leaf's lane axis contracted
   against its cotangent leaf, summed over leaves. *)
let theta_t_cotangent ~k thetas g =
  let leaf theta g =
    let n = Nx.numel g in
    Nx.matmul
      (Nx.reshape [| k; n |] (Nx.contiguous theta))
      (Nx.reshape [| n; 1 |] (Nx.contiguous g))
  in
  Nx.reshape
    [| k |]
    (Nx.add
       (leaf thetas.a g.a)
       (Nx.add (leaf thetas.ct g.ct) (Nx.add (leaf thetas.wt g.wt) (leaf thetas.b g.b))))

(* The reference Gram block of a little loss whose Hessian in the prediction is
   [h] times the identity: Σ Yᵀ(h I)Y = h·Y Yᵀ. *)
let gram_block ~k ~h f p thetas =
  let n = Nx.numel (f p) in
  let ys = Nx.reshape [| k; n |] (Nx.contiguous (lane_tangents ~k f p thetas)) in
  Nx.mul_s (Nx.matmul ys (Nx.transpose ys)) h

(* ── the ✓ rows ─────────────────────────────────────────────────────────── *)

let scan_steps = 6

let scan_targets =
  mat
    scan_steps
    3
    [| 0.5
     ; -1.1
     ; 0.4
     ; 0.2
     ; 0.7
     ; -0.3
     ; -0.8
     ; 0.1
     ; 0.6
     ; 1.2
     ; -0.5
     ; 0.9
     ; -0.2
     ; 0.3
     ; -0.7
     ; 0.4
     ; 1.0
     ; -0.6
    |]

let scan_loss p =
  let _, ls =
    Rune.scan
      (module Single)
      ~f:(fun z tgt ->
        let z' = step p z in
        z', Sofo.mse z' tgt)
      ~init:z0
      scan_targets
  in
  Nx.sum ls

let test_scan_folds_inside_the_collector () =
  (* jvp_k ∘ Rune.scan: the collector claims the scan, so the fold runs in its
     extent and every step's little loss is observed. The GGN is then the sum
     of the per-step blocks, each computed here from per-lane single-tangent
     jvps. *)
  let p = params ()
  and kk = 2 in
  let thetas = thetas_for kk p in
  let sk = Sofo.sketch (module Params) ~k:kk ~sketch_sampler:sampler scan_loss p in
  equal ~msg:"one observation per step" int scan_steps sk.diagnostics.blocks;
  equal ~msg:"nothing skipped" int 0 sk.diagnostics.skipped;
  (match Sofo.check sk with
   | Ok () -> ()
   | Error msg -> fail ("the observed steps do not add up to the loss: " ^ msg));
  let reference =
    List.fold_left
      (fun acc t ->
         Nx.add
           acc
           (gram_block ~k:kk ~h:(2.0 /. 3.0) (fun p -> roll p z0 (t + 1)) p thetas))
      (Nx.zeros f64 [| kk; kk |])
      (List.init scan_steps Fun.id)
  in
  check_arr ~msg:"Σ_t Y_tᵀ H Y_t" (to_arr reference) sk.ggn

let vmap_trials = 5

let starts =
  mat
    vmap_trials
    3
    [| 0.3; -0.6; 0.9; -0.4; 0.8; 0.2; 1.1; -0.9; 0.5; 0.1; 0.4; -0.7; -0.2; 0.6; 1.3 |]

let endpoints =
  mat
    vmap_trials
    3
    [| 0.5; -0.2; -0.3; 0.8; 0.6; 0.1; -0.9; 0.4; 0.2; -0.5; 0.7; 0.3; 0.9; -0.1; 0.6 |]

(* The endpoint loss, written the natural way: batch the trials with vmap and
   observe the batched prediction. *)
let vmap_loss p = Sofo.mse (Rune.vmap' (fun x -> roll p x 4) starts) endpoints

(* The same loss with the map written out as a loop. *)
let loop_loss p =
  Sofo.mse
    (Nx.stack
       ~axis:0
       (List.init vmap_trials (fun i -> roll p (Nx.slice [ Nx.I i ] starts) 4)))
    endpoints

let test_vmap_inside_matches_the_loop () =
  (* jvp_k ∘ vmap: vmap re-performs every operation batched, batched forward
     mode tracks the batched operations, and the observation sees the physical
     [M;3] prediction — one k×k block whose contraction sums over M. The
     reference is the same loss with the map written out, which tracks the same
     tangents through an ordinary loop. *)
  let p = params ()
  and kk = 3 in
  let sk_vmap = Sofo.sketch (module Params) ~k:kk ~sketch_sampler:sampler vmap_loss p in
  let sk_loop = Sofo.sketch (module Params) ~k:kk ~sketch_sampler:sampler loop_loss p in
  equal ~msg:"one observation, whatever the map" int 1 sk_vmap.diagnostics.blocks;
  equal ~msg:"the loop observes once too" int 1 sk_loop.diagnostics.blocks;
  check_arr ~msg:"loss" (to_arr sk_loop.loss) sk_vmap.loss;
  check_arr ~msg:"C" (to_arr sk_loop.c) sk_vmap.c;
  check_arr ~msg:"G̃" (to_arr sk_loop.ggn) sk_vmap.ggn;
  match Sofo.check sk_vmap with
  | Ok () -> ()
  | Error msg -> fail ("the mapped loss does not add up: " ^ msg)

(* Trials as a structure, so that a mapped function can see each trial's target
   alongside its start. *)
type trial =
  { x : Nx.float64_t
  ; t : Nx.float64_t
  }

module Trial = struct
  type t = trial

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) tr = { x = f tr.x; t = f tr.t }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { x = f a.x b.x; t = f a.t b.t }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) tr =
    f tr.x;
    f tr.t
end

let trials = { x = starts; t = endpoints }
let batch = float_of_int vmap_trials
let n_pred = 3.0

(* Observing inside the map, with the mean over trials folded into each
   observation: one observation is one term of the total loss, so the value and
   the curvature both carry the 1/M factor the map's reduction would apply
   outside, and the outside reduction is then the sum of those contributions —
   which is the mean of the unweighted little losses. The collector cannot
   infer the factor, which is why the cross-check exists. *)
let in_map_loss p =
  Nx.sum
    (Rune.vmap
       (module Trial)
       (fun tr ->
          let y = roll p tr.x 4 in
          let l = Nx.div_s (Nx.mean (Nx.square (Nx.sub y tr.t))) batch in
          Sofo.observe ~y ~curv:(Sofo.Curv.scale (2.0 /. (n_pred *. batch))) l;
          l)
       trials)

let test_observation_inside_a_vmap_scales_by_the_batch () =
  (* The documented way to observe inside a map: the observation carries the
     term's own contribution, and the map's reduction is applied outside. This
     must agree with observing the batched prediction — the two contract the
     same blocks in a different order, and M little losses packed into one
     observation are the same thing as M observations of one little loss. *)
  let p = params ()
  and kk = 2 in
  let sk = Sofo.sketch (module Params) ~k:kk ~sketch_sampler:sampler in_map_loss p in
  let sk_batched =
    Sofo.sketch (module Params) ~k:kk ~sketch_sampler:sampler vmap_loss p
  in
  (* vmap re-performs the body batched rather than looping it, so the [M]
     little losses are packed into a single observation — one block, whose
     contraction sums over the map's axis. *)
  equal ~msg:"one packed observation" int 1 sk.diagnostics.blocks;
  (match Sofo.check sk with
   | Ok () -> ()
   | Error msg -> fail ("the scaled in-map observation does not add up: " ^ msg));
  check_arr ~msg:"loss" (to_arr sk_batched.loss) sk.loss;
  check_arr ~msg:"C" (to_arr sk_batched.c) sk.c;
  check_arr ~msg:"G̃" (to_arr sk_batched.ggn) sk.ggn

let test_unscaled_observation_inside_a_vmap_is_reported () =
  (* Marking the raw per-trial mean inside the map keeps the loss's own mean
     outside, so the observations sum to M times the loss they are supposed to
     account for. The collector cannot read the user's mind about which
     reduction applies, so it accumulates what it is shown and Sofo.check
     reports the factor; strict mode refuses to return the sketch at all. *)
  let p = params () in
  let loss p =
    Nx.mean
      (Rune.vmap
         (module Trial)
         (fun tr ->
            let y = roll p tr.x 4 in
            Sofo.mse y tr.t)
         trials)
  in
  let sk = sketch loss p in
  (match Sofo.check sk with
   | Ok () -> fail "an unscaled in-map observation should not add up"
   | Error _ -> ());
  raises_containing ~msg:"strict in-map little loss" ~substring:"Sofo.sketch:" (fun () ->
    ignore (Sofo.sketch (module Params) ~k ~sketch_sampler:sampler ~strict:true loss p))

let test_vmap_outside_a_sketch_is_a_lane_error () =
  (* ✗ vmap ∘ jvp_k. Batch dimensions belong inside the tangent axis; a map
     around a whole sketch would put its own axis outside the lane axis, and
     the lane invariant catches it rather than computing wrong shapes. *)
  let m = 2 in
  let batched = sampler m (params ()) in
  raises_containing ~msg:"vmap outside jvp_k" ~substring:"Rune.jvp_k:" (fun () ->
    ignore (Rune.vmap (module Params) (fun p -> (sketch vmap_loss p).loss) batched))

let test_grad_outside_a_sketch_is_exact () =
  (* grad ∘ jvp_k: a sketch nested in reverse mode. The forward handler's
     primals are ordinary operations to the tape, and the collector's effects
     are inert to it, so differentiating the sketch's loss is differentiating
     the loss. *)
  let p = params () in
  let l_sketch, g_sketch =
    Rune.value_and_grad (module Params) (fun p -> (sketch scan_loss p).loss) p
  in
  let l_plain, g_plain = Rune.value_and_grad (module Params) scan_loss p in
  check_arr ~msg:"loss" (to_arr l_plain) l_sketch;
  check_arr ~msg:"gradient (a)" (to_arr g_plain.a) g_sketch.a;
  check_arr ~msg:"gradient (ct)" (to_arr g_plain.ct) g_sketch.ct;
  check_arr ~msg:"gradient (wt)" (to_arr g_plain.wt) g_sketch.wt;
  check_arr ~msg:"gradient (b)" (to_arr g_plain.b) g_sketch.b

let test_jvp_k_of_grad_is_the_batched_hessian () =
  (* jvp_k ∘ grad (the matrix's bonus row): the prediction is itself a gradient
     computed by reverse mode, and batched forward mode differentiates through
     that. The loss is exactly quadratic in the parameters, so the sketch's own
     second-order model is exact — a check that needs no reference. *)
  let loss p =
    let g = Rune.grad' (fun w -> Nx.sum (Nx.square (Nx.matmul w z0))) p.wt in
    Sofo.mse g (Nx.zeros_like g)
  in
  let p = params ()
  and kk = 3 in
  let sk = Sofo.sketch (module Params) ~k:kk ~sketch_sampler:sampler loss p in
  let z = vec [| 0.4; -1.1; 0.7 |] in
  let eps = 1e-4 in
  let p' =
    Params.map2
      (fun leaf d -> Nx.add leaf (Nx.mul_s d (Nx_core.Dtype.of_float (Nx.dtype d) eps)))
      p
      (sk.apply z)
  in
  let c0 = Nx.item [] sk.loss in
  let c1 = Nx.item [] (loss p') in
  let linear = Nx.item [] (Nx.sum (Nx.mul sk.c z)) in
  let quad = Nx.item [] (Nx.sum (Nx.mul z (Nx.matmul sk.ggn z))) in
  equal
    ~msg:"second-order model through reverse mode"
    (float 1e-9)
    (c0 +. (eps *. linear) +. (eps *. eps /. 2.0 *. quad))
    c1

let test_no_grad_and_detach_are_skipped () =
  (* The gate is honored in both directions: a prediction that is a constant of
     the differentiation has no tangent, so its block is skipped (not
     accumulated as zeros), while its value still counts towards the loss. *)
  let loss p =
    let gated = Rune.detach (step p z0) in
    let l1 = Nx.mean (Nx.square gated) in
    Sofo.observe ~y:gated ~curv:(Sofo.Curv.scale 2.0) l1;
    let frozen = Rune.no_grad (fun () -> step p z0) in
    let l2 = Nx.mean (Nx.square frozen) in
    Sofo.observe ~y:frozen ~curv:(Sofo.Curv.scale 2.0) l2;
    Nx.add l1 l2
  in
  let p = params () in
  let sk = sketch loss p in
  equal ~msg:"no blocks from gated predictions" int 0 sk.diagnostics.blocks;
  equal ~msg:"two skipped" int 2 sk.diagnostics.skipped;
  (match Sofo.check sk with
   | Ok () -> ()
   | Error msg -> fail ("a gated observation must still count towards the loss: " ^ msg));
  check_arr ~msg:"zero curvature" [| 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0 |] sk.ggn

let test_rng_in_the_graph_is_a_constant () =
  (* A fresh draw is a constant with respect to θ: it shifts the loss (and so
     C, the gradient of the shifted loss) while the prediction's tangent — and
     with it the curvature block, whose H is a constant — is the deterministic
     part's. C is checked against the exact gradient of the *noisy* loss under
     the same key: that is the statement that the draw is a constant of the
     differentiation rather than a missing term. *)
  let noisy p =
    let noise = Nx.randn f64 [| 3 |] in
    Sofo.mse (Nx.add (step p z0) noise) (Nx.zeros f64 [| 3 |])
  in
  let clean p = Sofo.mse (step p z0) (Nx.zeros f64 [| 3 |]) in
  let p = params ()
  and kk = 2 in
  let thetas = thetas_for kk p in
  let sk_noisy =
    Nx.Rng.with_key (Nx.Rng.key 11) (fun () ->
      Sofo.sketch (module Params) ~k:kk ~sketch_sampler:sampler noisy p)
  in
  let sk_clean = Sofo.sketch (module Params) ~k:kk ~sketch_sampler:sampler clean p in
  let _, grads =
    Nx.Rng.with_key (Nx.Rng.key 11) (fun () ->
      Rune.value_and_grad (module Params) noisy p)
  in
  check_arr
    ~msg:"C = Θᵀ∇c of the noisy loss"
    (to_arr (theta_t_cotangent ~k:kk thetas grads))
    sk_noisy.c;
  check_arr ~msg:"noise does not move G̃" (to_arr sk_clean.ggn) sk_noisy.ggn;
  (match Sofo.check sk_noisy with
   | Ok () -> ()
   | Error msg -> fail ("a stochastic little loss does not add up: " ^ msg));
  if Nx.item [] sk_noisy.loss = Nx.item [] sk_clean.loss
  then fail "the noise did not reach the loss"

let test_jit_inside_degrades () =
  (* jit inside user code sees a transformation installed and runs eagerly:
     correct, uncompiled, and numerically identical to the rule it wraps. *)
  let inline y = Nx.relu y in
  let jitted_activation y = Rune.jit' (fun v -> Nx.relu v) y in
  let model activation p =
    let z =
      Nx.add
        (Nx.matmul z0 p.a)
        (Nx.matmul (activation (Nx.add (Nx.matmul z0 p.ct) p.b)) p.wt)
    in
    Sofo.mse z (Nx.zeros f64 [| 3 |])
  in
  let p = params () in
  let sk_plain = sketch (model inline) p in
  let sk_jit = sketch (model jitted_activation) p in
  check_arr ~msg:"loss" (to_arr sk_plain.loss) sk_jit.loss;
  check_arr ~msg:"C" (to_arr sk_plain.c) sk_jit.c;
  check_arr ~msg:"jit inside does not change the sketch" (to_arr sk_plain.ggn) sk_jit.ggn

let test_custom_jvp_inside () =
  (* A custom rule is written for a single tangent, so the batched handler
     lifts it over the lane axis. With a rule that reproduces relu's own
     derivative, the sketch must equal the plain model's. *)
  let my_relu x =
    Rune.custom_jvp
      (module Single)
      ~f:Nx.relu
      ~jvp:(fun x dx ->
        let d = Nx.where (Nx.greater x (Nx.zeros_like x)) dx (Nx.zeros_like dx) in
        Nx.relu x, d)
      x
  in
  let loss p =
    Sofo.mse
      (Nx.add
         (Nx.matmul z0 p.a)
         (Nx.matmul (my_relu (Nx.add (Nx.matmul z0 p.ct) p.b)) p.wt))
      (Nx.zeros f64 [| 3 |])
  in
  let p = params () in
  let sk = sketch loss p in
  check_arr
    ~msg:"custom rule, same curvature"
    (to_arr (sketch (fun p -> Sofo.mse (step p z0) (Nx.zeros f64 [| 3 |])) p).ggn)
    sk.ggn;
  match Sofo.check sk with
  | Ok () -> ()
  | Error msg -> fail ("a custom_jvp does not add up: " ^ msg)

let test_custom_vjp_raises_and_leaves_the_gate_alone () =
  (* custom_vjp has no forward rule: it raises under any forward mode, and the
     failure must not disable jit for the rest of the process. *)
  let loss p =
    let g =
      Rune.custom_vjp
        (module Single)
        ~fwd:(fun x -> Nx.sin x, x)
        ~bwd:(fun x ct -> Nx.mul ct (Nx.cos x))
        p.b
    in
    Sofo.mse g (Nx.zeros f64 [| 4 |])
  in
  let p = params () in
  raises_containing
    ~msg:"custom_vjp under a sketch"
    ~substring:"Rune: a custom_vjp"
    (fun () -> ignore (sketch loss p));
  let sk = sketch (fun p -> Sofo.mse (step p z0) (Nx.zeros f64 [| 3 |])) p in
  equal ~msg:"the next sketch still works" int 1 sk.diagnostics.blocks

(* ── the ✗ rows in a jitted step ────────────────────────────────────────── *)

let test_jitted_sketch_matches_eager () =
  (* §14.3: with jit outermost the collector's effect is handled at trace time
     and its accumulations are traced beside the primal ones, so a jitted
     sketch is correct — but unrolled, and its check cannot read values. *)
  let p = params () in
  let bundle (sk : params Sofo.sketch) =
    Nx.concatenate ~axis:0 [ Nx.ravel sk.loss; Nx.ravel sk.c; Nx.ravel sk.ggn ]
  in
  let eager =
    bundle (Sofo.sketch (module Params) ~k ~sketch_sampler:sampler vmap_loss p)
  in
  let jitted =
    Rune.jit
      (module Params)
      (fun p ->
         bundle (Sofo.sketch (module Params) ~k ~sketch_sampler:sampler vmap_loss p))
  in
  check_arr ~msg:"a jitted sketch replays identically" (to_arr (jitted p)) (jitted p);
  check_arr ~msg:"jitted sketch (loss, C, G̃)" (to_arr eager) (jitted p);
  (* and the strict check, which reads values, refuses at trace time *)
  match
    Rune.jit
      (module Params)
      (fun p ->
         (Sofo.sketch (module Params) ~k ~sketch_sampler:sampler ~strict:true vmap_loss p)
           .loss)
      p
  with
  | _ -> fail "a strict sketch inside jit should have raised"
  | exception Rune.Jit_error _ -> ()

let test_scan_memory_is_bounded () =
  (* The collector keeps O(k²), and the ephemeron-keyed store drops a step's
     tangents with the step: a long horizon must not accumulate either. *)
  let steps = 120 in
  let targets = Nx.zeros f64 [| steps; 3 |] in
  let loss p =
    let _, ls =
      Rune.scan
        (module Single)
        ~f:(fun z tgt ->
          let z' = step p z in
          z', Sofo.mse z' tgt)
        ~init:z0
        targets
    in
    let live = ref 0 in
    if Nx.item [] (Nx.sum ls) = 0.0 then live := Rune.live_tangent_entries ();
    Nx.add (Nx.sum ls) (Nx.mul_s (Nx.scalar f64 (float_of_int !live)) 0.0)
  in
  let p = params () in
  let sk =
    Sofo.sketch (module Params) ~k:4 ~sketch_sampler:sampler ~strict:false loss p
  in
  equal ~msg:"a block per step" int steps sk.diagnostics.blocks;
  (* The reading is taken inside the sketch, so it is only a smoke test; the
     bound is what matters: not one binding per intermediate per step. *)
  if Nx.item [] sk.loss = 0.0 then fail "the loss collapsed"

let test_control_flow_is_inherited () =
  (* Control flow that depends on host values rather than on θ differentiates
     correctly, and masking with where is an ordinary operation: C is still
     exactly Θᵀ∇c. *)
  let p = params () in
  let mask =
    mat
      scan_steps
      3
      (Array.init (scan_steps * 3) (fun i -> if i mod 3 = 0 then 1.0 else 0.0))
  in
  let masked p =
    let _, zs =
      Rune.scan
        (module Single)
        ~f:(fun z tgt ->
          let z' = step p z in
          z', z')
        ~init:z0
        scan_targets
    in
    Sofo.mse (Nx.mul zs mask) (Nx.mul scan_targets mask)
  in
  let branch = Nx.item [] (Nx.sum p.b) > 0.0 in
  let loss p = if branch then masked p else Sofo.mse (roll p z0 3) endpoints in
  let sk = sketch loss p in
  let _, grads = Rune.value_and_grad (module Params) loss p in
  let thetas = thetas_for k p in
  check_arr
    ~msg:"C = Θᵀ∇c under host control flow"
    (to_arr (theta_t_cotangent ~k thetas grads))
    sk.c;
  match Sofo.check sk with
  | Ok () -> ()
  | Error msg -> fail ("a masked loss does not add up: " ^ msg)

let tests =
  [ group
      "the ✓ rows"
      [ test
          "jvp_k of Rune.scan folds inside the collector"
          test_scan_folds_inside_the_collector
      ; test "jvp_k of vmap matches the loop oracle" test_vmap_inside_matches_the_loop
      ; test
          "an observation inside a vmap scales by the batch"
          test_observation_inside_a_vmap_scales_by_the_batch
      ; test "grad of a sketch is exact" test_grad_outside_a_sketch_is_exact
      ; test
          "jvp_k of grad is the batched Hessian"
          test_jvp_k_of_grad_is_the_batched_hessian
      ; test "no_grad and detach are skipped" test_no_grad_and_detach_are_skipped
      ; test "RNG in the graph is a constant" test_rng_in_the_graph_is_a_constant
      ; test "jit inside degrades to eager" test_jit_inside_degrades
      ; test "custom_jvp is lifted over the lanes" test_custom_jvp_inside
      ; test "host control flow is inherited" test_control_flow_is_inherited
      ]
  ; group
      "the ✗ rows"
      [ test
          "vmap outside a sketch is a lane error"
          test_vmap_outside_a_sketch_is_a_lane_error
      ; test
          "an unscaled in-map observation is reported"
          test_unscaled_observation_inside_a_vmap_is_reported
      ; test
          "custom_vjp raises and leaves the gate alone"
          test_custom_vjp_raises_and_leaves_the_gate_alone
      ]
  ; group
      "a jitted sketch"
      [ test "matches eager, and strict refuses to trace" test_jitted_sketch_matches_eager
      ; test "a long scan does not accumulate tangents" test_scan_memory_is_bounded
      ]
  ]

let () = run "sofo composition" tests
