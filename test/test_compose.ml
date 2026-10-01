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
   checked for its error rather than its numbers, and so is the way a mark
   enters with weight one where the caller's reduction does not.

   Two of the checks are not about numbers: an observation inside [Rune.scan]
   must fire (a claim of the scan, not a coincidence of handler order), and a
   failed sketch must not leave the process unable to compile anything (the
   transformation gate is asked as an effect precisely so that it cannot
   leak). *)

open Windtrap

let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs
let mat r c xs = Nx.create f64 [| r; c |] xs

let check_arr ?(eps = 1e-9) ~msg expected actual =
  let actual = Nx.to_array actual in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e -> equal ~msg:(Printf.sprintf "%s[%d]" msg i) (float eps) e actual.(i))
    expected

(* [check_rel ~msg expected actual] compares two runs of the same computation
   that may have been optimized differently: the long rollout's values grow
   with the horizon, so a fixed absolute tolerance says nothing. *)
let check_rel ~msg expected actual =
  let expected = Nx.to_array expected and actual = Nx.to_array actual in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e ->
       let a = actual.(i) in
       let scale = Float.max (Float.abs e) (Float.abs a) in
       let scale = if scale = 0.0 then 1.0 else scale in
       equal
         ~msg:(Printf.sprintf "%s[%d]" msg i)
         (float 1e-9) 0.0 (Float.abs (e -. a) /. scale))
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
  type _ t = params

  let walk c p =
    let open Nx.Ptree.Walk in
    { a = field c "a" tensor p.a
    ; ct = field c "ct" tensor p.ct
    ; wt = field c "wt" tensor p.wt
    ; b = field c "b" tensor p.b
    }
end

let params_ptree : params Nx.Ptree.t = Nx.Ptree.instantiate (module Params)

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

let sketch ?(kk = k) loss p = Sofo.sketch params_ptree loss p (sampler kk p)

(* The reference tangent batch, computed independently of the lane machinery
   under test: one single-tangent [Rune.jvp] per lane, stacked. *)
let lane_tangents ~k (f : params -> Nx.float64_t) p thetas =
  Nx.stack
    ~axis:0
    (List.init k (fun i ->
       snd
         (Rune.jvp params_ptree Nx.Ptree.tensor f p
            (Nx.Ptree.map params_ptree (fun _ t -> Nx.slice [ Nx.I i ] t) thetas))))

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
    Rune.scan'
      ~f:(fun z tgt ->
        let z' = step p z in
        z', Sofo.mse ~target:tgt z')
      ~init:z0
      scan_targets
  in
  Nx.sum ls

let test_scan_threads_the_total () =
  (* Rune.scan inside the loss: the total's scope owns the fold, so every
     step's mark reaches it and the GGN is the sum of the per-step blocks, each
     computed here from per-lane single-tangent jvps. *)
  let p = params ()
  and kk = 2 in
  let thetas = thetas_for kk p in
  let sk = sketch ~kk scan_loss p in
  let reference =
    List.fold_left
      (fun acc t ->
         Nx.add
           acc
           (gram_block ~k:kk ~h:(2.0 /. 3.0) (fun p -> roll p z0 (t + 1)) p thetas))
      (Nx.zeros f64 [| kk; kk |])
      (List.init scan_steps Fun.id)
  in
  check_arr ~msg:"Σ_t Y_tᵀ H Y_t" (Nx.to_array reference) sk.ggn

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
let vmap_loss p =
  Sofo.mse ~target:endpoints (Rune.vmap' (fun x -> roll p x 4) starts)

(* The same loss with the map written out as a loop. *)
let loop_loss p =
  Sofo.mse
    ~target:endpoints
    (Nx.stack
       ~axis:0
       (List.init vmap_trials (fun i -> roll p (Nx.slice [ Nx.I i ] starts) 4)))

let test_vmap_inside_matches_the_loop () =
  (* vmap inside the loss: vmap re-performs every operation batched, forward
     mode tracks the batched operations, and the observation sees the physical
     [M;3] prediction — one k×k block whose contraction sums over M. The
     reference is the same loss with the map written out, which tracks the same
     tangents through an ordinary loop. *)
  let p = params ()
  and kk = 3 in
  let sk_vmap = sketch ~kk vmap_loss p in
  let sk_loop = sketch ~kk loop_loss p in
  check_arr ~msg:"loss" (Nx.to_array sk_loop.loss) sk_vmap.loss;
  check_arr ~msg:"C" (Nx.to_array sk_loop.c) sk_vmap.c;
  check_arr ~msg:"G̃" (Nx.to_array sk_loop.ggn) sk_vmap.ggn

(* Trials as a structure, so that a mapped function can see each trial's target
   alongside its start. *)
type trial =
  { x : Nx.float64_t
  ; t : Nx.float64_t
  }

module Trial = struct
  type _ t = trial

  let walk c tr =
    let open Nx.Ptree.Walk in
    { x = field c "x" tensor tr.x; t = field c "t" tensor tr.t }
end

let trial_ptree : trial Nx.Ptree.t = Nx.Ptree.instantiate (module Trial)

let trials = { x = starts; t = endpoints }
let batch = float_of_int vmap_trials
let n_pred = 3.0

(* Observing inside the map, with the mean over trials folded into each
   observation: one observation is one term of the total loss, so the value and
   the curvature both carry the 1/M factor the map's reduction would apply
   outside, and the outside reduction is then the sum of those contributions —
   which is the mean of the unweighted little losses. A mark enters with
   weight one, so the factor has to be written at the mark (law 6). *)
let in_map_loss p =
  Nx.sum
    (Rune.vmap
       Nx.Ptree.(trial_ptree @-> returns tensor)
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
  let sk = sketch ~kk in_map_loss p in
  let sk_batched =
    sketch ~kk vmap_loss p
  in
  (* vmap re-performs the body batched rather than looping it, so the [M]
     little losses are packed into a single observation whose block contracts
     over the map's axis. *)
  check_arr ~msg:"loss" (Nx.to_array sk_batched.loss) sk.loss;
  check_arr ~msg:"C" (Nx.to_array sk_batched.c) sk.c;
  check_arr ~msg:"G̃" (Nx.to_array sk_batched.ggn) sk.ggn

let test_an_unscaled_in_map_mean_overweights_the_block () =
  (* Law 6: a mark enters with weight one. Marking the raw per-trial mean
     inside the map and then averaging the map's outputs scales the loss but
     not the block, so the curvature is M times the loss's Gauss-Newton
     matrix. Scaling the little loss and its curvature by 1/M instead, as
     {!test_observation_inside_a_vmap_scales_by_the_batch} does, agrees with
     observing the batched prediction. *)
  let p = params () in
  let loss p =
    Nx.mean
      (Rune.vmap
         Nx.Ptree.(trial_ptree @-> returns tensor)
         (fun tr ->
            let y = roll p tr.x 4 in
            Sofo.mse ~target:tr.t y)
         trials)
  in
  let sk = sketch loss p in
  let scaled = sketch ~kk:k in_map_loss p in
  check_arr ~msg:"the loss is the same" (Nx.to_array scaled.loss) sk.loss;
  check_arr ~msg:"the block carries the missing factor of M"
    (Nx.to_array (Nx.mul_s scaled.ggn batch))
    sk.ggn

let test_vmap_outside_a_sketch_batches_the_sketches () =
  (* A map around a whole sketch: each mapped lane gets its own sketch of its
     own parameters, along the same sampled directions. The old batched forward
     mode made this a lane error; the vmap-inside-jvp composition batches the
     sketches naturally. Compare the whole bundle per lane. *)
  let m = 2 in
  let batched = sampler m (params ()) in
  let bundle (sk : params Sofo.sketch) =
    Nx.concatenate ~axis:0 [ Nx.ravel sk.loss; Nx.ravel sk.c; Nx.ravel sk.ggn ]
  in
  let expected =
    Nx.stack
      (List.init m (fun i ->
         let lane =
           Nx.Ptree.map params_ptree (fun _ t -> Nx.slice [ Nx.I i ] t) batched
         in
         bundle (sketch vmap_loss lane)))
  in
  let got =
    Rune.vmap
      Nx.Ptree.(params_ptree @-> returns tensor)
      (fun p -> bundle (sketch vmap_loss p))
      batched
  in
  check_arr
    ~msg:"a map outside the sketch batches the sketches"
    (Nx.to_array expected)
    got

let test_grad_outside_a_sketch_is_exact () =
  (* grad of a sketch: reverse mode outside the forward pass. The forward
     handler's primals are ordinary operations to the tape, and the mark is a
     unit-result custom_jvp whose tangent map reverse mode never applies, so
     differentiating the sketch's loss is differentiating the loss. *)
  let p = params () in
  let l_sketch, g_sketch =
    Rune.value_and_grad params_ptree (fun p -> (sketch scan_loss p).loss) p
  in
  let l_plain, g_plain = Rune.value_and_grad params_ptree scan_loss p in
  check_arr ~msg:"loss" (Nx.to_array l_plain) l_sketch;
  check_arr ~msg:"gradient (a)" (Nx.to_array g_plain.a) g_sketch.a;
  check_arr ~msg:"gradient (ct)" (Nx.to_array g_plain.ct) g_sketch.ct;
  check_arr ~msg:"gradient (wt)" (Nx.to_array g_plain.wt) g_sketch.wt;
  check_arr ~msg:"gradient (b)" (Nx.to_array g_plain.b) g_sketch.b

let test_sketch_of_grad_is_the_batched_hessian () =
  (* a sketch of grad (the matrix's bonus row): the prediction is itself a
     gradient computed by reverse mode, and the batched forward pass
     differentiates through that. The loss is exactly quadratic in the
     parameters, so the sketch's own second-order model is exact — a check
     that needs no reference. *)
  let loss p =
    let g = Rune.grad' (fun w -> Nx.sum (Nx.square (Nx.matmul w z0))) p.wt in
    Sofo.mse ~target:(Nx.zeros_like g) g
  in
  let p = params ()
  and kk = 3 in
  let sk = sketch ~kk loss p in
  let z = vec [| 0.4; -1.1; 0.7 |] in
  let eps = 1e-4 in
  let p' =
    Nx.Ptree.map2 params_ptree
      (fun _ leaf d -> Nx.add leaf (Nx.mul_s d (Nx_dtype.of_float (Nx.dtype d) eps)))
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

let test_detach_is_skipped () =
  (* A prediction that is a constant of the differentiation has no tangent, so
     its block is skipped (not accumulated as zeros), while its value still
     counts towards the loss. *)
  let loss p =
    let gated = Rune.detach (step p z0) in
    let l = Nx.mean (Nx.square gated) in
    Sofo.observe ~y:gated ~curv:(Sofo.Curv.scale 2.0) l;
    l
  in
  let p = params () in
  let sk = sketch loss p in
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
    Sofo.mse ~target:(Nx.zeros f64 [| 3 |]) (Nx.add (step p z0) noise)
  in
  let clean p = Sofo.mse ~target:(Nx.zeros f64 [| 3 |]) (step p z0) in
  let p = params ()
  and kk = 2 in
  let thetas = thetas_for kk p in
  let sk_noisy =
    Nx.Rng.with_key (Nx.Rng.key 11) (fun () ->
      sketch ~kk noisy p)
  in
  let sk_clean = sketch ~kk clean p in
  let _, grads =
    Nx.Rng.with_key (Nx.Rng.key 11) (fun () ->
      Rune.value_and_grad params_ptree noisy p)
  in
  check_arr
    ~msg:"C = Θᵀ∇c of the noisy loss"
    (Nx.to_array (theta_t_cotangent ~k:kk thetas grads))
    sk_noisy.c;
  check_arr ~msg:"noise does not move G̃" (Nx.to_array sk_clean.ggn) sk_noisy.ggn;
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
    Sofo.mse ~target:(Nx.zeros f64 [| 3 |]) z
  in
  let p = params () in
  let sk_plain = sketch (model inline) p in
  let sk_jit = sketch (model jitted_activation) p in
  check_arr ~msg:"loss" (Nx.to_array sk_plain.loss) sk_jit.loss;
  check_arr ~msg:"C" (Nx.to_array sk_plain.c) sk_jit.c;
  check_arr
    ~msg:"jit inside does not change the sketch"
    (Nx.to_array sk_plain.ggn)
    sk_jit.ggn

let test_custom_jvp_inside () =
  (* A custom rule is written for a single tangent, so the batched handler
     lifts it over the lane axis. With a rule that reproduces relu's own
     derivative, the sketch must equal the plain model's. *)
  let my_relu x =
    Rune.custom_jvp
      Nx.Ptree.tensor
      Nx.Ptree.tensor
      (fun x ->
        ( Nx.relu x
        , fun dx -> Nx.where (Nx.greater x (Nx.zeros_like x)) dx (Nx.zeros_like dx) ))
      x
  in
  let loss p =
    Sofo.mse
      ~target:(Nx.zeros f64 [| 3 |])
      (Nx.add
         (Nx.matmul z0 p.a)
         (Nx.matmul (my_relu (Nx.add (Nx.matmul z0 p.ct) p.b)) p.wt))
  in
  let p = params () in
  let sk = sketch loss p in
  check_arr
    ~msg:"custom rule, same curvature"
    (Nx.to_array
       (sketch (fun p -> Sofo.mse ~target:(Nx.zeros f64 [| 3 |]) (step p z0)) p).ggn)
    sk.ggn

let test_custom_vjp_raises_and_leaves_the_gate_alone () =
  (* custom_vjp has no forward rule: it raises under any forward mode, and the
     failure must not disable jit for the rest of the process. *)
  let loss p =
    let g =
      Rune.custom_vjp
        Nx.Ptree.tensor
        Nx.Ptree.tensor
        (fun x -> Nx.sin x, fun ct -> Nx.mul ct (Nx.cos x))
        p.b
    in
    Sofo.mse ~target:(Nx.zeros f64 [| 4 |]) g
  in
  let p = params () in
  raises_containing
    ~msg:"custom_vjp under a sketch"
    ~substring:"Rune.jvp: a custom_vjp rule has no forward derivative"
    (fun () -> ignore (sketch loss p));
  ignore (sketch (fun p -> Sofo.mse ~target:(Nx.zeros f64 [| 3 |]) (step p z0)) p)

(* ── the ✗ rows in a jitted step ────────────────────────────────────────── *)

let test_jitted_sketch_matches_eager () =
  (* With jit outermost the total's scope is traced with the primal ones, so a
     jitted sketch is correct: it draws its directions from a key carried as an
     input leaf, and the scan inside the loss stays a loop. *)
  let p = params () in
  let dirs = sampler k p in
  let bundle (sk : params Sofo.sketch) =
    Nx.concatenate ~axis:0 [ Nx.ravel sk.loss; Nx.ravel sk.c; Nx.ravel sk.ggn ]
  in
  let eager = bundle (Sofo.sketch params_ptree vmap_loss p dirs) in
  let jitted =
    Rune.jit
      Nx.Ptree.(params_ptree @-> returns tensor)
      (fun p -> bundle (Sofo.sketch params_ptree vmap_loss p dirs))
  in
  check_arr
    ~msg:"a jitted sketch replays identically"
    (Nx.to_array (jitted p))
    (jitted p);
  check_arr ~msg:"jitted sketch (loss, C, G̃)" (Nx.to_array eager) (jitted p)

let test_a_long_scan_compiles () =
  (* A long horizon: the scope's total rides the staged loop, so a compiled
     sketch over many steps equals the eager one. *)
  let steps = 120 in
  let targets = Nx.zeros f64 [| steps; 3 |] in
  let loss p =
    let _, ls =
      Rune.scan'
        ~f:(fun z tgt ->
          let z' = step p z in
          z', Sofo.mse ~target:tgt z')
        ~init:z0
        targets
    in
    Nx.sum ls
  in
  let p = params () in
  let dirs = sampler 4 p in
  let bundle (sk : params Sofo.sketch) =
    Nx.concatenate ~axis:0 [ Nx.ravel sk.loss; Nx.ravel sk.c; Nx.ravel sk.ggn ]
  in
  let eager = bundle (Sofo.sketch params_ptree loss p dirs) in
  let jitted =
    Rune.jit
      Nx.Ptree.(params_ptree @-> returns tensor)
      (fun p -> bundle (Sofo.sketch params_ptree loss p dirs))
  in
  check_rel ~msg:"a long compiled sketch equals the eager one" eager (jitted p)

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
      Rune.scan'
        ~f:(fun z tgt ->
          let z' = step p z in
          z', z')
        ~init:z0
        scan_targets
    in
    Sofo.mse ~target:(Nx.mul scan_targets mask) (Nx.mul zs mask)
  in
  let branch = Nx.item [] (Nx.sum p.b) > 0.0 in
  let loss p = if branch then masked p else Sofo.mse ~target:endpoints (roll p z0 3) in
  let sk = sketch loss p in
  let _, grads = Rune.value_and_grad params_ptree loss p in
  let thetas = thetas_for k p in
  check_arr
    ~msg:"C = Θᵀ∇c under host control flow"
    (Nx.to_array (theta_t_cotangent ~k thetas grads))
    sk.c

let tests =
  [ group
      "the ✓ rows"
      [ test
          "the total threads a scan inside the loss"
          test_scan_threads_the_total
      ; test "vmap inside the loss matches the loop oracle" test_vmap_inside_matches_the_loop
      ; test
          "an observation inside a vmap scales by the batch"
          test_observation_inside_a_vmap_scales_by_the_batch
      ; test "grad of a sketch is exact" test_grad_outside_a_sketch_is_exact
      ; test
          "a sketch of grad is the batched Hessian"
          test_sketch_of_grad_is_the_batched_hessian
      ; test "detach is skipped" test_detach_is_skipped
      ; test "RNG in the graph is a constant" test_rng_in_the_graph_is_a_constant
      ; test "jit inside degrades to eager" test_jit_inside_degrades
      ; test "custom_jvp is lifted over the lanes" test_custom_jvp_inside
      ; test "host control flow is inherited" test_control_flow_is_inherited
      ; test
          "vmap outside a sketch batches the sketches"
          test_vmap_outside_a_sketch_batches_the_sketches
      ]
  ; group
      "the ✗ rows"
      [ test
          "an unscaled in-map mean overweights the block"
          test_an_unscaled_in_map_mean_overweights_the_block
      ; test
          "custom_vjp raises and leaves the gate alone"
          test_custom_vjp_raises_and_leaves_the_gate_alone
      ]
  ; group
      "a jitted sketch"
      [ test "matches eager" test_jitted_sketch_matches_eager
      ; test "a long scan compiles" test_a_long_scan_compiles
      ]
  ]

let () = exit (run "sofo composition" tests)
