(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Lorenz task of the SOFO paper (§4.1, Appendix E.1), in miniature: a
   3-dimensional RNN with an inverted bottleneck is trained to reach, after 32
   steps, the state the Lorenz system would have reached — the trajectory
   between the endpoints is unsupervised, and only the endpoint is scored.

   The point of the example is that one loss function is driven three ways,

   - [value_and_grad] — the exact gradient,
   - [jvp_k] alone — the first-order subspace method (FGD at K = 1),
   - [Sofo.sketch] — loss, C and the sketched GGN,

   and that the sketch is measured against the composition the plan calls its
   reference (a per-lane [vmap ∘ jvp] plus an explicit Σ YᵀHY). It reports
   wall-clock, a peak-live-tangent count (the O(1)-memory-in-T claim), and what
   a jitted step can do today: a sketch's scan unrolls into the trace, so a
   compiled sketch is correct but grows with the horizon, where a plain
   compiled rollout stages the same scan as a loop.

   The SOFO update rule of Algorithm 1 (SVD, relative damping, the subspace
   solve) is deliberately out of scope — it is the next milestone, and this
   example is the interface it will be built on. The FGD loop here is the
   first-order method the paper compares against, and it steps with [apply] and
   [C] only. *)

let f64 = Nx.float64

(* ── configuration ───────────────────────────────────────────────────────── *)

type config =
  { k : int
  ; trials : int
  ; horizon : int
  ; hidden : int
  ; steps : int
  ; runs : int
  ; seed : int
  ; lr : float
  ; jit : bool
  }

(* ── timing ──────────────────────────────────────────────────────────────── *)

let median xs =
  let xs = List.sort compare xs in
  List.nth xs (List.length xs / 2)

let time_ms ?(runs = 5) ?(warmup = 1) f =
  for _ = 1 to warmup do
    ignore (f ())
  done;
  median
    (List.init runs (fun _ ->
       let t0 = Unix.gettimeofday () in
       let r = f () in
       let t1 = Unix.gettimeofday () in
       ignore r;
       (t1 -. t0) *. 1e3))

let fmt_ms x = Printf.sprintf "%8.1f ms" x

(* ── the Lorenz system, integrated with RK4 and z-scored (E.1) ───────────── *)

let lorenz_trajectory ~n ~dt =
  let deriv (x, y, z) =
    10.0 *. (y -. x), (x *. (28.0 -. z)) -. y, (x *. y) -. (8.0 /. 3.0 *. z)
  in
  let step (x, y, z) =
    let k1 = deriv (x, y, z) in
    let x1, y1, z1 = k1 in
    let k2 =
      deriv (x +. (0.5 *. dt *. x1), y +. (0.5 *. dt *. y1), z +. (0.5 *. dt *. z1))
    in
    let x2, y2, z2 = k2 in
    let k3 =
      deriv (x +. (0.5 *. dt *. x2), y +. (0.5 *. dt *. y2), z +. (0.5 *. dt *. z2))
    in
    let x3, y3, z3 = k3 in
    let k4 = deriv (x +. (dt *. x3), y +. (dt *. y3), z +. (dt *. z3)) in
    let x4, y4, z4 = k4 in
    ( x +. (dt /. 6.0 *. (x1 +. (2.0 *. x2) +. (2.0 *. x3) +. x4))
    , y +. (dt /. 6.0 *. (y1 +. (2.0 *. y2) +. (2.0 *. y3) +. y4))
    , z +. (dt /. 6.0 *. (z1 +. (2.0 *. z2) +. (2.0 *. z3) +. z4)) )
  in
  let rec go acc state i =
    if i = n then List.rev acc else go (state :: acc) (step state) (i + 1)
  in
  Array.of_list (go [] (1.0, 1.0, 1.0) 0)

let to_rows path = Array.map (fun (x, y, z) -> [| x; y; z |]) path

let zscore path =
  let n = Array.length path in
  let mean j =
    Array.fold_left (fun acc row -> acc +. row.(j)) 0.0 path /. float_of_int n
  in
  let sd j =
    let m = mean j in
    sqrt
      (Array.fold_left (fun acc row -> acc +. ((row.(j) -. m) ** 2.0)) 0.0 path
       /. float_of_int n)
  in
  let m = Array.init 3 mean
  and s = Array.init 3 sd in
  Array.map (fun row -> Array.init 3 (fun j -> (row.(j) -. m.(j)) /. s.(j))) path

(* One batch of trials: each starts at a random point of the attractor, and its
   label is where the *system* is after [horizon] steps. *)
let batch ~config ~path =
  let n = Array.length path in
  let rng = Random.State.make [| config.seed |] in
  let last = n - config.horizon - 1 in
  let starts = Array.init config.trials (fun _ -> Random.State.int rng last) in
  let gather offset =
    Nx.create
      f64
      [| config.trials; 3 |]
      (Array.concat
         (Array.to_list (Array.map (fun i -> Array.copy path.(i + offset)) starts)))
  in
  gather 0, gather config.horizon

(* ── the model ───────────────────────────────────────────────────────────── *)

(* Weights are stored transposed: every matmul has the (possibly batched)
   activation on the left and a matrix on the right, which is the form
   [Rune.vmap] can translate. *)
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

(* The number of parameters, for the K/P ratio the paper quotes. *)
let num_params ~hidden = (3 * 3) + (3 * hidden) + (hidden * 3) + hidden

let init ~hidden ~seed =
  Nx.Rng.with_key (Nx.Rng.key seed) (fun () ->
    { a = Nx.mul_s (Nx.eye f64 3) 0.98
    ; ct = Nx.mul_s (Nx.randn f64 [| 3; hidden |]) (1.0 /. sqrt 3.0)
    ; wt = Nx.mul_s (Nx.randn f64 [| hidden; 3 |]) (0.05 /. sqrt (float_of_int hidden))
    ; b = Nx.zeros f64 [| hidden |]
    })

let step p z =
  Nx.add (Nx.matmul z p.a) (Nx.matmul (Nx.relu (Nx.add (Nx.matmul z p.ct) p.b)) p.wt)

module Single = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t
  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b = f a b
  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

(* The rollout is written with [Rune.scan] rather than a loop so that a staging
   [jit] can compile it as a loop: under a sketch the collector claims the scan
   and it unrolls (below), which is exactly the cost this example reports. *)
let rollout p x steps =
  fst
    (Rune.scan
       (module Single)
       ~f:(fun z _ ->
         let z' = step p z in
         z', z')
       ~init:x
       steps)

(* ── the loss: one function, three drivers ───────────────────────────────── *)

let trials = ref (Nx.zeros f64 [| 1; 3 |])
let labels = ref (Nx.zeros f64 [| 1; 3 |])
let steps_tensor = ref (Nx.zeros f64 [| 1; 1 |])
let horizon = ref 1

(* The endpoint loss, written the natural way: batch the trials with vmap and
   observe the batched prediction. One observation, so one k×k block whose
   contraction sums over the batch. *)
let loss p =
  let preds = Rune.vmap' (fun x -> rollout p x !steps_tensor) !trials in
  Sofo.mse preds !labels

(* A single-trial loss, for the jit section: no vmap, so the scan stands or
   falls on its own claim. *)
let single_loss p = Sofo.mse (rollout p !trials !steps_tensor) !labels

(* Directions: Gaussian, drawn from a named key so that every mode and every
   run sees the same Θ. (The default sampler of [Sofo.sketch] would do the same
   under [Nx.Rng.with_key]; naming the key here makes the three drivers
   comparable by construction.) *)
let gaussian_sampler k p =
  Nx.Rng.with_key (Nx.Rng.key 2024) (fun () ->
    let draw t =
      Nx.cast (Nx.dtype t) (Nx.randn f64 (Array.append [| k |] (Nx.shape t)))
    in
    { a = draw p.a; ct = draw p.ct; wt = draw p.wt; b = draw p.b })

(* A sampler with no randomness, for use inside a [Rune.jit]ed sketch: a key or
   a scope the transform closes over is a compile-time constant, and the tracer
   refuses to freeze a draw into the program. Directions that are constants of
   the trace are legal. *)
let constant_sampler k p =
  let fill t =
    Nx.mul_s
      (Nx.ones (Nx.dtype t) (Array.append [| k |] (Nx.shape t)))
      (1.0 /. float_of_int (1 + k))
  in
  { a = fill p.a; ct = fill p.ct; wt = fill p.wt; b = fill p.b }

(* Θᵀ∇c, for checking the sketched gradient against the exact one. *)
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

let max_abs t = Nx.item [] (Nx.max (Nx.abs t))

(* ── the exact-GGN reference: vmap ∘ jvp plus the explicit Σ YᵀHY ─────────── *)

let reference_ggn ~k p thetas =
  let n = Nx.numel !labels in
  let preds q = Rune.vmap' (fun x -> rollout q x !steps_tensor) !trials in
  (* One single-tangent jvp per lane, stacked: the reference the plan
     prescribes. Composing vmap around jvp would be the tidy way to write it,
     but vmap's matmul rule passes a batched operand through to the backend
     untranslated, which cannot express the mixed-rank cases (§13.3). *)
  let y =
    Nx.stack
      ~axis:0
      (List.init k (fun i ->
         snd
           (Rune.jvp
              (module Params)
              preds
              p
              (Params.map (fun t -> Nx.slice [ Nx.I i ] t) thetas))))
  in
  let y = Nx.reshape [| k; n |] (Nx.contiguous y) in
  (* the mean-squared little loss has H = (2/n)·I over the batched prediction *)
  Nx.mul_s (Nx.matmul y (Nx.transpose y)) (2.0 /. float_of_int n)

(* ── probes ──────────────────────────────────────────────────────────────── *)

(* Peak live tangent bindings under a sketch, with a major collection before
   each reading so the number reflects what is reachable rather than what the
   collector has not got round to. Two shapes of the same recurrence:

   - a plain loop, where each step's intermediates die with the step — this is
     the plan's O(1)-in-T claim;
   - [Rune.scan], whose eager fold stacks every step's output before returning
     it, so the store (and the heap) holds one binding per step for as long as
     the fold runs. That is the scan's contract, not the differentiation's
     doing: the activations inside a step never accumulate. *)
let peak_live_tangents ~config ~scan p =
  let peak = ref 0 in
  let targets = Nx.zeros f64 [| config.horizon; 3 |] in
  let loss p =
    if scan
    then (
      let _, ls =
        Rune.scan
          (module Single)
          ~f:(fun z tgt ->
            let z' = step p z in
            Gc.full_major ();
            peak := max !peak (Rune.live_tangent_entries ());
            z', Sofo.mse z' tgt)
          ~init:(Nx.reshape [| 3 |] (Nx.slice [ Nx.I 0 ] !trials))
          targets
      in
      Nx.sum ls)
    else (
      let z = ref (Nx.reshape [| 3 |] (Nx.slice [ Nx.I 0 ] !trials)) in
      let acc = ref (Nx.zeros f64 [||]) in
      for t = 0 to config.horizon - 1 do
        let z' = step p !z in
        Gc.full_major ();
        peak := max !peak (Rune.live_tangent_entries ());
        acc := Nx.add !acc (Sofo.mse z' (Nx.slice [ Nx.I t ] targets));
        z := z'
      done;
      !acc)
  in
  ignore (Sofo.sketch (module Params) ~k:8 ~sketch_sampler:gaussian_sampler loss p);
  !peak

(* ── the report ──────────────────────────────────────────────────────────── *)

let run config =
  (* data and model *)
  let path =
    zscore (to_rows (lorenz_trajectory ~n:(2000 + config.horizon + 2) ~dt:0.01))
  in
  let starts, targets = batch ~config ~path in
  trials := starts;
  labels := targets;
  horizon := config.horizon;
  steps_tensor := Nx.zeros f64 [| config.horizon; 1 |];
  let p = init ~hidden:config.hidden ~seed:config.seed in
  let parameters = num_params ~hidden:config.hidden in
  let thetas = gaussian_sampler config.k p in
  Printf.printf
    "Lorenz endpoint task: %d trials x %d steps, hidden %d\n\
     %d parameters, K = %d (%.1f%% of P), %d timing runs\n\n"
    config.trials
    config.horizon
    config.hidden
    parameters
    config.k
    (100.0 *. float_of_int config.k /. float_of_int parameters)
    config.runs;
  (* the three modes *)
  let loss_exact, grads = Rune.value_and_grad (module Params) loss p in
  let loss_fgd, c_fgd = Rune.jvp_k (module Params) loss p thetas in
  let sk =
    Sofo.sketch (module Params) ~k:config.k ~sketch_sampler:gaussian_sampler loss p
  in
  let c_exact = theta_t_cotangent ~k:config.k thetas grads in
  Printf.printf "one loss function, three drivers\n";
  Printf.printf
    "  loss  exact grad %12.6f   jvp_k %12.6f   sketch %12.6f\n"
    (Nx.item [] loss_exact)
    (Nx.item [] loss_fgd)
    (Nx.item [] sk.loss);
  let rel a b = max_abs (Nx.sub a b) /. Float.max 1e-30 (max_abs b) in
  Printf.printf "  C vs Θᵀ∇c        : max |Δ| %.3g (relative)\n" (rel sk.c c_exact);
  Printf.printf
    "  C sketch vs jvp_k: max |Δ| %.3g (relative)\n"
    (rel sk.c (Nx.reshape [| config.k |] c_fgd));
  (match Sofo.check sk with
   | Ok () -> Printf.printf "  cross-check      : observed little losses add up\n"
   | Error msg -> Printf.printf "  cross-check      : FAILED — %s\n" msg);
  Printf.printf
    "  GGN              : symmetric to %.3g (relative), trace %.6g, diag range      \
     %.3g..%.3g\n\n"
    (rel sk.ggn (Nx.transpose sk.ggn))
    (Nx.item [] (Nx.trace sk.ggn))
    (Nx.item [] (Nx.min (Nx.diag sk.ggn)))
    (Nx.item [] (Nx.max (Nx.diag sk.ggn)));
  (* wall-clock *)
  let t_exact =
    time_ms ~runs:config.runs (fun () -> Rune.value_and_grad (module Params) loss p)
  in
  let t_fgd =
    time_ms ~runs:config.runs (fun () -> Rune.jvp_k (module Params) loss p thetas)
  in
  let t_sofo =
    time_ms ~runs:config.runs (fun () ->
      Sofo.sketch (module Params) ~k:config.k ~sketch_sampler:gaussian_sampler loss p)
  in
  let t_ref = time_ms ~runs:config.runs (fun () -> reference_ggn ~k:config.k p thetas) in
  Printf.printf "wall-clock (median of %d)\n" config.runs;
  Printf.printf "  exact value_and_grad       %s\n" (fmt_ms t_exact);
  Printf.printf "  FGD  jvp_k (%d lanes)      %s   (C only)\n" config.k (fmt_ms t_fgd);
  Printf.printf
    "  SOFO sketch (%d lanes)     %s   (loss, C, GGN)\n"
    config.k
    (fmt_ms t_sofo);
  Printf.printf
    "  vmap∘jvp + Σ YᵀHY          %s   (%.2f× the sketch)\n\n"
    (fmt_ms t_ref)
    (t_ref /. t_sofo);
  (* memory *)
  let long_config = { config with horizon = 4 * config.horizon } in
  let loop_short = peak_live_tangents ~config ~scan:false p in
  let loop_long = peak_live_tangents ~config:long_config ~scan:false p in
  let scan_short = peak_live_tangents ~config ~scan:true p in
  let scan_long = peak_live_tangents ~config:long_config ~scan:true p in
  Printf.printf "memory\n";
  Printf.printf
    "  dominant tangent batch         %6.2f MiB   (k*M*hidden*8 B, the paper's App. C \
     bound)\n"
    (float_of_int (config.k * config.trials * config.hidden * 8) /. 1048576.0);
  Printf.printf
    "  live tangent bindings, peak    %3d (T=%-3d) -> %3d (T=%-3d)  plain loop: O(1) in T\n"
    loop_short
    config.horizon
    loop_long
    long_config.horizon;
  Printf.printf
    "                                 %3d (T=%-3d) -> %3d (T=%-3d)  Rune.scan: one \
     binding per output it returns\n"
    scan_short
    config.horizon
    scan_long
    long_config.horizon;
  Printf.printf
    "  held by a sketch               %6.1f KiB   (Theta k*P = %d scalars, plus C k and \
     GGN k*k; the tensor bytes are off-heap)\n\n"
    (float_of_int (8 * ((config.k * parameters) + config.k + (config.k * config.k)))
     /. 1024.0)
    (config.k * parameters);
  (* jit *)
  if config.jit
  then (
    (* Warm the device and the kernel compiler before timing: the first
       [Rune.jit] in a process pays for both, and would otherwise look like a
       horizon effect. Set JITCACHE=0 for cold compile numbers. *)
    let _ = Rune.jit (module Params) (fun p -> single_loss p) p in
    let jit_probe horizon =
      let steps = Nx.zeros f64 [| horizon; 1 |] in
      let single p = Sofo.mse (rollout p starts steps) !labels in
      let plain = Rune.jit (module Params) (fun p -> single p) in
      let t_plain =
        let t0 = Unix.gettimeofday () in
        ignore (plain p);
        (Unix.gettimeofday () -. t0) *. 1e3
      in
      let t_replay = time_ms ~runs:5 ~warmup:1 (fun () -> plain p) in
      let sketch =
        Rune.jit
          (module Params)
          (fun p ->
             (Sofo.sketch
                (module Params)
                ~k:config.k
                ~sketch_sampler:constant_sampler
                (fun p -> Sofo.mse (rollout p starts steps) !labels)
                p)
               .loss)
      in
      let t_sketch =
        let t0 = Unix.gettimeofday () in
        ignore (sketch p);
        (Unix.gettimeofday () -. t0) *. 1e3
      in
      t_plain, t_replay, t_sketch
    in
    let tp1, tr1, ts1 = jit_probe config.horizon in
    let tp2, tr2, ts2 = jit_probe (2 * config.horizon) in
    Printf.printf "jit (single trial: the scan stands or falls on its own claim)\n";
    Printf.printf
      "  plain rollout      trace+compile %s (T=%d) → %s (T=%d), replay %s\n"
      (fmt_ms tp1)
      config.horizon
      (fmt_ms tp2)
      (2 * config.horizon)
      (fmt_ms tr1);
    Printf.printf
      "  sketched rollout   trace+compile %s (T=%d) → %s (T=%d)\n"
      (fmt_ms ts1)
      config.horizon
      (fmt_ms ts2)
      (2 * config.horizon);
    Printf.printf
      "  (a scan stages as a loop under jit alone; a sketch claims it, so it unrolls \
       into the trace — see §13.5/§14.3)\n\n";
    ignore tr2);
  (* FGD: the first-order subspace method the paper compares against. It needs
     no GGN and no update rule — θ ← θ − η·ΘΘᵀ∇c, with C from jvp_k alone and
     [apply] turning the coordinates back into a parameter-space direction. *)
  if config.steps > 0
  then (
    Printf.printf
      "FGD smoke (eta = %g, first-order subspace step; the sketch also computes a GGN \
       that FGD ignores)\n"
      config.lr;
    let rec go p i =
      if i > config.steps
      then ()
      else (
        let sk =
          Sofo.sketch (module Params) ~k:config.k ~sketch_sampler:gaussian_sampler loss p
        in
        if i = 1 || i mod 5 = 0 || i = config.steps
        then Printf.printf "  step %3d  loss %10.6f\n" i (Nx.item [] sk.loss);
        (* ΘΘᵀ∇c, normalized: with Gaussian directions the projection's length
           scales with k, so an unnormalized step would need η ∼ 1/k. *)
        let dir = sk.apply sk.c in
        let sum_sq = ref 0.0 in
        Params.iter
          (fun d ->
             sum_sq
             := !sum_sq
                +. Nx.item [] (Nx.reshape [||] (Nx.cast f64 (Nx.sum (Nx.square d)))))
          dir;
        let scale = config.lr /. Float.max 1e-30 (sqrt !sum_sq) in
        let p' =
          Params.map2
            (fun pi d ->
               Nx.sub pi (Nx.mul_s d (Nx_core.Dtype.of_float (Nx.dtype d) scale)))
            p
            dir
        in
        go p' (i + 1))
    in
    go p 1)

let make_config k trials horizon hidden steps runs seed lr jit =
  { k; trials; horizon; hidden; steps; runs; seed; lr; jit }

let config_term =
  let open Cmdliner in
  let k =
    Arg.(
      value
      & opt int 32
      & info [ "lanes"; "k" ] ~docv:"K" ~doc:"Number of sketch directions (default 32).")
  in
  let trials =
    Arg.(
      value
      & opt int 64
      & info [ "trials"; "m" ] ~docv:"M" ~doc:"Trials per batch (default 64).")
  in
  let horizon =
    Arg.(
      value
      & opt int 32
      & info [ "horizon"; "T" ] ~docv:"T" ~doc:"Steps from start to label (default 32).")
  in
  let hidden =
    Arg.(
      value
      & opt int 128
      & info [ "hidden"; "H" ] ~docv:"H" ~doc:"Inverted-bottleneck width (default 128).")
  in
  let steps =
    Arg.(
      value
      & opt int 15
      & info
          [ "steps"; "n" ]
          ~docv:"N"
          ~doc:"FGD demo steps; 0 skips the training loop (default 15).")
  in
  let runs =
    Arg.(
      value
      & opt int 5
      & info [ "runs"; "r" ] ~docv:"N" ~doc:"Timing repetitions (default 5).")
  in
  let seed =
    Arg.(value & opt int 0 & info [ "seed"; "s" ] ~docv:"S" ~doc:"Data and init seed.")
  in
  let lr =
    Arg.(
      value
      & opt float 0.05
      & info [ "lr"; "l" ] ~docv:"eta" ~doc:"FGD step size (default 0.05).")
  in
  let jit =
    Arg.(
      value
      & opt bool true
      & info
          [ "jit"; "j" ]
          ~docv:"BOOL"
          ~doc:
            "Measure the jitted step (default $(b,true); pass $(b,--jit=false) to skip).")
  in
  Term.(
    const make_config $ k $ trials $ horizon $ hidden $ steps $ runs $ seed $ lr $ jit)

let main config = Nx.Rng.with_key (Nx.Rng.key config.seed) (fun () -> run config)

let () =
  let doc = "Sketch the Lorenz endpoint task with sofo (SOFO / FGD / exact grad)." in
  let info = Cmdliner.Cmd.info "lorenz" ~doc in
  exit (Cmdliner.Cmd.eval (Cmdliner.Cmd.v info Cmdliner.Term.(const main $ config_term)))
