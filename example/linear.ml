(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Student-teacher linear regression, trained with the SOFO update of
   Algorithm 1, structured the way that update has to be deployed:

   compile the sketch  →  invert the sketched GGN eagerly  →  step

   The first stage is the expensive one, and it is what [Rune.jit] compiles:
   the forward pass, its K tangent batches, and the collector's accumulation of
   C and G̃. The second needs [Nx.svd], which does not compile, so it runs on
   the host — where eager CPU linalg is the right tool anyway, since G̃ is K×K
   and Alg. 1's whole point is that inverting it costs O(K³) against a model
   with P ≫ K parameters. The parameter update is then applied eagerly.

   The directions Θ are drawn *inside* the compiled step, from an RNG key
   threaded as an input leaf of the carried structure: the trace sees a tensor,
   not a captured constant, so one compilation serves the whole run and every
   replay draws fresh directions. The compiled step returns both the sketch's
   (c, C, G̃) bundle and the directions it used, because the update that
   consumes them happens outside it.

   The task is small enough that the loss is exactly quadratic in the
   parameters, so the sketch's second-order model

   c(θ − η·Θz) = c − η·⟨C, z⟩ + (η²/2)·zᵀG̃z

   is exact. The example checks it after every step: that validates C, G̃, the
   sampled Θ and the update jointly, with no reference implementation. A
   first-order control then replays the *same* compiled sketch with z = C and
   no SVD, which is the comparison the GGN is supposed to win.

   The update is deliberately *not* in the sofo library yet: this example is
   the bootstrap for it, kept self-contained so the interface can settle before
   the API does. *)

let f64 = Nx.float64

(* ── configuration ───────────────────────────────────────────────────────── *)

type config =
  { dim : int (* input dimension *)
  ; out : int (* output dimension *)
  ; samples : int
  ; k : int (* subspace dimension *)
  ; steps : int
  ; lr : float
  ; damping : float (* λ, relative to the largest singular value *)
  ; fgd_lr : float
  ; compare : bool
  ; seed : int
  }

let median xs =
  let xs = List.sort compare xs in
  List.nth xs (List.length xs / 2)

(* ── the carried structure: the student's weights and an RNG key ───────────

   One structure feeds the compiled step. The key rides along as an ordinary
   int32 leaf — invisible to the model, but an input of the trace, which is
   what lets a single compiled program draw fresh directions on every replay.
   The sampler gives it a zero lane batch, as [Sofo.sketch]'s own default does
   for non-float leaves. *)

type carry =
  { w : Nx.float64_t (* [dim; out] *)
  ; key : Nx.int32_t (* [2] *)
  }

module Carry = struct
  type t = carry

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) c = { w = f c.w; key = f c.key }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { w = f a.w b.w; key = f a.key b.key }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) c =
    f c.w;
    f c.key
end

(* What the compiled step hands back: the sketch's numbers, and the directions
   they were measured along. *)
type step_out =
  { bundle : Nx.float64_t (* [1 + k + k*k]: c, C, G̃ *)
  ; dirs : Nx.float64_t (* Θ for the weight leaf, [k; dim; out] *)
  }

module Step_out = struct
  type t = step_out

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) o =
    { bundle = f o.bundle; dirs = f o.dirs }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { bundle = f a.bundle b.bundle; dirs = f a.dirs b.dirs }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) o =
    f o.bundle;
    f o.dirs
end

let draw ~k (c : carry) =
  { w = Nx.Rng.normal c.key f64 (Array.append [| k |] (Nx.shape c.w))
  ; key = Nx.zeros (Nx.dtype c.key) (Array.append [| k |] (Nx.shape c.key))
  }

(* ── the model, the loss, the compiled step ──────────────────────────────── *)

let squared_error x y w = Sofo.mse (Nx.matmul x w) y

(* [compiled ~k ~x ~y] is the sketched step, traced once and replayed on every
   call. Note what is *not* here: no SVD, no parameter update — those are the
   host's half of the algorithm, below. *)
let compiled ~k ~x ~y =
  Rune.jit2
    (module Carry)
    (module Step_out)
    (fun c ->
       let dirs = draw ~k c in
       let sk =
         Sofo.sketch
           (module Carry)
           ~k
           ~sketch_sampler:(fun _ _ -> dirs)
           (fun c -> squared_error x y c.w)
           c
       in
       { bundle =
           Nx.concatenate ~axis:0 [ Nx.ravel sk.loss; Nx.ravel sk.c; Nx.ravel sk.ggn ]
       ; dirs = dirs.w
       })

(* ── the host's half: two ways to turn a sketch into a step ──────────────── *)

let split_bundle ~k b =
  let c = Nx.slice [ Nx.R (1, 1 + k) ] b in
  let ggn = Nx.reshape [| k; k |] (Nx.slice [ Nx.R (1 + k, 1 + k + (k * k)) ] b) in
  c, ggn

(* [sofo_coordinates ~k ~damping bundle] is the solution z of the damped,
   sketched normal equations: z = U (S + λ·s_max I)⁻¹ Vᵀ C, from the SVD of G̃ —
   Alg. 1 lines 10–12, with the sketch's own C. G̃ is symmetric PSD, so V = U up
   to signs; the SVD form is kept because that is what the algorithm says. *)
let sofo_coordinates ~k ~damping bundle =
  let c, ggn = split_bundle ~k bundle in
  let u, s, vt = Nx.svd ggn in
  let smax =
    Nx.item [ 0 ] s
    (* singular values descend *)
  in
  let damped = Nx.add s (Nx.mul_s (Nx.ones_like s) (damping *. smax)) in
  let rhs = Nx.matmul vt (Nx.reshape [| k; 1 |] c) in
  Nx.reshape [| k |] (Nx.matmul u (Nx.div rhs (Nx.reshape [| k; 1 |] damped)))

(* Θz: the parameter-space direction the coordinates denote. *)
let apply_dirs ~k z dirs =
  let shape = Nx.shape dirs in
  let lead = Array.make (Array.length shape - 1) 1 in
  let zr = Nx.reshape (Array.append [| k |] lead) (Nx.contiguous z) in
  Nx.sum (Nx.mul zr dirs) ~axes:[ 0 ]

let l2 t = Nx.item [] (Nx.sqrt (Nx.sum (Nx.square t)))

(* ── the run ─────────────────────────────────────────────────────────────── *)

let max_abs t = Nx.item [] (Nx.max (Nx.abs t))

let run config =
  let k0 = Nx.Rng.key config.seed in
  let x = Nx.Rng.normal k0 f64 [| config.samples; config.dim |] in
  let scale = 1.0 /. sqrt (float_of_int config.dim) in
  let teacher =
    Nx.mul_s (Nx.Rng.normal (Nx.Rng.fold_in k0 1) f64 [| config.dim; config.out |]) scale
  in
  let student0 =
    Nx.mul_s (Nx.Rng.normal (Nx.Rng.fold_in k0 2) f64 [| config.dim; config.out |]) scale
  in
  let y = Nx.matmul x teacher in
  let parameters = config.dim * config.out in
  let carry0 = { w = student0; key = k0 } in
  Printf.printf
    "student-teacher linear regression: y = x·W*, %d samples, x:[%d], W:[%d;%d]\n\
     P = %d parameters, K = %d (%.1f%%), eta = %g, lambda = %g\n\n"
    config.samples
    config.dim
    config.dim
    config.out
    parameters
    config.k
    (100.0 *. float_of_int config.k /. float_of_int parameters)
    config.lr
    config.damping;
  (* One eager sketch first: Sofo.check reads values to compare the observed
     little losses with the loss, which a compiled trace must not do. *)
  let sk0 =
    Sofo.sketch
      (module Carry)
      ~k:config.k
      ~sketch_sampler:(fun k c -> draw ~k c)
      (fun c -> squared_error x y c.w)
      carry0
  in
  (match Sofo.check sk0 with
   | Ok () -> Printf.printf "cross-check: the observed little losses add up\n"
   | Error msg -> Printf.printf "cross-check FAILED: %s\n" msg);
  (* Warm the device and the kernel compiler, so that the compile time reported
     below is this sketch's rather than the process's first jit. Set JITCACHE=0
     for a cold number. *)
  let _ = Rune.jit (module Carry) (fun c -> Nx.sum c.w) carry0 in
  let step = compiled ~k:config.k ~x ~y in
  let t0 = Unix.gettimeofday () in
  ignore (step carry0);
  let compile_ms = (Unix.gettimeofday () -. t0) *. 1e3 in
  (* ── the SOFO loop (Alg. 1) ── *)
  let rec loop carry i (losses, updates, residuals) =
    if i > config.steps
    then carry, List.rev losses, List.rev updates, List.rev residuals
    else (
      let out = step carry in
      let loss = Nx.item [] (Nx.reshape [||] (Nx.slice [ Nx.I 0 ] out.bundle)) in
      (* the host's half: invert G̃, solve, step. Everything inside the clock is
         eager, host-side, and O(K³). *)
      let t0 = Unix.gettimeofday () in
      let z = sofo_coordinates ~k:config.k ~damping:config.damping out.bundle in
      let dw = apply_dirs ~k:config.k z out.dirs in
      let w = Nx.sub carry.w (Nx.mul_s dw (Nx_core.Dtype.of_float f64 config.lr)) in
      let update_ms = (Unix.gettimeofday () -. t0) *. 1e3 in
      let carry = { w; key = Nx.Rng.fold_in carry.key i } in
      (* the sketch's second-order model of the step just taken; exact for this
         loss, so the residual should sit at rounding *)
      let _, ggn = split_bundle ~k:config.k out.bundle in
      let c, _ = split_bundle ~k:config.k out.bundle in
      let linear = Nx.item [] (Nx.sum (Nx.mul c z)) in
      let quad = Nx.item [] (Nx.sum (Nx.mul z (Nx.matmul ggn z))) in
      let after = Nx.item [] (squared_error x y carry.w) in
      let predicted =
        loss -. (config.lr *. linear) +. (config.lr *. config.lr /. 2.0 *. quad)
      in
      let residual =
        Float.abs (after -. predicted) /. Float.max 1e-30 (Float.abs after)
      in
      if i = 1 || i mod 5 = 0 || i = config.steps
      then
        Printf.printf
          "  step %3d  loss %12.6g  |W-W*|max %10.4g  model residual %.2e\n"
          i
          loss
          (max_abs (Nx.sub carry.w teacher))
          residual;
      loop carry (i + 1) (loss :: losses, update_ms :: updates, residual :: residuals))
  in
  let carry, losses, update_ms, residuals = loop carry0 1 ([], [], []) in
  let final = Nx.item [] (squared_error x y carry.w) in
  let initial = Nx.item [] sk0.loss in
  Printf.printf
    "\nSOFO: loss %.6g → %.6g (%.1e× smaller) in %d steps; |W - W*|max %.4g → %.4g\n"
    initial
    final
    (initial /. Float.max 1e-30 final)
    config.steps
    (max_abs (Nx.sub student0 teacher))
    (max_abs (Nx.sub carry.w teacher));
  Printf.printf
    "  second-order model residual: max %.2e over %d steps (the loss is exactly \
     quadratic, so this is rounding)\n"
    (List.fold_left Float.max 0.0 residuals)
    (List.length residuals);
  (* ── first-order control: the same compiled step, C straight into the step ──
     [ΘΘᵀ∇c] with no curvature and no SVD — the first-order subspace method the
     paper compares against. The sketch still computes G̃; the control ignores
     it, which is the point: the compiled half is shared. *)
  if config.compare
  then (
    let rec fgd carry i =
      if i > config.steps
      then carry
      else (
        let out = step carry in
        let c, _ = split_bundle ~k:config.k out.bundle in
        let dw = apply_dirs ~k:config.k c out.dirs in
        let scale = config.fgd_lr /. Float.max 1e-30 (l2 dw) in
        let w = Nx.sub carry.w (Nx.mul_s dw (Nx_core.Dtype.of_float f64 scale)) in
        fgd { w; key = Nx.Rng.fold_in carry.key i } (i + 1))
    in
    let fgd_carry = fgd carry0 1 in
    let fgd_final = Nx.item [] (squared_error x y fgd_carry.w) in
    Printf.printf
      "\n\
       first-order control (same sketch, z = C, normalized, eta = %g): loss %.6g → %.6g\n"
      config.fgd_lr
      initial
      fgd_final);
  (* ── timings ── *)
  let replays =
    List.init 7 (fun i -> { w = carry.w; key = Nx.Rng.fold_in carry.key (1000 + i) })
  in
  let times =
    List.map
      (fun c ->
         let t0 = Unix.gettimeofday () in
         let out = step c in
         ignore (Nx.item [] (Nx.reshape [||] (Nx.slice [ Nx.I 0 ] out.bundle)));
         (Unix.gettimeofday () -. t0) *. 1e3)
      replays
  in
  Printf.printf "\ntimings\n";
  Printf.printf
    "  compiled sketch: first call (trace + compile + run)  %8.1f ms\n"
    compile_ms;
  Printf.printf
    "    (kernel compilation is cached on disk; JITCACHE=0 measures it cold)\n";
  Printf.printf
    "  compiled sketch: replay                              %8.1f ms\n"
    (median times);
  Printf.printf
    "  host update: SVD of %d×%d, solve, step                 %8.1f ms\n"
    config.k
    config.k
    (median update_ms);
  ignore losses

(* ── command line ────────────────────────────────────────────────────────── *)

let make_config dim out samples k steps lr damping fgd_lr compare seed =
  { dim; out; samples; k; steps; lr; damping; fgd_lr; compare; seed }

let config_term =
  let open Cmdliner in
  let dim =
    Arg.(value & opt int 32 & info [ "dim"; "d" ] ~docv:"D" ~doc:"Input dimension.")
  in
  let out =
    Arg.(value & opt int 8 & info [ "out"; "o" ] ~docv:"O" ~doc:"Output dimension.")
  in
  let samples =
    Arg.(value & opt int 512 & info [ "samples"; "n" ] ~docv:"N" ~doc:"Training samples.")
  in
  let k =
    Arg.(value & opt int 32 & info [ "lanes"; "k" ] ~docv:"K" ~doc:"Subspace dimension.")
  in
  let steps =
    Arg.(value & opt int 40 & info [ "steps" ] ~docv:"T" ~doc:"Training iterations.")
  in
  let lr =
    Arg.(
      value & opt float 1.0 & info [ "lr"; "l" ] ~docv:"eta" ~doc:"SOFO learning rate.")
  in
  let damping =
    Arg.(
      value
      & opt float 1e-6
      & info
          [ "damping"; "r" ]
          ~docv:"lambda"
          ~doc:"Damping relative to the largest singular value.")
  in
  let fgd_lr =
    Arg.(
      value
      & opt float 0.05
      & info
          [ "fgd-lr"; "g" ]
          ~docv:"eta1"
          ~doc:"First-order control's step (a unit direction).")
  in
  let compare =
    Arg.(
      value
      & opt bool true
      & info
          [ "compare"; "c" ]
          ~docv:"BOOL"
          ~doc:"Also train the first-order control (default true).")
  in
  let seed = Arg.(value & opt int 0 & info [ "seed"; "s" ] ~docv:"S" ~doc:"Seed.") in
  Term.(
    const make_config
    $ dim
    $ out
    $ samples
    $ k
    $ steps
    $ lr
    $ damping
    $ fgd_lr
    $ compare
    $ seed)

let main config = Nx.Rng.with_key (Nx.Rng.key config.seed) (fun () -> run config)

let () =
  let doc = "Student-teacher linear regression with the SOFO update (Alg. 1)." in
  let info = Cmdliner.Cmd.info "linear" ~doc in
  exit (Cmdliner.Cmd.eval (Cmdliner.Cmd.v info Cmdliner.Term.(const main $ config_term)))
