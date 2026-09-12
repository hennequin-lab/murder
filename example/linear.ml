(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Student-teacher linear regression, trained with [Sofo.Optim] — Algorithm 1's
   update — on the deployment the library's two halves describe:

   compile the sketching half  →  solve eagerly  →  step eagerly

   [Sofo.Optim.Compiled (Params).sketch] is the sketching computation as a pure
   function of [(params, key)], which [Rune.jit2] compiles once and replays for
   the whole run: the directions Θ are drawn inside it from the key the caller
   threads, so a fresh random subspace costs nothing. The update needs the SVD
   of the sketched GGN, which does not compile, so it runs on the host where
   eager CPU linalg is the right tool anyway — G̃ is K×K and Alg. 1's whole
   point is that inverting it costs O(K³) against a model with P ≫ K
   parameters.

   The task is small enough that the loss is exactly quadratic in the
   parameters, so the sketch's second-order model

   c(θ − η·Θz) = c − η·⟨C, z⟩ + (η²/2)·zᵀG̃z

   is exact. The example checks it after every step: that validates C, G̃, the
   sampled Θ and the update jointly, with no reference implementation. A
   first-order control then replays the *same* compiled sketch with z = C and
   no SVD, which is the comparison the GGN is supposed to win. *)

let f64 = Nx.float64

(* ── configuration ───────────────────────────────────────────────────────── *)

type config =
  { dim : int (* input dimension *)
  ; out : int (* output dimension *)
  ; samples : int
  ; k : int (* subspace dimension *)
  ; steps : int
  ; lr : float
  ; damping : Sofo.Optim.damping (* how γ is scaled; see Sofo.Optim *)
  ; fgd_lr : float
  ; compare : bool
  ; seed : int
  }

let median xs =
  let xs = List.sort compare xs in
  List.nth xs (List.length xs / 2)

let damping_to_string = function
  | `Absolute value -> Printf.sprintf "absolute %.3g" value
  | `Relative_from_top factor -> Printf.sprintf "%.3g x s_max" factor
  | `Relative_from_bottom factor -> Printf.sprintf "%.3g x s_min" factor

(* ── the student ─────────────────────────────────────────────────────────── *)

type params = { w : Nx.float64_t (* [dim; out] *) }

module Params = struct
  type t = params

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p = { w = f p.w }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { w = f a.w b.w }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) p = f p.w
end

(* The two halves, tied to the parameter structure: [O] is the jittable
   sketching step, [Sofo.Optim] the eager update. *)
module O = Sofo.Optim.Compiled (Params)

let max_abs t = Nx.item [] (Nx.max (Nx.abs t))
let l2 t = Nx.item [] (Nx.sqrt (Nx.sum (Nx.square t)))

(* ── the run ─────────────────────────────────────────────────────────────── *)

let run config =
  let k0 = Nx.Rng.key config.seed in
  let x = Nx.Rng.normal k0 f64 [| config.samples; config.dim |] in
  let scale = 1.0 /. sqrt (float_of_int config.dim) in
  let teacher =
    Nx.mul_s (Nx.Rng.normal (Nx.Rng.fold_in k0 1) f64 [| config.dim; config.out |]) scale
  in
  let w0 =
    Nx.mul_s (Nx.Rng.normal (Nx.Rng.fold_in k0 2) f64 [| config.dim; config.out |]) scale
  in
  let student0 = { w = w0 } in
  let y = Nx.matmul x teacher in
  let objective (p : params) = Sofo.mse (Nx.matmul x p.w) y in
  let parameters = config.dim * config.out in
  let state = Sofo.Optim.init ~key:k0 () in
  Printf.printf
    "student-teacher linear regression: y = x·W*, %d samples, x:[%d], W:[%d;%d]\n\
     P = %d parameters, K = %d (%.1f%%), eta = %g, damping %s\n\n"
    config.samples
    config.dim
    config.dim
    config.out
    parameters
    config.k
    (100.0 *. float_of_int config.k /. float_of_int parameters)
    config.lr
    (damping_to_string config.damping);
  (* One eager sketch first: Sofo.check reads values to compare the observed
     little losses with the loss, which a compiled trace must not do. *)
  let sk0 =
    Sofo.sketch
      (module Params)
      ~k:config.k
      ~sketch_sampler:(fun k p ->
        Sofo.Optim.directions (module Params) ~key:state.Sofo.Optim.key ~k p)
      objective
      student0
  in
  (match Sofo.check sk0 with
   | Ok () -> Printf.printf "cross-check: the observed little losses add up\n"
   | Error msg -> Printf.printf "cross-check FAILED: %s\n" msg);
  (* Warm the device and the kernel compiler, so that the compile time reported
     below is this sketch's rather than the process's first jit. Set JITCACHE=0
     for a cold number. *)
  let _ =
    Rune.jit2
      (module O.In)
      (module O.Out)
      (fun (i : O.in_) ->
         { O.loss = Nx.sum i.O.params.w
         ; c = Nx.zeros f64 [| 1 |]
         ; ggn = Nx.zeros f64 [| 1; 1 |]
         ; dirs = i.O.params
         ; observed_loss = Nx.zeros f64 [||]
         ; observed_c = Nx.zeros f64 [||]
         })
      { O.params = student0; key = state.key }
  in
  (* One trace for the whole run. *)
  let sketch_step =
    Rune.jit2 (module O.In) (module O.Out) (O.sketch ~k:config.k objective)
  in
  let t0 = Unix.gettimeofday () in
  let out0 = sketch_step { O.params = student0; key = state.key } in
  let compile_ms = (Unix.gettimeofday () -. t0) *. 1e3 in
  (match O.check out0 with
   | Ok () -> ()
   | Error msg -> Printf.printf "compiled cross-check FAILED: %s\n" msg);
  (* ── the SOFO loop (Alg. 1) ── *)
  let rec loop params state i (losses, updates, residuals) =
    if i > config.steps
    then params, state, List.rev losses, List.rev updates, List.rev residuals
    else
      let open Sofo.Optim in
      let out = sketch_step { O.params; key = state.key } in
      let l = Nx.item [] out.O.loss in
      (* the host's half: solve, then step. Everything inside the clock is
         eager, host-side, and O(K³). *)
      let t0 = Unix.gettimeofday () in
      let params = O.update ~lr:config.lr ~damping:config.damping params out in
      let update_ms = (Unix.gettimeofday () -. t0) *. 1e3 in
      (* the sketch's second-order model of the step just taken; exact for this
         loss, so the residual should sit at rounding. Re-solving for the
         coordinates outside the clock keeps the timed region to one solve. *)
      let z = coordinates ~damping:config.damping out.O.ggn out.O.c in
      let linear = Nx.item [] (Nx.sum (Nx.mul out.O.c z)) in
      let quad = Nx.item [] (Nx.sum (Nx.mul z (Nx.matmul out.O.ggn z))) in
      let after = Nx.item [] (objective params) in
      let predicted =
        l -. (config.lr *. linear) +. (config.lr *. config.lr /. 2.0 *. quad)
      in
      let residual =
        Float.abs (after -. predicted) /. Float.max 1e-30 (Float.abs after)
      in
      if i = 1 || i mod 5 = 0 || i = config.steps
      then
        Printf.printf
          "  step %3d  loss %12.6g  |W-W*|max %10.4g  model residual %.2e\n"
          i
          l
          (max_abs (Nx.sub params.w teacher))
          residual;
      let state = next state in
      loop params state (i + 1) (l :: losses, update_ms :: updates, residual :: residuals)
  in
  let params, state, losses, update_ms, residuals = loop student0 state 1 ([], [], []) in
  let final = Nx.item [] (objective params) in
  let initial = Nx.item [] sk0.loss in
  Printf.printf
    "\nSOFO: loss %.6g → %.6g (%.1e× smaller) in %d steps; |W - W*|max %.4g → %.4g\n"
    initial
    final
    (initial /. Float.max 1e-30 final)
    config.steps
    (max_abs (Nx.sub student0.w teacher))
    (max_abs (Nx.sub params.w teacher));
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
    let rec fgd state params i =
      if i > config.steps
      then params
      else
        let open Sofo.Optim in
        let out = sketch_step { O.params; key = state.key } in
        (* z = C: the gradient sketch is already a coordinate vector *)
        let dw = apply (module Params) ~k:config.k out.O.dirs out.O.c in
        let scale = config.fgd_lr /. Float.max 1e-30 (l2 dw.w) in
        let params =
          Params.map2
            (fun p d -> Nx.sub p (Nx.mul_s d (Nx_core.Dtype.of_float (Nx.dtype d) scale)))
            params
            dw
        in
        fgd (next state) params (i + 1)
    in
    let fgd_params = fgd state student0 1 in
    let fgd_final = Nx.item [] (objective fgd_params) in
    Printf.printf
      "\n\
       first-order control (same sketch, z = C, normalized, eta = %g): loss %.6g → %.6g\n"
      config.fgd_lr
      initial
      fgd_final);
  (* ── timings ── *)
  let times =
    List.map
      (fun i ->
         let state = Sofo.Optim.next state in
         let t0 = Unix.gettimeofday () in
         let out = sketch_step { O.params; key = state.Sofo.Optim.key } in
         ignore i;
         ignore (Nx.item [] out.O.loss);
         (Unix.gettimeofday () -. t0) *. 1e3)
      (List.init 7 Fun.id)
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

let make_config dim out samples k steps lr damping damping_mode fgd_lr compare seed =
  let damping =
    match damping_mode with
    | `Top -> `Relative_from_top damping
    | `Bottom -> `Relative_from_bottom damping
    | `Absolute -> `Absolute damping
  in
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
      & info [ "damping"; "r" ] ~docv:"lambda" ~doc:"Damping factor.")
  in
  let damping_mode =
    Arg.(
      value
      & opt (enum [ "top", `Top; "bottom", `Bottom; "absolute", `Absolute ]) `Top
      & info
          [ "damping-mode" ]
          ~docv:"MODE"
          ~doc:
            "What lambda is relative to: $(b,top) (default, Algorithm 1's lambda*s_max), \
             $(b,bottom) (lambda*s_min), or $(b,absolute) (lambda itself).")
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
    $ damping_mode
    $ fgd_lr
    $ compare
    $ seed)

let main config = Nx.Rng.with_key (Nx.Rng.key config.seed) (fun () -> run config)

let () =
  let doc = "Student-teacher linear regression with the SOFO update (Alg. 1)." in
  let info = Cmdliner.Cmd.info "linear" ~doc in
  exit (Cmdliner.Cmd.eval (Cmdliner.Cmd.v info Cmdliner.Term.(const main $ config_term)))
