(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Lorenz-63 forecasting with the damped GGN.

   The task is the one of [sofo/example/lorenz.ml]: integrate the Lorenz-63
   attractor, chop the trajectory into chunks of [horizon] steps, and train a
   small recurrent model to predict each chunk from its first state. The
   optimizer preconditions the gradient with the second-order orbit average of
   the GGN,

     (S_g + λI)^{-1} g            (the inverse damping, the default)   or
     (S_g^{1/2} + λI)^{-1} g      (the root damping, optional)

   where [S_g] is built from the gradients of the model's predicted outputs
   ({!Symo.Optim.S.ggn}). Each variant sweeps [λ] relative to the metric's top
   eigenvalue and the learning rate, keeps the best final validation loss, and
   draws the loss curves with hugin. [Adam] (the same batches and schedule) is
   the baseline: [-only Adam] runs only it, with its own learning-rate
   default. *)

open Base
open Nx
open Symo

let in_dir = Cmdargs.in_dir "-d"

(* ------------------------------------------------------------------------
   Data: Lorenz-63, RK4, standardized chunks
   ------------------------------------------------------------------------ *)

let dt = 0.01
let d = 3
let horizon = 8
let n_train = 4096
let n_val = 1024
let n_chunks = n_train + n_val

type data =
  { x0 : host float32_t (* [n; 3] *)
  ; xt : host float32_t (* [n; horizon; 3] *)
  }

(* One chunk is [horizon] transitions: its initial state and the [horizon]
   states that follow. [Lorenz.snippets] integrates one long run, cuts it into
   [n_chunks] consecutive chunks of [horizon + 1] states and shuffles them; the
   states are standardized per coordinate over the whole record. *)
let make_data () =
  let snippets =
    Classic_attractors.Lorenz.snippets
      ~discard_first:1000
      ~key:(Rng.key 42)
      float32
      ~dt
      ~n_steps:(horizon + 1)
      n_chunks
  in
  let mu = mean ~axes:[ 0; 1 ] snippets in
  let sd = std ~axes:[ 0; 1 ] snippets in
  let z = Infix.((snippets - mu) / sd) in
  let all =
    { x0 = slice [ I 0; A; A ] z
    ; xt = slice [ R (1, horizon + 1); A; A ] z |> transpose ~axes:[ 1; 0; 2 ]
    }
  in
  ( { x0 = slice [ R (0, n_train); A ] all.x0; xt = slice [ R (0, n_train); A; A ] all.xt }
  , { x0 = slice [ R (n_train, n_chunks); A ] all.x0
    ; xt = slice [ R (n_train, n_chunks); A; A ] all.xt
    } )

let minibatch ~key ~bs (data : data) =
  let indices = Rng.permutation ~key n_train |> slice [ Nx.R (0, bs) ] in
  take ~axis:0 indices data.x0, take ~axis:0 indices data.xt

(* ------------------------------------------------------------------------
   Model: the hidden units are exchangeable
   ------------------------------------------------------------------------ *)

module Model = struct
  type ('a, _) t =
    { w : 'a (* hidden x d *)
    ; c : 'a (* (d + 1) x hidden *)
    ; a : 'a (* d x d *)
    }
  [@@deriving ptree, sexp]

  let hidden = 64
  let n_w = hidden * d
  let n_c = (d + 1) * hidden
  let n_a = d * d
  let n_params = n_w + n_c + n_a
  let dims : (int list, _) t = { w = [ hidden; d ]; c = [ d + 1; hidden ]; a = [ d; d ] }

  (* Permuting hidden unit [h] permutes row [h] of [w] and column [h] of [c];
     the other axes, including both axes of [a], are free. *)
  let symmetries : (Symmetry.spec list, _) t =
    { w = [ Symmetry.Perm 0; Symmetry.Id ]
    ; c = [ Symmetry.Id; Symmetry.Perm 0 ]
    ; a = [ Symmetry.Id; Symmetry.Id ]
    }

  let init ~key =
    let w = Rng.normal ~key:(Rng.fold_in key 0) float32 [| hidden; d |] in
    let c = Rng.normal ~key:(Rng.fold_in key 1) float32 [| d + 1; hidden |] in
    let w_scale = 0.1 /. Float.sqrt (Float.of_int hidden) in
    let c_scale = 1. /. Float.sqrt (Float.of_int (d + 1)) in
    { w = Infix.(w *$ w_scale); c = Infix.(c *$ c_scale); a = eye float32 d }
end

let relu x = maximum x (zeros_like x)

(* The batch functions are per sample: [vmap] adds the sample axis for the
   training loss and for the metric. *)
let step1 (p : (host float32_t, host) Model.t) (x : host float32_t) =
  let open Infix in
  let inp = concatenate ~axis:0 [ x; ones_at (beside x) float32 [| 1 |] ] in
  (x *@ p.a) + (relu (inp *@ p.c) *@ p.w)

let forward1 ~horizon (p : (host float32_t, host) Model.t) x0 =
  let _, acc =
    List.fold (List.range 0 horizon) ~init:(x0, []) ~f:(fun (x, acc) _ ->
      let x = step1 p x in
      x, x :: acc)
  in
  stack ~axis:0 (List.rev acc)

let loss1 ~horizon p x0 xt =
  let open Infix in
  mean (square (forward1 ~horizon p x0 - xt))

let batch_loss ~horizon p (x0, xt) =
  Rune.vmap
    Nx.Ptree.(tensor @-> tensor @-> returns tensor)
    (fun x0 xt -> loss1 ~horizon p x0 xt)
    x0
    xt
  |> mean

(* The metric's samples. *)
module S = Make (Model)

(* The GGN's samples: one per element of the model's predictions on the metric
   batch, through the library's Jacobian. *)
let batch_forward params x0 =
  Rune.vmap
    Nx.Ptree.(tensor @-> returns tensor)
    (fun x0 -> forward1 ~horizon params x0)
    x0

let samples params x0 = S.ggn ~outputs:(fun p -> batch_forward p x0) params

(* ------------------------------------------------------------------------
   Adam: the baseline
   ------------------------------------------------------------------------ *)

module Adam = struct
  type t =
    { m : (host float32_t, host) Model.t
    ; v : (host float32_t, host) Model.t
    ; n : int
    }

  let map f = Nx.Ptree.Payload.map (module Model) (fun _ x -> f x)
  let map2 f = Nx.Ptree.Payload.map2 (module Model) (fun _ a b -> f a b)
  let init p = { m = map zeros_like p; v = map zeros_like p; n = 0 }
  let adam_beta_1 = 0.9
  let adam_beta_2 = 0.999
  let eps = 1e-8

  (* The bias-corrected step, as in the paper. *)
  let step t ~lr ~params ~grads =
    let n = t.n + 1 in
    let open Infix in
    let m = map2 (fun m g -> (m *$ adam_beta_1) + (g *$ (1. -. adam_beta_1))) t.m grads in
    let v =
      map2 (fun v g -> (v *$ adam_beta_2) + (g * g *$ (1. -. adam_beta_2))) t.v grads
    in
    let bc_1 = 1. -. Float.(adam_beta_1 ** Float.of_int n) in
    let bc_2 = 1. -. Float.(adam_beta_2 ** Float.of_int n) in
    let update =
      map2
        (fun m v ->
           let m_hat = m /$ bc_1 in
           let v_hat = v /$ bc_2 in
           m_hat / (sqrt v_hat +$ eps) *$ lr)
        m
        v
    in
    let theta = map2 (fun p d -> p - d) params update in
    { m; v; n }, theta
end

(* ------------------------------------------------------------------------
   The metric's scale, and the sweep
   ------------------------------------------------------------------------ *)

type variant =
  { name : string
  ; optimizer : [ `Adam | `Symo ]
  ; form : [ `Inverse | `Root ] (* ignored by Adam *)
  }

let variants =
  [ { name = "GGN, inverse"; optimizer = `Symo; form = `Inverse }
  ; { name = "GGN, root"; optimizer = `Symo; form = `Root }
  ; { name = "Adam"; optimizer = `Adam; form = `Inverse }
  ]

(* The top eigenvalue of the first batch's metric: the scale the sweep's
   relative damping is relative to. *)
let metric_scale ~params ~x0 ~xt =
  let samples = samples params x0 in
  let config : S.config =
    { learning_rate = None; beta_1 = 0.9; beta_2 = 0.0; damping = 1.0; form = `Inverse }
  in
  let state = S.init ~config params in
  let _, grads =
    Rune.value_and_grad S.ptree (fun p -> batch_loss ~horizon p (x0, xt)) params
  in
  let mid = S.prepare ~config ~state ~grads ~samples in
  let blocks =
    S.Second_order.blocks_of_factors (S.Second_order.unpack mid.S.Mid.sigma_g)
  in
  List.concat_map blocks ~f:(fun b ->
    let w, _ = eigh float64 (cast float64 b) in
    Array.to_list (to_array (cast float32 w)))
  |> List.fold ~init:0.0 ~f:Float.max

let flag_int name ~default = Option.value (Cmdargs.get_int name) ~default
let beta_1 = 0.9
let beta_2 = 0.99
let seed = flag_int "-seed" ~default:7
let iters = flag_int "-iters" ~default:300
let batch = flag_int "-batch" ~default:128
let metric_batch = flag_int "-metric-batch" ~default:8
let log_every = flag_int "-log-every" ~default:5

let floats_of_option name ~default =
  match Cmdargs.get_string name with
  | None -> default
  | Some s -> String.split s ~on:',' |> List.map ~f:Float.of_string

let lrs = floats_of_option "-lrs" ~default:[]
let rhos = floats_of_option "-rhos" ~default:[ 0.0003; 0.001; 0.003; 0.01; 0.03; 0.1 ]

(* The symo step is preconditioned, Adam's is not: their learning rates live on
   different scales, so each has its own default grid. *)
let lrs_for variant =
  match lrs with
  | _ :: _ -> lrs
  | [] ->
    (match variant.optimizer with
     | `Adam -> [ 0.001; 0.003; 0.01; 0.03; 0.1 ]
     | `Symo -> [ 0.1; 0.2; 0.4 ])

let only = Cmdargs.get_string "-only"

type run =
  { variant : variant
  ; lr : float
  ; rho : float
  ; damping : float
  ; points : (int * float) list
  }

(* The mean of the last few validation points: the final loss of a single
   batch is noisy. *)
let final_loss run =
  let n = List.length run.points in
  let tail = List.drop run.points (Int.max 0 (n - 3)) in
  List.fold tail ~init:0. ~f:(fun acc (_, l) -> acc +. l)
  /. Float.of_int (List.length tail)

let train_adam ~lr ~params ~data ~val_data =
  let state = ref (Adam.init params) in
  let theta = ref params in
  let validate () = item [] (batch_loss ~horizon !theta (val_data.x0, val_data.xt)) in
  let points = ref [ 0, validate () ] in
  for i = 1 to iters do
    let key = Rng.fold_in (Rng.key seed) i in
    let x0, xt = minibatch ~key ~bs:batch data in
    let _, grads =
      Rune.value_and_grad S.ptree (fun p -> batch_loss ~horizon p (x0, xt)) !theta
    in
    let state', theta' = Adam.step !state ~lr ~params:!theta ~grads in
    state := state';
    theta := theta';
    if Int.(i % log_every = 0) then points := (i, validate ()) :: !points
  done;
  List.rev !points

let train_symo ~variant ~lr ~damping ~params ~data ~val_data =
  let config : S.config =
    match variant.form with
    | `Inverse -> S.default ~learning_rate:(Some lr) ~damping
    | `Root -> { (S.default ~learning_rate:(Some lr) ~damping) with form = `Root }
  in
  let state = ref (S.init ~config params) in
  let validate () =
    item [] (batch_loss ~horizon !state.S.State.theta (val_data.x0, val_data.xt))
  in
  let points = ref [ 0, validate () ] in
  for i = 1 to iters do
    let key = Rng.fold_in (Rng.key seed) i in
    let x0, xt = minibatch ~key ~bs:batch data in
    let _, grads =
      Rune.value_and_grad
        S.ptree
        (fun p -> batch_loss ~horizon p (x0, xt))
        !state.S.State.theta
    in
    let x0m = slice [ Nx.R (0, metric_batch); Nx.A ] x0 in
    let samples = samples !state.S.State.theta x0m in
    state := S.step ~config ~state:!state ~grads ~samples;
    if Int.(i % log_every = 0) then points := (i, validate ()) :: !points
  done;
  List.rev !points

let train ~variant ~lr ~damping ~params ~data ~val_data =
  match variant.optimizer with
  | `Adam -> train_adam ~lr ~params ~data ~val_data
  | `Symo -> train_symo ~variant ~lr ~damping ~params ~data ~val_data

let () =
  let params = Model.init ~key:(Rng.key seed) in
  let train_data, val_data = make_data () in
  let x0m, xtm = minibatch ~key:(Rng.key 11) ~bs:metric_batch train_data in
  Stdio.printf
    "Lorenz-63: %d hidden units, horizon %d, %d/%d chunks\n%!"
    Model.hidden
    horizon
    n_train
    n_val;
  let variants =
    match only with
    | None -> variants
    | Some s -> List.filter variants ~f:(fun v -> String.is_substring v.name ~substring:s)
  in
  let runs =
    List.concat_map variants ~f:(fun variant ->
      match variant.optimizer with
      | `Adam ->
        Stdio.printf
          "\nAdam (beta_1 %.2g, beta_2 %.3g):\n%!"
          Adam.adam_beta_1
          Adam.adam_beta_2;
        List.map (lrs_for variant) ~f:(fun lr ->
          let points =
            train ~variant ~lr ~damping:0. ~params ~data:train_data ~val_data
          in
          let run = { variant; lr; rho = 0.; damping = 0.; points } in
          Stdio.printf "  lr %-6g final loss %.5f\n%!" lr (final_loss run);
          run)
      | `Symo ->
        let scale = metric_scale ~params ~x0:x0m ~xt:xtm in
        Stdio.printf "\n%s (metric top eigenvalue %.3g):\n%!" variant.name scale;
        List.concat_map (lrs_for variant) ~f:(fun lr ->
          List.map rhos ~f:(fun rho ->
            let damping =
              match variant.form with
              | `Root -> rho *. Float.sqrt scale
              | `Inverse -> rho *. scale
            in
            let points = train ~variant ~lr ~damping ~params ~data:train_data ~val_data in
            let run = { variant; lr; rho; damping; points } in
            Stdio.printf
              "  lr %-4g rho %-5g  damping %-10.4g  final loss %.5f\n%!"
              lr
              rho
              damping
              (final_loss run);
            run)))
  in
  let best =
    List.map variants ~f:(fun variant ->
      List.filter runs ~f:(fun r -> Poly.equal r.variant.name variant.name)
      |> List.min_elt ~compare:(fun a b ->
        let key r =
          if Float.is_nan (final_loss r) then Float.infinity else final_loss r
        in
        Float.compare (key a) (key b))
      |> Option.value_exn)
  in
  Stdio.printf "\nBest:\n";
  List.iter best ~f:(fun r ->
    match r.variant.optimizer with
    | `Adam ->
      Stdio.printf
        "  %-14s lr %-5g final loss %.5f\n%!"
        r.variant.name
        r.lr
        (final_loss r)
    | `Symo ->
      Stdio.printf
        "  %-14s lr %-5g rho %-6g damping %-10.4g final loss %.5f\n%!"
        r.variant.name
        r.lr
        r.rho
        r.damping
        (final_loss r));
  let n = List.length (List.hd_exn best).points in
  let steps =
    create
      float32
      [| n |]
      (Array.of_list
         (List.map (List.hd_exn best).points ~f:(fun (s, _) -> Float.of_int s)))
  in
  let losses =
    create
      float32
      [| List.length best; n |]
      (Array.concat
         (List.map best ~f:(fun r ->
            Array.of_list (List.map r.points ~f:(fun (_, l) -> l)))))
  in
  let codes =
    create
      int32
      [| List.length best; 1 |]
      (Array.init (List.length best) ~f:Int32.of_int_exn)
  in
  let labels =
    Array.of_list
      (List.map best ~f:(fun r ->
         match r.variant.optimizer with
         | `Adam -> Printf.sprintf "%s (lr %g)" r.variant.name r.lr
         | `Symo -> Printf.sprintf "%s (lr %g, λ %.3g)" r.variant.name r.lr r.damping))
  in
  let figure =
    Hugin.line
      ~x:(Hugin.num ~title:"step" steps)
      ~y:(Hugin.num ~title:"validation loss" ~scale:(Hugin.Scale.log ()) losses)
      ~stroke:(Hugin.cat ~title:"metric" ~labels codes)
      ()
    |> Hugin.title "Lorenz-63: damped GGN and Adam, best learning rate and damping"
  in
  Hugin.save ~size:(Hugin.Size.figure 720. 400.) (in_dir "lorenz_loss.png") figure;
  Nx_io.save_txt (in_dir "lorenz_loss") losses;
  Stdio.printf "\nwrote %s\n%!" (in_dir "lorenz_loss.png")
