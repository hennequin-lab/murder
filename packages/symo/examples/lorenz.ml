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
   draws the loss curves with hugin. *)

open Base
open Nx
open Symo

let in_dir = Cmdargs.in_dir "-d"

(* ------------------------------------------------------------------------
   Data: Lorenz-63, RK4, standardized chunks
   ------------------------------------------------------------------------ *)

let dt = 0.01
let sigma = 10.
let rho = 28.
let beta = 8. /. 3.
let d = 3
let horizon = 8
let n_train = 4096
let n_val = 1024
let n_chunks = n_train + n_val

let drift y =
  [| sigma *. (y.(1) -. y.(0))
   ; (y.(0) *. (rho -. y.(2))) -. y.(1)
   ; (y.(0) *. y.(1)) -. (beta *. y.(2))
  |]

let rk4 y =
  let h = dt in
  let moved base k c = Array.map2_exn base k ~f:(fun y d -> y +. (c *. h *. d)) in
  let k1 = drift y in
  let k2 = drift (moved y k1 0.5) in
  let k3 = drift (moved y k2 0.5) in
  let k4 = drift (moved y k3 1.0) in
  Array.mapi y ~f:(fun i y ->
    y +. (h /. 6. *. (k1.(i) +. (2. *. k2.(i)) +. (2. *. k3.(i)) +. k4.(i))))

type data =
  { x0 : host float32_t (* [n; 3] *)
  ; xt : host float32_t (* [n; horizon; 3] *)
  }

(* One chunk is [horizon] transitions: its initial state and the [horizon]
   states that follow. The states are standardized per coordinate over the
   whole record. *)
let make_data () =
  let n_states = (n_chunks * horizon) + 1 in
  let y = ref [| 1.; 1.; 1. |] in
  for _ = 1 to 1000 do
    y := rk4 !y
  done;
  let states = Array.create ~len:(n_states * d) 0. in
  for i = 0 to n_states - 1 do
    let yi = !y in
    for j = 0 to d - 1 do
      states.((i * d) + j) <- yi.(j)
    done;
    y := rk4 yi
  done;
  let raw = create float32 [| n_states; d |] states in
  let mu = mean ~axes:[ 0 ] raw in
  let sd = std ~axes:[ 0 ] raw in
  let z = to_array Infix.((raw - mu) / sd) in
  let x0 = Array.create ~len:(n_chunks * d) 0. in
  let xt = Array.create ~len:(n_chunks * horizon * d) 0. in
  for b = 0 to n_chunks - 1 do
    for j = 0 to d - 1 do
      x0.((b * d) + j) <- z.((b * horizon * d) + j)
    done;
    for t = 0 to horizon - 1 do
      for j = 0 to d - 1 do
        xt.((((b * horizon) + t) * d) + j) <- z.((((b * horizon) + t + 1) * d) + j)
      done
    done
  done;
  let all =
    { x0 = create float32 [| n_chunks; d |] x0
    ; xt = create float32 [| n_chunks; horizon; d |] xt
    }
  in
  ( { x0 = slice [ Nx.R (0, n_train); Nx.A ] all.x0
    ; xt = slice [ Nx.R (0, n_train); Nx.A; Nx.A ] all.xt
    }
  , { x0 = slice [ Nx.R (n_train, n_chunks); Nx.A ] all.x0
    ; xt = slice [ Nx.R (n_train, n_chunks); Nx.A; Nx.A ] all.xt
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
   The metric's scale, and the sweep
   ------------------------------------------------------------------------ *)

type variant =
  { name : string
  ; form : [ `Inverse | `Root ]
  }

let variants =
  [ { name = "GGN, inverse"; form = `Inverse }; { name = "GGN, root"; form = `Root } ]

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
let iters = flag_int "-iters" ~default:300
let batch = flag_int "-batch" ~default:128
let metric_batch = flag_int "-metric-batch" ~default:8
let log_every = flag_int "-log-every" ~default:5

let floats_of_option name ~default =
  match Cmdargs.get_string name with
  | None -> default
  | Some s -> String.split s ~on:',' |> List.map ~f:Float.of_string

let lrs = floats_of_option "-lrs" ~default:[ 0.1; 0.2; 0.4 ]
let rhos = floats_of_option "-rhos" ~default:[ 0.0003; 0.001; 0.003; 0.01; 0.03; 0.1 ]
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

let train ~variant ~lr ~damping ~params ~data ~val_data =
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
    let key = Rng.fold_in (Rng.key 7) i in
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

let () =
  let params = Model.init ~key:(Rng.key 42) in
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
      let scale = metric_scale ~params ~x0:x0m ~xt:xtm in
      Stdio.printf "\n%s (metric top eigenvalue %.3g):\n%!" variant.name scale;
      List.concat_map lrs ~f:(fun lr ->
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
         Printf.sprintf "%s (lr %g, λ %.3g)" r.variant.name r.lr r.damping))
  in
  let figure =
    Hugin.line
      ~x:(Hugin.num ~title:"step" steps)
      ~y:(Hugin.num ~title:"validation loss" ~scale:(Hugin.Scale.log ()) losses)
      ~stroke:(Hugin.cat ~title:"metric" ~labels codes)
      ()
    |> Hugin.title "Lorenz-63: inverse-damped GGN, best learning rate and damping"
  in
  Hugin.save ~size:(Hugin.Size.figure 720. 400.) (in_dir "lorenz_loss.png") figure;
  Nx_io.save_txt (in_dir "lorenz_loss") losses;
  Stdio.printf "\nwrote %s\n%!" (in_dir "lorenz_loss.png")
