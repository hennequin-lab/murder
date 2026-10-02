open Base
open Nx

let in_dir = Cmdargs.in_dir "-d"
let print s = Stdio.print_endline (Sexp.to_string_hum s)

(* Parameters *)

let total_bs = 4000
let bs = 256
let horizon = 32
let d, d_hidden = 3, 400
let max_iter = 10_000
let lr = 0.3
let n_tangents = 128
let damping : Sofo.Optim.damping = `Relative_from_top 1e-5

(* [--device] lists the devices to try, in order, as [Devices.first] reads
   them: "cpu" (the default), "cuda", "cuda:1", "cuda,cpu". The sketch
   accumulates in float64, which Metal cannot compute. *)
let device = Devices.first Cmdargs.(get_string "--device" |> default "cpu")

(* Data generation: simulate the Lorenz attractor for a very long time using RK4,
   and chop the resulting sequence into [total_bs] chunks *)

let lorenz_trajs =
  let file = in_dir "data.npy" in
  try Nx.unpack float32 (Nx_io.load_npy file) with
  | _ ->
    let dt = 0.01 in
    let sigma = 10. in
    let rho = 28. in
    let beta = 8. /. 3. in
    let lorenz =
      Rune.jit
        Nx.Ptree.(tensor @-> returns tensor)
        (fun y ->
           let open Infix in
           let x = slice [ A; I 0 ] y
           and yc = slice [ A; I 1 ] y
           and z = slice [ A; I 2 ] y in
           let dx = (yc - x) *$ sigma
           and dy = (x * (-z +$ rho)) - yc
           and dz = (x * yc) - (z *$ beta) in
           stack ~axis:1 [ dx; dy; dz ])
    in
    Rng.with_key (Rng.key 42)
    @@ fun () ->
    let n_bins = Int.((total_bs + 10) * horizon) in
    let ys =
      Laguz.solve
        ~stepper:Laguz.rk4
        ~output:`Every_step
        ~max_steps:n_bins
        ~dt
        ~rhs:(fun _ y -> lorenz y)
        ~t0:0.
        ~t1:Float.(dt * of_int Int.(n_bins - 1))
        ~y0:(ones float32 [| 1; 3 |])
        ()
      |> Laguz.ys
      |> Option.value_exn
    in
    let n_bins = dim 0 ys in
    let ys =
      ys
      |> slice Int.[ R (n_bins - 1 - (total_bs * horizon), -1) ]
      |> reshape [| total_bs; horizon; 3 |]
      |> transpose ~axes:[ 1; 0; 2 ]
      |> fun x -> Infix.((x - mean ~axes:[ 0; 1 ] x) / std ~axes:[ 0; 1 ] x)
    in
    Nx_io.save_npy file ys;
    ys (* [ horizon, total_bs, 3 ] *)

let _ = print [%message (shape lorenz_trajs : int array)]

let minibatch key bs =
  let indices = Rng.permutation key total_bs |> slice [ R (0, bs) ] in
  let x0 = take ~axis:0 ~indices (slice [ I 0 ] lorenz_trajs) in
  let xT = take ~axis:0 ~indices (slice [ I (horizon - 1) ] lorenz_trajs) in
  x0, xT

(* Model specifications *)

module Model = struct
  module P = struct
    type t =
      { w : float32_t
      ; c : float32_t
      ; a : float32_t
      }
    [@@deriving ptree]
  end

  let init ~d ~d_hidden =
    let open Infix in
    let w = randn float32 [| d_hidden; d |] *$ Float.(0.1 / sqrt (of_int d_hidden)) in
    let c =
      randn float32 [| Int.(d + 1); d_hidden |] /$ Float.(sqrt (of_int Int.(d + 1)))
    in
    let a = eye float32 d in
    P.{ w; c; a }

  let step (theta : P.t) x =
    let open Infix in
    let bs = dim 0 x in
    (x *@ theta.a)
    + (relu (concatenate ~axis:1 [ x; ones float32 [| bs; 1 |] ] *@ theta.c) *@ theta.w)

  let forward ~horizon (theta : P.t) x0 =
    Rune.scan'
      ~f:(fun x t -> step theta x, t)
      ~init:x0
      (Nx.zeros float32 [| horizon - 1; 1 |])
    |> fst
end

let params =
  Rng.with_key (Rng.key 42) (fun () -> Model.init ~d ~d_hidden)
  |> Nx.Ptree.place Model.P.ptree (Nx.Placement.on device)

module Aux = struct
  type t = float32_t * float32_t [@@deriving ptree]
end

module O = Sofo.Optim.Compiled (Model.P) (Aux)

let objective params (x0, xf) =
  let open Infix in
  let pred = Model.forward ~horizon params x0 in
  Sofo.mse ~w:(scalar float32 Float.(1. / of_int horizon)) ~target:xf pred

(* The sketch compiles for the device the parameters are placed on; the
   minibatch and the keys are host values that join them on each call. *)
let sketch_step = Rune.jit O.signature (O.sketch ~k:n_tangents objective)

let rec loop_sofo ~i ~out (params : Model.P.t) (state : Sofo.Optim.state) =
  if i >= max_iter
  then params
  else (
    let key_data = Rng.fold_in state.key 0 in
    let data = minibatch key_data bs in
    let sketch = sketch_step { params; key = state.key; aux = data } in
    let params, state = O.update ~lr ~damping state params sketch in
    let loss = item [] sketch.loss in
    Stdio.printf "[%05i] loss = %.6f\n%!" i loss;
    if i % 10 = 0 then Nx_io.save_txt ~append:true out (create float32 [| 1 |] [| loss |]);
    loop_sofo ~i:(i + 1) ~out params state)

let state = Sofo.Optim.init ~key:(Rng.key 1985) ()

let _ =
  Stdio.printf "device: %s\n%!" (Nx.Device.name device);
  let out = in_dir "loss" in
  let () = Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore in
  loop_sofo ~i:0 ~out params state

(* Adam version *)

module In = struct
  type t =
    { params : Model.P.t
    ; data : Aux.t
    }
  [@@deriving ptree]
end

module Out = struct
  type t = float32_t * Model.P.t [@@deriving ptree]
end

let value_and_grad_jit =
  Rune.jit
    Nx.Ptree.(In.ptree @-> returns Out.ptree)
    (fun In.{ params; data } ->
       Rune.value_and_grad Model.P.ptree (fun model -> objective model data) params)

let rec loop_adam ~i ~out (params : Model.P.t) state key =
  if i >= max_iter
  then params
  else (
    let keys = Rng.split key ~n:2 in
    let data = minibatch keys.(0) bs in
    let loss_val, grads = value_and_grad_jit { params; data } in
    let params, state =
      Vega.adam_step Model.P.ptree ~lr:(scalar float32 0.0002) state ~params ~grads
    in
    let loss = item [] loss_val in
    Stdio.printf "[%05i] loss = %.6f\n%!" i loss;
    if i % 10 = 0 then Nx_io.save_txt ~append:true out (create float32 [| 1 |] [| loss |]);
    loop_adam ~i:(i + 1) ~out params state keys.(1))

let state = Vega.adam_init Model.P.ptree params

(*
   let _ =
   let out = in_dir "loss_adam" in
   let () = Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore in
   loop_adam ~i:0 ~out params state (Rng.key 1985)
*)
