open Base
open Nx
open Jera

let in_dir = Cmdargs.in_dir "-d"
let print s = Stdio.print_endline (Sexp.to_string_hum s)

let device =
  let gpu =
    match Cmdargs.get_string "-dev" with
    | Some "amd" | Some "AMD" ->
      Rig.open_ (Rig_amd_amdgpu.gpu 0) |> Stdlib.Result.get_ok |> Option.some
    | Some "nv" | Some "NV" ->
      Rig.open_ (Rig_nv_nvidia.gpu 0) |> Stdlib.Result.get_ok |> Option.some
    | _ -> None
  in
  match gpu with
  | Some g -> g
  | None -> Rig.host

module D = (val Nx.devices [ device ])

let placement = D.on

(* [Nx.relu] is gone from [Nx] and activations now live in [Kaun]; the example
   keeps its own rather than depend on [Kaun] for one line. *)
let relu x =
  let zero = scalar_like x 0. in
  where Infix.(x <= zero) zero x

(* Parameters *)

let total_bs = 4000
let bs = 256
let horizon = 32
let d, d_hidden = 3, 400
let max_iter = 10_000
let lr = 0.3
let n_tangents = 128
let damping : Sofo.Optim.damping = `Relative_from_top 1e-5

(* Data generation: simulate the Lorenz attractor for a very long time using RK4,
   and chop the resulting sequence into [total_bs] chunks *)

let lorenz_trajs =
  let file = in_dir "data.npy" in
  try unpack float32 (Nx_io.load_npy file) with
  | _ ->
    let dt = 0.01 in
    let sigma = 10. in
    let rho = 28. in
    let beta = 8. /. 3. in
    let lorenz _t y =
      let open Nx.Infix in
      let x = slice [ A; I 0 ] y
      and yc = slice [ A; I 1 ] y
      and z = slice [ A; I 2 ] y in
      let dx = (yc - x) *$ sigma
      and dy = (x * (-z +$ rho)) - yc
      and dz = (x * yc) - (z *$ beta) in
      stack ~axis:1 [ dx; dy; dz ]
    in
    (* The trajectory is the march's own output: [at] holds one time per step
       and every step is kept, so nothing is interpolated. Its times are
       float64, the steps as exact as the stride allows. *)
    let march =
      Rune.jit
        Nx.Ptree.(tensor @-> tensor @-> returns tensor)
        (fun at y0 -> Ode.march Nx.Ptree.tensor Ode.rk4 ~steps:1 lorenz ~at y0)
    in
    Rng.with_key (Rng.key 42)
    @@ fun () ->
    let n_bins = Int.((total_bs + 10) * horizon) in
    let y0 = ones float32 [| 1; 3 |] in
    let at = linspace float64 0. Float.(dt * of_int Int.(n_bins - 1)) n_bins in
    Stdio.printf "integrating the lorenz ODE... %!";
    let ys = march at y0 in
    Stdio.printf "done.\n%!";
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
  let indices = Rng.permutation ~key total_bs |> slice [ R (0, bs) ] in
  let x0 = take ~axis:0 indices (slice [ I 0 ] lorenz_trajs) in
  let xT = take ~axis:0 indices (slice [ I (horizon - 1) ] lorenz_trajs) in
  place placement x0, place placement xT

let data_ptree = Ptree.(pair tensor tensor)

(* Model specifications *)

module Model = struct
  module P = struct
    type (_, 'd) t =
      { w : 'd float32_t
      ; c : 'd float32_t
      ; a : 'd float32_t
      }
    [@@deriving ptree]
  end

  let ptree : ((D.d float32_t, D.d) P.t, D.d) Nx.Ptree.t = Nx.Ptree.instantiate (module P)

  let init ~d ~d_hidden =
    let open Nx.Infix in
    let w =
      Rng.normal float32 [| d_hidden; d |] *$ Float.(0.1 / sqrt (of_int d_hidden))
    in
    let c =
      Rng.normal float32 [| Int.(d + 1); d_hidden |] /$ Float.(sqrt (of_int Int.(d + 1)))
    in
    let a = eye float32 d in
    P.{ w; c; a }

  let step (theta : _ P.t) x =
    let open Nx.Infix in
    let bs = dim 0 x in
    (x *@ theta.a)
    + (relu (concatenate ~axis:1 [ x; ones_at (beside x) float32 [| bs; 1 |] ] *@ theta.c)
       *@ theta.w)

  let forward ~horizon (theta : _ P.t) x0 =
    Rune.scan
      Ptree.tensor
      Ptree.tensor
      Ptree.unit
      (fun x _t -> step theta x, ())
      ~init:x0
      (zeros_at (beside x0) float32 [| horizon - 1 |])
    |> fst
end

let params =
  Rng.with_key (Rng.key 42) (fun () -> Model.init ~d ~d_hidden)
  |> Nx.Ptree.place (module Model.P) placement

let objective params (x0, xf) =
  let pred = Model.forward ~horizon params x0 in
  Sofo.mse ~w:(scalar_like pred Float.(1. / of_int horizon)) ~target:xf pred

(* The sketch compiles for the device the parameters are placed on; the
   minibatch is placed there too, and the keys are host values that join them
   on each call. *)
let sketch_step =
  Sofo.Optim.sketch_jit
    ~beam:2
    ~parallel:8
    ~k:n_tangents
    placement
    Model.ptree
    data_ptree
    objective

let rec loop_sofo ~i ~out ~losses (params : _ Model.P.t) (state : Sofo.Optim.state) =
  if i >= max_iter
  then params
  else (
    let key_data = Rng.fold_in state.key 0 in
    let data = minibatch key_data bs in
    let sketch = sketch_step ~key:state.key ~aux:data params in
    let params, state =
      Sofo.Optim.update ~lr ~damping placement Model.ptree state params sketch
    in
    let loss = item [] sketch.loss in
    Stdio.printf "[%05i] loss = %.6f\n%!" i loss;
    (* A save replaces its file, so every tenth loss so far is saved as one
       column. *)
    let losses = if i % 10 = 0 then loss :: losses else losses in
    if i % 10 = 0
    then (
      let n = List.length losses in
      Nx_io.save_txt out (create float32 [| n; 1 |] (Array.of_list (List.rev losses))));
    loop_sofo ~i:(i + 1) ~out ~losses params state)

let state = Sofo.Optim.init ~key:(Rng.key 1985) ()

let _ =
  Stdio.printf "device: %s\n%!" (Rig.name device);
  let out = in_dir "loss" in
  let () = Bos.Cmd.(v "rm" % "-f" % out) |> Bos.OS.Cmd.run |> ignore in
  loop_sofo ~i:0 ~out ~losses:[] params state
