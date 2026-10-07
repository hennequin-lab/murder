(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Types

type ('w, 'p) t =
  { n : int
  ; m : int
  ; structure : 'p Nx.Ptree.t
  ; params : 'p
  ; dyn : 'p -> k:'w value -> x:'w value -> u:'w value -> 'w value
  }

let integrate (d : ('w, 'p) t) (p : 'p) ~(x0 : 'w value) ~(u : 'w value) =
  let horizon = (Nx.shape u).(0) in
  let ks = Nx.arange (Nx.dtype x0) 0 horizon 1 in
  let _, xs =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      ~f:(fun x (u_k, k) ->
        let x' = d.dyn p ~k ~x ~u:u_k in
        x', x')
      ~init:x0
      (u, ks)
  in
  (* [x0] may be one state or a batch of states; broadcast it to the carry's
     batch shape before stacking the initial state in front of the path. *)
  let batch_shape =
    Array.sub (Nx.shape xs) ~pos:1 ~len:(Array.length (Nx.shape xs) - 1)
  in
  let x0 =
    Nx.reshape (Array.append [| 1 |] batch_shape) (Nx.broadcast_to batch_shape x0)
  in
  Nx.concatenate ~axis:0 [ x0; xs ]

(* [randn ?key dtype shape] is a standard normal draw, from [key] when given
   and from the implicit random scope otherwise. *)
let randn ?key dtype shape =
  match key with
  | Some key -> Nx.Rng.normal key dtype shape
  | None -> Nx.randn dtype shape

(* [scaled_randn ?key ~sigma dtype shape] is a normal draw of standard
   deviation [sigma]. *)
let scaled_randn ?key ~sigma dtype shape = Nx.mul_s (randn ?key dtype shape) sigma

(* [first_step k] is a boolean scalar that is [true] at step 0. A dynamics
   written for these constructors selects on it with [Nx.where], since [k] is
   a tensor under the encoder's scans. *)
let first_step (k : 'w value) = Nx.equal k (Nx.scalar_like k 0.0)

(* [input_term ~dt_over_tau ~k ~u] is the input contribution of one step: the
   projected input itself at the first step, [dt_over_tau] times it
   afterwards. Applying the whole scaled input at step 0 is the convention of
   the reference implementation. *)
let input_term ~dt_over_tau ~k ~u =
  let scaled =
    match dt_over_tau with
    | `Scalar dt -> Nx.mul_s u dt
    | `Vector dt -> Nx.mul u dt
  in
  Nx.where (first_step k) u scaled

(* {1 Linear} *)

module Linear = struct
  type 'a params =
    { dt_over_tau : float [@ptree.skip]
    ; w : 'a
    ; b : 'a option
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type 'a t = 'a params

        let walk = walk_params
      end)

  let matrix { dt_over_tau; w; _ } =
    let identity = Nx.eye (Nx.dtype w) (Array.get (Nx.shape w) 0) in
    Nx.add identity (Nx.mul_s (Nx.sub w identity) dt_over_tau)

  let dyn p ~k ~x ~u =
    let u =
      match p.b with
      | None -> u
      | Some b -> Nx.matmul u b
    in
    let input = input_term ~dt_over_tau:(`Scalar p.dt_over_tau) ~k ~u in
    Nx.add (Nx.matmul x (matrix p)) input

  let make ?key ?(alpha = 1.0) ~n ~m dtype =
    if n <= 0 then invalid_arg "Dynamics.Linear.make: n must be positive";
    if m <= 0 then invalid_arg "Dynamics.Linear.make: m must be positive";
    let w = Nx.mul_s (Nx.eye dtype n) alpha in
    let b = if Int.equal n m then None else Some (randn ?key dtype [| m; n |]) in
    { n; m; structure = structure (); params = { dt_over_tau = 0.1; w; b }; dyn }
end

(* {1 Linear–nonlinear} *)

module Linear_nonlinear = struct
  type 'a params =
    { dt_over_tau : 'a
    ; a0 : 'a
    ; a1 : 'a
    ; a2 : 'a
    ; bias1 : 'a
    ; bias2 : 'a
    ; b : 'a option
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type 'a t = 'a params

        let walk = walk_params
      end)

  let dyn ~phi p ~k ~x ~u =
    let hidden = phi (Nx.add (Nx.matmul x p.a1) p.bias1) in
    let passive =
      Nx.add
        x
        (Nx.mul
           (Nx.add (Nx.sub (Nx.matmul x p.a0) x) (Nx.add (Nx.matmul hidden p.a2) p.bias2))
           p.dt_over_tau)
    in
    let u =
      match p.b with
      | None -> u
      | Some b -> Nx.matmul u b
    in
    Nx.add passive (input_term ~dt_over_tau:(`Vector p.dt_over_tau) ~k ~u)

  let make ?key ?(radius = 0.1) ~dt ~tau ~n ~nh ~m ~phi dtype =
    if n <= 0 then invalid_arg "Dynamics.Linear_nonlinear.make: n must be positive";
    if nh <= 0 then invalid_arg "Dynamics.Linear_nonlinear.make: nh must be positive";
    if m <= 0 then invalid_arg "Dynamics.Linear_nonlinear.make: m must be positive";
    if not (Float.( > ) dt 0.0 && Float.( > ) tau 0.0)
    then invalid_arg "Dynamics.Linear_nonlinear.make: dt and tau must be positive";
    let params =
      { dt_over_tau = Nx.full dtype [| n |] (dt /. tau)
      ; a0 = Nx.mul_s (Nx.eye dtype n) 0.5
      ; a1 =
          scaled_randn ?key ~sigma:(1.0 /. Float.sqrt (Float.of_int n)) dtype [| n; nh |]
      ; a2 =
          scaled_randn
            ?key
            ~sigma:(radius /. Float.sqrt (Float.of_int nh))
            dtype
            [| nh; n |]
      ; bias1 = Nx.sub_s (Nx.mul_s (Nx.rand dtype [| nh |]) 2.0) 1.0
      ; bias2 = Nx.zeros dtype [| n |]
      ; b =
          Some
            (scaled_randn
               ?key
               ~sigma:(1.0 /. Float.sqrt (Float.of_int m))
               dtype
               [| m; n |])
      }
    in
    { n
    ; m
    ; structure = structure ()
    ; params
    ; dyn = (fun p ~k ~x ~u -> dyn ~phi p ~k ~x ~u)
    }
end

(* {1 Gated nonlinear} *)

module Gnode = struct
  type 'a params =
    { dt_over_tau : float [@ptree.skip]
    ; g_a1 : 'a
    ; g_bias1 : 'a
    ; h_a1 : 'a
    ; h_a2 : 'a
    ; h_bias1 : 'a
    ; h_bias2 : 'a
    ; b : 'a option
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type 'a t = 'a params

        let walk = walk_params
      end)

  let dyn ~phi p ~k ~x ~u =
    let g = phi (Nx.add (Nx.matmul x p.g_a1) p.g_bias1) in
    let h =
      Nx.add (Nx.matmul (phi (Nx.add (Nx.matmul x p.h_a1) p.h_bias1)) p.h_a2) p.h_bias2
    in
    let passive = Nx.add x (Nx.mul_s (Nx.mul g (Nx.sub h x)) p.dt_over_tau) in
    let u =
      match p.b with
      | None -> u
      | Some b -> Nx.matmul u b
    in
    Nx.add passive (input_term ~dt_over_tau:(`Scalar p.dt_over_tau) ~k ~u)

  let make ?key ?(radius = 0.1) ?m ~dt ~tau ~n ~nh ~phi dtype =
    if n <= 0 then invalid_arg "Dynamics.Gnode.make: n must be positive";
    if nh <= 0 then invalid_arg "Dynamics.Gnode.make: nh must be positive";
    if not (Float.( > ) dt 0.0 && Float.( > ) tau 0.0)
    then invalid_arg "Dynamics.Gnode.make: dt and tau must be positive";
    let m_input = m in
    let m = Option.value m ~default:n in
    let params =
      { dt_over_tau = dt /. tau
      ; g_a1 = Nx.zeros dtype [| n; n |]
      ; g_bias1 = Nx.zeros dtype [| n |]
      ; h_a1 =
          scaled_randn ?key ~sigma:(1.0 /. Float.sqrt (Float.of_int n)) dtype [| n; nh |]
      ; h_a2 =
          scaled_randn
            ?key
            ~sigma:(radius /. Float.sqrt (Float.of_int nh))
            dtype
            [| nh; n |]
      ; h_bias1 = Nx.zeros dtype [| nh |]
      ; h_bias2 = Nx.zeros dtype [| n |]
      ; b = Option.map m_input ~f:(fun m -> randn ?key dtype [| m; n |])
      }
    in
    { n
    ; m
    ; structure = structure ()
    ; params
    ; dyn = (fun p ~k ~x ~u -> dyn ~phi p ~k ~x ~u)
    }
end
