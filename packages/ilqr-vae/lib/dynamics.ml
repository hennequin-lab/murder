(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Types

type ('w, 'p) t =
  { n : int
  ; m : int
  ; structure : 'p Nx.Ptree.t
  ; params : 'p
  ; dyn : 'p -> k:'w value -> x:'w value -> u:'w value -> 'w value
  }

let shape_string shape =
  String.concat ~sep:"; " (Array.to_list (Array.map shape ~f:Int.to_string))

(* [integrate_along d p ~time_axis ~x0 ~u] is the rollout when [time_axis] is
   the horizon's position in [u]. [Rune.scan] walks axis 0, so the horizon is
   moved there for the scan and the result put back where it belongs. *)
let integrate_along (d : ('w, 'p) t) (p : 'p) ~time_axis ~(x0 : 'w value) ~(u : 'w value) =
  let horizon = (shape u).(time_axis) in
  let u = if Int.equal time_axis 0 then u else moveaxis time_axis 0 u in
  let ks = arange (Nx.dtype x0) 0 horizon 1 in
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
  let batch_shape = Array.sub (shape xs) ~pos:1 ~len:(Array.length (shape xs) - 1) in
  let x0 = reshape (Array.append [| 1 |] batch_shape) (broadcast_to batch_shape x0) in
  let path = concatenate ~axis:0 [ x0; xs ] in
  if Int.equal time_axis 0 then path else moveaxis 0 time_axis path

let integrate (d : ('w, 'p) t) (p : 'p) ~(x0 : 'w value) ~(u : 'w value) =
  let rank = Array.length (shape u) in
  if rank < 2
  then
    invalid_arg
      (Printf.sprintf
         "Dynamics.integrate: u must be [batch…; n_steps; %d], got rank %d"
         d.m
         rank);
  let time_axis = rank - 2 in
  if Array.length (shape x0) <> time_axis + 1
  then
    invalid_arg
      (Printf.sprintf
         "Dynamics.integrate: x0 must be [batch…; %d] for u with %d batch axes, got [%s]"
         d.n
         time_axis
         (shape_string (shape x0)));
  integrate_along d p ~time_axis ~x0 ~u

let integrate_samples (d : ('w, 'p) t) (p : 'p) ~(x0 : 'w value) ~(u : 'w value) =
  let rank = Array.length (shape u) in
  if rank < 3
  then
    invalid_arg
      (Printf.sprintf
         "Dynamics.integrate_samples: u must be [batch…; n_steps; n_samples; %d], got \
          rank %d"
         d.m
         rank);
  let time_axis = rank - 3 in
  if Array.length (shape x0) <> time_axis + 2
  then
    invalid_arg
      (Printf.sprintf
         "Dynamics.integrate_samples: x0 must be [batch…; n_samples; %d] for u with %d \
          batch axes, got [%s]"
         d.n
         time_axis
         (shape_string (shape x0)));
  integrate_along d p ~time_axis ~x0 ~u

(* [randn ?key dtype shape] is a standard normal draw, from [key] when given
   and from the implicit random scope otherwise. *)
let randn ?key dtype shape =
  match key with
  | Some key -> Nx.Rng.normal key dtype shape
  | None -> Nx.randn dtype shape

(* [scaled_randn ?key ~sigma dtype shape] is a normal draw of standard
   deviation [sigma]. *)
let scaled_randn ?key ~sigma dtype shape = Infix.(randn ?key dtype shape *$ sigma)

(* [first_step k] is a boolean scalar that is [true] at step 0. A dynamics
   written for these constructors selects on it with [Nx.where], since [k] is
   a tensor under the encoder's scans. *)
let first_step (k : 'w value) = Nx.equal k (scalar_like k 0.0)

(* [input_term ~dt_over_tau ~k ~u] is the input contribution of one step: the
   projected input itself at the first step, [dt_over_tau] times it
   afterwards. Applying the whole scaled input at step 0 is the convention of
   the reference implementation. *)
let input_term ~dt_over_tau ~k ~u =
  let scaled =
    match dt_over_tau with
    | `Scalar dt -> Infix.(u *$ dt)
    | `Vector dt -> Infix.(u * dt)
  in
  where (first_step k) u scaled

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
    let identity = eye (Nx.dtype w) (Array.get (shape w) 0) in
    Infix.(identity + ((w - identity) *$ dt_over_tau))

  let dyn p ~k ~x ~u =
    let u =
      match p.b with
      | None -> u
      | Some b -> Infix.(u *@ b)
    in
    let input = input_term ~dt_over_tau:(`Scalar p.dt_over_tau) ~k ~u in
    Infix.((x *@ matrix p) + input)

  let make ?key ?(alpha = 1.0) ~n ~m dtype =
    if n <= 0 then invalid_arg "Dynamics.Linear.make: n must be positive";
    if m <= 0 then invalid_arg "Dynamics.Linear.make: m must be positive";
    let w = Infix.(eye dtype n *$ alpha) in
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
    let hidden = phi Infix.((x *@ p.a1) + p.bias1) in
    let passive =
      Infix.(x + (((x *@ p.a0) - x + ((hidden *@ p.a2) + p.bias2)) * p.dt_over_tau))
    in
    let u =
      match p.b with
      | None -> u
      | Some b -> Infix.(u *@ b)
    in
    Infix.(passive + input_term ~dt_over_tau:(`Vector p.dt_over_tau) ~k ~u)

  let make ?key ?(radius = 0.1) ~dt ~tau ~n ~nh ~m ~phi dtype =
    if n <= 0 then invalid_arg "Dynamics.Linear_nonlinear.make: n must be positive";
    if nh <= 0 then invalid_arg "Dynamics.Linear_nonlinear.make: nh must be positive";
    if m <= 0 then invalid_arg "Dynamics.Linear_nonlinear.make: m must be positive";
    if not (Float.( > ) dt 0.0 && Float.( > ) tau 0.0)
    then invalid_arg "Dynamics.Linear_nonlinear.make: dt and tau must be positive";
    let params =
      { dt_over_tau = full dtype [| n |] (dt /. tau)
      ; a0 = Infix.(eye dtype n *$ 0.5)
      ; a1 =
          scaled_randn ?key ~sigma:(1.0 /. Float.sqrt (Float.of_int n)) dtype [| n; nh |]
      ; a2 =
          scaled_randn
            ?key
            ~sigma:(radius /. Float.sqrt (Float.of_int nh))
            dtype
            [| nh; n |]
      ; bias1 = Infix.((rand dtype [| nh |] *$ 2.0) -$ 1.0)
      ; bias2 = zeros dtype [| n |]
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
    let g = phi Infix.((x *@ p.g_a1) + p.g_bias1) in
    let h = Infix.((phi ((x *@ p.h_a1) + p.h_bias1) *@ p.h_a2) + p.h_bias2) in
    let passive = Infix.(x + (g * (h - x) *$ p.dt_over_tau)) in
    let u =
      match p.b with
      | None -> u
      | Some b -> Infix.(u *@ b)
    in
    Infix.(passive + input_term ~dt_over_tau:(`Scalar p.dt_over_tau) ~k ~u)

  let make ?key ?(radius = 0.1) ?m ~dt ~tau ~n ~nh ~phi dtype =
    if n <= 0 then invalid_arg "Dynamics.Gnode.make: n must be positive";
    if nh <= 0 then invalid_arg "Dynamics.Gnode.make: nh must be positive";
    if not (Float.( > ) dt 0.0 && Float.( > ) tau 0.0)
    then invalid_arg "Dynamics.Gnode.make: dt and tau must be positive";
    let m_input = m in
    let m = Option.value m ~default:n in
    let params =
      { dt_over_tau = dt /. tau
      ; g_a1 = zeros dtype [| n; n |]
      ; g_bias1 = zeros dtype [| n |]
      ; h_a1 =
          scaled_randn ?key ~sigma:(1.0 /. Float.sqrt (Float.of_int n)) dtype [| n; nh |]
      ; h_a2 =
          scaled_randn
            ?key
            ~sigma:(radius /. Float.sqrt (Float.of_int nh))
            dtype
            [| nh; n |]
      ; h_bias1 = zeros dtype [| nh |]
      ; h_bias2 = zeros dtype [| n |]
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

(* {1 Minimal gated unit} *)

module Mgu = struct
  type 'a params =
    { uf : 'a
    ; uh : 'a
    ; w : 'a
    ; bf : 'a
    ; bh : 'a
    }
  [@@deriving ptree]

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type 'a t = 'a params

        let walk = walk_params
      end)

  (* The paper's Appendix B.2 (S7)-(S9), with the [t - 1] subscript typo of
     (S9) corrected and the biases it omits made explicit. [b_f] is not
     optional: at the encoder's zero initial state a biasless gate is
     [phi 0 = 0] and the state never leaves the origin. *)
  let dyn ~phi p ~k:_ ~x ~u =
    let f = phi Infix.((x *@ p.uf) + p.bf) in
    let h = Infix.(phi ((f * x *@ p.uh) + p.bh) + (u *@ p.w)) in
    Infix.(((ones_like f - f) * x) + (f * h))

  let make ?key ?(bias_f = 0.5) ~n ~m ~phi dtype =
    if n <= 0 then invalid_arg "Dynamics.Mgu.make: n must be positive";
    if m <= 0 then invalid_arg "Dynamics.Mgu.make: m must be positive";
    let params =
      { uf = zeros dtype [| n; n |]
      ; uh = zeros dtype [| n; n |]
      ; w = scaled_randn ?key ~sigma:(1.0 /. Float.sqrt (Float.of_int m)) dtype [| m; n |]
      ; bf = full dtype [| n |] bias_f
      ; bh = zeros dtype [| n |]
      }
    in
    { n
    ; m
    ; structure = structure ()
    ; params
    ; dyn = (fun p ~k ~x ~u -> dyn ~phi p ~k ~x ~u)
    }
end
