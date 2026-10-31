(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Taylor optimizer.

   The estimator is checked on an invariant quadratic: the exchange identity
   [S_g = H S_w H] between the second-order moments, the damped inverse of
   [Solve] on a known operator, and a full Newton step that must land on the
   minimizer when the gradient is confined to the non-invariant subspace. A
   small two-layer fixture then trains with [Rune.value_and_grad]. *)

open Base
open Nx
open Windtrap
open Symo

(* One [2; 3] weight whose columns are permuted: the commutant is
   [M_2(R) ⊗ span {I_3, J_3}], so a Hessian [A ⊗ (3 I + 0.7 J)] exercises the
   row-space structure that the blocks carry. *)
module Quad = struct
  type ('a, _) t = { w : 'a } [@@deriving ptree]

  let dims : (int list, _) t = { w = [ 2; 3 ] }
  let symmetries : (Symmetry.spec list, _) t = { w = [ Id; Perm 0 ] }
end

module Q = Make (Quad)

let relu x = maximum x (zeros_like x)

(* A two-layer MLP with a hidden-permutation symmetry: the same group element
   permutes the rows of [w1] and the columns of [w2]. *)
module Mlp = struct
  type ('a, _) t =
    { w1 : 'a
    ; w2 : 'a
    }
  [@@deriving ptree]

  let dims : (int list, _) t = { w1 = [ 3; 2 ]; w2 = [ 1; 3 ] }

  let symmetries : (Symmetry.spec list, _) t =
    { w1 = [ Perm 0; Id ]; w2 = [ Id; Perm 0 ] }
end

module M = Make (Mlp)

let mat32 shape values = create float32 shape values

let check ~msg ~tol expected got =
  equal ~msg (array (float tol)) (to_array expected) (to_array got)

(* The large Hessian [A ⊗ C] of the quadratic, as a 6×6 matrix: [a] is 2×2
   and [c] is 3×3. *)
let kron23 a c =
  let a = to_array a
  and c = to_array c in
  create
    float32
    [| 6; 6 |]
    (Array.init 36 ~f:(fun k ->
       let row = k / 6
       and col = Int.rem k 6 in
       let r1 = row / 3
       and r2 = Int.rem row 3 in
       let c1 = col / 3
       and c2 = Int.rem col 3 in
       a.((r1 * 2) + c1) *. c.((r2 * 3) + c2)))

(* ------------------------------------------------------------------------
   Tests
   ------------------------------------------------------------------------ *)

let test_schedules () =
  let s v = scalar float32 v in
  equal ~msg:"momentum" (float 1e-6) 1.5 (item [] (Optim.ema ~beta:0.5 (s 2.) (s 1.)));
  equal ~msg:"debias" (float 1e-6) 6.0 (item [] (Optim.debias (s 0.5) (s 3.)));
  equal ~msg:"counter decay" (float 1e-6) 0.05 (item [] (Optim.bump_tensor (s 0.1) 0.5));
  equal ~msg:"counter floor" (float 1e-6) 1e-4 (item [] (Optim.bump_tensor (s 1e-6) 0.5))

let test_solve_identity () =
  let sigma_w = mat32 [| 2; 2 |] [| 1.5; 0.2; 0.2; 0.8 |] in
  let a = mat32 [| 2; 2 |] [| 2.0; 0.3; 0.3; 1.2 |] in
  let sigma_g = Infix.(a *@ sigma_w *@ transpose a) in
  let h_inv = Solve.hessian_inverse ~damping:0.0 sigma_w sigma_g in
  let h = Solve.hessian ~damping:0.0 sigma_w sigma_g in
  check ~msg:"H_inv is the inverse" ~tol:1e-4 (eye float32 2) Infix.(h_inv *@ a);
  check ~msg:"H H_inv = I" ~tol:1e-4 (eye float32 2) Infix.(h *@ h_inv);
  (* Damping regularises a singular sigma_w without producing NaNs. *)
  let singular = mat32 [| 2; 2 |] [| 1.0; 1.0; 1.0; 1.0 |] in
  let sigma_g0 = zeros float32 [| 2; 2 |] in
  let regularized = Solve.hessian_inverse ~damping:1e-3 singular sigma_g0 in
  is_true
    ~msg:"damped singular solve is finite"
    (Array.for_all (to_array regularized) ~f:Float.is_finite)

let test_exchange_identity () =
  let a = mat32 [| 2; 2 |] [| 2.0; 0.5; 0.5; 1.0 |] in
  let c = Infix.((eye float32 3 *$ 3.0) + (ones float32 [| 3; 3 |] *$ 0.7)) in
  let h_large = kron23 a c in
  let delta = Nx.Rng.normal ~key:(Nx.Rng.key 1) float32 [| 2; 3 |] in
  (* A non-invariant offset: the orbit average removes the row means. *)
  let w = Infix.(delta - mean ~axes:[ 1 ] ~keepdims:true delta) in
  let g = Infix.(a *$ 3.0 *@ w) in
  let compiled = (Q.Second_order.large ()).w.(0) in
  let sw =
    compiled.dense_block
      ~factors:(Q.Second_order.factors_of_pair { Quad.w } { Quad.w }).w.(0)
  in
  let sg =
    compiled.dense_block
      ~factors:(Q.Second_order.factors_of_pair { Quad.w = g } { Quad.w = g }).w.(0)
  in
  let expected = Infix.(h_large *@ sw *@ h_large) in
  check ~msg:"S_g = H S_w H" ~tol:1e-4 expected sg

let test_newton_step () =
  let a = mat32 [| 2; 2 |] [| 2.0; 0.5; 0.5; 1.0 |] in
  let theta_star = create float32 [| 2; 3 |] [| 0.4; 0.4; 0.4; -0.2; -0.2; -0.2 |] in
  let delta = Nx.Rng.normal ~key:(Nx.Rng.key 2) float32 [| 2; 3 |] in
  let w = Infix.(delta - mean ~axes:[ 1 ] ~keepdims:true delta) in
  let theta = Infix.(theta_star + w) in
  let g = Infix.(a *$ 3.0 *@ w) in
  let config : Q.config =
    { learning_rate = Some 1.0; beta_1 = 0.0; beta_2 = 0.0; damping = 1e-6 }
  in
  let state = Q.init ~config { Quad.w = theta } in
  let state = Q.step ~config ~state ~grads:{ Quad.w = g } in
  let loss x = Infix.(sum (x * (a *$ 3.0 *@ x)) *$ 0.5) in
  let before = item [] (loss theta) in
  let after = item [] (loss state.theta.w) in
  is_true ~msg:"loss decreased" Float.(after < before);
  check ~msg:"Newton step lands on the minimizer" ~tol:2e-3 theta_star state.theta.w;
  (* [learning_rate = None] measures without shifting the parameters. *)
  let measure : Q.config =
    { learning_rate = None; beta_1 = 0.0; beta_2 = 0.0; damping = 1e-6 }
  in
  let state' =
    Q.step
      ~config:measure
      ~state:(Q.init ~config:measure { Quad.w = theta })
      ~grads:{ Quad.w = g }
  in
  check ~msg:"measure-only keeps theta" ~tol:1e-6 theta state'.theta.w

(* ------------------------------------------------------------------------
   End-to-end fixture
   ------------------------------------------------------------------------ *)

let xs = Nx.Rng.normal ~key:(Nx.Rng.key 7) float32 [| 8; 2 |]

let ys =
  let w = mat32 [| 2; 1 |] [| 1.0; -0.5 |] in
  Infix.((xs *@ w) + (Nx.Rng.normal ~key:(Nx.Rng.key 8) float32 [| 8; 1 |] *$ 0.05))

let mlp_loss (p : (Nx.host Nx.float32_t, Nx.host) Mlp.t) =
  let open Infix in
  let h = relu (xs *@ transpose p.w1) in
  let out = h *@ transpose p.w2 in
  let d = out - ys in
  mean (d * d)

let test_end_to_end () =
  let params =
    { Mlp.w1 = Nx.Rng.normal ~key:(Nx.Rng.key 9) float32 [| 3; 2 |]
    ; w2 = Nx.Rng.normal ~key:(Nx.Rng.key 10) float32 [| 1; 3 |]
    }
  in
  let config : M.config =
    { learning_rate = Some 0.5; beta_1 = 0.9; beta_2 = 0.99; damping = 1e-4 }
  in
  let state = M.init ~config params in
  let loss params = item [] (mlp_loss params) in
  let before = loss params in
  let state =
    List.fold (List.range 0 20) ~init:state ~f:(fun state _ ->
      let _, grads = Rune.value_and_grad M.ptree mlp_loss state.M.State.theta in
      M.step ~config ~state ~grads)
  in
  is_true ~msg:"loss decreased" Float.(loss state.M.State.theta < before)

(* Transport across widths: the assembled blocks of an operator estimated at
   one hidden size are the blocks of its re-encoding at another. *)
let test_transport_blocks () =
  let theta =
    { Mlp.w1 = Nx.Rng.normal ~key:(Nx.Rng.key 41) float32 [| 3; 2 |]
    ; w2 = Nx.Rng.normal ~key:(Nx.Rng.key 42) float32 [| 1; 3 |]
    }
  in
  let factors = M.Second_order.factors_of_pair theta theta in
  let blocks = M.Second_order.blocks_of_factors factors in
  let wide = { Mlp.w1 = [ 5; 2 ]; w2 = [ 1; 5 ] } in
  let transported = M.Second_order.transport ~dims:wide factors in
  let blocks_wide = M.Second_order.blocks_of_factors transported in
  List.iter2_exn blocks blocks_wide ~f:(fun a b ->
    check ~msg:"transported blocks" ~tol:1e-4 a b)

(* A set of its own over the host's device: its arrays are not host values,
   so a step there places the compiled tables beside them and brings the
   solve's result back. The kernels are the host's, so it computes what a
   step on the host computes. *)
module Other = (val Nx.devices (Nx.rigs Nx.Host.v))

let test_other_set () =
  let config : M.config =
    { learning_rate = Some 0.5; beta_1 = 0.9; beta_2 = 0.99; damping = 1e-4 }
  in
  let tree k =
    { Mlp.w1 = Nx.Rng.normal ~key:(Nx.Rng.key k) float32 [| 3; 2 |]
    ; w2 = Nx.Rng.normal ~key:(Nx.Rng.key (k + 1)) float32 [| 1; 3 |]
    }
  in
  let theta = tree 61
  and grads = tree 63 in
  let other x = Nx.Ptree.place (module Mlp) Other.on x in
  let two_steps ~theta ~grads =
    let state = M.init ~config theta in
    let state = M.step ~config ~state ~grads in
    M.step ~config ~state ~grads
  in
  let on_host = two_steps ~theta ~grads in
  let on_other = two_steps ~theta:(other theta) ~grads:(other grads) in
  check ~msg:"w1" ~tol:1e-6 on_host.theta.w1 on_other.theta.w1;
  check ~msg:"w2" ~tol:1e-6 on_host.theta.w2 on_other.theta.w2;
  check ~msg:"sigma_g_avg" ~tol:1e-6 on_host.sigma_g_avg on_other.sigma_g_avg;
  check ~msg:"beta_2_t" ~tol:1e-6 on_host.beta_2_t on_other.beta_2_t

let tests =
  [ test "EMA and debias schedules" test_schedules
  ; test "Solve inverts a known operator" test_solve_identity
  ; test "exchange identity S_g = H S_w H" test_exchange_identity
  ; test "Newton step on an invariant quadratic" test_newton_step
  ; test "transport preserves the blocks" test_transport_blocks
  ; test "end-to-end MLP training" test_end_to_end
  ; test "a step on another set equals the host's" test_other_set
  ]

let () = Stdlib.exit (run "symo optim" tests)
