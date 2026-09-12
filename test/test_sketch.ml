(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* End-to-end sketches.

   The checks are independent of the implementation wherever they can be: C
   against the exact gradient contracted with Θ, the curvature against an
   explicit Σ YᵀHY built from a vmap∘jvp reference and rune's Hessians, and
   against an exact JᵀHJ on a model small enough to write down. The strongest
   check needs no reference at all: for a loss that is exactly quadratic in the
   parameters, the sketch's own second-order model reproduces the loss. *)

open Windtrap

let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs
let mat r c xs = Nx.create f64 [| r; c |] xs
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

let check_arr ?(eps = 1e-8) ~msg expected actual =
  let actual = to_arr actual in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e -> equal ~msg:(Printf.sprintf "%s[%d]" msg i) (float eps) e actual.(i))
    expected

(* A deterministic lane batch [k; shape t]. *)
let lanes ~k t =
  let s = Nx.shape t in
  let n = Nx.numel t in
  let data =
    Array.init (k * n) (fun i ->
      let l = i / n
      and j = i mod n in
      float_of_int ((((j * 7) + (3 * l)) mod 11) - 5) /. 4.0)
  in
  Nx.create f64 (Array.append [| k |] s) data

type params =
  { w : Nx.float64_t
  ; b : Nx.float64_t
  }

module Params = struct
  type t = params

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; b } = { w = f w; b = f b }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w = f p.w q.w; b = f p.b q.b }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; b } =
    f w;
    f b
end

let x_data = mat 4 3 [| 0.2; -0.4; 0.6; 0.1; -0.7; 0.5; 0.3; 0.9; -0.2; 0.8; 0.4; -0.1 |]
let target = mat 4 2 [| 0.5; -0.2; -0.3; 0.8; 0.6; 0.1; -0.9; 0.4 |]

let params () =
  { w = mat 3 2 [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |]; b = vec [| 0.3; -0.7 |] }

(* A linear readout: the squared error is then exactly quadratic in the
   parameters, which makes the second-order model check exact. *)
let predict p = Nx.add (Nx.matmul x_data p.w) p.b
let data_loss p = Sofo.mse (predict p) target

(* Two little losses, both observed: the readout's error and a ridge on w. *)
let loss_all p = Nx.add (Sofo.mse (predict p) target) (Sofo.mse p.w (Nx.zeros_like p.w))

(* Deterministic directions, so every reference uses the same Θ. *)
let fixed_sampler k p = { w = lanes ~k p.w; b = lanes ~k p.b }

(* Θᵀ[·] for a parameter structure: each leaf's lane axis contracted against
   the corresponding leaf of the cotangent, summed over leaves. Reimplemented
   here rather than borrowed from the library. *)
(* Θᵀ[·]: each leaf's lane axis contracted against its cotangent leaf — lane i
   of the result is the inner product of lane i's directions with the
   cotangent — summed over leaves. Reimplemented here rather than borrowed from
   the library. *)
let theta_t_cotangent ~k thetas g =
  let leaf theta g =
    let n = Nx.numel g in
    Nx.matmul
      (Nx.reshape [| k; n |] (Nx.contiguous theta))
      (Nx.reshape [| n; 1 |] (Nx.contiguous g))
  in
  Nx.reshape [| k |] (Nx.add (leaf thetas.w g.w) (leaf thetas.b g.b))

let test_c_and_loss () =
  let p = params ()
  and k = 3 in
  let thetas = fixed_sampler k p in
  let sk = Sofo.sketch (module Params) ~k ~sketch_sampler:fixed_sampler loss_all p in
  let y, grads = Rune.value_and_grad (module Params) loss_all p in
  equal ~msg:"loss" (float 1e-10) (Nx.item [] y) (Nx.item [] sk.loss);
  check_arr ~msg:"C = Θᵀ∇c" (to_arr (theta_t_cotangent ~k thetas grads)) sk.c

let test_quadratic_expansion () =
  (* c is exactly quadratic in the parameters, so the sketch's model is exact
     for any step: c(θ + εΘz) = c + ε⟨C,z⟩ + ε²/2 zᵀG̃z. *)
  let p = params ()
  and k = 4 in
  let sk = Sofo.sketch (module Params) ~k ~sketch_sampler:fixed_sampler loss_all p in
  let z = vec [| 0.4; -1.1; 0.7; 0.2 |] in
  let eps = 1e-3 in
  let p' =
    Params.map2
      (fun pl dp -> Nx.add pl (Nx.mul_s dp (Nx_core.Dtype.of_float (Nx.dtype dp) eps)))
      p
      (sk.apply z)
  in
  let c0 = Nx.item [] sk.loss in
  let c1 = Nx.item [] (loss_all p') in
  let linear = Nx.item [] (Nx.sum (Nx.mul sk.c z)) in
  let quad = Nx.item [] (Nx.sum (Nx.mul z (Nx.matmul sk.ggn z))) in
  equal
    ~msg:"second-order model"
    (float 1e-9)
    (c0 +. (eps *. linear) +. (eps *. eps /. 2.0 *. quad))
    c1

(* The reference Gram: per observation, the tangent batch from the vmap∘jvp
   reference and the Hessian from rune, contracted into a k×k block. *)
let test_ggn_matches_explicit_gram () =
  let p = params ()
  and k = 2 in
  let thetas = fixed_sampler k p in
  let sk = Sofo.sketch (module Params) ~k ~sketch_sampler:fixed_sampler loss_all p in
  let tangent_batch f =
    Rune.vmap (module Params) (fun th -> snd (Rune.jvp (module Params) f p th)) thetas
  in
  let block y_tangent y h =
    let n = Nx.numel y in
    let y_r = Nx.reshape [| k; n |] (Nx.contiguous y_tangent) in
    let h_r = Nx.reshape [| n; n |] (Nx.contiguous h) in
    Nx.matmul (Nx.matmul y_r h_r) (Nx.transpose y_r)
  in
  let y1 = predict p
  and y2 = p.w in
  let h1 =
    Rune.hessian' (fun y -> Nx.mean (Nx.mul (Nx.sub y target) (Nx.sub y target))) y1
  in
  let h2 = Rune.hessian' (fun y -> Nx.mean (Nx.mul y y)) y2 in
  let reference =
    Nx.add
      (block (tangent_batch predict) y1 h1)
      (block (tangent_batch (fun p -> p.w)) y2 h2)
  in
  check_arr ~msg:"collector = Σ YᵀHY" (to_arr reference) sk.ggn

(* The exact GGN of a single-observation model, from its Jacobian. *)
let test_ggn_matches_exact_jacobian () =
  let p = params ()
  and k = 3 in
  let thetas = fixed_sampler k p in
  let sk = Sofo.sketch (module Params) ~k ~sketch_sampler:fixed_sampler data_loss p in
  let n = Nx.numel (predict p) in
  let flat w b = Nx.ravel (Nx.add (Nx.matmul x_data w) b) in
  let jw =
    Rune.jacrev' (fun w -> flat (Nx.reshape (Nx.shape p.w) w) p.b) (Nx.ravel p.w)
  in
  let jb = Rune.jacrev' (fun b -> flat p.w b) p.b in
  let j = Nx.concatenate ~axis:1 [ jw; jb ] in
  (* the mean-squared loss has H = (2/n)·I *)
  let h = Nx.mul_s (Nx.eye f64 n) (2.0 /. float_of_int n) in
  let flatten_theta t = Nx.reshape [| k; Nx.numel t / k |] (Nx.contiguous t) in
  let theta = Nx.concatenate ~axis:1 [ flatten_theta thetas.w; flatten_theta thetas.b ] in
  let reference =
    Nx.matmul
      (Nx.matmul theta (Nx.matmul (Nx.matmul (Nx.transpose j) h) j))
      (Nx.transpose theta)
  in
  check_arr ~msg:"G̃ = ΘᵀJᵀHJΘ" (to_arr reference) sk.ggn

let test_apply_is_the_sampled_directions () =
  (* apply is Θ, so it must return the sampled lanes themselves: the basis
     vectors pick them out one at a time. *)
  let p = params ()
  and k = 3 in
  let thetas = fixed_sampler k p in
  let sk = Sofo.sketch (module Params) ~k ~sketch_sampler:fixed_sampler loss_all p in
  List.iter
    (fun i ->
       let z = vec (Array.init k (fun j -> if i = j then 1.0 else 0.0)) in
       let step = sk.apply z in
       check_arr
         ~msg:"apply e_i is lane i (w)"
         (to_arr (Nx.slice [ Nx.I i ] thetas.w))
         step.w;
       check_arr
         ~msg:"apply e_i is lane i (b)"
         (to_arr (Nx.slice [ Nx.I i ] thetas.b))
         step.b)
    [ 0; 1; 2 ]

let test_zero_curvature_is_first_order () =
  (* A sketch whose observations carry no curvature is the first-order
     subspace sketch: C and the loss are unchanged, the GGN is zero. *)
  let loss p =
    let y = predict p in
    let l = Nx.mean (Nx.mul (Nx.sub y target) (Nx.sub y target)) in
    Sofo.observe ~y ~curv:(Sofo.Curv.scale 0.0) l;
    l
  in
  let p = params ()
  and k = 2 in
  let thetas = fixed_sampler k p in
  let sk = Sofo.sketch (module Params) ~k ~sketch_sampler:fixed_sampler loss p in
  check_arr ~msg:"no curvature" [| 0.0; 0.0; 0.0; 0.0 |] sk.ggn;
  let _, grads = Rune.value_and_grad (module Params) loss p in
  check_arr
    ~msg:"C still the gradient sketch"
    (to_arr (theta_t_cotangent ~k thetas grads))
    sk.c

let test_unobserved_term_is_flagged () =
  (* The ridge is accumulated into the loss but never observed: the sketch's
     curvature is short one block, and the check says so. *)
  let loss p =
    Nx.add (Sofo.mse (predict p) target) (Nx.mul_s (Nx.sum (Nx.mul p.w p.w)) 0.01)
  in
  let p = params ()
  and k = 2 in
  let sk = Sofo.sketch (module Params) ~k ~sketch_sampler:fixed_sampler loss p in
  (match Sofo.check sk with
   | Ok () -> fail "an unobserved loss term was not flagged"
   | Error _ -> ());
  raises_match Exn.invalid_arg (fun () ->
    ignore
      (Sofo.sketch (module Params) ~k ~strict:true ~sketch_sampler:fixed_sampler loss p))

let test_fully_observed_passes_the_check () =
  let p = params ()
  and k = 2 in
  let sk = Sofo.sketch (module Params) ~k ~sketch_sampler:fixed_sampler loss_all p in
  (match Sofo.check sk with
   | Ok () -> ()
   | Error msg -> fail ("unexpected mismatch: " ^ msg));
  equal ~msg:"both losses contributed a block" int 2 sk.diagnostics.blocks;
  equal ~msg:"nothing skipped" int 0 sk.diagnostics.skipped

let test_constant_prediction_is_skipped () =
  (* A little loss whose prediction is a constant of the differentiation has a
     zero block by construction: it is skipped, not accumulated as zeros, and
     its value still counts towards the loss. *)
  let loss p =
    let c = Nx.ones_like p.w in
    let l = Nx.mean (Nx.mul c c) in
    Sofo.observe ~y:c ~curv:(Sofo.Curv.scale 2.0) l;
    Nx.add (Sofo.mse (predict p) target) l
  in
  let p = params ()
  and k = 2 in
  let sk = Sofo.sketch (module Params) ~k ~sketch_sampler:fixed_sampler loss p in
  equal ~msg:"one block" int 1 sk.diagnostics.blocks;
  equal ~msg:"one skip" int 1 sk.diagnostics.skipped;
  (match Sofo.check sk with
   | Ok () -> ()
   | Error msg -> fail ("a constant observation should not break the check: " ^ msg));
  (* and the skipped term contributes nothing to the curvature *)
  let sk' = Sofo.sketch (module Params) ~k ~sketch_sampler:fixed_sampler data_loss p in
  check_arr ~msg:"same GGN as without the constant term" (to_arr sk'.ggn) sk.ggn

let test_reproducible_with_an_rng_key () =
  let p = params ()
  and k = 3 in
  let sketch_under key =
    Nx.Rng.with_key (Nx.Rng.key key) (fun () -> Sofo.sketch (module Params) ~k loss_all p)
  in
  let a = sketch_under 42
  and b = sketch_under 42
  and c = sketch_under 7 in
  check_arr ~msg:"same C" (to_arr a.c) b.c;
  check_arr ~msg:"same GGN" (to_arr a.ggn) b.ggn;
  if to_arr c.ggn = to_arr a.ggn then fail "a different key gave the same sketch"

let test_rejects_a_non_scalar_loss () =
  raises_match Exn.invalid_arg (fun () ->
    ignore
      (Sofo.sketch
         (module Params)
         ~k:2
         ~sketch_sampler:fixed_sampler
         (fun p -> predict p)
         (params ())))

let test_rejects_zero_lanes () =
  raises_match Exn.invalid_arg (fun () ->
    ignore
      (Sofo.sketch
         (module Params)
         ~k:0
         ~sketch_sampler:fixed_sampler
         data_loss
         (params ())))

let tests =
  [ test "the loss and C are exact" test_c_and_loss
  ; test "the second-order model is exact for a quadratic loss" test_quadratic_expansion
  ; test "the collector equals the explicit Σ YᵀHY" test_ggn_matches_explicit_gram
  ; test "the GGN equals an exact ΘᵀJᵀHJΘ" test_ggn_matches_exact_jacobian
  ; test "apply is the sampled directions" test_apply_is_the_sampled_directions
  ; test "zero curvature gives the first-order sketch" test_zero_curvature_is_first_order
  ; test "an unobserved term is flagged" test_unobserved_term_is_flagged
  ; test "a fully observed loss passes the check" test_fully_observed_passes_the_check
  ; test "a constant prediction is skipped" test_constant_prediction_is_skipped
  ; test "an RNG key makes the sketch reproducible" test_reproducible_with_an_rng_key
  ; test "a non-scalar loss is rejected" test_rejects_a_non_scalar_loss
  ; test "k = 0 is rejected" test_rejects_zero_lanes
  ]

let () = run "sofo sketch" tests
