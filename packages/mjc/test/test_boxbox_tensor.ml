(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The tensor box-box against the host port, on poses no corpus contains.

   [Collision_box] is a statement-for-statement port of MuJoCo's [mjc_BoxBox]
   and matches the C engine on the [boxes2] corpus. The tensor version is a
   different program — masks over the fifteen axes rather than an early-exit
   search, a fixed twelve-slot clip rather than a polygon — and the only honest
   way to call it the same algorithm is to compare them on random poses. *)

open Base
module Printf = Stdlib.Printf

(* A deterministic generator: the test has to be the same failure tomorrow. *)
let state = ref 0x2f6e2b1

let rand () =
  state := ((!state * 1103515245) + 12345) land 0x3fffffff;
  Float.of_int !state /. Float.of_int 0x3fffffff

let uniform lo hi = lo +. (rand () *. (hi -. lo))

let quat () =
  (* a random unit quaternion, by the rejection-free method: normalise four
     gaussians-ish numbers *)
  let g () = (rand () *. 2.) -. 1. in
  let a = g ()
  and b = g ()
  and c = g ()
  and d = g () in
  let n = Float.sqrt ((a *. a) +. (b *. b) +. (c *. c) +. (d *. d)) in
  [| a /. n; b /. n; c /. n; d /. n |]

let mat_of_quat q =
  let w = q.(0)
  and x = q.(1)
  and y = q.(2)
  and z = q.(3) in
  [| 1. -. (2. *. ((y *. y) +. (z *. z)))
   ; 2. *. ((x *. y) -. (w *. z))
   ; 2. *. ((x *. z) +. (w *. y))
   ; 2. *. ((x *. y) +. (w *. z))
   ; 1. -. (2. *. ((x *. x) +. (z *. z)))
   ; 2. *. ((y *. z) -. (w *. x))
   ; 2. *. ((x *. z) -. (w *. y))
   ; 2. *. ((y *. z) +. (w *. x))
   ; 1. -. (2. *. ((x *. x) +. (y *. y)))
  |]

let n = 400
let failures = ref 0

let check ~msg ok =
  if not ok
  then (
    Int.incr failures;
    Printf.printf "FAIL %s\n%!" msg)

let () =
  let pos1 = Array.init n ~f:(fun _ -> Array.init 3 ~f:(fun _ -> uniform (-0.6) 0.6))
  and pos2 = Array.init n ~f:(fun _ -> Array.init 3 ~f:(fun _ -> uniform (-0.6) 0.6))
  and mat1 = Array.init n ~f:(fun _ -> mat_of_quat (quat ()))
  and mat2 = Array.init n ~f:(fun _ -> mat_of_quat (quat ()))
  and size1 = Array.init n ~f:(fun _ -> Array.init 3 ~f:(fun _ -> uniform 0.05 0.4))
  and size2 = Array.init n ~f:(fun _ -> Array.init 3 ~f:(fun _ -> uniform 0.05 0.4))
  and margin = Array.init n ~f:(fun _ -> 0.) in
  let flat f =
    Array.concat (Array.to_list (Array.map ~f (Stdlib.Array.init n (fun i -> i))))
  in
  let pos1_t = Nx.create Nx.float64 [| n; 3 |] (flat (fun i -> pos1.(i)))
  and pos2_t = Nx.create Nx.float64 [| n; 3 |] (flat (fun i -> pos2.(i)))
  and mat1_t = Nx.create Nx.float64 [| n; 9 |] (flat (fun i -> mat1.(i)))
  and mat2_t = Nx.create Nx.float64 [| n; 9 |] (flat (fun i -> mat2.(i)))
  and size1_t = Nx.create Nx.float64 [| n; 3 |] (flat (fun i -> size1.(i)))
  and size2_t = Nx.create Nx.float64 [| n; 3 |] (flat (fun i -> size2.(i)))
  and margin_t = Nx.create Nx.float64 [| n |] margin in
  let batch =
    Mjc.Collision_box_tensor.
      { pos1 = pos1_t
      ; mat1 = mat1_t
      ; size1 = size1_t
      ; pos2 = pos2_t
      ; mat2 = mat2_t
      ; size2 = size2_t
      ; margin = margin_t
      }
  in
  let s = Mjc.Collision_box_tensor.sat batch in
  let code = Nx.to_array s.Mjc.Collision_box_tensor.code
  and sep = Nx.to_array s.sep
  and separated = Nx.to_array s.separated in
  let face_max, edge_max, tol = Mjc.Collision_box_tensor.debug batch in
  let face_max = Nx.to_array face_max
  and edge_max = Nx.to_array edge_max
  and tol = Nx.to_array tol in
  let contacts = ref 0 in
  for i = 0 to n - 1 do
    let host =
      Mjc.Collision_box.sat
        ~pos1:pos1.(i)
        ~mat1:mat1.(i)
        ~size1:size1.(i)
        ~pos2:pos2.(i)
        ~mat2:mat2.(i)
        ~size2:size2.(i)
        ~margin:margin.(i)
    in
    let label tag = Printf.sprintf "pair %d %s" i tag in
    check
      ~msg:
        (label
           (Printf.sprintf
              "separated %b vs %g [host code %d sep %.17g | tensor code %g sep %.17g | \
               max face %.17g max edge %.17g tol %.3g]"
              host.separated
              separated.(i)
              host.code
              host.sep
              code.(i)
              sep.(i)
              face_max.(i)
              edge_max.(i)
              tol.(i)))
      (Bool.equal host.separated (Float.to_int separated.(i) = 1));
    if not host.separated
    then (
      Int.incr contacts;
      check
        ~msg:(label (Printf.sprintf "code %d vs %g" host.code code.(i)))
        (Int.equal host.code (Float.to_int code.(i)));
      check
        ~msg:(label (Printf.sprintf "sep %g vs %g" host.sep sep.(i)))
        (Float.( <= ) (Float.abs (host.sep -. sep.(i))) 1e-12))
  done;
  (* the manifold: edge-code pairs only, because the face generator is not
     written yet *)
  let m = Mjc.Collision_box_tensor.manifold batch in
  let md = Nx.to_array m.Mjc.Collision_box_tensor.dist
  and mp = Nx.to_array m.pos
  and mn = Nx.to_array m.normal
  and ml = Nx.to_array m.live in
  let slots = Mjc.Collision_box_tensor.max_contacts in
  let edge_pairs = ref 0 in
  for i = 0 to n - 1 do
    let code, separated =
      let h =
        Mjc.Collision_box.sat
          ~pos1:pos1.(i)
          ~mat1:mat1.(i)
          ~size1:size1.(i)
          ~pos2:pos2.(i)
          ~mat2:mat2.(i)
          ~size2:size2.(i)
          ~margin:margin.(i)
      in
      h.code, h.separated
    in
    if (not separated) && Int.( >= ) code 6
    then (
      Int.incr edge_pairs;
      let label tag = Printf.sprintf "pair %d edge %s" i tag in
      let host =
        Mjc.Collision_box.manifold
          ~pos1:pos1.(i)
          ~mat1:mat1.(i)
          ~size1:size1.(i)
          ~pos2:pos2.(i)
          ~mat2:mat2.(i)
          ~size2:size2.(i)
          ~margin:margin.(i)
      in
      check ~msg:(label "contact count") (Int.equal (Array.length host) 1);
      check ~msg:(label "live") (Float.equal ml.(i * slots) 1.);
      match Array.length host with
      | 1 ->
        let h = host.(0) in
        check
          ~msg:(label (Printf.sprintf "dist %g vs %g" h.dist md.(i * slots)))
          (Float.( <= ) (Float.abs (h.dist -. md.(i * slots))) 1e-12);
        for k = 0 to 2 do
          check
            ~msg:(label "pos")
            (Float.( <= ) (Float.abs (h.pos.(k) -. mp.((i * slots * 3) + k))) 1e-12);
          check
            ~msg:(label "normal")
            (Float.( <= ) (Float.abs (h.normal.(k) -. mn.((i * slots * 3) + k))) 1e-12)
        done
      | _ -> ())
  done;
  Printf.printf "edge-code pairs compared: %d\n%!" !edge_pairs;
  if !failures > 0
  then (
    Printf.printf "%d checks failed\n%!" !failures;
    Stdlib.exit 1)
  else (
    Printf.printf
      "box-box sat: %d pairs, %d in contact, code and separation match the host port\n%!"
      n
      !contacts;
    Stdlib.exit 0)
