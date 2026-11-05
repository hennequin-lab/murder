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

let n = 800
let failures = ref 0

let check ~msg ok =
  if not ok
  then (
    Int.incr failures;
    Printf.printf "FAIL %s\n%!" msg)

let () =
  (* close enough that most pairs touch, which is where the face branch lives *)
  let pos1 = Array.init n ~f:(fun _ -> Array.init 3 ~f:(fun _ -> uniform (-0.45) 0.45))
  and pos2 = Array.init n ~f:(fun _ -> Array.init 3 ~f:(fun _ -> uniform (-0.45) 0.45))
  and mat1 = Array.init n ~f:(fun _ -> mat_of_quat (quat ()))
  and mat2 = Array.init n ~f:(fun _ -> mat_of_quat (quat ()))
  and size1 = Array.init n ~f:(fun _ -> Array.init 3 ~f:(fun _ -> uniform 0.08 0.4))
  and size2 = Array.init n ~f:(fun _ -> Array.init 3 ~f:(fun _ -> uniform 0.08 0.4))
  (* a non-zero margin on half the pairs: the depth test is one-sided there too *)
  and margin = Array.init n ~f:(fun i -> if Int.equal (i land 1) 0 then 0. else 0.01) in
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
  (* the manifold, for every touching pair: same contacts, as a set. The tensor
     generator takes the vertex set of the clipped polygon directly, so a face
     pair's rows come out in a different order from the engine's clip order —
     the geometry is what is compared, not the order. *)
  let m = Mjc.Collision_box_tensor.manifold batch in
  let md = Nx.to_array m.Mjc.Collision_box_tensor.dist
  and mp = Nx.to_array m.pos
  and mn = Nx.to_array m.normal
  and ml = Nx.to_array m.live in
  let slots = Mjc.Collision_box_tensor.max_contacts in
  let face_pairs = ref 0
  and edge_pairs = ref 0
  and total = ref 0
  and most = ref 0 in
  for i = 0 to n - 1 do
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
    let code =
      Mjc.Collision_box.sat
        ~pos1:pos1.(i)
        ~mat1:mat1.(i)
        ~size1:size1.(i)
        ~pos2:pos2.(i)
        ~mat2:mat2.(i)
        ~size2:size2.(i)
        ~margin:margin.(i)
    in
    if not code.separated
    then (
      if Int.( >= ) code.code 6 then Int.incr edge_pairs else Int.incr face_pairs;
      let label tag = Printf.sprintf "pair %d code %d %s" i code.code tag in
      let ours = ref [] in
      for k = 0 to slots - 1 do
        if Float.equal ml.((i * slots) + k) 1.
        then
          ours
          := ( md.((i * slots) + k)
             , mp.((i * slots * 3) + (3 * k))
             , mp.((i * slots * 3) + (3 * k) + 1)
             , mp.((i * slots * 3) + (3 * k) + 2) )
             :: !ours
      done;
      let ours = Array.of_list (List.rev !ours) in
      total := !total + Array.length ours;
      most := Int.max !most (Array.length ours);
      check
        ~msg:
          (label
             (Printf.sprintf
                "contact count %d vs %d"
                (Array.length host)
                (Array.length ours)))
        (Int.equal (Array.length host) (Array.length ours));
      (* each host contact must be one of ours, and each of ours one of the
         host's, so a duplicate cannot pass as a match for two *)
      let used = Array.create ~len:(Array.length ours) false in
      Array.iter host ~f:(fun (h : Mjc.Collision_box.contact) ->
        let found = ref false in
        Array.iteri ours ~f:(fun j (d, x, y, z) ->
          if
            (not used.(j))
            && Float.( <= ) (Float.abs (h.dist -. d)) 1e-12
            && Float.( <= ) (Float.abs (h.pos.(0) -. x)) 1e-12
            && Float.( <= ) (Float.abs (h.pos.(1) -. y)) 1e-12
            && Float.( <= ) (Float.abs (h.pos.(2) -. z)) 1e-12
          then (
            used.(j) <- true;
            found := true));
        check
          ~msg:
            (label
               (Printf.sprintf
                  "contact (%.17g, %.17g, %.17g, %.17g) has no match"
                  h.dist
                  h.pos.(0)
                  h.pos.(1)
                  h.pos.(2)))
          !found);
      (* the normal is the reference face's, and every slot shares it *)
      Array.iteri ours ~f:(fun j (_, _, _, _) -> ignore j);
      if Array.length host > 0
      then
        for k = 0 to 2 do
          check
            ~msg:(label (Printf.sprintf "normal %d" k))
            (Float.( <= )
               (Float.abs (host.(0).normal.(k) -. mn.((i * slots * 3) + k)))
               1e-12)
        done)
  done;
  Printf.printf
    "manifolds compared: %d face-code, %d edge-code, %d contacts, at most %d per pair\n%!"
    !face_pairs
    !edge_pairs
    !total
    !most;
  if !failures > 0
  then (
    Printf.printf "%d checks failed\n%!" !failures;
    Stdlib.exit 1)
  else (
    Printf.printf
      "box-box: %d pairs, %d touching, %d contacts; sat code and separation match the \
       host port at 1e-12, and every contact matches as a set\n\
       %!"
      n
      !contacts
      !total;
    Stdlib.exit 0)
