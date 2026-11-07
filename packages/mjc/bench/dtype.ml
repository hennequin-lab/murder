(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What fp32 would buy, measured before any of it is built.

   The solver is 98.5% of a device step, and its round is one batched Cholesky of
   the row matrix plus two triangular solves. Making the engine fp32 is a
   type-level refactor the size of the device refactor, so it gets the same
   treatment: measure the workload it would speed up, in both dtypes, on both
   devices, and let the ratio decide.

   Both prices are reported, because both are on the table: the round's cost in
   milliseconds, and how far the f32 answer lands from the f64 one.

   dune exec packages/mjc/bench/dtype.exe *)

open Base
module Printf = Stdlib.Printf

let now () = Unix.gettimeofday ()

let round : ('v, 's, 'd) Nx.t * ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t =
  fun (a, b) ->
  let l = Nx.cholesky a in
  Nx.solve_triangular
    ~triangle:Lower
    ~transpose:Transpose
    l
    (Nx.solve_triangular ~triangle:Lower l b)

(* One timing routine per dtype: the same source compiled at f64 and at f32,
   once at the host's brand and once at the brand a device set mints. The dtype
   is a static part of the type, so this cannot be one function. *)
let time_f64 place (a_host, b_host) reps =
  let a = Nx.place place a_host
  and b = Nx.place place b_host in
  let f = Rune.jit Nx.Ptree.(pair tensor tensor @-> returns tensor) round in
  let out = ref (f (a, b)) in
  let sync () = Nx.sync (Nx.slice [ I 0 ] (Nx.reshape [| -1 |] !out)) in
  sync ();
  let t0 = now () in
  for _ = 1 to reps do
    out := f (a, b)
  done;
  sync ();
  let ms = (now () -. t0) /. Float.of_int reps *. 1e3 in
  let back = Nx.place Nx.Host.on !out in
  ms, Nx.to_array (Nx.reshape [| -1 |] back)

let time_f32 place (a_host, b_host) reps =
  let a = Nx.place place a_host
  and b = Nx.place place b_host in
  let f = Rune.jit Nx.Ptree.(pair tensor tensor @-> returns tensor) round in
  let out = ref (f (a, b)) in
  let sync () = Nx.sync (Nx.slice [ I 0 ] (Nx.reshape [| -1 |] !out)) in
  sync ();
  let t0 = now () in
  for _ = 1 to reps do
    out := f (a, b)
  done;
  sync ();
  let ms = (now () -. t0) /. Float.of_int reps *. 1e3 in
  let back = Nx.place Nx.Host.on !out in
  ms, Nx.to_array (Nx.reshape [| -1 |] back)

(* [B Bᵀ + nb I] in the given dtype, with a right-hand side: the shape of the
   system one active-set round solves, at the row counts the corpus uses. *)
let problem dtype nb lanes =
  let lead = [| lanes; nb; nb |] in
  let b =
    Nx.create
      dtype
      lead
      (Array.init (lanes * nb * nb) ~f:(fun i -> Float.sin (Float.of_int i)))
  in
  let a =
    Nx.add
      (Nx.matmul b (Nx.matrix_transpose b))
      (Nx.mul_s (Nx.broadcast_to lead (Nx.eye dtype nb)) (Float.of_int nb))
  in
  let rhs =
    Nx.create
      dtype
      [| lanes; nb; 1 |]
      (Array.init (lanes * nb) ~f:(fun i -> Float.cos (Float.of_int i)))
  in
  a, rhs

let worst_rel (got : float array) (want : float array) =
  let n = Stdlib.min (Array.length got) (Array.length want) in
  let acc = ref 0.0 in
  for i = 0 to n - 1 do
    let y = want.(i) in
    acc := Float.max !acc (Float.abs (got.(i) -. y) /. Float.max 1.0 (Float.abs y))
  done;
  !acc

let amd_gpus () =
  match Rig_amd_amdgpu.count () with
  | Error _ -> []
  | Ok n ->
    List.filter_map (List.init n ~f:Fn.id) ~f:(fun i ->
      match Rig.open_ (Rig_amd_amdgpu.gpu i) with
      | Ok dev when Rig.computes dev -> Some dev
      | _ -> None)

let () =
  let gpus = amd_gpus () in
  Printf.printf
    "| rows | lanes | device | f64 ms | f32 ms | f64/f32 | f32 vs f64, worst rel |\n";
  Printf.printf "| ---: | ---: | --- | ---: | ---: | ---: | ---: |\n%!";
  let lanes = 512
  and reps = 5 in
  List.iter [ 24; 40; 76 ] ~f:(fun nb ->
    let a64, b64 = problem Nx.float64 nb lanes in
    let a32, b32 = problem Nx.float32 nb lanes in
    let row device f64_ms f32_ms rel =
      Printf.printf
        "| %d | %d | %s | %.2f | %.2f | %.2fx | %s |\n%!"
        nb
        lanes
        device
        f64_ms
        f32_ms
        (f64_ms /. f32_ms)
        rel
    in
    let h64, out64 = time_f64 Nx.Host.on (a64, b64) reps in
    let h32, out32 = time_f32 Nx.Host.on (a32, b32) reps in
    row "host" h64 h32 (Printf.sprintf "%.1e" (worst_rel out32 out64));
    match gpus with
    | [] -> ()
    | dev :: _ ->
      let module D = (val Nx.devices [ dev ] : Nx.Devices) in
      let go () =
        let g64, g_out64 = time_f64 D.on (a64, b64) reps in
        let g32, g_out32 = time_f32 D.on (a32, b32) reps in
        g64, g32, worst_rel g_out32 g_out64
      in
      (try
         let g64, g32, rel = go () in
         row "gpu" g64 g32 (Printf.sprintf "%.1e" rel)
       with
       | e ->
         Printf.printf
           "| %d | %d | gpu | no number: %s | | | |\n%!"
           nb
           lanes
           (Exn.to_string e)))
