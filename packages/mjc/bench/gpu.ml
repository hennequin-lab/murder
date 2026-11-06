(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPU numbers for the solver's round, before any device-polymorphism work.

   The engine's own types are the host's, so its step cannot be placed on a
   device. What can be placed is the computation one active-set round is made
   of -- a batched Cholesky of the row matrix and its two triangular solves --
   on a batch the size of a training workload. This runs that one program on
   the host and, where the amdgpu path opens a GPU, on the machine's GPU, times
   both, and compares the two devices' answers.

   Written as rune's own GPU benches are (`rune/bench/bench_rune.ml`): the
   program is a record with a polymorphic field, so the same source is compiled
   at the host's brand and at the brand a device set mints at run time. A jit'd
   program is compiled per device, which is why it cannot be built outside.

   dune exec packages/mjc/bench/gpu.exe
   BEAM=2 dune exec packages/mjc/bench/gpu.exe *)

open Base
module Printf = Stdlib.Printf

let now () = Unix.gettimeofday ()

(* One active-set round's linear algebra: factor the row matrix, solve it. *)
let round ((a : ('v, 's, 'd) Nx.t), (b : ('v, 's, 'd) Nx.t)) =
  let l = Nx.cholesky a in
  Nx.solve_triangular
    ~triangle:Lower
    ~transpose:Transpose
    l
    (Nx.solve_triangular ~triangle:Lower l b)

(* [run place (a, b) reps] places the host's inputs at [place], compiles the
   round for that device, times [reps] calls of it, then brings the result back:
   the timings are of the compute alone, one copy each way outside them. *)
type run =
  { run :
      'd.
      'd Nx.Placement.t
      -> Nx.host Nx.float64_t * Nx.host Nx.float64_t
      -> int
      -> float * float array
  }

let prog =
  { run =
      (fun place (a_host, b_host) reps ->
        let a = Nx.place place a_host
        and b = Nx.place place b_host in
        (* Inline, not a top-level value: a `Ptree` signature is an application,
           so its brand would be a weak variable and the field could not
           generalise over the device. *)
        let f = Rune.jit Nx.Ptree.(pair tensor tensor @-> returns tensor) round in
        ignore (f (a, b));
        let t0 = now () in
        for _ = 1 to reps do
          ignore (f (a, b))
        done;
        let ms = (now () -. t0) /. Float.of_int reps *. 1e3 in
        let back = Nx.place Nx.Host.on (f (a, b)) in
        ms, Nx.to_array (Nx.reshape [| -1 |] back))
  }

(* The question this bench now answers: an op with no `_at` variant, `Nx.init`,
   builds its result on the host -- but the host value can be placed. Here the
   identity is built that way *inside* the compiled program, which is where the
   stage code needs it, and the round that uses it must agree with the one that
   does not. *)
let round_init ((a : ('v, 's, 'd) Nx.t), (b : ('v, 's, 'd) Nx.t)) =
  let at = Nx.placement a in
  let n = (Nx.shape a).(Array.length (Nx.shape a) - 1) in
  let id =
    Nx.place
      at
      (Nx.init Nx.float64 [| n; n |] (fun i -> if Int.equal i.(0) i.(1) then 1.0 else 0.0))
  in
  (* Used, so it cannot be dropped, and scaled by zero, so it changes nothing. *)
  round (Nx.add a (Nx.mul_s id 0.0), b)

let prog_init =
  { run =
      (fun place (a_host, b_host) reps ->
        let a = Nx.place place a_host
        and b = Nx.place place b_host in
        let f = Rune.jit Nx.Ptree.(pair tensor tensor @-> returns tensor) round_init in
        ignore (f (a, b));
        let t0 = now () in
        for _ = 1 to reps do
          ignore (f (a, b))
        done;
        let ms = (now () -. t0) /. Float.of_int reps *. 1e3 in
        let back = Nx.place Nx.Host.on (f (a, b)) in
        ms, Nx.to_array (Nx.reshape [| -1 |] back))
  }

let worst_rel got want =
  let n = Stdlib.min (Array.length got) (Array.length want) in
  let acc = ref 0.0 in
  for i = 0 to n - 1 do
    let y = want.(i) in
    acc := Float.max !acc (Float.abs (got.(i) -. y) /. Float.max 1.0 (Float.abs y))
  done;
  !acc

let amd_gpus () =
  match Rig_amd_amdgpu.count () with
  | Error e ->
    Printf.printf "amdgpu path: %s\n%!" e;
    []
  | Ok n ->
    List.filter_map (List.init n ~f:Fn.id) ~f:(fun i ->
      match Rig.open_ (Rig_amd_amdgpu.gpu i) with
      | Ok dev ->
        Printf.printf
          "  gpu %d: %s  arch %s  computes %b\n%!"
          i
          (Rig.name dev)
          (Rig.arch dev)
          (Rig.computes dev);
        Some dev
      | Error e ->
        Printf.printf "  gpu %d: %s\n%!" i e;
        None)

let () =
  Printf.printf "AMD GPUs visible to the amdgpu path:\n%!";
  let gpus = amd_gpus () in
  (* One active-set round at the corpus's row-buffer size: 76 rows, which is
     walker's `nefcmax`, over a batch of 64 environments. *)
  let nb = 76
  and lanes = 64
  and reps = 5 in
  let lead = [| lanes; nb; nb |] in
  let b =
    Nx.create
      Nx.float64
      lead
      (Array.init (lanes * nb * nb) ~f:(fun i -> Float.sin (Float.of_int i)))
  in
  let a_host =
    Nx.add
      (Nx.matmul b (Nx.matrix_transpose b))
      (Nx.mul_s (Nx.broadcast_to lead (Nx.eye Nx.float64 nb)) (Float.of_int nb))
  in
  let b_host =
    Nx.create
      Nx.float64
      [| lanes; nb; 1 |]
      (Array.init (lanes * nb) ~f:(fun i -> Float.cos (Float.of_int i)))
  in
  let host_ms, host_out = prog.run Nx.Host.on (a_host, b_host) reps in
  Printf.printf "\none active-set round: %d batched %dx%d Cholesky solves\n%!" lanes nb nb;
  Printf.printf "  host  %9.2f ms\n%!" host_ms;
  match gpus with
  | [] -> Printf.printf "  no GPU to compare\n%!"
  | dev :: _ ->
    (* A device compiles its program through tolk, whose AMD backend needs
       AMD's code object manager. Where that is missing the device still opens
       and the number is simply not available; say so rather than crash. *)
    let go () =
      let module D = (val Nx.devices [ dev ] : Nx.Devices) in
      prog.run D.on (a_host, b_host) reps, prog_init.run D.on (a_host, b_host) 1
    in
    (try
       let (gpu_ms, gpu_out), (_, init_gpu) = go () in
       Printf.printf "  gpu   %9.2f ms  (%.2fx the host)\n%!" gpu_ms (host_ms /. gpu_ms);
       Printf.printf
         "  gpu vs host, worst relative difference %.3g\n%!"
         (worst_rel gpu_out host_out);
       Printf.printf
         "  Nx.init on the host then placed, inside the program: %.3g from the round \
          that does not use it\n\
          %!"
         (worst_rel init_gpu host_out)
     with
     | e ->
       Printf.printf "  gpu   no number: %s\n%!" (Exn.to_string e);
       Printf.printf
         "  (tolk's AMD backend compiles through libamd_comgr: install `comgr`)\n%!")
