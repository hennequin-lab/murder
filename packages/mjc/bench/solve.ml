(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Is the blocked solve wrong, or is the system ill-conditioned?

   `bench/la.ml` says the blocked Cholesky in {!Linalg} is 1.25-4.3x faster than
   the [Nx.cholesky] kernel at the corpus's row counts, which would make it the
   engine's best remaining throughput lever. It is not the default because on one
   recorded state of `limited` it moved the solved force by 9e-5 where the kernel
   reproduces MuJoCo to 1e-9, and that was put down to conditioning.

   The residual decides it. Both solves run on every recorded corpus state's own
   matrix -- rebuilt here exactly as [Solver.forward] builds it -- and each
   answer's [|A x - b| / |b|] is reported beside the distance between the two
   answers. A much worse residual from the blocked solve is a bug; alike
   residuals mean the two answers are equally good and the difference is the
   matrix's.

   dune exec packages/mjc/bench/solve.exe *)

open Base
module Printf = Stdlib.Printf

let load name =
  match Mjc.Mjcb.read (Printf.sprintf "test/fixtures/%s.mjcb" name) with
  | Ok m -> m
  | Error e -> failwith e

let norm (x : Nx.host Nx.float64_t) =
  Nx.to_array x |> Array.fold ~init:0.0 ~f:(fun a v -> Float.max a (Float.abs v))

let rel (a : float array) (b : float array) =
  let n = Stdlib.min (Array.length a) (Array.length b) in
  let acc = ref 0.0 in
  for i = 0 to n - 1 do
    acc := Float.max !acc (Float.abs (a.(i) -. b.(i)))
  done;
  !acc

let () =
  Printf.printf
    "| model | states with rows | worst blocked residual | worst kernel residual | worst \
     difference | where |\n";
  Printf.printf "| --- | ---: | ---: | ---: | ---: | ---: |\n%!";
  List.iter [ "hopper"; "walker"; "humanoid" ] ~f:(fun name ->
    let m = load name in
    let path = Printf.sprintf "test/fixtures/%s.g4.mjbf" name in
    let qpos, qvel, _, act, ctrl, _, _, (steps, _) = G4lib.fixture path in
    let nrows = ref 0
    and r_blk = ref 0.0
    and r_ker = ref 0.0
    and diff = ref 0.0
    and where = ref 0 in
    for k = 0 to steps do
      let d0 = Mjc.Data.create m in
      let d =
        { d0 with
          Mjc.Data.qpos = Nx.slice [ I k; A ] qpos
        ; qvel = Nx.slice [ I k; A ] qvel
        ; act = Nx.slice [ I k; A ] act
        ; ctrl = Nx.slice [ I k; A ] ctrl
        }
      in
      let d = Mjc.Forward.forward m d in
      let nb = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.nefc) in
      if nb > 0
      then (
        Int.incr nrows;
        let nv = (Nx.shape d.Mjc.Data.qvel).(0) in
        let l = Mjc.Mass.factor d.Mjc.Data.mass_matrix in
        let j = Nx.slice [ R (0, nb); R (0, nv) ] d.Mjc.Data.efc_J in
        let a =
          Nx.add
            (Nx.matmul j (Mjc.Mass.solve l (Nx.matrix_transpose j)))
            (Nx.diag (Nx.slice [ R (0, nb) ] d.Mjc.Data.efc_R))
        in
        let b =
          Nx.reshape
            [| nb; 1 |]
            (Nx.sub
               (Nx.matmul j d.Mjc.Data.qacc)
               (Nx.slice [ R (0, nb) ] d.Mjc.Data.efc_aref))
        in
        let res x =
          let e = Nx.sub (Nx.matmul a x) b in
          norm e /. Float.max 1e-300 (norm b)
        in
        let x_ker =
          let cl = Nx.cholesky a in
          Nx.solve_triangular
            ~triangle:Lower
            ~transpose:Transpose
            cl
            (Nx.solve_triangular ~triangle:Lower cl b)
        in
        let x_blk = Mjc.Linalg.solve_spd a b in
        let dd = rel (Nx.to_array x_blk) (Nx.to_array x_ker) in
        if Float.( > ) dd !diff
        then (
          diff := dd;
          where := k);
        r_blk := Float.max !r_blk (res x_blk);
        r_ker := Float.max !r_ker (res x_ker))
    done;
    Printf.printf
      "| %s | %d | %.1e | %.1e | %.1e | state %d |\n%!"
      name
      !nrows
      !r_blk
      !r_ker
      !diff
      !where)
