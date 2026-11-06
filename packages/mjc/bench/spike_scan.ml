(*****************************************************************************)
(* The claim the scan makes on the row solve: its round — a Cholesky
   factorisation, two triangular solves, and reductions — is legal inside a
   {!Rune.scan} body on this machine, in float64. `spike_scan` is the evidence
   for the compile-cost claim in `lib/solver.ml`: the round that the scan
   traces once per compile. *)
open Base

(* Probes the scan-row shape contract as well. *)
let () =
  let nb = 64 in
  let length = 4 in
  let phases = Nx.create Nx.float64 [| length; 1 |] [| 0.; 1.; 2.; 1. |] in
  let mask_f = Nx.ones Nx.float64 [| nb |] in
  let body (lambda, free) (phase : Nx.host Nx.float64_t) =
    Stdlib.Printf.printf
      "  row shape=%s  lambda=%s  free=%s\n%!"
      (String.concat
         ~sep:" "
         (List.map (Array.to_list (Nx.shape phase)) ~f:Int.to_string))
      (String.concat
         ~sep:" "
         (List.map (Array.to_list (Nx.shape lambda)) ~f:Int.to_string))
      (String.concat ~sep:" " (List.map (Array.to_list (Nx.shape free)) ~f:Int.to_string));
    let p = Nx.broadcast_to [| nb; 1 |] phase in
    let is_first = Nx.equal_s p 0.0 in
    let at_boundary = Nx.logical_or is_first (Nx.equal_s p 1.0) in
    let flag f = Nx.broadcast_to [| nb |] (Nx.slice [ I 0; A ] f) in
    let free_new =
      Nx.where (flag at_boundary) (Nx.where (flag is_first) mask_f free) free
    in
    (Nx.add lambda free_new, free_new), Nx.reshape [| 1 |] (Nx.sum lambda)
  in
  let (l, f), _ =
    Rune.scan
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      Nx.Ptree.tensor
      body
      ~init:(Nx.zeros Nx.float64 [| nb |], Nx.ones Nx.float64 [| nb |])
      phases
  in
  Stdlib.Printf.printf
    "lambda[0]=%.6g free[0]=%.6g\n%!"
    (Nx.item [ 0 ] l)
    (Nx.item [ 0 ] f)
