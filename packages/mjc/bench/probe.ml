(* Where the contact stage's time goes: eager, then traced, then compiled. *)
open Base
module Printf = Stdlib.Printf
module Data = Mjc.Data

let now () = Unix.gettimeofday ()

let () =
  let name =
    if Array.length (Sys.get_argv ()) > 1 then (Sys.get_argv ()).(1) else "humanoid"
  in
  let m =
    Mjc.Mjcb.read (Printf.sprintf "test/fixtures/%s.mjcb" name) |> Result.ok_or_failwith
  in
  let d0 = Data.create m in
  let ncand = Array.length (Mjc.Collision.candidates m) in
  Printf.printf
    "%s: ngeom %d, candidates %d, nv %d, nefcmax %d\n%!"
    name
    m.Mjc.Model.topo.Mjc.Topo.ngeom
    ncand
    m.Mjc.Model.topo.Mjc.Topo.nv
    m.Mjc.Model.topo.Mjc.Topo.nefcmax;
  let t0 = now () in
  let d1 = Mjc.Collision.contacts m d0 in
  Printf.printf
    "eager contacts: %.1f ms (ncon %d)\n%!"
    ((now () -. t0) *. 1e3)
    (Stdlib.Int32.to_int (Nx.item [] d1.Data.ncon));
  let ptree = Nx.Ptree.( @-> ) (Data.ptree ()) (Nx.Ptree.returns (Data.ptree ())) in
  let f = Rune.jit ptree (fun d -> Mjc.Collision.contacts m d) in
  let t0 = now () in
  let out = f d0 in
  Printf.printf
    "jit contacts: %.1f ms (ncon %d)\n%!"
    ((now () -. t0) *. 1e3)
    (Stdlib.Int32.to_int (Nx.item [] out.Data.ncon))
