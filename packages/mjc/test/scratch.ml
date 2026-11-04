open Base

let () =
  let m =
    match Mjc.Mjcb.read "packages/mjc/test/fixtures/pendulum.mjcb" with
    | Ok m -> m
    | Error e -> failwith e
  in
  Stdlib.Printf.printf "loaded%!\n";
  let d = Mjc.Data.create m in
  Stdlib.Printf.printf "created%!\n";
  let t0 = Stdlib.Sys.time () in
  let d = ref d in
  for _ = 1 to 10 do
    d := Mjc.Step.step m !d
  done;
  Stdlib.Printf.printf "10 eager steps %.3f s%!\n" (Stdlib.Sys.time () -. t0);
  let f = Mjc.Step.compile m in
  let t0 = Stdlib.Sys.time () in
  let _ = Mjc.Step.run f !d in
  Stdlib.Printf.printf "one compiled %.3f s%!\n" (Stdlib.Sys.time () -. t0)
