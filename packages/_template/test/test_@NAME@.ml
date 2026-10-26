open Windtrap

let () =
  exit
    (run
       "@NAME@"
       [ test "placeholder" (fun () -> equal int 1 1) ])
