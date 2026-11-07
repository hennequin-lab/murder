let () =
  let m =
    match Mjc.Mjcb.read "test/fixtures/quad.mjcb" with
    | Ok m -> m
    | Error e -> failwith e
  in
  let d = Mjc.Forward.forward m (Mjc.Data.create m) in
  let nq = (Nx.shape d.Mjc.Data.qpos).(0) in
  let steps = 1000 in
  let d = ref d in
  for k = 1 to steps do
    let ctrl =
      let nu = (Nx.shape !d.Mjc.Data.ctrl).(0) in
      Nx.init Nx.float64 [| nu |] (fun i ->
        let p = float_of_int (i.(0) / 2) *. 1.57 in
        Float.sin ((float_of_int k *. 0.004) +. p) *. 0.6)
    in
    d := Mjc.Step.step m { !d with Mjc.Data.ctrl }
  done;
  let h = Nx.to_array (Nx.slice [ I 2 ] !d.Mjc.Data.qpos) in
  Printf.printf
    "after %d steps: torso z = %.4f, ncon = %d, nq = %d\n"
    steps
    h.(0)
    (Stdlib.Int32.to_int (Nx.item [] !d.Mjc.Data.ncon))
    nq
