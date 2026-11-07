let () =
  let m =
    match Mjc.Mjcb.read "test/fixtures/quad.mjcb" with
    | Ok m -> m
    | Error e -> failwith e
  in
  let d = Mjc.Forward.forward m (Mjc.Data.create m) in
  let d = ref d in
  for k = 1 to 100 do
    let t = Nx.item [] !d.Mjc.Data.time in
    let nu = (Nx.shape !d.Mjc.Data.ctrl).(0) in
    let ctrl =
      Nx.init Nx.float64 [| nu |] (fun i ->
        let i = float_of_int i.(0) in
        0.25 *. Float.sin ((2. *. Float.pi *. t /. 0.6) +. (Float.pi *. i)))
    in
    d := Mjc.Step.step m { !d with Mjc.Data.ctrl };
    if k mod 25 = 0
    then
      Printf.printf
        "k %d: t = %.4f, z = %.4f, ctrl0 = %.4f\n"
        k
        (Nx.item [] !d.Mjc.Data.time)
        (Nx.to_array (Nx.slice [ I 2 ] !d.Mjc.Data.qpos)).(0)
        (Nx.to_array !d.Mjc.Data.ctrl).(0)
  done
