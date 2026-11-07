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
        let t = float_of_int k *. 0.002 in
        let i = float_of_int i.(0) in
        0.25 *. Float.sin ((2. *. Float.pi *. t /. 0.6) +. (Float.pi *. i)))
    in
    d := Mjc.Step.step m { !d with Mjc.Data.ctrl }
  done;
  let h = Nx.to_array (Nx.slice [ I 2 ] !d.Mjc.Data.qpos) in
  Printf.printf
    "after %d steps: torso z = %.4f, ncon = %d, nq = %d\n"
    steps
    h.(0)
    (Stdlib.Int32.to_int (Nx.item [] !d.Mjc.Data.ncon))
    nq;
  (* The wiggle, in numbers: a standing model keeps its height and its feet. *)
  let lo = ref 1e9
  and hi = ref (-1e9)
  and touching = ref 0 in
  let d = ref (Mjc.Forward.forward m (Mjc.Data.create m)) in
  for k = 1 to 600 do
    let t = float_of_int k *. 0.002 in
    let nu = (Nx.shape !d.Mjc.Data.ctrl).(0) in
    let ctrl =
      Nx.init Nx.float64 [| nu |] (fun i ->
        let i = float_of_int i.(0) in
        0.25 *. Float.sin ((2. *. Float.pi *. t /. 0.6) +. (Float.pi *. i)))
    in
    d := Mjc.Step.step m { !d with Mjc.Data.ctrl };
    let z = (Nx.to_array (Nx.slice [ I 2 ] !d.Mjc.Data.qpos)).(0) in
    lo := Float.min !lo z;
    hi := Float.max !hi z;
    if Stdlib.Int32.to_int (Nx.item [] !d.Mjc.Data.ncon) > 0 then incr touching
  done;
  Printf.printf
    "over 600 steps of antiphase drive: torso z %.4f..%.4f, in contact %d of 600\n"
    !lo
    !hi
    !touching
