open Base

let () =
  let m = Mjc.Mjcb.read "packages/mjc/test/fixtures/looped.mjcb" in
  let m =
    match m with
    | Ok m -> m
    | Error e -> failwith e
  in
  let d = Mjc.Data.create m in
  let q =
    Nx.reshape
      [| 1; 4 |]
      (Nx.create Nx.float64 [| 4 |] [| -0.528225; 0.536093; 0.657729; 0.03124 |])
  in
  let id = Nx.create Nx.float64 [| 4 |] [| 1.0; 0.0; 0.0; 0.0 |] in
  let one = Nx.create Nx.float64 [| 1; 4 |] [| 1.0; 0.0; 0.0; 0.0 |] in
  let show name (x : (float, Bigarray.float64_elt, Nx.host) Nx.t) =
    Stdlib.Printf.printf
      "%s: %s\n%!"
      name
      (String.concat
         (Array.to_list (Array.map (Nx.to_array x) ~f:Float.to_string))
         ~sep:",")
  in
  show "q*rel" (Mjc.Rigid.quat_mul q one);
  show "conj(id)*q" (Mjc.Rigid.quat_mul (Mjc.Rigid.quat_conj one) q);
  show "q" q;
  ignore id;
  let d = Mjc.Kinematics.forward m d in
  let arr x =
    String.concat (Array.to_list (Array.map (Nx.to_array x) ~f:Float.to_string)) ~sep:","
  in
  Stdlib.Printf.printf
    "obj1id=%s obj2id=%s data=%s\n%!"
    (arr m.Mjc.Model.params.eq_obj1id)
    (arr m.Mjc.Model.params.eq_obj2id)
    (arr m.Mjc.Model.params.eq_data);
  Stdlib.Printf.printf
    "xpos bolt=%s world=%s\n%!"
    (arr (Nx.slice [ I 2; A ] d.Mjc.Data.xpos))
    (arr (Nx.slice [ I 0; A ] d.Mjc.Data.xpos));
  let d = Mjc.Constraint.equality m d in
  let pos = Nx.to_array d.Mjc.Data.efc_pos in
  Stdlib.Printf.printf
    "pos: %s\n%!"
    (String.concat (Array.to_list (Array.map pos ~f:Float.to_string)) ~sep:",")
