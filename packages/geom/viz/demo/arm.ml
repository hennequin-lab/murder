open Geom
open Geom_viz
module P = Primitive

let p3 = P.p3
let lengths = [| 0.65; 0.55; 0.45; 0.35; 0.30; 0.25 |]
let amps = [| 0.50; 0.60; 0.55; 0.70; 0.60; 0.80 |]
let omegas = [| 0.9; 1.2; 1.5; 1.8; 2.1; 2.4 |]
let phases = [| 0.0; 0.7; 1.3; 2.0; 0.4; 1.1 |]

let axis i =
  let x, y, z = if i land 1 = 0 then 0., 0., 1. else 0., 1., 0. in
  Nx.create Motor.dtype [| 3 |] [| x; y; z |]

let origin_of m =
  let p = Cga.apply_motor m (Cga.point3 0. 0. 0.) |> Cga.coords |> Nx.to_array in
  p3 p.(0) p.(1) p.(2)

let kinematics t =
  let q i = amps.(i) *. sin ((omegas.(i) *. t) +. phases.(i)) in
  let rec go i m acc =
    if i >= 6
    then List.rev (m :: acc)
    else (
      let r = Motor.rotate ~axis:(axis i) ~angle:(Nx.scalar Motor.dtype (q i)) in
      let link = Motor.translate3 0. 0. lengths.(i) in
      let m' = Motor.mul m (Motor.mul r link) in
      go (i + 1) m' (m :: acc))
  in
  let motors = go 0 Motor.one [] in
  let origins = List.map origin_of motors in
  let end_motor = List.nth motors 6 in
  origins, end_motor

let link_style =
  { Draw.default with
    fill = None
  ; stroke = Some (Hugin_gg.Color.v 0.22 0.27 0.33)
  ; width = 14.0
  }

let joint_style =
  { Draw.default with
    fill = Some (Hugin_gg.Color.v ~alpha:0.95 0.95 0.48 0.15)
  ; stroke = Some (Hugin_gg.Color.v 0.45 0.25 0.05)
  ; width = 1.2
  }

let () =
  let camera =
    Camera.create
      ~eye:(p3 3.9 (-3.9) 2.9)
      ~target:(p3 0. 0. 0.75)
      ~up:(p3 0. 0. 1.)
      ~fov_y:(Float.pi /. 4.2)
      ~width:800.
      ~height:600.
      ()
  in
  let plane =
    Draw.static
      ~layer:0
      ~style:
        { Draw.default with
          fill = Some (Hugin_gg.Color.v ~alpha:0.10 0.45 0.50 0.55)
        ; stroke = Some (Hugin_gg.Color.v ~alpha:0.25 0.25 0.35 0.40)
        ; width = 1.0
        }
      (P.Plane
         { normal = p3 0. 0. 1.
         ; offset = 0.
         ; patch =
             [ p3 (-2.5) (-2.5) 0.; p3 2.5 (-2.5) 0.; p3 2.5 2.5 0.; p3 (-2.5) 2.5 0. ]
         })
  in
  let links =
    List.init 6 (fun i ->
      Draw.moving_with
        ~layer:2
        ~style:link_style
        ~motion:(fun t _ ->
          let origins, _ = kinematics t in
          P.Segment (List.nth origins i, List.nth origins (i + 1)))
        (P.Segment (p3 0. 0. 0., p3 0. 0. 0.)))
  in
  let joints =
    List.init 7 (fun i ->
      Draw.moving_with
        ~layer:3
        ~style:joint_style
        ~motion:(fun t _ ->
          let origins, _ = kinematics t in
          let c = List.nth origins i in
          P.Sphere { center = c; radius = (if i = 0 then 0.12 else 0.09) })
        (P.Sphere { center = p3 0. 0. 0.; radius = 0.09 }))
  in
  let end_frame =
    Draw.moving_with
      ~layer:4
      ~style:Draw.default
      ~motion:(fun t _ ->
        let origins, m = kinematics t in
        let origin = List.nth origins 6 in
        let dir x y z =
          let p = Cga.apply_motor m (Cga.point3 x y z) |> Cga.coords |> Nx.to_array in
          p3 p.(0) p.(1) p.(2)
        in
        ignore origin;
        P.Frame
          { origin = List.nth origins 6
          ; x = dir 0.32 0. 0.
          ; y = dir 0. 0.32 0.
          ; z = dir 0. 0. 0.32
          })
      (P.Frame
         { origin = p3 0. 0. 0.; x = p3 0.2 0. 0.; y = p3 0. 0.2 0.; z = p3 0. 0. 0.2 })
  in
  let scene = Draw.create camera ((plane :: links) @ joints @ [ end_frame ]) in
  let picture = Draw.picture_at scene ~time:0.6 in
  let frame = Frame.create ~density:1.5 ~width:800. ~height:600. picture in
  Frame.save_png frame "viz/demo/out/arm.png";
  Frame.save_svg frame "viz/demo/out/arm.svg";
  let spec = { Movie.duration = 3.; fps = 24; scene; density = 1.5 } in
  Movie.to_mp4 spec ~path:"viz/demo/out/arm.mp4";
  Printf.printf "wrote viz/demo/out/arm.{png,svg,mp4}\n"
