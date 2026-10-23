open Geom
open Geom_viz
module P = Primitive

let p3 = P.p3

let () =
  let camera =
    Camera.create
      ~eye:(p3 4.2 (-4.2) 3.0)
      ~target:(p3 0. 0. 0.6)
      ~up:(p3 0. 0. 1.)
      ~fov_y:(Float.pi /. 3.6)
      ~width:640.
      ~height:480.
      ()
  in
  let plane_patch = [ p3 (-3.) (-3.) 0.; p3 3. (-3.) 0.; p3 3. 3. 0.; p3 (-3.) 3. 0. ] in
  let plane =
    Draw.static
      ~layer:0
      ~style:
        { Draw.default with
          fill = Some (Hugin_gg.Color.v ~alpha:0.14 0.45 0.50 0.55)
        ; stroke = Some (Hugin_gg.Color.v ~alpha:0.30 0.25 0.35 0.40)
        ; width = 1.0
        }
      (P.Plane { normal = p3 0. 0. 1.; offset = 0.; patch = plane_patch })
  in
  let path_circle =
    Draw.static
      ~layer:1
      ~style:
        (Draw.outline ~width:1.2 ~dash:[ 4.; 6. ] (Hugin_gg.Color.gray ~alpha:0.55 0.45))
      (P.Circle { center = p3 0. 0. 0.02; normal = p3 0. 0. 1.; radius = 1.5 })
  in
  let trail =
    List.init 16 (fun i ->
      let t = 2. *. Float.pi *. float_of_int i /. 16. in
      Draw.static
        ~layer:2
        ~style:
          { Draw.default with
            fill = Some (Hugin_gg.Color.v ~alpha:0.55 0.25 0.25 0.30)
          ; stroke = None
          ; width = 1.0
          }
        (P.Point (p3 (1.5 *. cos t) (1.5 *. sin t) 0.03)))
  in
  let motion t =
    let a = 2. *. Float.pi *. t in
    let axis = Nx.create Motor.dtype [| 3 |] [| 0.; 0.; 1. |] in
    let angle = Nx.scalar Motor.dtype a in
    Motor.mul
      (Motor.translate3 (1.5 *. cos a) (1.5 *. sin a) 0.6)
      (Motor.rotate ~axis ~angle)
  in
  let sphere =
    Draw.moving
      ~layer:3
      ~motion
      ~style:
        { Draw.default with
          fill = Some (Hugin_gg.Color.v ~alpha:0.88 0.22 0.50 0.82)
        ; stroke = Some (Hugin_gg.Color.v 0.10 0.24 0.42)
        ; width = 1.6
        }
      (P.Sphere { center = p3 0. 0. 0.; radius = 0.35 })
  in
  let frame =
    let a = 0.6 in
    Draw.moving
      ~layer:4
      ~motion
      ~style:Draw.default
      (P.Frame { origin = p3 0. 0. 0.; x = p3 a 0. 0.; y = p3 0. a 0.; z = p3 0. 0. a })
  in
  let scene = Draw.create camera ([ plane; path_circle ] @ trail @ [ sphere; frame ]) in
  let spec = { Movie.duration = 2.; fps = 24; scene; density = 1.5 } in
  Movie.to_mp4 spec ~path:"viz/demo/out/motor.mp4";
  Printf.printf "wrote viz/demo/out/motor.mp4\n"
