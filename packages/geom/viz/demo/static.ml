open Geom_viz
module P = Primitive

let p3 = P.p3

let () =
  let camera =
    Camera.create
      ~eye:(p3 4. (-4.) 3.)
      ~target:(p3 0. 0. 0.5)
      ~up:(p3 0. 0. 1.)
      ~fov_y:(Float.pi /. 4.)
      ~width:800.
      ~height:600.
      ()
  in
  let plane_patch = [ p3 (-3.) (-3.) 0.; p3 3. (-3.) 0.; p3 3. 3. 0.; p3 (-3.) 3. 0. ] in
  let plane_style =
    { Draw.default with
      fill = Some (Hugin_gg.Color.v ~alpha:0.16 0.45 0.50 0.55)
    ; stroke = Some (Hugin_gg.Color.v ~alpha:0.35 0.25 0.35 0.40)
    ; width = 1.0
    }
  in
  let grid =
    List.init 7 (fun i ->
      let x = -3. +. (float_of_int i *. 1.) in
      Draw.static
        ~layer:0
        ~style:(Draw.outline ~width:0.5 (Hugin_gg.Color.gray ~alpha:0.25 0.55))
        (P.Segment (p3 x (-3.) 0., p3 x 3. 0.)))
    @ List.init 7 (fun i ->
      let y = -3. +. (float_of_int i *. 1.) in
      Draw.static
        ~layer:0
        ~style:(Draw.outline ~width:0.5 (Hugin_gg.Color.gray ~alpha:0.25 0.55))
        (P.Segment (p3 (-3.) y 0., p3 3. y 0.)))
  in
  let sphere =
    Draw.static
      ~layer:1
      ~style:
        { Draw.default with
          fill = Some (Hugin_gg.Color.v ~alpha:0.82 0.22 0.50 0.82)
        ; stroke = Some (Hugin_gg.Color.v 0.10 0.24 0.42)
        ; width = 1.8
        }
      (P.Sphere { center = p3 0. 0. 1.; radius = 1. })
  in
  let circle =
    Draw.static
      ~layer:1
      ~style:(Draw.outline ~width:2.0 (Hugin_gg.Color.v 0.10 0.55 0.50))
      (P.Circle { center = p3 0. 0. 0.01; normal = p3 0. 0. 1.; radius = 1.5 })
  in
  let points =
    List.init 9 (fun i ->
      let t = 2. *. Float.pi *. float_of_int i /. 9. in
      let x = 1.5 *. cos t
      and y = 1.5 *. sin t in
      Draw.static
        ~layer:2
        ~style:
          { Draw.default with
            fill = Some (Hugin_gg.Color.v 0.15 0.15 0.18)
          ; stroke = None
          ; width = 1.0
          }
        (P.Point (p3 x y 0.02)))
  in
  let frame =
    let o = p3 1.4 (-0.8) 0.9 in
    let a = 0.65 in
    Draw.static
      ~layer:3
      ~style:Draw.default
      (P.Frame
         { origin = o
         ; x = p3 (o.P.x +. a) o.P.y o.P.z
         ; y = p3 o.P.x (o.P.y +. a) o.P.z
         ; z = p3 o.P.x o.P.y (o.P.z +. a)
         })
  in
  let scene =
    Draw.create
      camera
      (Draw.static
         ~layer:0
         ~style:plane_style
         (P.Plane { normal = p3 0. 0. 1.; offset = 0.; patch = plane_patch })
       :: (grid @ [ sphere; circle; frame ] @ points))
  in
  let picture = Draw.picture_at scene ~time:0. in
  let frame = Frame.create ~density:1.5 ~width:800. ~height:600. picture in
  Frame.save_png frame "viz/demo/out/static.png";
  Frame.save_svg frame "viz/demo/out/static.svg";
  Printf.printf "wrote viz/demo/out/static.png and .svg\n"
