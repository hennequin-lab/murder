open Geom
open Geom_viz
module P = Primitive

let p3 = P.p3
let point_tensor x y z = Nx.create Cga.dtype [| 3 |] [| x; y; z |]

let ghost =
  { Draw.default with
    fill = Some (Hugin_gg.Color.v ~alpha:0.18 0.30 0.34 0.40)
  ; stroke = Some (Hugin_gg.Color.v ~alpha:0.28 0.25 0.30 0.36)
  ; width = 1.0
  }

let () =
  let camera =
    Camera.create
      ~eye:(p3 4.6 (-4.6) 3.4)
      ~target:(p3 0. 0. 0.7)
      ~up:(p3 0. 0. 1.)
      ~fov_y:(Float.pi /. 4.)
      ~width:800.
      ~height:600.
      ()
  in
  let plane_obj =
    Cga.plane ~normal:(point_tensor 0. 0. 1.) ~offset:(Nx.scalar Cga.dtype 0.)
  in
  let plane = P.of_cga_plane plane_obj in
  let sphere_obj =
    Cga.sphere ~center:(point_tensor 1.0 0. 0.9) ~radius:(Nx.scalar Cga.dtype 0.55)
  in
  let sphere = P.of_cga_sphere sphere_obj in
  let line_obj = Cga.line (Cga.point3 (-1.6) (-0.9) 0.55) (Cga.point3 1.6 0.9 0.55) in
  let line = P.of_cga_line line_obj in
  let cc = p3 0. 0. 0.8 in
  let nn = P.normalize (p3 1. 1. 1.) in
  let uu = P.normalize (P.cross nn (p3 1. 0. 0.)) in
  let vv = P.cross nn uu in
  let r = 0.85 in
  let pt t =
    let p = P.add (P.scale uu (r *. cos t)) (P.scale vv (r *. sin t)) in
    let q = P.add cc p in
    Cga.point3 q.P.x q.P.y q.P.z
  in
  let circle_obj =
    Cga.circle (pt 0.) (pt (2. *. Float.pi /. 3.)) (pt (4. *. Float.pi /. 3.))
  in
  let circle = P.of_cga_circle circle_obj in
  let points =
    [ Cga.point3 1.0 0. 0.9
    ; Cga.point3 (-1.0) 0.5 0.55
    ; Cga.point3 0. 1.2 0.55
    ; Cga.point3 (-0.6) (-1.0) 0.55
    ]
    |> List.map P.of_cga_point
  in
  let motion t =
    let a = 2. *. Float.pi *. t in
    let axis = Nx.create Motor.dtype [| 3 |] [| 0.; 0.; 1. |] in
    let rz = Motor.rotate ~axis ~angle:(Nx.scalar Motor.dtype a) in
    let bob = Motor.translate3 0. 0. (0.12 *. sin a) in
    Motor.mul bob rz
  in
  let moving style layer prim = Draw.moving ~layer ~style ~motion prim in
  let sphere_style =
    { Draw.default with
      fill = Some (Hugin_gg.Color.v ~alpha:0.82 0.22 0.50 0.82)
    ; stroke = Some (Hugin_gg.Color.v 0.10 0.24 0.42)
    ; width = 1.6
    }
  in
  let line_style = Draw.outline ~width:2.2 (Hugin_gg.Color.v 0.85 0.35 0.10) in
  let circle_style = Draw.outline ~width:2.2 (Hugin_gg.Color.v 0.10 0.55 0.50) in
  let point_style =
    { Draw.default with
      fill = Some (Hugin_gg.Color.v 0.14 0.14 0.18)
    ; stroke = None
    ; width = 1.0
    }
  in
  let frame =
    let a = 0.55 in
    P.Frame { origin = p3 0. 0. 0.; x = p3 a 0. 0.; y = p3 0. a 0.; z = p3 0. 0. a }
  in
  let entities =
    [ Draw.static
        ~layer:0
        ~style:
          { Draw.default with
            fill = Some (Hugin_gg.Color.v ~alpha:0.13 0.45 0.50 0.55)
          ; stroke = Some (Hugin_gg.Color.v ~alpha:0.30 0.25 0.35 0.40)
          ; width = 1.0
          }
        plane
    ; Draw.static ~layer:1 ~style:ghost sphere
    ; Draw.static ~layer:1 ~style:ghost line
    ; Draw.static ~layer:1 ~style:ghost circle
    ]
    @ List.map (fun p -> Draw.static ~layer:1 ~style:ghost p) points
    @ [ moving sphere_style 2 sphere
      ; moving line_style 2 line
      ; moving circle_style 2 circle
      ]
    @ List.map (moving point_style 3) points
    @ [ moving Draw.default 4 frame ]
  in
  let scene = Draw.create camera entities in
  let picture = Draw.picture_at scene ~time:0.2 in
  let frame = Frame.create ~density:1.5 ~width:800. ~height:600. picture in
  Frame.save_png frame "viz/demo/out/zoo.png";
  Frame.save_svg frame "viz/demo/out/zoo.svg";
  let spec = { Movie.duration = 2.; fps = 24; scene; density = 1.5 } in
  Movie.to_mp4 spec ~path:"viz/demo/out/zoo.mp4";
  Printf.printf "wrote viz/demo/out/zoo.png, zoo.svg and zoo.mp4\n"
