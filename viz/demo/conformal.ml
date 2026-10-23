open Geom
open Geom_viz
module P = Primitive

let p3 = P.p3

type obj =
  | Point of Cga.t
  | Sphere of Cga.t
  | Circle of Cga.t

let to_primitive v = function
  | Point p -> P.of_cga_point (Cga.sandwich v p)
  | Sphere s -> P.of_cga_sphere (Cga.sandwich v s)
  | Circle c -> P.of_cga_circle (Cga.sandwich v c)

let () =
  let camera =
    Camera.create
      ~eye:(p3 5.0 (-5.0) 3.6)
      ~target:(p3 0. 0. 0.75)
      ~up:(p3 0. 0. 1.)
      ~fov_y:(Float.pi /. 4.2)
      ~width:640.
      ~height:480.
      ()
  in
  let plane =
    Draw.static
      ~layer:0
      ~style:
        { Draw.default with
          fill = Some (Hugin_gg.Color.v ~alpha:0.12 0.45 0.50 0.55)
        ; stroke = Some (Hugin_gg.Color.v ~alpha:0.28 0.25 0.35 0.40)
        ; width = 1.0
        }
      (P.Plane
         { normal = p3 0. 0. 1.
         ; offset = 0.
         ; patch = [ p3 (-3.) (-3.) 0.; p3 3. (-3.) 0.; p3 3. 3. 0.; p3 (-3.) 3. 0. ]
         })
  in
  let sphere_obj =
    Cga.sphere
      ~center:(Nx.create Cga.dtype [| 3 |] [| 0.; 0.; 1.0 |])
      ~radius:(Nx.scalar Cga.dtype 0.45)
  in
  let point_objs =
    [ Cga.point3 1.0 0. 0.6
    ; Cga.point3 (-1.0) 0.2 0.6
    ; Cga.point3 0. 1.0 0.6
    ; Cga.point3 0.3 (-1.0) 0.6
    ]
  in
  let cc = p3 0. 0. 0.65 in
  let nn = P.normalize (p3 1. 1. 1.) in
  let uu = P.normalize (P.cross nn (p3 1. 0. 0.)) in
  let vv = P.cross nn uu in
  let radius = 0.65 in
  let pt t =
    let p =
      P.add cc (P.add (P.scale uu (radius *. cos t)) (P.scale vv (radius *. sin t)))
    in
    Cga.point3 p.P.x p.P.y p.P.z
  in
  let circle_obj =
    Cga.circle (pt 0.) (pt (2. *. Float.pi /. 3.)) (pt (4. *. Float.pi /. 3.))
  in
  let objects =
    [ Sphere sphere_obj ] @ List.map (fun p -> Point p) point_objs @ [ Circle circle_obj ]
  in
  let ghost =
    { Draw.default with
      fill = Some (Hugin_gg.Color.v ~alpha:0.14 0.30 0.34 0.40)
    ; stroke = Some (Hugin_gg.Color.v ~alpha:0.25 0.25 0.30 0.36)
    ; width = 1.0
    }
  in
  let style_of = function
    | Sphere _ ->
      { Draw.default with
        fill = Some (Hugin_gg.Color.v ~alpha:0.80 0.22 0.50 0.82)
      ; stroke = Some (Hugin_gg.Color.v 0.10 0.24 0.42)
      ; width = 1.5
      }
    | Circle _ -> Draw.outline ~width:2.0 (Hugin_gg.Color.v 0.10 0.55 0.50)
    | Point _ ->
      { Draw.default with
        fill = Some (Hugin_gg.Color.v 0.14 0.14 0.18)
      ; stroke = None
      ; width = 1.0
      }
  in
  let scene_for versor =
    let ghosts =
      List.map
        (fun o -> Draw.static ~layer:1 ~style:ghost (to_primitive Cga.one o))
        objects
    in
    let movers =
      List.map
        (fun o ->
           Draw.moving_with
             ~layer:2
             ~style:(style_of o)
             ~motion:(fun t _ -> to_primitive (versor t) o)
             (to_primitive Cga.one o))
        objects
    in
    Draw.create camera ((plane :: ghosts) @ movers)
  in
  let render name versor =
    let scene = scene_for versor in
    let picture = Draw.picture_at scene ~time:0.25 in
    let frame = Frame.create ~density:1.5 ~width:640. ~height:480. picture in
    Frame.save_png frame ("viz/demo/out/" ^ name ^ ".png");
    Frame.save_svg frame ("viz/demo/out/" ^ name ^ ".svg");
    let spec = { Movie.duration = 2.; fps = 24; scene; density = 1.5 } in
    Movie.to_mp4 spec ~path:("viz/demo/out/" ^ name ^ ".mp4");
    Printf.printf "wrote viz/demo/out/%s.{png,svg,mp4}\n" name
  in
  let dilation t =
    Cga.dilator (Nx.scalar Cga.dtype (0.30 *. sin (2. *. Float.pi *. t)))
  in
  let transversion t =
    Cga.transversion
      (Nx.create Cga.dtype [| 3 |] [| 0.25 *. sin (2. *. Float.pi *. t); 0.; 0. |])
  in
  render "conformal-dilation" dilation;
  render "conformal-transversion" transversion
