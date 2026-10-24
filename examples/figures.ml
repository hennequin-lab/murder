(* README figures, drawn with geom-viz.

   Run from geom/:  dune exec examples/figures.exe

   Writes doc/figures/*.svg (committed, referenced by the READMEs) and
   doc/figures/*.png (a local preview; PNGs are gitignored). *)

open Geom
open Geom_viz
module P = Primitive
module C = Camera
module Gg = Hugin_gg
module Vg = Hugin_vg
module F = Hugin_font

let width = 700.
let height = 520.
let svg_dir = "doc/figures"

(* -------------------------------------------------------------------------- *)
(* Small drawing helpers                                                      *)
(* -------------------------------------------------------------------------- *)

let p3 = P.p3
let ink = Gg.Color.v 0.13 0.15 0.18
let muted = Gg.Color.v 0.40 0.43 0.48
let e1_color = Gg.Color.v 0.80 0.18 0.13
let e2_color = Gg.Color.v 0.12 0.55 0.22
let rotor_color = Gg.Color.v 0.55 0.20 0.65
let plane_line = Gg.Color.v ~alpha:0.40 0.24 0.36 0.45
let grid_style = Draw.outline ~width:0.5 (Gg.Color.gray ~alpha:0.22 0.55)

(* [text] lays out ASCII [s] in Inter and draws the glyph outlines. Outlines
   keep the SVG independent of the viewer's fonts; the labels are short, so the
   loss of selectable text does not matter. *)
let text ?(color = ink) ?(size = 12.) ?(font = F.Font.regular) ~x ~y s =
  let chars = ref [] in
  String.iter (fun c -> chars := Uchar.of_char c :: !chars) s;
  let glyphs = Array.of_list (List.rev_map (F.Font.glyph font) !chars) in
  let n = Array.length glyphs in
  let xs = Array.make n 0. in
  for i = 1 to n - 1 do
    xs.(i)
    <- xs.(i - 1)
       +. (size
           *. (F.Font.advance font glyphs.(i - 1)
               +. F.Font.kerning font glyphs.(i - 1) glyphs.(i)))
  done;
  let glyph i =
    Vg.Picture.transform
      Gg.Affine.(translate (x +. xs.(i)) y * scale size size)
      (Vg.Picture.fill color (F.Font.outline font glyphs.(i)))
  in
  Vg.Picture.group (List.init n glyph)

let project (c : C.t) p =
  match C.project c p with
  | Some (x, y, _) -> Some (x, y)
  | None -> None

let label ?(color = ink) ?(size = 12.) ?(dx = 8.) ?(dy = -8.) (c : C.t) p s =
  match project c p with
  | None -> Vg.Picture.empty
  | Some (x, y) -> text ~color ~size ~x:(x +. dx) ~y:(y +. dy) s

let label2d ?(color = ink) ?(size = 12.) ~x ~y s = text ~color ~size ~x ~y s

let arrow2d ?(color = ink) ?(width = 2.2) ?(head = 10.) (c : C.t) a b =
  match project c a, project c b with
  | Some (x0, y0), Some (x1, y1) ->
    let dx = x1 -. x0
    and dy = y1 -. y0 in
    let len = sqrt ((dx *. dx) +. (dy *. dy)) in
    if len < 1e-6
    then Vg.Picture.empty
    else (
      let ux = dx /. len
      and uy = dy /. len in
      let bx = x1 -. (head *. ux)
      and by = y1 -. (head *. uy) in
      let nx = -.uy
      and ny = ux in
      let half = head *. 0.45 in
      let tri =
        Gg.Path.polygon
          [| x1; bx +. (half *. nx); bx -. (half *. nx) |]
          [| y1; by +. (half *. ny); by -. (half *. ny) |]
      in
      Vg.Picture.group
        [ Vg.Picture.stroke
            (Gg.Stroke.v width)
            color
            (Gg.Path.polyline [| x0; bx |] [| y0; by |])
        ; Vg.Picture.fill color tri
        ])
  | _ -> Vg.Picture.empty

let polyline3 ?(color = ink) ?(width = 1.5) ?(dash = []) (c : C.t) pts =
  let pts = List.filter_map (fun p -> project c p) pts in
  match pts with
  | [] -> Vg.Picture.empty
  | _ ->
    let xs = Array.of_list (List.map fst pts)
    and ys = Array.of_list (List.map snd pts) in
    let stroke = if dash = [] then Gg.Stroke.v width else Gg.Stroke.v ~dash width in
    Vg.Picture.stroke stroke color (Gg.Path.polyline xs ys)

let disc ?(color = ink) ?(radius = 4.) (c : C.t) p =
  match project c p with
  | None -> Vg.Picture.empty
  | Some (x, y) -> Vg.Picture.fill color (Gg.Path.circle (Gg.P2.v x y) radius)

let ring ?(color = ink) ?(radius = 4.) ?(width = 1.4) (c : C.t) p =
  match project c p with
  | None -> Vg.Picture.empty
  | Some (x, y) ->
    Vg.Picture.stroke (Gg.Stroke.v width) color (Gg.Path.circle (Gg.P2.v x y) radius)

let ground ~size style =
  Draw.static
    ~layer:0
    ~style
    (P.Plane
       { normal = p3 0. 0. 1.
       ; offset = 0.
       ; patch =
           [ p3 (-.size) (-.size) 0.
           ; p3 size (-.size) 0.
           ; p3 size size 0.
           ; p3 (-.size) size 0.
           ]
       })

let grid ~layer ~size ~step =
  let n = int_of_float (2. *. size /. step) in
  List.init (n + 1) (fun i ->
    let x = -.size +. (float_of_int i *. step) in
    [ Draw.static ~layer ~style:grid_style (P.Segment (p3 x (-.size) 0., p3 x size 0.))
    ; Draw.static ~layer ~style:grid_style (P.Segment (p3 (-.size) x 0., p3 size x 0.))
    ])
  |> List.concat

let ground_style =
  { Draw.default with
    fill = Some (Gg.Color.v ~alpha:0.05 0.35 0.45 0.40)
  ; stroke = Some plane_line
  ; width = 0.7
  }

let save name picture =
  let frame = Frame.create ~density:1.6 ~width ~height picture in
  Frame.save_svg frame (Filename.concat svg_dir (name ^ ".svg"));
  Frame.save_png frame (Filename.concat svg_dir (name ^ ".png"));
  Printf.printf "wrote %s/%s.{svg,png}\n" svg_dir name

let scene_picture scene = Draw.picture_at scene ~time:0.

let p3_of_coords t =
  let a = Nx.to_array t in
  p3 a.(0) a.(1) a.(2)

let p3_of_cga p = p3_of_coords (Cga.coords p)

(* -------------------------------------------------------------------------- *)
(* Figure 1: the geometric product and a rotor in E3                          *)
(* -------------------------------------------------------------------------- *)

let figure_e3 () =
  let camera =
    C.create
      ~eye:(p3 1.55 (-2.20) 1.35)
      ~target:(p3 0.42 0.38 (-0.28))
      ~up:(p3 0. 0. 1.)
      ~fov_y:(Float.pi /. 4.8)
      ~width
      ~height
      ()
  in
  let square =
    P.Polygon [ p3 0. 0. 0.001; p3 1. 0. 0.001; p3 1. 1. 0.001; p3 0. 1. 0.001 ]
  in
  let square_style =
    { Draw.default with
      fill = Some (Gg.Color.v ~alpha:0.18 0.22 0.45 0.82)
    ; stroke = Some (Gg.Color.v ~alpha:0.55 0.18 0.30 0.55)
    ; width = 1.1
    }
  in
  let angle = 2. *. Float.pi /. 3. in
  let rotor =
    E3.rotor
      ~axis:(Nx.create Nx.float32 [| 3 |] [| 0.; 0.; 1. |])
      ~angle:(Nx.scalar Nx.float32 angle)
  in
  let rotated =
    E3.apply_rotor rotor (E3.vector3 1. 0. 0.) |> E3.to_vector |> Nx.to_array
  in
  let rotated = p3 rotated.(0) rotated.(1) rotated.(2) in
  let arc =
    List.init 33 (fun i ->
      let a = angle *. float_of_int i /. 32. in
      p3 (cos a) (sin a) 0.002)
  in
  let origin = p3 0. 0. 0. in
  let dots =
    [ disc ~color:ink ~radius:3.5 camera origin
    ; disc ~color:e1_color ~radius:3.5 camera (p3 1. 0. 0.)
    ; disc ~color:e2_color ~radius:3.5 camera (p3 0. 1. 0.)
    ; disc ~color:rotor_color ~radius:4. camera rotated
    ]
  in
  let entities =
    [ ground ~size:1.5 ground_style ]
    @ grid ~layer:0 ~size:1.5 ~step:0.5
    @ [ Draw.static ~layer:1 ~style:square_style square ]
  in
  let scene = Draw.create camera entities in
  let overlay =
    Vg.Picture.group
      [ arrow2d ~color:e1_color ~width:2.6 camera origin (p3 1.05 0. 0.)
      ; arrow2d ~color:e2_color ~width:2.6 camera origin (p3 0. 1.05 0.)
      ; arrow2d ~color:rotor_color ~width:3.0 camera origin rotated
      ; polyline3 ~color:rotor_color ~width:1.3 ~dash:[ 4.; 4. ] camera arc
      ; Vg.Picture.group dots
      ; label ~color:e1_color ~size:14. ~dx:4. ~dy:18. camera (p3 1. 0. 0.) "e1"
      ; label ~color:e2_color ~size:14. ~dx:8. ~dy:(-4.) camera (p3 0. 1. 0.) "e2"
      ; label ~color:rotor_color ~size:14. ~dx:(-72.) ~dy:4. camera rotated "R e1 R~"
      ; label
          ~color:muted
          ~size:12.
          ~dx:8.
          ~dy:14.
          camera
          (p3 (0.62 *. cos (angle /. 2.)) (0.62 *. sin (angle /. 2.)) 0.)
          "120 deg"
      ; label2d
          ~color:muted
          ~size:13.
          ~x:22.
          ~y:34.
          "the product e1 e2 is a bivector: the oriented plane they span"
      ]
  in
  save "e3-products" (Vg.Picture.group [ scene_picture scene; overlay ])

(* -------------------------------------------------------------------------- *)
(* Figure 2: the CGA object zoo                                               *)
(* -------------------------------------------------------------------------- *)

let figure_cga () =
  let camera =
    C.create
      ~eye:(p3 3.5 (-4.1) 2.7)
      ~target:(p3 0.05 0. 0.88)
      ~up:(p3 0. 0. 1.)
      ~fov_y:(Float.pi /. 4.5)
      ~width
      ~height
      ()
  in
  let v3 x y z = Nx.create Nx.float32 [| 3 |] [| x; y; z |] in
  let sc x = Nx.scalar Nx.float32 x in
  let p = Cga.point3 in
  let sphere_obj = Cga.sphere ~center:(v3 1.0 (-0.45) 0.78) ~radius:(sc 0.45) in
  let plane_obj = Cga.plane ~normal:(v3 1. 0. 0.) ~offset:(sc (-1.3)) in
  (* A circle in a tilted plane, defined by three of its points. *)
  let normal = P.normalize (p3 0.30 0.20 1.) in
  let u = P.normalize (P.cross normal (p3 0. 1. 0.)) in
  let v = P.cross normal u in
  let center = p3 0. 0. 1.53 in
  let radius = 0.82 in
  let on_circle t =
    let q =
      P.add center (P.add (P.scale u (radius *. cos t)) (P.scale v (radius *. sin t)))
    in
    p q.P.x q.P.y q.P.z
  in
  let c1 = on_circle (Float.pi /. 2.)
  and c2 = on_circle ((Float.pi /. 2.) +. (2. *. Float.pi /. 3.))
  and c3 = on_circle ((Float.pi /. 2.) +. (4. *. Float.pi /. 3.)) in
  let circle_obj = Cga.circle c1 c2 c3 in
  (* A line through two points on it. *)
  let a = p3 (-1.5) (-1.5) 0.06
  and b = p3 1.5 (-1.5) 0.06 in
  let line_obj = Cga.line (p a.P.x a.P.y a.P.z) (p b.P.x b.P.y b.P.z) in
  let ld = P.p3_of_tensor (Cga.line_direction line_obj) in
  let lm = P.p3_of_tensor (Cga.line_moment line_obj) in
  let lc = P.cross ld lm in
  let at t = P.add lc (P.scale ld t) in
  let sphere = P.of_cga_sphere sphere_obj in
  let plane = P.of_cga_plane plane_obj in
  let circle = P.of_cga_circle circle_obj in
  let line = P.of_cga_line ~extent:1.05 line_obj in
  let sphere_style =
    { Draw.default with
      fill = Some (Gg.Color.v ~alpha:0.55 0.85 0.25 0.55)
    ; stroke = Some (Gg.Color.v 0.45 0.10 0.35)
    ; width = 1.4
    }
  in
  let plane_style =
    { Draw.default with
      fill = Some (Gg.Color.v ~alpha:0.16 0.25 0.50 0.58)
    ; stroke = Some plane_line
    ; width = 1.0
    }
  in
  let circle_style = Draw.outline ~width:2.6 (Gg.Color.v 0.90 0.45 0.05) in
  let line_style = Draw.outline ~width:2.6 (Gg.Color.v 0.05 0.45 0.55) in
  let entities =
    [ ground ~size:2.3 ground_style ]
    @ grid ~layer:0 ~size:2.0 ~step:0.5
    @ [ Draw.static ~layer:1 ~style:plane_style plane
      ; Draw.static ~layer:2 ~style:sphere_style sphere
      ; Draw.static ~layer:2 ~style:line_style line
      ; Draw.static ~layer:3 ~style:circle_style circle
      ]
  in
  let scene = Draw.create camera entities in
  let overlay =
    Vg.Picture.group
      [ ring ~color:(Gg.Color.v 0.90 0.45 0.05) ~radius:4.2 camera (p3_of_cga c1)
      ; ring ~color:(Gg.Color.v 0.90 0.45 0.05) ~radius:4.2 camera (p3_of_cga c2)
      ; ring ~color:(Gg.Color.v 0.90 0.45 0.05) ~radius:4.2 camera (p3_of_cga c3)
      ; disc ~color:(Gg.Color.v 0.05 0.45 0.55) ~radius:3.6 camera (at (-0.65))
      ; disc ~color:(Gg.Color.v 0.05 0.45 0.55) ~radius:3.6 camera (at 0.65)
      ; disc ~color:ink ~radius:3.6 camera (p3 1.0 (-0.45) 1.23)
      ; label ~color:ink ~size:13. ~dx:12. ~dy:0. camera (p3 0. 0. 1.53) "circle"
      ; label
          ~color:ink
          ~size:13.
          ~dx:102.
          ~dy:38.
          camera
          (p3_of_cga (Cga.sphere_center sphere_obj))
          "sphere"
      ; label ~color:ink ~size:13. ~dx:8. ~dy:(-8.) camera (at 1.12) "line"
      ; label ~color:ink ~size:13. ~dx:10. ~dy:(-10.) camera (p3 (-1.3) 1.0 1.55) "plane"
      ; label2d
          ~color:muted
          ~size:13.
          ~x:22.
          ~y:34.
          "one algebra, five kinds of object; each is a Cga.t"
      ]
  in
  save "cga-objects" (Vg.Picture.group [ scene_picture scene; overlay ])

(* -------------------------------------------------------------------------- *)
(* Figure 3: PGA meet                                                        *)
(* -------------------------------------------------------------------------- *)

let figure_pga () =
  let camera =
    C.create
      ~eye:(p3 3.35 (-3.75) 2.55)
      ~target:(p3 0. 0. 0.88)
      ~up:(p3 0. 0. 1.)
      ~fov_y:(Float.pi /. 4.5)
      ~width
      ~height
      ()
  in
  let pi1 = Pga.plane3 1. 0. 0. 0. in
  let pi2 = Pga.plane3 0. 1. 0. 0. in
  let pi3 = Pga.plane3 0. 0. 1. 1.9 in
  let l = Pga.meet pi1 pi2 in
  let d = Pga.line_direction l |> Nx.to_array in
  let m = Pga.line_moment l |> Nx.to_array in
  let cx = (d.(1) *. m.(2)) -. (d.(2) *. m.(1))
  and cy = (d.(2) *. m.(0)) -. (d.(0) *. m.(2))
  and cz = (d.(0) *. m.(1)) -. (d.(1) *. m.(0)) in
  let on_line = p3 cx cy (cz +. 0.8) in
  let corner = Pga.meet (Pga.meet pi1 pi2) pi3 in
  let corner = Pga.point_coords corner |> Nx.to_array in
  let corner = p3 corner.(0) corner.(1) corner.(2) in
  let pi1_patch = [ p3 0. (-1.6) 0.; p3 0. 1.6 0.; p3 0. 1.6 1.9; p3 0. (-1.6) 1.9 ] in
  let pi2_patch = [ p3 (-1.6) 0. 0.; p3 1.6 0. 0.; p3 1.6 0. 1.9; p3 (-1.6) 0. 1.9 ] in
  let pi3_patch =
    [ p3 (-1.6) (-1.6) 1.9; p3 1.6 (-1.6) 1.9; p3 1.6 1.6 1.9; p3 (-1.6) 1.6 1.9 ]
  in
  let plane_style fill =
    { Draw.default with fill = Some fill; stroke = Some plane_line; width = 0.9 }
  in
  let line_prim = P.Segment (p3 cx cy cz, p3 cx cy (cz +. 1.9)) in
  let entities =
    [ ground ~size:2.3 ground_style ]
    @ grid ~layer:0 ~size:2.0 ~step:0.5
    @ [ Draw.static
          ~layer:1
          ~style:(plane_style (Gg.Color.v ~alpha:0.20 0.20 0.45 0.85))
          (P.Polygon pi1_patch)
      ; Draw.static
          ~layer:1
          ~style:(plane_style (Gg.Color.v ~alpha:0.20 0.20 0.45 0.85))
          (P.Polygon pi2_patch)
      ; Draw.static
          ~layer:1
          ~style:(plane_style (Gg.Color.v ~alpha:0.16 0.55 0.55 0.85))
          (P.Polygon pi3_patch)
      ; Draw.static
          ~layer:3
          ~style:(Draw.outline ~width:3.2 (Gg.Color.v 0.75 0.10 0.10))
          line_prim
      ]
  in
  let scene = Draw.create camera entities in
  let overlay =
    Vg.Picture.group
      [ disc ~color:(Gg.Color.v 0.75 0.10 0.10) ~radius:5. camera corner
      ; disc ~color:ink ~radius:4. camera on_line
      ; disc ~color:muted ~radius:4. camera (p3 0.55 0.55 1.45)
      ; label ~color:ink ~size:13. ~dx:9. ~dy:(-6.) camera corner "p = pi1 ^ pi2 ^ pi3"
      ; label ~color:ink ~size:13. ~dx:9. ~dy:(-6.) camera on_line "on pi1 ^ pi2"
      ; label ~color:muted ~size:13. ~dx:9. ~dy:(-6.) camera (p3 0.55 0.55 1.45) "q"
      ; label ~color:muted ~size:13. ~dx:6. ~dy:(-6.) camera (p3 0. 1.45 1.2) "pi1"
      ; label ~color:muted ~size:13. ~dx:6. ~dy:(-6.) camera (p3 1.45 0. 1.2) "pi2"
      ; label ~color:muted ~size:13. ~dx:6. ~dy:(-10.) camera (p3 1.25 1.25 1.9) "pi3"
      ; label2d
          ~color:muted
          ~size:13.
          ~x:22.
          ~y:34.
          "two planes meet in a line, three planes in a point"
      ]
  in
  save "pga-meet" (Vg.Picture.group [ scene_picture scene; overlay ])

(* -------------------------------------------------------------------------- *)
(* Figure 4: ScLERP along a screw                                             *)
(* -------------------------------------------------------------------------- *)

let figure_slerp () =
  let camera =
    C.create
      ~eye:(p3 2.35 (-3.05) 1.95)
      ~target:(p3 0.78 0.0 0.35)
      ~up:(p3 0. 0. 1.)
      ~fov_y:(Float.pi /. 4.9)
      ~width
      ~height
      ()
  in
  let v3 x y z = Nx.create Nx.float32 [| 3 |] [| x; y; z |] in
  let sc x = Nx.scalar Nx.float32 x in
  let axis = v3 0. 0. 1. in
  let m1 =
    Motor.mul
      (Motor.translate3 1.3 0. 0.55)
      (Motor.rotate ~axis ~angle:(sc (Float.pi *. 0.85)))
  in
  let m0 = Motor.one in
  let pose t = Motor.slerp m0 m1 (sc t) in
  let origin t =
    Cga.apply_motor (pose t) (Cga.point3 0. 0. 0.)
    |> Cga.coords
    |> Nx.to_array
    |> fun a -> p3 a.(0) a.(1) a.(2)
  in
  let frame_at t size =
    let m = pose t in
    let dir x y z =
      Cga.apply_motor m (Cga.point3 x y z)
      |> Cga.coords
      |> Nx.to_array
      |> fun a -> p3 a.(0) a.(1) a.(2)
    in
    P.Frame
      { origin = origin t; x = dir size 0. 0.; y = dir 0. size 0.; z = dir 0. 0. size }
  in
  let arc = List.init 41 (fun i -> origin (float_of_int i /. 40.)) in
  let steps = List.init 9 (fun i -> float_of_int i /. 8.) in
  let sphere t =
    let o = origin t in
    P.Sphere { center = p3 o.P.x o.P.y (o.P.z +. 0.42); radius = 0.11 }
  in
  let alphas = List.init 9 (fun i -> 0.14 +. (0.72 *. float_of_int i /. 8.)) in
  let ghost =
    { Draw.default with
      fill = Some (Gg.Color.v ~alpha:0.10 0.30 0.34 0.35)
    ; stroke = Some (Gg.Color.v ~alpha:0.20 0.25 0.30 0.25)
    ; width = 1.0
    }
  in
  let spheres =
    List.map2
      (fun t a ->
         Draw.static
           ~layer:2
           ~style:
             { Draw.default with
               fill = Some (Gg.Color.v ~alpha:a 0.55 0.20 0.65)
             ; stroke = Some (Gg.Color.v ~alpha:0.75 0.35 0.10 0.50)
             ; width = 1.2
             }
           (sphere t))
      steps
      alphas
  in
  let entities =
    [ ground ~size:2.3 ground_style ]
    @ grid ~layer:0 ~size:2.0 ~step:0.5
    @ [ Draw.static ~layer:1 ~style:ghost (frame_at 0. 0.5)
      ; Draw.static ~layer:1 ~style:ghost (frame_at 1. 0.5)
      ]
    @ spheres
  in
  let scene = Draw.create camera entities in
  let overlay =
    Vg.Picture.group
      ([ polyline3
           ~color:(Gg.Color.v ~alpha:0.55 0.35 0.30 0.45)
           ~width:1.4
           ~dash:[ 5.; 5. ]
           camera
           arc
       ; disc ~color:(Gg.Color.v 0.55 0.20 0.65) ~radius:4. camera (origin 0.)
       ; disc ~color:(Gg.Color.v 0.85 0.30 0.15) ~radius:4. camera (origin 1.)
       ]
       @ List.map
           (fun t ->
              label
                ~color:ink
                ~size:12.
                ~dx:14.
                ~dy:(-10.)
                camera
                (origin t)
                (Printf.sprintf "t=%.2g" t))
           [ 0.; 0.5; 1. ]
       @ [ label2d
             ~color:muted
             ~size:13.
             ~x:22.
             ~y:34.
             "ScLERP sweeps one screw motion from M0 to M1"
         ])
  in
  save "slerp-screw" (Vg.Picture.group [ scene_picture scene; overlay ])

(* -------------------------------------------------------------------------- *)
(* Figure 5: a six-joint kinematic chain                                      *)
(* -------------------------------------------------------------------------- *)

let lengths = [| 0.65; 0.55; 0.45; 0.35; 0.30; 0.25 |]
let angles = [| 0.55; -0.75; 0.95; 0.35; -0.85; 0.65 |]

let figure_arm () =
  let camera =
    C.create
      ~eye:(p3 3.75 (-3.85) 2.95)
      ~target:(p3 0. 0. 1.02)
      ~up:(p3 0. 0. 1.)
      ~fov_y:(Float.pi /. 4.6)
      ~width
      ~height
      ()
  in
  let v3 x y z = Nx.create Nx.float32 [| 3 |] [| x; y; z |] in
  let sc x = Nx.scalar Nx.float32 x in
  let axis i = if i land 1 = 0 then v3 0. 0. 1. else v3 0. 1. 0. in
  let motors =
    let rec go i m acc =
      if i >= Array.length lengths
      then List.rev (m :: acc)
      else (
        let r = Motor.rotate ~axis:(axis i) ~angle:(sc angles.(i)) in
        let link = Motor.translate3 0. 0. lengths.(i) in
        go (i + 1) (Motor.mul m (Motor.mul r link)) (m :: acc))
    in
    go 0 Motor.one []
  in
  let end_motor = List.nth motors (Array.length lengths) in
  let pos m =
    Cga.apply_motor m (Cga.point3 0. 0. 0.)
    |> Cga.coords
    |> Nx.to_array
    |> fun a -> p3 a.(0) a.(1) a.(2)
  in
  let tips = List.map pos motors in
  let link_style =
    { Draw.default with
      fill = None
    ; stroke = Some (Gg.Color.v 0.20 0.24 0.30)
    ; width = 13.0
    }
  in
  let joint_style =
    { Draw.default with
      fill = Some (Gg.Color.v ~alpha:0.95 0.95 0.75 0.20)
    ; stroke = Some (Gg.Color.v 0.50 0.30 0.05)
    ; width = 1.1
    }
  in
  let entities =
    [ ground ~size:2.3 ground_style ]
    @ grid ~layer:0 ~size:2.0 ~step:0.5
    @ List.init (Array.length lengths) (fun i ->
      Draw.static
        ~layer:2
        ~style:link_style
        (P.Segment (List.nth tips i, List.nth tips (i + 1))))
    @ List.mapi
        (fun i p ->
           Draw.static
             ~layer:3
             ~style:joint_style
             (P.Sphere { center = p; radius = (if i = 0 then 0.12 else 0.085) }))
        tips
    @ [ Draw.static
          ~layer:4
          ~style:Draw.default
          (let o = List.nth tips 6 in
           let dir x y z =
             Cga.apply_motor end_motor (Cga.point3 x y z)
             |> Cga.coords
             |> Nx.to_array
             |> fun a -> p3 a.(0) a.(1) a.(2)
           in
           P.Frame
             { origin = o; x = dir 0.42 0. 0.; y = dir 0. 0.42 0.; z = dir 0. 0. 0.42 })
      ]
  in
  let scene = Draw.create camera entities in
  let ee = List.nth tips 6 in
  let overlay =
    Vg.Picture.group
      [ label ~color:ink ~size:13. ~dx:16. ~dy:(-16.) camera ee "end effector"
      ; label ~color:muted ~size:13. ~dx:14. ~dy:22. camera (List.nth tips 0) "base"
      ; label2d
          ~color:muted
          ~size:13.
          ~x:22.
          ~y:34.
          "six motors, one product: M = M1 M2 ... M6"
      ]
  in
  save "motor-arm" (Vg.Picture.group [ scene_picture scene; overlay ])

let () =
  figure_e3 ();
  figure_cga ();
  figure_pga ();
  figure_slerp ();
  figure_arm ()
