open Geom

let failf fmt = Printf.ksprintf failwith fmt
let check name b = if not b then failf "FAIL %s" name
let approx a b = Float.abs (a -. b) < 1e-4

let arr3 v =
  let a = Cga.coords v |> Nx.to_array in
  a.(0), a.(1), a.(2)

let check3 name v (x, y, z) =
  let a, b, c = arr3 v in
  check name (approx a x && approx b y && approx c z)

let scalar v = Nx.item [] v

let cross a b =
  [| (a.(1) *. b.(2)) -. (a.(2) *. b.(1))
   ; (a.(2) *. b.(0)) -. (a.(0) *. b.(2))
   ; (a.(0) *. b.(1)) -. (a.(1) *. b.(0))
  |]

let vnorm a = sqrt ((a.(0) *. a.(0)) +. (a.(1) *. a.(1)) +. (a.(2) *. a.(2)))

let normalize a =
  let s = vnorm a in
  [| a.(0) /. s; a.(1) /. s; a.(2) /. s |]

let add a b = [| a.(0) +. b.(0); a.(1) +. b.(1); a.(2) +. b.(2) |]
let scale a s = [| a.(0) *. s; a.(1) *. s; a.(2) *. s |]
let dot a b = (a.(0) *. b.(0)) +. (a.(1) *. b.(1)) +. (a.(2) *. b.(2))

let () =
  let p = Cga.point3 1. 2. 3. in
  check3 "coords roundtrip" p (1., 2., 3.);
  check "point null" (approx (scalar (Cga.norm2 p)) 0.);
  let origin = Cga.point3 0. 0. 0. in
  check "distance origin" (approx (scalar (Cga.distance p origin)) (sqrt 14.));
  let s =
    Cga.sphere_of_point ~center:(Cga.point3 1. 0. 0.) ~radius:(Nx.scalar Cga.dtype 2.)
  in
  check "sphere radius2" (approx (scalar (Cga.sphere_radius2 s)) 4.);
  check3 "sphere center" (Cga.sphere_center s) (1., 0., 0.);
  check "sphere incident on" (scalar (Cga.incident (Cga.point3 3. 0. 0.) s));
  check "sphere incident off" (not (scalar (Cga.incident (Cga.point3 3.1 0. 0.) s)));
  let s2 =
    Cga.sphere
      ~center:(Nx.create Cga.dtype [| 3 |] [| 1.; 0.; 0. |])
      ~radius:(Nx.scalar Cga.dtype 2.)
  in
  check "sphere coords radius2" (approx (scalar (Cga.sphere_radius2 s2)) 4.);
  check "sphere round" (scalar (Cga.is_round s2));
  let pi =
    Cga.plane
      ~normal:(Nx.create Cga.dtype [| 3 |] [| 0.; 0.; 1. |])
      ~offset:(Nx.scalar Cga.dtype 1.)
  in
  check "plane flat" (scalar (Cga.is_flat pi));
  check "plane on" (scalar (Cga.incident (Cga.point3 0. 0. 1.) pi));
  check "plane off" (not (scalar (Cga.incident (Cga.point3 0. 0. 0.) pi)));
  let a = Cga.point3 0. 0. 0.
  and b = Cga.point3 1. 0. 0. in
  let l = Cga.line a b in
  check "line flat" (scalar (Cga.is_flat l));
  check "line on" (scalar (Cga.incident (Cga.point3 0.5 0. 0.) l));
  check "line off" (not (scalar (Cga.incident (Cga.point3 0. 1. 0.) l)));
  let c1 = Cga.point3 1. 0. 0.
  and c2 = Cga.point3 0. 1. 0.
  and c3 = Cga.point3 (-1.) 0. 0. in
  let c = Cga.circle c1 c2 c3 in
  check "circle round" (scalar (Cga.is_round c));
  check "circle on" (scalar (Cga.incident c2 c));
  check "circle off" (not (scalar (Cga.incident (Cga.point3 0. 0. 1.) c)));
  let px =
    Cga.plane
      ~normal:(Nx.create Cga.dtype [| 3 |] [| 1.; 0.; 0. |])
      ~offset:(Nx.scalar Cga.dtype 0.)
  in
  let py =
    Cga.plane
      ~normal:(Nx.create Cga.dtype [| 3 |] [| 0.; 1.; 0. |])
      ~offset:(Nx.scalar Cga.dtype 0.)
  in
  let l2 = Cga.meet px py in
  check "meet planes on" (scalar (Cga.incident (Cga.point3 0. 0. 1.) l2));
  check "meet planes off" (not (scalar (Cga.incident (Cga.point3 1. 0. 0.) l2)));
  let s1 =
    Cga.sphere
      ~center:(Nx.create Cga.dtype [| 3 |] [| 0.; 0.; 0. |])
      ~radius:(Nx.scalar Cga.dtype 1.)
  in
  let s2 =
    Cga.sphere
      ~center:(Nx.create Cga.dtype [| 3 |] [| 1.; 0.; 0. |])
      ~radius:(Nx.scalar Cga.dtype 1.)
  in
  let c2 = Cga.meet s1 s2 in
  check "meet spheres on" (scalar (Cga.incident (Cga.point3 0.5 (sqrt 0.75) 0.) c2));
  check "meet spheres off" (not (scalar (Cga.incident (Cga.point3 0. 0. 0.) c2)));
  let dil = Cga.dilator (Nx.scalar Cga.dtype (log 2.)) in
  check3 "dilation point" (Cga.sandwich dil (Cga.point3 1. 0. 0.)) (2., 0., 0.);
  let sp =
    Cga.sphere_of_point ~center:(Cga.point3 1. 0. 0.) ~radius:(Nx.scalar Cga.dtype 0.5)
  in
  let spd = Cga.sandwich dil sp in
  check3 "dilation sphere center" (Cga.sphere_center spd) (2., 0., 0.);
  check "dilation sphere radius" (approx (scalar (Cga.sphere_radius2 spd)) 1.);
  let tv = Cga.transversion (Nx.create Cga.dtype [| 3 |] [| 0.5; 0.; 0. |]) in
  let ptv = Cga.sandwich tv (Cga.point3 1. 0. 0.) in
  check3 "transversion" ptv (1.5 /. 2.25, 0., 0.);
  let line_a = Cga.point3 1. 2. 3.
  and line_b = Cga.point3 1. 2. 4. in
  let line = Cga.line line_a line_b in
  let d = Cga.line_direction line |> Nx.to_array in
  let m = Cga.line_moment line |> Nx.to_array in
  let dz = d.(2) in
  check "line direction" (approx (Float.abs dz) 1. && approx d.(0) 0. && approx d.(1) 0.);
  check
    "line moment"
    (approx m.(0) (2. *. dz) && approx m.(1) (-1. *. dz) && approx m.(2) 0.);
  let cup =
    Cga.circle (Cga.point3 1. 0. 0.) (Cga.point3 0. 1. 0.) (Cga.point3 (-1.) 0. 0.)
  in
  check3 "circle center unit" (Cga.circle_center cup) (0., 0., 0.);
  check "circle radius unit" (approx (scalar (Cga.circle_radius2 cup)) 1.);
  let nc = Cga.circle_normal cup |> Nx.to_array in
  check "circle normal unit" (approx (Float.abs nc.(2)) 1.);
  let csh =
    Cga.circle (Cga.point3 3. 2. 1.) (Cga.point3 1. 4. 1.) (Cga.point3 (-1.) 2. 1.)
  in
  check3 "shifted circle center" (Cga.circle_center csh) (1., 2., 1.);
  check "shifted circle radius" (approx (scalar (Cga.circle_radius2 csh)) 4.);
  let cc = [| 0.5; 0.5; 0.5 |] in
  let nn = normalize [| 1.; 1.; 1. |] in
  let uu = normalize (cross nn [| 1.; 0.; 0. |]) in
  let vv = cross nn uu in
  let r = 1. in
  let pt t =
    let a = cc
    and b = add (scale uu (r *. cos t)) (scale vv (r *. sin t)) in
    let q = add a b in
    Cga.point3 q.(0) q.(1) q.(2)
  in
  let ct = Cga.circle (pt 0.) (pt (2. *. Float.pi /. 3.)) (pt (4. *. Float.pi /. 3.)) in
  check3 "tilted circle center" (Cga.circle_center ct) (0.5, 0.5, 0.5);
  check "tilted circle radius" (approx (scalar (Cga.circle_radius2 ct)) 1.);
  let nct = Cga.circle_normal ct |> Nx.to_array in
  check "tilted circle normal" (approx (Float.abs (dot nct nn)) 1.);
  let t = Motor.translate3 0. 0. 1. in
  let moved = Cga.apply_motor t p in
  check3 "translate" moved (1., 2., 4.);
  check "translate null" (approx (scalar (Cga.norm2 moved)) 0.);
  let r =
    Motor.rotate
      ~axis:(Nx.create Motor.dtype [| 3 |] [| 0.; 0.; 1. |])
      ~angle:(Nx.scalar Motor.dtype (Float.pi /. 2.))
  in
  let rotated = Cga.apply_motor r (Cga.point3 1. 0. 0.) in
  check3 "rotate" rotated (0., 1., 0.);
  let tr = Motor.exp (Nx.create Motor.dtype [| 6 |] [| 0.; 0.; 0.; 0.; 0.; 1. |]) in
  check3 "exp translate" (Cga.apply_motor tr origin) (0., 0., 1.);
  let rot =
    Motor.exp (Nx.create Motor.dtype [| 6 |] [| 0.; 0.; Float.pi /. 2.; 0.; 0.; 0. |])
  in
  check3 "exp rotate" (Cga.apply_motor rot (Cga.point3 1. 0. 0.)) (0., 1., 0.);
  let m = Nx.create Motor.dtype [| 3; 3 |] [| 0.; -1.; 0.; 1.; 0.; 0.; 0.; 0.; 1. |] in
  let rm = Motor.of_rotation m in
  check3 "of_rotation" (Cga.apply_motor rm (Cga.point3 1. 0. 0.)) (0., 1., 0.);
  let pts = Nx.create Cga.dtype [| 2; 3 |] [| 1.; 2.; 3.; -1.; 0.; 0.5 |] in
  let batch = Cga.point pts in
  check "batch shape" (Nx.shape (Cga.coords batch) = [| 2; 3 |]);
  let ca = Cga.coords batch |> Nx.to_array in
  check
    "batch coords"
    (approx ca.(0) 1.
     && approx ca.(1) 2.
     && approx ca.(2) 3.
     && approx ca.(3) (-1.)
     && approx ca.(4) 0.
     && approx ca.(5) 0.5);
  Printf.printf "CGA OK\n"
