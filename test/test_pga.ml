open Geom

let failf fmt = Printf.ksprintf failwith fmt
let check name b = if not b then failf "FAIL %s" name
let approx a b = Float.abs (a -. b) < 1e-4
let scalar v = Nx.item [] v

let coords p =
  let a = Pga.point_coords p |> Nx.to_array in
  a.(0), a.(1), a.(2)

let check3 name p (x, y, z) =
  let a, b, c = coords p in
  check name (approx a x && approx b y && approx c z)

let () =
  let p = Pga.point3 1. 2. 3. in
  check3 "pga coords" p (1., 2., 3.);
  check "pga weight" (approx (scalar (Pga.weight p)) 1.);
  let pi = Pga.plane3 0. 0. 1. 1. in
  check "pga plane on" (scalar (Pga.point_on_plane (Pga.point3 0. 0. 1.) pi));
  check "pga plane off" (not (scalar (Pga.point_on_plane (Pga.point3 0. 0. 0.) pi)));
  let lv = Pga.line_through (Pga.point3 1. 2. 3.) (Pga.point3 1. 2. 4.) in
  let ld = Pga.line_direction lv |> Nx.to_array in
  let lm = Pga.line_moment lv |> Nx.to_array in
  check
    "pga line direction"
    (approx (Float.abs ld.(2)) 1. && approx ld.(0) 0. && approx ld.(1) 0.);
  check
    "pga line moment"
    (approx lm.(0) (2. *. ld.(2)) && approx lm.(1) (-1. *. ld.(2)) && approx lm.(2) 0.);
  let a = Pga.point3 0. 0. 0.
  and b = Pga.point3 1. 0. 0. in
  let l = Pga.line_through a b in
  check "pga line on" (scalar (Pga.point_on_line (Pga.point3 0.5 0. 0.) l));
  check "pga line off" (not (scalar (Pga.point_on_line (Pga.point3 0. 1. 0.) l)));
  let parallel = Pga.meet pi l in
  check "pga parallel meet ideal" (scalar (Pga.is_ideal parallel));
  let zline = Pga.line_through (Pga.point3 0. 0. 0.) (Pga.point3 0. 0. 1.) in
  let m = Pga.meet (Pga.plane3 0. 0. 1. 0.5) zline in
  check3 "pga meet plane line" m (0., 0., 0.5);
  let l2 = Pga.line_of_planes (Pga.plane3 1. 0. 0. 0.) (Pga.plane3 0. 1. 0. 0.) in
  check "pga line in plane" (scalar (Pga.line_in_plane l2 (Pga.plane3 1. 0. 0. 0.)));
  check
    "pga ideal"
    (scalar
       (Pga.is_ideal (Pga.ideal_point (Nx.create Pga.dtype [| 3 |] [| 1.; 0.; 0. |]))));
  let t = Motor.translate3 0. 0. 1. in
  let pm = Pga.apply_motor t p in
  check3 "pga motor" pm (1., 2., 4.);
  let r =
    Motor.rotate
      ~axis:(Nx.create Motor.dtype [| 3 |] [| 0.; 0.; 1. |])
      ~angle:(Nx.scalar Motor.dtype (Float.pi /. 2.))
  in
  check3 "pga rotate" (Pga.apply_motor r (Pga.point3 1. 0. 0.)) (0., 1., 0.);
  let r =
    E3.rotor
      ~axis:(Nx.create E3.dtype [| 3 |] [| 0.; 0.; 1. |])
      ~angle:(Nx.scalar E3.dtype (Float.pi /. 2.))
  in
  let v = E3.apply_rotor r (E3.vector3 1. 0. 0.) in
  let vv = E3.to_vector v |> Nx.to_array in
  check "e3 rotor" (approx vv.(0) 0. && approx vv.(1) 1. && approx vv.(2) 0.);
  Printf.printf "PGA/E3 OK\n"
