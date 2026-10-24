(* The numerical tour quoted by ../README.md.

   Run from geom/:  dune exec examples/tour.exe

   The values printed here are the values the README shows. *)

open Geom

let pi = 4. *. atan 1.
let v3 x y z = Nx.create Nx.float32 [| 3 |] [| x; y; z |]
let v6 a b c d e f = Nx.create Nx.float32 [| 6 |] [| a; b; c; d; e; f |]
let sc x = Nx.scalar Nx.float32 x
let item x = Nx.item [] x
let bool b = if item b then "true" else "false"

(* A compact printer for the nonzero blades of a multivector, in the
   orthonormal basis e1, e2, ... of the model. *)
let term x b =
  if Blade.is_scalar b
  then Printf.sprintf "%.4g" x
  else Printf.sprintf "%.4g %s" x (Blade.to_string b)

let show ~n a =
  let blades = Blade.all n in
  let terms =
    Array.to_list a
    |> List.mapi (fun i x -> blades.(i), x)
    |> List.filter (fun (_, x) -> Float.abs x > 1e-5)
  in
  match terms with
  | [] -> "0"
  | (b, x) :: rest ->
    let sign x = if x < 0. then " - " else " + " in
    List.fold_left
      (fun acc (b, x) -> acc ^ sign x ^ term (Float.abs x) b)
      (if x < 0. then "-" ^ term (-.x) b else term x b)
      rest

let clean x = if Float.abs x < 1e-6 then 0. else x

let vec v =
  Array.to_list (Nx.to_array v)
  |> List.map (fun x -> Printf.sprintf "%.4g" (clean x))
  |> String.concat ", "
  |> Printf.sprintf "[%s]"

let header s = Printf.printf "\n### %s\n" s
let out label s = Printf.printf "%-36s %s\n" (label ^ ":") s

(* -------------------------------------------------------------------------- *)
(* 1. The geometric product in E3                                             *)
(* -------------------------------------------------------------------------- *)

let section_e3 () =
  header "E3: the geometric product";
  let e1 = E3.vector3 1. 0. 0.
  and e2 = E3.vector3 0. 1. 0. in
  out "show (gp e1 e2)" (show ~n:3 (E3.to_array (E3.gp e1 e2)));
  out "show (gp e1 e1)" (show ~n:3 (E3.to_array (E3.gp e1 e1)));
  out
    "show (gp e1 e2 e1 e2)"
    (show ~n:3 (E3.to_array (E3.gp (E3.gp e1 e2) (E3.gp e1 e2))));
  let u = E3.vector3 1. 0. 0. in
  let v = E3.vector3 (1. /. sqrt 2.) (1. /. sqrt 2.) 0. in
  out "u . v" (Printf.sprintf "%.4g" (item (E3.scalar_prod u v)));
  out "show (op u v)" (show ~n:3 (E3.to_array (E3.op u v)));
  out "show (gp u v)" (show ~n:3 (E3.to_array (E3.gp u v)));
  let in_plane = E3.vector3 1. 1. 0. in
  let e12 = E3.op e1 e2 in
  out "show (op (vector 1 1 0) e12)" (show ~n:3 (E3.to_array (E3.op in_plane e12)));
  out "show (op e3 e12)" (show ~n:3 (E3.to_array (E3.op E3.e3 e12)));
  out "show (inner e1 e12)" (show ~n:3 (E3.to_array (E3.inner e1 e12)));
  out "show (inner e2 e12)" (show ~n:3 (E3.to_array (E3.inner e2 e12)));
  let r90 = E3.rotor ~axis:(v3 0. 0. 1.) ~angle:(sc (pi /. 2.)) in
  out "show r90" (show ~n:3 (E3.to_array r90));
  out "e1 rotated 90 about z" (vec (E3.to_vector (E3.apply_rotor r90 e1)));
  let r45 = E3.rotor ~axis:(v3 0. 0. 1.) ~angle:(sc (pi /. 4.)) in
  let r135 = E3.gp r90 r45 in
  out "show (r90 * r45)" (show ~n:3 (E3.to_array r135));
  out "e1 rotated 135 about z" (vec (E3.to_vector (E3.apply_rotor r135 e1)))

(* -------------------------------------------------------------------------- *)
(* 2. CGA: points, distances, spheres, planes                                 *)
(* -------------------------------------------------------------------------- *)

let section_cga () =
  header "CGA: points and distances";
  let x = Cga.point3 1. 1. 0.
  and y = Cga.point3 4. 5. 0. in
  out "norm2 x" (Printf.sprintf "%.4g" (item (Cga.norm2 x)));
  out "scalar_prod einf x" (Printf.sprintf "%.4g" (item (Cga.scalar_prod Cga.einf x)));
  out "distance2 x y" (Printf.sprintf "%.4g" (item (Cga.distance2 x y)));
  out "distance x y" (Printf.sprintf "%.4g" (item (Cga.distance x y)));
  header "CGA: spheres";
  let s = Cga.sphere ~center:(v3 0. 0. 0.) ~radius:(sc 2.) in
  out "incident (point 2 0 0) s" (bool (Cga.incident (Cga.point3 2. 0. 0.) s));
  out "incident (point 1 0 0) s" (bool (Cga.incident (Cga.point3 1. 0. 0.) s));
  out "sphere_radius2 s" (Printf.sprintf "%.4g" (item (Cga.sphere_radius2 s)));
  out "sphere_center s" (vec (Cga.coords (Cga.sphere_center s)));
  out
    "scalar_prod (point 3 0 0) s"
    (Printf.sprintf "%.4g" (item (Cga.scalar_prod (Cga.point3 3. 0. 0.) s)));
  header "CGA: planes";
  let pi = Cga.plane ~normal:(v3 0. 0. 1.) ~offset:(sc 1.) in
  out "incident (point 4 5 1) pi" (bool (Cga.incident (Cga.point3 4. 5. 1.) pi));
  out
    "scalar_prod (point 4 5 3) pi"
    (Printf.sprintf "%.4g" (item (Cga.scalar_prod (Cga.point3 4. 5. 3.) pi)));
  out "is_flat pi" (bool (Cga.is_flat pi));
  out "is_round s" (bool (Cga.is_round s));
  header "CGA: objects from points";
  let a = Cga.point3 1. 0. 0.
  and b = Cga.point3 0. 1. 0.
  and c = Cga.point3 (-1.) 0. 0. in
  let circle = Cga.circle a b c in
  out "circle a b c radius2" (Printf.sprintf "%.4g" (item (Cga.circle_radius2 circle)));
  out "circle a b c normal" (vec (Cga.circle_normal circle));
  out "circle a b c center" (vec (Cga.coords (Cga.circle_center circle)));
  out
    "incident (point 0 -1 0) circle"
    (bool (Cga.incident (Cga.point3 0. (-1.) 0.) circle));
  out "incident (point 0 0 1) circle" (bool (Cga.incident (Cga.point3 0. 0. 1.) circle));
  let line = Cga.line (Cga.point3 1. 0. 1.) (Cga.point3 (-1.) 0. 1.) in
  out "line direction" (vec (Cga.line_direction line));
  out "line moment" (vec (Cga.line_moment line));
  out "incident (point 0 0 1) line" (bool (Cga.incident (Cga.point3 0. 0. 1.) line));
  out "incident (point 0 1 1) line" (bool (Cga.incident (Cga.point3 0. 1. 1.) line));
  let pp = Cga.point_pair (Cga.point3 0. 0. 0.) (Cga.point3 2. 0. 0.) in
  out
    "point_pair squared"
    (Printf.sprintf "%.4g" (item (Cga.coeff (Cga.gp pp pp) (Blade.of_int 0))))

(* -------------------------------------------------------------------------- *)
(* 3. PGA: meet and join                                                      *)
(* -------------------------------------------------------------------------- *)

let section_pga () =
  header "PGA: meet and join";
  let pi1 = Pga.plane3 1. 0. 0. 0.
  and pi2 = Pga.plane3 0. 1. 0. 0. in
  let l = Pga.meet pi1 pi2 in
  out "meet pi1 pi2 direction" (vec (Pga.line_direction l));
  out "point_on_line (0,0,2) l" (bool (Pga.point_on_line (Pga.point3 0. 0. 2.) l));
  out "point_on_line (1,0,2) l" (bool (Pga.point_on_line (Pga.point3 1. 0. 2.) l));
  out "point_coords (0,3,1)" (vec (Pga.point_coords (Pga.point3 0. 3. 1.)));
  out "point_on_plane (0,3,1) pi1" (bool (Pga.point_on_plane (Pga.point3 0. 3. 1.) pi1));
  let p = Pga.point3 0. 0. 2.
  and q = Pga.point3 1. 0. 2. in
  let l2 = Pga.join p q in
  out "join p q direction" (vec (Pga.line_direction l2));
  let pi3 = Pga.plane3 0. 0. 1. 2. in
  let corner = Pga.meet (Pga.meet pi1 pi2) pi3 in
  out "meet (meet pi1 pi2) pi3" (vec (Pga.point_coords corner))

(* -------------------------------------------------------------------------- *)
(* 4. Motors                                                                  *)
(* -------------------------------------------------------------------------- *)

let section_motor () =
  header "Motors";
  let rz = Motor.rotate ~axis:(v3 0. 0. 1.) ~angle:(sc (pi /. 2.)) in
  let tz = Motor.translate3 0. 0. 1. in
  let m = Motor.mul tz rz in
  out "to_array rz" (vec (Nx.reshape [| 8 |] rz));
  out "to_array (translate 0 0 1)" (vec (Nx.reshape [| 8 |] tz));
  out
    "coords (m (point 1 0 0))"
    (vec (Cga.coords (Cga.apply_motor m (Cga.point3 1. 0. 0.))));
  let inv = Motor.inverse m in
  out
    "coords (m^-1 (m x))"
    (vec (Cga.coords (Cga.apply_motor inv (Cga.apply_motor m (Cga.point3 1. 0. 0.)))));
  header "Motors: exponential, logarithm, interpolation";
  let twist = v6 0. 0. (pi /. 2.) 0. 0. 1. in
  let m = Motor.exp twist in
  out "log (exp twist)" (vec (Motor.log m));
  out
    "coords (m (point 1 0 0))"
    (vec (Cga.coords (Cga.apply_motor m (Cga.point3 1. 0. 0.))));
  let m0 = Motor.one in
  let m1 =
    Motor.mul
      (Motor.translate3 1.3 0. 0.55)
      (Motor.rotate ~axis:(v3 0. 0. 1.) ~angle:(sc (pi *. 0.85)))
  in
  let positions =
    List.init 5 (fun i ->
      let t = float_of_int i /. 4. in
      Motor.slerp m0 m1 (sc t)
      |> fun mi -> Cga.apply_motor mi (Cga.point3 0. 0. 0.) |> Cga.coords |> vec)
  in
  List.iteri
    (fun i s -> out (Printf.sprintf "slerp t=%.2f" (float_of_int i /. 4.)) s)
    positions;
  header "Motors: forward kinematics";
  let lengths = [| 0.65; 0.55; 0.45; 0.35; 0.30; 0.25 |] in
  let q = [| 0.55; -0.75; 0.95; 0.35; -0.85; 0.65 |] in
  let axis i = if i land 1 = 0 then v3 0. 0. 1. else v3 0. 1. 0. in
  let joint i =
    Motor.mul
      (Motor.rotate ~axis:(axis i) ~angle:(sc q.(i)))
      (Motor.translate3 0. 0. lengths.(i))
  in
  let rec arm i =
    if i = Array.length lengths then Motor.one else Motor.mul (joint i) (arm (i + 1))
  in
  out
    "end effector position"
    (vec (Cga.coords (Cga.apply_motor (arm 0) (Cga.point3 0. 0. 0.))))

(* -------------------------------------------------------------------------- *)
(* 5. Batching and autodiff                                                   *)
(* -------------------------------------------------------------------------- *)

let section_stack () =
  header "Batching";
  let centers =
    Nx.create Nx.float32 [| 4; 3 |] [| 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]
  in
  let xs = Cga.point centers in
  out
    "shape (point centers)"
    (Printf.sprintf
       "[%s]"
       (Array.to_list (Nx.shape xs) |> List.map string_of_int |> String.concat "; "));
  out "distance2 xs origin" (vec (Cga.distance2 xs (Cga.point3 0. 0. 0.)));
  header "Autodiff";
  let f p = Cga.distance2 (Cga.point p) (Cga.point3 3. 4. 0.) in
  let value, grad = Rune.value_and_grad' f (v3 0. 0. 0.) in
  out "value" (Printf.sprintf "%.4g" (item value));
  out "grad" (vec grad);
  let grad = Rune.grad' f (v3 1. 1. 1.) in
  out "grad at (1,1,1)" (vec grad);
  header "Compilation";
  let jit = Rune.jit' f in
  out "jit at (1,1,1)" (Printf.sprintf "%.4g" (item (jit (v3 1. 1. 1.))))

let () =
  section_e3 ();
  section_cga ();
  section_pga ();
  section_motor ();
  section_stack ()
