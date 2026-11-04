(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
module Format = Stdlib.Format

type p3 =
  { x : float
  ; y : float
  ; z : float
  }

type colour =
  { r : float
  ; g : float
  ; b : float
  ; a : float
  }

type shape =
  | Sphere of
      { centre : p3
      ; radius : float
      }
  | Capsule of
      { a : p3
      ; b : p3
      ; radius : float
      }
  | Box of
      { centre : p3
      ; axes : p3 * p3 * p3
      ; half : float * float * float
      }
  | Cylinder of
      { centre : p3
      ; axis : p3
      ; radius : float
      ; half_height : float
      }
  | Ellipsoid of
      { centre : p3
      ; axes : p3 * p3 * p3
      ; radii : float * float * float
      }
  | Plane of
      { normal : p3
      ; offset : float
      ; half : float * float
      }
  | Unsupported of string

type geom =
  { name : string
  ; shape : shape
  ; colour : colour
  }

type frame =
  { origin : p3
  ; x : p3
  ; y : p3
  ; z : p3
  }

type body =
  { name : string
  ; frame : frame
  }

type contact =
  { point : p3
  ; normal : p3
  ; tangents : p3 * p3
  ; depth : float
  ; dim : int
  ; force : float
  ; tangent_force : p3
  ; friction : float
  }

type arrow =
  { from : p3
  ; vector : p3
  }

type jnt =
  | Hinge
  | Slide
  | Ball
  | Free

type inertia =
  { centre : p3
  ; axes : p3 * p3 * p3
  ; half : float * float * float
  }

type joint =
  { name : string
  ; kind : jnt
  ; anchor : p3
  ; axis : p3
  }

type energy =
  { kinetic : float
  ; potential : float
  }

type t =
  { time : float
  ; bodies : body list
  ; geoms : geom list
  ; contacts : contact list
  ; forces : arrow list
  ; inertia : inertia list
  ; com : p3 list
  ; joints : joint list
  ; velocity : arrow list
  ; energy : energy option
  }

(* {1 Vector arithmetic} *)

let add (a : p3) (b : p3) = { x = a.x +. b.x; y = a.y +. b.y; z = a.z +. b.z }
let sub (a : p3) (b : p3) = { x = a.x -. b.x; y = a.y -. b.y; z = a.z -. b.z }
let smul (a : p3) s = { x = a.x *. s; y = a.y *. s; z = a.z *. s }
let dot (a : p3) (b : p3) = (a.x *. b.x) +. (a.y *. b.y) +. (a.z *. b.z)

(* {1 Translation from the engine}

   Everything here is host data already: the pose arrays of {!Mjc.Data} are
   read once per frame, which is the one copy the viewer is allowed. *)

let point a i = { x = a.((3 * i) + 0); y = a.((3 * i) + 1); z = a.((3 * i) + 2) }

(* The [k]-th column of a row-major 3×3 block: a world direction. *)
let axis m i k = { x = m.((9 * i) + k); y = m.((9 * i) + 3 + k); z = m.((9 * i) + 6 + k) }

let quat_axes q i =
  let w = q.((4 * i) + 0)
  and x = q.((4 * i) + 1)
  and y = q.((4 * i) + 2)
  and z = q.((4 * i) + 3) in
  let ax k =
    match k with
    | 0 ->
      { x = 1. -. (2. *. ((y *. y) +. (z *. z)))
      ; y = 2. *. ((x *. y) +. (w *. z))
      ; z = 2. *. ((x *. z) -. (w *. y))
      }
    | 1 ->
      { x = 2. *. ((x *. y) -. (w *. z))
      ; y = 1. -. (2. *. ((x *. x) +. (z *. z)))
      ; z = 2. *. ((y *. z) +. (w *. x))
      }
    | _ ->
      { x = 2. *. ((x *. z) +. (w *. y))
      ; y = 2. *. ((y *. z) -. (w *. x))
      ; z = 1. -. (2. *. ((x *. x) +. (y *. y)))
      }
  in
  ax

(* The shape of geom [i], in world coordinates. A capsule carries its two
   endpoints so that no backend has to know it lies along its local z. *)
let shape_of (t : Mjc.Topo.t) ~pos ~mat ~size i =
  let centre = point pos i in
  let z = axis mat i 2 in
  let s k = size.((3 * i) + k) in
  match t.geom_type.(i) with
  | Mjc.Topo.Sphere -> Sphere { centre; radius = s 0 }
  | Mjc.Topo.Capsule ->
    let half = smul z (s 1) in
    Capsule { a = sub centre half; b = add centre half; radius = s 0 }
  | Mjc.Topo.Box ->
    Box { centre; axes = axis mat i 0, axis mat i 1, axis mat i 2; half = s 0, s 1, s 2 }
  | Mjc.Topo.Cylinder -> Cylinder { centre; axis = z; radius = s 0; half_height = s 1 }
  | Mjc.Topo.Ellipsoid ->
    Ellipsoid
      { centre; axes = axis mat i 0, axis mat i 1, axis mat i 2; radii = s 0, s 1, s 2 }
  | Mjc.Topo.Plane -> Plane { normal = z; offset = dot z centre; half = s 0, s 1 }
  | Mjc.Topo.Hfield | Mjc.Topo.Mesh -> Unsupported "height field or mesh"

let as_jnt = function
  | Mjc.Topo.Hinge -> Hinge
  | Mjc.Topo.Slide -> Slide
  | Mjc.Topo.Ball -> Ball
  | Mjc.Topo.Free -> Free

(* The box with the same diagonal inertia as the body, which is what MuJoCo's
   own viewer draws: h_x^2 = 3/(2m) (I_y + I_z - I_x) and around. A body whose
   inertia does not satisfy the triangle inequality has no such box; the
   negative roots are clamped at a hair rather than drawn inside out. *)
let half_extents ~mass ~inertia =
  let h i =
    let j = Int.rem (i + 1) 3
    and k = Int.rem (i + 2) 3 in
    let x = 3. /. (2. *. mass) *. (inertia.(j) +. inertia.(k) -. inertia.(i)) in
    Float.sqrt (Float.max x 1e-12)
  in
  h 0, h 1, h 2

let bodies_of ~mass (t : Mjc.Topo.t) =
  List.filter (List.init t.nbody ~f:Fn.id) ~f:(fun b ->
    Stdlib.( && ) (Int.( > ) b 0) (Float.( > ) mass.(b) 0.))

let inertia_boxes (m : Mjc.Model.host) (d : Mjc.Data.host) =
  let t = m.topo in
  let mass = Nx.to_array m.params.Mjc.Model.Params.body_mass
  and inert = Nx.to_array m.params.Mjc.Model.Params.body_inertia
  and xipos = Nx.to_array d.Mjc.Data.xipos
  and ximat = Nx.to_array d.Mjc.Data.ximat in
  List.map (bodies_of ~mass t) ~f:(fun b ->
    let i = Array.init 3 ~f:(fun k -> inert.((3 * b) + k)) in
    { centre = point xipos b
    ; axes = axis ximat b 0, axis ximat b 1, axis ximat b 2
    ; half = half_extents ~mass:mass.(b) ~inertia:i
    })

let subtree_com (m : Mjc.Model.host) (d : Mjc.Data.host) =
  let t = m.topo in
  let sub = Nx.to_array m.params.Mjc.Model.Params.body_subtreemass
  and com = Nx.to_array d.Mjc.Data.subtree_com in
  List.filter (List.init t.nbody ~f:Fn.id) ~f:(fun b -> Float.( > ) sub.(b) 0.)
  |> List.map ~f:(fun b -> point com b)

let joint_axes (m : Mjc.Model.host) (d : Mjc.Data.host) =
  let t = m.topo in
  if t.njnt = 0
  then []
  else (
    let an = Nx.to_array d.Mjc.Data.xanchor
    and ax = Nx.to_array d.Mjc.Data.xaxis in
    List.init t.njnt ~f:(fun j ->
      { name = Mjc.Topo.jnt_name t j
      ; kind = as_jnt t.jnt_type.(j)
      ; anchor = point an j
      ; axis = point ax j
      }))

(* The velocity of a body's own centre of mass, from the Jacobian of that
   point: [J qvel], which is exact and costs [O(nv)] per body. The engine's
   [cvel] is com-based over a body's *subtree*, and composing it down a chain
   needs a shift this overlay does not want to get subtly wrong. *)
let com_velocity (m : Mjc.Model.host) (d : Mjc.Data.host) b =
  let pos = Nx.get [ b ] d.Mjc.Data.xipos in
  let jacp, jacr = Mjc.Jacobian.point m d ~body:b ~pos in
  (* the Jacobians are [nv;3], so the velocity is its transpose times qvel *)
  let v = Nx.dot (Nx.transpose jacp) d.Mjc.Data.qvel
  and w = Nx.dot (Nx.transpose jacr) d.Mjc.Data.qvel in
  point (Nx.to_array v) 0, point (Nx.to_array w) 0

let velocities ?(scale = 1.) (m : Mjc.Model.host) (d : Mjc.Data.host) =
  let t = m.topo in
  let mass = Nx.to_array m.params.Mjc.Model.Params.body_mass
  and xipos = Nx.to_array d.Mjc.Data.xipos in
  List.map (bodies_of ~mass t) ~f:(fun b ->
    let v, _ = com_velocity m d b in
    { from = point xipos b; vector = smul v scale })

(* The kinetic energy is a sum over bodies rather than [1/2 qvel' M qvel]: it
   is the same number — the mass matrix is the assembly of exactly these terms —
   and it costs [O(nbody nv)] instead of building an [nv; nv] matrix at display
   rate. Each body contributes [1/2 m |v|^2] for its centre of mass and
   [1/2 w' I w] for its rotation, with [v] and [w] from the Jacobian of the com
   and [I] the body's principal inertia read in the body's principal frame. *)
let mechanical_energy (m : Mjc.Model.host) (d : Mjc.Data.host) =
  let t = m.topo in
  let mass = Nx.to_array m.params.Mjc.Model.Params.body_mass
  and inert = Nx.to_array m.params.Mjc.Model.Params.body_inertia
  and xipos = Nx.to_array d.Mjc.Data.xipos
  and ximat = Nx.to_array d.Mjc.Data.ximat
  and g = Nx.to_array m.params.Mjc.Model.Params.gravity in
  let gravity = { x = g.(0); y = g.(1); z = g.(2) } in
  let kinetic, potential =
    List.fold (bodies_of ~mass t) ~init:(0., 0.) ~f:(fun (k, u) b ->
      let v, w = com_velocity m d b in
      (* omega in the body's principal frame: the columns of ximat *)
      let wi k =
        let a = axis ximat b k in
        (w.x *. a.x) +. (w.y *. a.y) +. (w.z *. a.z)
      in
      let rot =
        (inert.(3 * b) *. wi 0 *. wi 0)
        +. (inert.((3 * b) + 1) *. wi 1 *. wi 1)
        +. (inert.((3 * b) + 2) *. wi 2 *. wi 2)
      in
      ( k +. (0.5 *. mass.(b) *. dot v v) +. (0.5 *. rot)
      , u -. (mass.(b) *. dot gravity (point xipos b)) ))
  in
  { kinetic; potential }

(* The geom list, from the arrays a state or a recorded frame provides. *)
let geoms_of (t : Mjc.Topo.t) ~pos ~mat ~size ~rgba =
  List.init t.ngeom ~f:(fun i ->
    let base = 4 * i in
    { name = Mjc.Topo.geom_name t i
    ; shape = shape_of t ~pos ~mat ~size i
    ; colour =
        { r = rgba.(base); g = rgba.(base + 1); b = rgba.(base + 2); a = rgba.(base + 3) }
    })

let empty_overlays time bodies geoms contacts forces =
  { time
  ; bodies
  ; geoms
  ; contacts
  ; forces
  ; inertia = []
  ; com = []
  ; joints = []
  ; velocity = []
  ; energy = None
  }

let of_poses (m : Mjc.Model.host) ~time ~(geom : float array) ~contacts =
  let t = m.topo in
  let pos =
    Array.init (3 * t.ngeom) ~f:(fun k ->
      let i = k / 3
      and j = Int.rem k 3 in
      geom.((12 * i) + j))
  and mat =
    Array.init (9 * t.ngeom) ~f:(fun k ->
      let i = k / 9
      and j = Int.rem k 9 in
      geom.((12 * i) + 3 + j))
  in
  let size = Nx.to_array m.params.Mjc.Model.Params.geom_size
  and rgba = Nx.to_array m.params.Mjc.Model.Params.geom_rgba in
  empty_overlays time [] (geoms_of t ~pos ~mat ~size ~rgba) contacts []

let of_state
      ?(frames = false)
      ?(contacts = false)
      ?(forces = false)
      ?(inertia = false)
      ?(com = false)
      ?(joints = false)
      ?(velocity = false)
      ?(energy = false)
      (m : Mjc.Model.host)
      (d : Mjc.Data.host)
  =
  let t = m.topo
  and p = m.params in
  let pos = Nx.to_array d.Mjc.Data.geom_xpos
  and mat = Nx.to_array d.Mjc.Data.geom_xmat
  and size = Nx.to_array p.Mjc.Model.Params.geom_size
  and rgba = Nx.to_array p.Mjc.Model.Params.geom_rgba in
  let geoms = geoms_of t ~pos ~mat ~size ~rgba in
  let bpos = Nx.to_array d.Mjc.Data.xpos
  and bquat = Nx.to_array d.Mjc.Data.xquat in
  let bodies =
    if not frames
    then []
    else
      List.init t.nbody ~f:(fun b ->
        let origin = point bpos b in
        let ax = quat_axes bquat b in
        let scale = 0.12 in
        { name = Mjc.Topo.body_name t b
        ; frame =
            { origin
            ; x = add origin (smul (ax 0) scale)
            ; y = add origin (smul (ax 1) scale)
            ; z = add origin (smul (ax 2) scale)
            }
        })
  in
  let ncon = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.ncon) in
  let cpos = Nx.to_array d.Mjc.Data.contact_pos
  and cn = Nx.to_array d.Mjc.Data.contact_normal
  and cdist = Nx.to_array d.Mjc.Data.contact_dist
  and cdim = Nx.to_array d.Mjc.Data.contact_dim in
  (* Which constraint rows belong to which contact, and with what force: the
     rows are tagged, so this does not depend on the order they were written
     in, only on the contact id. Within a contact the assembler writes the
     normal row first (dim 1) or the pyramid's four edges in axis-major order. *)
  let rows_of_contact =
    if not contacts
    then [||]
    else (
      let nefc = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.nefc) in
      let acc = Array.create ~len:ncon [] in
      if nefc > 0
      then (
        let types = Nx.to_array d.Mjc.Data.efc_type
        and ids = Nx.to_array d.Mjc.Data.efc_id
        and force = Nx.to_array d.Mjc.Data.efc_force in
        for r = nefc - 1 downto 0 do
          let ty = Stdlib.Int32.to_int types.(r) in
          if ty = 5 || ty = 6
          then (
            let c = Stdlib.Int32.to_int ids.(r) in
            if c >= 0 && c < ncon then acc.(c) <- (r, force.(r)) :: acc.(c))
        done);
      Array.map acc ~f:Array.of_list)
  in
  let cframe = Nx.to_array d.Mjc.Data.contact_frame in
  let cfriction = Nx.to_array d.Mjc.Data.contact_friction in
  let contacts =
    if not contacts
    then []
    else
      List.init ncon ~f:(fun i ->
        let n = point cn i in
        (* the frame is three rows: normal, tangent, tangent *)
        let row k =
          { x = cframe.((9 * i) + (3 * k))
          ; y = cframe.((9 * i) + (3 * k) + 1)
          ; z = cframe.((9 * i) + (3 * k) + 2)
          }
        in
        let t1 = row 1
        and t2 = row 2 in
        let mu = cfriction.(5 * i) in
        let rows = rows_of_contact.(i) in
        let force = Array.fold rows ~init:0. ~f:(fun acc (_, f) -> acc +. f) in
        (* The pyramid's edges are [n +/- mu t1] and [n +/- mu t2]: the sum
           along the normal is the normal force, and what does not cancel along
           the normal is the friction. *)
        let tangent_force =
          if Stdlib.Array.length rows < 4
          then { x = 0.; y = 0.; z = 0. }
          else (
            let f k = rows.(k) |> snd in
            let a = mu *. (f 0 -. f 1)
            and b = mu *. (f 2 -. f 3) in
            add (smul t1 a) (smul t2 b))
        in
        { point = point cpos i
        ; normal = n
        ; tangents = t1, t2
        ; depth = cdist.(i)
        ; dim = Stdlib.Int32.to_int cdim.(i)
        ; force
        ; tangent_force
        ; friction = mu
        })
  in
  let forces =
    if not forces
    then []
    else (
      let nefc = Stdlib.Int32.to_int (Nx.item [] d.Mjc.Data.nefc) in
      let ef = Nx.to_array d.Mjc.Data.efc_force in
      (* Every row of a contact pushes along its normal, so the sum of the
         rows' forces is the normal impulse an arrow should show. *)
      List.init ncon ~f:(fun i ->
        let per = if Stdlib.Int32.to_int cdim.(i) >= 3 then 4 else 1 in
        let total =
          List.init per ~f:(fun k ->
            let row = (i * per) + k in
            if Stdlib.( < ) row nefc then ef.(row) else 0.)
          |> List.fold ~init:0. ~f:( +. )
        in
        { from = add (point cpos i) (smul (point cn i) 0.02)
        ; vector = smul (point cn i) (total *. 0.002)
        }))
  in
  { time = Nx.item [] d.Mjc.Data.time
  ; bodies
  ; geoms
  ; contacts
  ; forces
  ; inertia = (if inertia then inertia_boxes m d else [])
  ; com = (if com then subtree_com m d else [])
  ; joints = (if joints then joint_axes m d else [])
  ; velocity = (if velocity then velocities m d else [])
  ; energy = (if energy then Some (mechanical_energy m d) else None)
  }

let scale p k = smul p k

(* Green at contact, red at a centimetre of penetration, amber in between: one
   colour channel is enough to read, and the eye is good at that. *)
let depth_colour depth =
  let t = Float.max 0. (Float.min 1. (Float.neg depth /. 0.01)) in
  { r = 0.15 +. (0.75 *. t); g = 0.65 -. (0.55 *. t); b = 0.2 -. (0.15 *. t); a = 0.95 }

(* {1 Bounds and framing}

   Both backends open on the same view, so this lives here rather than once per
   backend: a window that framed differently would look like a disagreement
   about where the model is. Planes are left out of the box — they are the
   ground, and including a 10-metre floor makes the subject a speck — and so is
   anything v1 cannot draw. *)
let extent (s : t) =
  let lo = ref { x = Float.infinity; y = Float.infinity; z = Float.infinity }
  and hi =
    ref { x = Float.neg_infinity; y = Float.neg_infinity; z = Float.neg_infinity }
  in
  let seen = ref false in
  let eat (p : p3) =
    seen := true;
    lo := { x = Float.min p.x !lo.x; y = Float.min p.y !lo.y; z = Float.min p.z !lo.z };
    hi := { x = Float.max p.x !hi.x; y = Float.max p.y !hi.y; z = Float.max p.z !hi.z }
  in
  List.iter s.geoms ~f:(fun g ->
    match g.shape with
    | Sphere { centre; radius; _ } ->
      eat { centre with x = centre.x +. radius };
      eat { centre with x = centre.x -. radius }
    | Capsule { a; b; _ } ->
      eat a;
      eat b
    | Box { centre; _ } | Cylinder { centre; _ } | Ellipsoid { centre; _ } -> eat centre
    | Plane _ | Unsupported _ -> ());
  if not !seen
  then { x = -1.; y = -1.; z = -1. }, { x = 1.; y = 1.; z = 1. }
  else !lo, !hi

let framing (s : t) =
  let lo, hi = extent s in
  let centre =
    { x = (lo.x +. hi.x) /. 2.; y = (lo.y +. hi.y) /. 2.; z = (lo.z +. hi.z) /. 2. }
  in
  let size = Float.max (hi.x -. lo.x) (Float.max (hi.y -. lo.y) (hi.z -. lo.z)) in
  let d = Float.max size 0.5 in
  ( { x = centre.x +. (2.0 *. d); y = centre.y -. (2.8 *. d); z = centre.z +. (1.6 *. d) }
  , centre )

let pp ppf (s : t) =
  Format.fprintf
    ppf
    "@[<v>time %.3f@,geoms %d@,contacts %d@,forces %d@,inertia %d@]"
    s.time
    (List.length s.geoms)
    (List.length s.contacts)
    (List.length s.forces)
    (List.length s.inertia)
