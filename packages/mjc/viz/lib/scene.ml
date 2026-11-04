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
  ; depth : float
  ; dim : int
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
  let geoms =
    List.init t.ngeom ~f:(fun i ->
      let base = 4 * i in
      { name = Mjc.Topo.geom_name t i
      ; shape = shape_of t ~pos ~mat ~size i
      ; colour =
          { r = rgba.(base)
          ; g = rgba.(base + 1)
          ; b = rgba.(base + 2)
          ; a = rgba.(base + 3)
          }
      })
  in
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
  let contacts =
    if not contacts
    then []
    else
      List.init ncon ~f:(fun i ->
        { point = point cpos i
        ; normal = point cn i
        ; depth = cdist.(i)
        ; dim = Stdlib.Int32.to_int cdim.(i)
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

let pp ppf (s : t) =
  Format.fprintf
    ppf
    "@[<v>time %.3f@,geoms %d@,contacts %d@,forces %d@,inertia %d@]"
    s.time
    (List.length s.geoms)
    (List.length s.contacts)
    (List.length s.forces)
    (List.length s.inertia)
