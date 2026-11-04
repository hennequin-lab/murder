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

type t =
  { time : float
  ; bodies : body list
  ; geoms : geom list
  ; contacts : contact list
  ; forces : arrow list
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

let of_state
      ?(frames = false)
      ?(contacts = false)
      ?(forces = false)
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
  { time = Nx.item [] d.Mjc.Data.time; bodies; geoms; contacts; forces }

let pp ppf (s : t) =
  Format.fprintf
    ppf
    "@[<v>time %.3f@,geoms %d@,contacts %d@,forces %d@]"
    s.time
    (List.length s.geoms)
    (List.length s.contacts)
    (List.length s.forces)
