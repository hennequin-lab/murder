(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module type MOTION = sig
  type kind
  type t = (float, kind) Nx.t

  val dtype : (float, kind) Nx.dtype
  val one : t
  val mul : t -> t -> t
  val inverse : t -> t
  val normalize : t -> t
  val translate : (float, kind) Nx.t -> t
  val translate3 : float -> float -> float -> t
  val rotate : axis:(float, kind) Nx.t -> angle:(float, kind) Nx.t -> t
  val exp : (float, kind) Nx.t -> t
  val log : t -> (float, kind) Nx.t
  val adjoint : t -> (float, kind) Nx.t
  val translation : t -> (float, kind) Nx.t
  val rotation : t -> (float, kind) Nx.t
end

module Float32 = struct
  type kind = Nx.float32_elt
  type t = Nx.float32_t

  let dtype = Nx.float32
  let one = Geom.Motor.one
  let mul = Geom.Motor.mul
  let inverse = Geom.Motor.inverse
  let normalize = Geom.Motor.normalize
  let translate = Geom.Motor.translate
  let translate3 = Geom.Motor.translate3
  let rotate = Geom.Motor.rotate
  let exp = Geom.Motor.exp
  let log = Geom.Motor.log
  let adjoint = Geom.Motor.adjoint
  let translation = Geom.Motor.translation
  let rotation = Geom.Motor.rotation
end

module Make (Mo : MOTION) = struct
  type motion = Mo.t
  type twist = (float, Mo.kind) Nx.t
  type q = (float, Mo.kind) Nx.t
  type axis = float * float * float

  type kind =
    | Fixed
    | Revolute of axis
    | Prismatic of axis

  type joint =
    { jname : string
    ; kind : kind
    ; parent : int (* link index *)
    ; child : int (* link index *)
    ; origin : motion (* M_{parent, joint} *)
    ; screw : twist (* unit twist in the joint frame, shape [|6|] *)
    ; generator : motion (* the packed [screw], shape [|8|] *)
    ; lower : float
    ; upper : float
    }

  type frame_rec =
    { fname : string
    ; flink : int
    ; fpose : motion
    }

  type t =
    { link_names : string array
    ; joints : joint array
    ; frames : frame_rec array
    ; link_parent : int array (* parent link of a link, [-1] for the root *)
    ; link_joint : int array (* the joint driving a link, [-1] for the root *)
    ; dof_index : int array (* the column of a joint's coordinate, [-1] if fixed *)
    ; dof : int
    }

  type frame = int

  type joint_spec =
    { name : string
    ; kind : kind
    ; parent : string
    ; child : string
    ; origin : motion
    ; limits : (float * float) option
    }

  type frame_spec =
    { name : string
    ; link : string
    ; pose : motion
    }

  (* Shapes *)

  let size = 8
  let prod a = Array.fold_left ( * ) 1 a
  let i64 n = Nx.scalar Nx.int64 (Int64.of_int n)
  let take_axis ~axis idx x = Nx.take ~axis ~indices:idx x

  let batch_shape x =
    let s = Nx.shape x in
    if Array.length s = 0 then invalid_arg "Robot: expected an axis to batch over";
    Array.sub s 0 (Array.length s - 1)

  let expand batch x = Nx.reshape (Array.append batch [| 1 |]) x
  let column x i = take_axis ~axis:(-1) (i64 i) x

  (* The packed generator of a unit screw.

     A twist [(omega, v)] is the Lie-algebra element, and
     [d/dq (exp (q g)) = G_g * exp (q g)] with [G_g] the bivector of [g] in the
     packing of [Motor]. In the basis
     [[1, e12, e13, e23, e1 inf, e2 inf, e3 inf, e123 inf]] that is

     {[
      G_g = (1/2) * (0, -w_z, w_y, -w_x, -v_x, -v_y, -v_z, 0).
     ]}

     The rotational and the translational families are pinned by the packings of
     [Motor.rotate] and [Motor.translate]; the map is linear in [g], so that
     determines it. The test suite checks the identity against a finite
     difference of [Mo.exp] on a general screw. *)
  let generator_of_screw a =
    let h v = 0.5 *. v in
    Nx.create
      Mo.dtype
      [| size |]
      [| 0.
       ; h (-.a.(2))
       ; h a.(1)
       ; h (-.a.(0))
       ; h (-.a.(3))
       ; h (-.a.(4))
       ; h (-.a.(5))
       ; 0.
      |]

  (* Construction *)

  exception Invalid of string

  let invalid fmt = Printf.ksprintf (fun msg -> raise (Invalid msg)) fmt

  let find name names =
    let rec go i =
      if i >= Array.length names
      then None
      else if String.equal names.(i) name
      then Some i
      else go (i + 1)
    in
    go 0

  let unit_axis (x, y, z) =
    let n = Float.sqrt ((x *. x) +. (y *. y) +. (z *. z)) in
    if (not (Float.is_finite n)) || n < 1e-9
    then invalid "the axis must be a non-zero, finite vector"
    else x /. n, y /. n, z /. n

  let screw_of_kind = function
    | Fixed -> Nx.zeros Mo.dtype [| 6 |]
    | Revolute axis ->
      let x, y, z = unit_axis axis in
      Nx.create Mo.dtype [| 6 |] [| x; y; z; 0.; 0.; 0. |]
    | Prismatic axis ->
      let x, y, z = unit_axis axis in
      Nx.create Mo.dtype [| 6 |] [| 0.; 0.; 0.; x; y; z |]

  let create ~links ~joints ?(frames = []) () =
    try
      let link_names = Array.of_list links in
      let nlinks = Array.length link_names in
      if nlinks = 0 then invalid "no links: a model has at least a root link";
      Array.iteri
        (fun i name ->
           if String.trim name = "" then invalid "link %d has an empty name" i;
           for j = i + 1 to nlinks - 1 do
             if String.equal name link_names.(j)
             then invalid "link name %S is used twice" name
           done)
        link_names;
      let specs : joint_spec array = Array.of_list joints in
      let njoints = Array.length specs in
      Array.iteri
        (fun i (s : joint_spec) ->
           for j = i + 1 to njoints - 1 do
             if String.equal s.name specs.(j).name
             then invalid "joint name %S is used twice" s.name
           done)
        specs;
      let link_of joint name =
        match find name link_names with
        | Some k -> k
        | None -> invalid "joint %S: no link named %S among the model's links" joint name
      in
      let joint_of = Array.make nlinks (-1) in
      let parent_of = Array.make nlinks (-1) in
      let child_of_joint = Array.make njoints (-1) in
      let resolved = Array.make nlinks false in
      resolved.(0) <- true;
      Array.iteri
        (fun i (s : joint_spec) ->
           let parent = link_of s.name s.parent in
           let child = link_of s.name s.child in
           if child = 0
           then invalid "joint %S: the root link %S cannot be driven" s.name s.child;
           if joint_of.(child) >= 0
           then
             invalid
               "joint %S: link %S is already driven by joint %S"
               s.name
               s.child
               specs.(joint_of.(child)).name;
           if not resolved.(parent)
           then
             invalid
               "joint %S: parent link %S is not attached to the root by an earlier \
                joint; order the joints topologically"
               s.name
               s.parent;
           (match s.limits with
            | Some (lo, hi) when lo > hi -> invalid "joint %S: limits are reversed" s.name
            | Some _ | None -> ());
           resolved.(child) <- true;
           parent_of.(child) <- parent;
           joint_of.(child) <- i;
           child_of_joint.(i) <- child)
        specs;
      Array.iteri
        (fun i r ->
           if not r
           then
             invalid
               "link %S is not connected to the root: every link but the root must be a \
                joint's child"
               link_names.(i))
        resolved;
      let dof_index = Array.make njoints (-1) in
      let dof = ref 0 in
      let built =
        Array.mapi
          (fun j (s : joint_spec) ->
             let screw = screw_of_kind s.kind in
             (match s.kind with
              | Fixed -> ()
              | Revolute _ | Prismatic _ ->
                dof_index.(j) <- !dof;
                incr dof);
             let lower, upper =
               match s.limits with
               | None -> Float.neg_infinity, Float.infinity
               | Some (lo, hi) -> lo, hi
             in
             { jname = s.name
             ; kind = s.kind
             ; parent = parent_of.(child_of_joint.(j))
             ; child = child_of_joint.(j)
             ; origin = s.origin
             ; screw
             ; generator = generator_of_screw (Nx.to_array screw)
             ; lower
             ; upper
             })
          specs
      in
      let frames : frame_rec array =
        Array.of_list
          (List.map
             (fun (f : frame_spec) ->
                let flink =
                  match find f.link link_names with
                  | Some k -> k
                  | None -> invalid "frame %S: no link named %S" f.name f.link
                in
                { fname = f.name; flink; fpose = f.pose })
             frames)
      in
      Array.iteri
        (fun i f ->
           for j = i + 1 to Array.length frames - 1 do
             if String.equal f.fname frames.(j).fname
             then invalid "frame name %S is used twice" f.fname
           done)
        frames;
      Ok
        { link_names
        ; joints = built
        ; frames
        ; link_parent = parent_of
        ; link_joint = joint_of
        ; dof_index
        ; dof = !dof
        }
    with
    | Invalid msg -> Error msg

  (* Inspection *)

  let dof m = m.dof
  let link_count m = Array.length m.link_names
  let joint_count m = Array.length m.joints
  let link_names m = Array.copy m.link_names
  let joint_names m = Array.map (fun j -> j.jname) m.joints
  let kind_of_joint m j = m.joints.(j).kind
  let limits m j = m.joints.(j).lower, m.joints.(j).upper
  let frame_names m = Array.map (fun (f : frame_rec) -> f.fname) m.frames

  let frame m name =
    match find name (frame_names m) with
    | Some i -> i
    | None ->
      invalid_arg
        (Printf.sprintf
           "Robot.frame: no frame named %S; the model has %s"
           name
           (match Array.to_list (frame_names m) with
            | [] -> "no frames"
            | names -> String.concat ", " (List.map (Printf.sprintf "%S") names)))

  (* Forward kinematics *)

  let joint_motion m j q =
    let jt = m.joints.(j) in
    match jt.kind with
    | Fixed -> jt.origin
    | Revolute _ | Prismatic _ ->
      let qi = column q m.dof_index.(j) in
      let scaled = Nx.mul jt.screw (expand (Nx.shape qi) qi) in
      Mo.mul jt.origin (Mo.exp scaled)

  let link_motors m q =
    let batch = batch_shape q in
    let motors = Array.make (Array.length m.link_names) Mo.one in
    motors.(0) <- Nx.broadcast_to (Array.append batch [| size |]) Mo.one;
    Array.iteri
      (fun j (jt : joint) ->
         motors.(jt.child) <- Mo.mul motors.(jt.parent) (joint_motion m j q))
      m.joints;
    Nx.stack ~axis:(Array.length batch) (Array.to_list motors)

  let take_link lm batch l = take_axis ~axis:(Array.length batch) (i64 l) lm

  let frame_motors m q =
    let batch = batch_shape q in
    let lm = link_motors m q in
    let nf = Array.length m.frames in
    if nf = 0
    then Nx.zeros Mo.dtype (Array.append batch [| 0; size |])
    else (
      let links =
        Nx.create
          Nx.int64
          [| nf |]
          (Array.map (fun (f : frame_rec) -> Int64.of_int f.flink) m.frames)
      in
      let selected = take_axis ~axis:(Array.length batch) links lm in
      let poses =
        Nx.stack
          ~axis:0
          (Array.to_list (Array.map (fun (f : frame_rec) -> f.fpose) m.frames))
      in
      Mo.mul selected poses)

  let pose m ~frame:fi q =
    let batch = batch_shape q in
    let lm = link_motors m q in
    let f = m.frames.(fi) in
    Mo.mul (take_link lm batch f.flink) f.fpose

  let relative m ~from:a ~to_:b q =
    Mo.mul (Mo.inverse (pose m ~frame:a q)) (pose m ~frame:b q)

  (* Differential kinematics *)

  (* The joints from the root to [link], in order. *)
  let chain m link =
    let rec go acc l =
      if l <= 0 then acc else go (m.link_joint.(l) :: acc) m.link_parent.(l)
    in
    go [] link

  let is_moving (jt : joint) =
    match jt.kind with
    | Fixed -> false
    | Revolute _ | Prismatic _ -> true

  (* Every moving joint, in coordinate order. *)
  let moving m =
    List.filter
      (fun j -> is_moving m.joints.(j))
      (List.init (Array.length m.joints) Fun.id)

  let frame_state m batch q fi =
    let lm = link_motors m q in
    let f = m.frames.(fi) in
    lm, Mo.mul (take_link lm batch f.flink) f.fpose

  (* One column per coordinate, in [q]'s order, so that a Jacobian multiplies a
     whole [qdot]: a joint off the frame's chain contributes zeros. *)
  let columns m ~frame:fi ~zero ~col =
    let on_chain = chain m m.frames.(fi).flink in
    List.map (fun j -> if List.mem j on_chain then col j else zero ()) (moving m)

  let motion_jacobian m ~frame:fi q =
    let batch = batch_shape q in
    let lm, mframe = frame_state m batch q fi in
    let cols =
      columns
        m
        ~frame:fi
        ~zero:(fun () -> Nx.zeros Mo.dtype (Array.append batch [| size |]))
        ~col:(fun j ->
          let jt = m.joints.(j) in
          (* [w] is the world pose of the joint's frame, so the column is the
             world screw [w g w^-1] acting on the frame's pose. *)
          let w = Mo.mul (take_link lm batch jt.parent) jt.origin in
          let tail = Mo.mul (Mo.inverse w) mframe in
          Mo.mul (Mo.mul w jt.generator) tail)
    in
    if cols = []
    then Nx.zeros Mo.dtype (Array.append batch [| size; 0 |])
    else Nx.stack ~axis:(Array.length batch + 1) cols

  let jacobian m ?(space = `World) ~frame:fi q =
    let batch = batch_shape q in
    let bl = Array.length batch in
    let lm, mframe = frame_state m batch q fi in
    let joints = moving m in
    let n = List.length joints in
    if n = 0
    then Nx.zeros Mo.dtype (Array.append batch [| 6; 0 |])
    else (
      (* Re-reference the twists to the frame's origin, and, for [`Body], rotate
         them onto the frame's axes. The re-referencing is an adjoint; the
         rotation is the adjoint of the rotation part alone, not of the whole
         frame motion, because [Mo.adjoint] moves the point of reference to the
         body origin as well as the axes. Applying the full adjoint here would
         shift a twist that the first step has already shifted. *)
      let shift =
        let here = Mo.translate (Nx.neg (Mo.translation mframe)) in
        match space with
        | `World -> here
        | `Body ->
          let rotation_only = Mo.mul here mframe in
          Mo.mul (Mo.inverse rotation_only) here
      in
      let columns =
        columns
          m
          ~frame:fi
          ~zero:(fun () -> Nx.zeros Mo.dtype (Array.append batch [| 6; 6 |]))
          ~col:(fun j ->
            let jt = m.joints.(j) in
            let w = Mo.mul (take_link lm batch jt.parent) jt.origin in
            Mo.adjoint (Mo.mul shift w))
      in
      let screws = Nx.stack ~axis:0 (List.map (fun j -> m.joints.(j).screw) joints) in
      let stacked = Nx.reshape [| prod batch; n; 6; 6 |] (Nx.stack ~axis:bl columns) in
      let out = Nx.einsum "bnij,nj->bin" [| stacked; screws |] in
      Nx.reshape (Array.append batch [| 6; n |]) out)

  (* Differentiable kinematics *)

  let pose_tangent m ~frame:fi q dq =
    let batch = batch_shape q in
    let n = m.dof in
    if n = 0
    then Nx.zeros Mo.dtype (Array.append batch [| size |])
    else (
      let jm = motion_jacobian m ~frame:fi q in
      let bsz = prod batch in
      let cols = Nx.reshape [| bsz; size; n |] jm in
      let tangents = Nx.reshape [| bsz; n |] dq in
      let out = Nx.einsum "bij,bj->bi" [| cols; tangents |] in
      Nx.reshape (Array.append batch [| size |]) out)

  let pose_diff m ~frame:fi =
    Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun q ->
      pose m ~frame:fi q, fun dq -> pose_tangent m ~frame:fi q dq)

  (* Standard robots *)

  let joint name kind ~parent ~child ~origin =
    { name; kind; parent; child; origin; limits = None }

  let planar2 ?(l1 = 1.) ?(l2 = 1.) () =
    let z = Revolute (0., 0., 1.) in
    let links = [ "base"; "upper"; "fore" ] in
    let joints =
      [ joint "shoulder" z ~parent:"base" ~child:"upper" ~origin:Mo.one
      ; joint "elbow" z ~parent:"upper" ~child:"fore" ~origin:(Mo.translate3 l1 0. 0.)
      ]
    in
    let frames = [ { name = "ee"; link = "fore"; pose = Mo.translate3 l2 0. 0. } ] in
    match create ~links ~joints ~frames () with
    | Ok m -> m
    | Error msg -> invalid_arg ("Robot.planar2: " ^ msg)

  let six_dof ?(lengths = [| 0.65; 0.55; 0.45; 0.35; 0.30; 0.25 |]) () =
    if Array.length lengths <> 6
    then invalid_arg "Robot.six_dof: lengths must have six entries";
    let link i = if i = 0 then "base" else Printf.sprintf "link%d" i in
    let links = List.init 7 link in
    let axis i =
      match i with
      | 0 | 2 | 4 -> Revolute (0., 0., 1.)
      | _ -> Revolute (0., 1., 0.)
    in
    let joints =
      List.init 6 (fun i ->
        joint
          (Printf.sprintf "j%d" i)
          (axis i)
          ~parent:(link i)
          ~child:(link (i + 1))
          ~origin:(if i = 0 then Mo.one else Mo.translate3 0. 0. lengths.(i - 1)))
    in
    let frames =
      [ { name = "ee"; link = "link6"; pose = Mo.translate3 0. 0. lengths.(5) } ]
    in
    match create ~links ~joints ~frames () with
    | Ok m -> m
    | Error msg -> invalid_arg ("Robot.six_dof: " ^ msg)

  (* Printing *)

  let pp ppf m =
    let kind_name = function
      | Fixed -> "fixed"
      | Revolute (x, y, z) -> Printf.sprintf "revolute (%.3g, %.3g, %.3g)" x y z
      | Prismatic (x, y, z) -> Printf.sprintf "prismatic (%.3g, %.3g, %.3g)" x y z
    in
    let limits j =
      if Float.is_infinite j.lower || Float.is_infinite j.upper
      then "unbounded"
      else Printf.sprintf "[%.3g, %.3g]" j.lower j.upper
    in
    Format.fprintf
      ppf
      "@[<v 0>robot: %d links, %d joints, %d coordinates, %d frames"
      (Array.length m.link_names)
      (Array.length m.joints)
      m.dof
      (Array.length m.frames);
    Array.iteri (fun i name -> Format.fprintf ppf "@,  link %d: %s" i name) m.link_names;
    Array.iteri
      (fun i j ->
         let column =
           if m.dof_index.(i) < 0 then "-" else string_of_int m.dof_index.(i)
         in
         Format.fprintf
           ppf
           "@,  joint %s: %s, %s -> %s, q[%s], %s"
           j.jname
           (kind_name j.kind)
           m.link_names.(j.parent)
           m.link_names.(j.child)
           column
           (limits j))
      m.joints;
    Array.iteri
      (fun i (f : frame_rec) ->
         Format.fprintf ppf "@,  frame %s on %s" f.fname m.link_names.(f.flink))
      m.frames;
    Format.fprintf ppf "@]"
end

include Make (Float32)
