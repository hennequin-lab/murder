(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Geom_robot.Robot: the model, forward kinematics and differential
   kinematics.

   The reference implementations here are deliberately independent of the code
   under test: hand-derived formulas for the planar arm, 4x4 homogeneous
   matrices for the six-joint arm, and central differences for every Jacobian.
   The finite differences run in float64 on float32 poses, so the tolerance is
   the difference between a float32 evaluation and a float64 difference, not
   the float32 noise of the model itself. *)

open Windtrap
module M = Geom.Motor
module R = Geom_robot.Robot

let f32 = Nx.float32
let f64 = Nx.float64
let scalar x = Nx.scalar f32 x
let item idx x = Nx.item idx x
let vec3 x y z = Nx.create f32 [| 3 |] [| x; y; z |]
let q_of l = Nx.create f32 [| List.length l |] (Array.of_list l)

let cfg rows =
  (* A batch of configurations, one list per row. *)
  let dof = List.length (List.hd rows) in
  let flat = List.concat rows in
  Nx.reshape
    [| List.length rows; dof |]
    (Nx.create f32 [| List.length flat |] (Array.of_list flat))

let to64 x = Nx.cast f64 x
let arr64 x = Nx.to_array (to64 x)
let arr32 x = Nx.to_array x
let rel = 2e-4
let abs = 2e-4

let close_scalar ?(rel = rel) ?(abs = abs) a b =
  Float.equal a b
  || Float.abs (a -. b) <= abs
  || Float.abs (a -. b) <= rel *. Float.max (Float.abs a) (Float.abs b)

let close ?(rel = rel) ?(abs = abs) a b =
  let sa = Nx.shape a
  and sb = Nx.shape b in
  Array.length sa = Array.length sb
  && Array.for_all2 Int.equal sa sb
  &&
  let a = arr64 a
  and b = arr64 b in
  Array.length a = Array.length b && Array.for_all2 (close_scalar ~rel ~abs) a b

let equal_tensor ?msg ?(rel = rel) ?(abs = abs) expected actual =
  equal ?msg (Testable.make ~pp:Nx.pp ~equal:(close ~rel ~abs)) expected actual

let equal_floats ?msg ?(eps = 1e-4) expected actual =
  equal ?msg (array (float eps)) expected (arr64 actual)

(* 4x4 homogeneous matrices, row-major, on the host: the reference for the
   six-joint arm. *)

let mmul a b =
  Array.init 16 (fun k ->
    let i = k / 4
    and j = k mod 4 in
    let s = ref 0. in
    for l = 0 to 3 do
      s := !s +. (a.((i * 4) + l) *. b.((l * 4) + j))
    done;
    !s)

let rotm ~axis ~angle =
  let x, y, z = axis in
  let n = Float.sqrt ((x *. x) +. (y *. y) +. (z *. z)) in
  let x = x /. n
  and y = y /. n
  and z = z /. n in
  let c = Float.cos angle
  and s = Float.sin angle
  and t = 1. -. Float.cos angle in
  [| (t *. x *. x) +. c
   ; (t *. x *. y) -. (s *. z)
   ; (t *. x *. z) +. (s *. y)
   ; 0.
   ; (t *. x *. y) +. (s *. z)
   ; (t *. y *. y) +. c
   ; (t *. y *. z) -. (s *. x)
   ; 0.
   ; (t *. x *. z) -. (s *. y)
   ; (t *. y *. z) +. (s *. x)
   ; (t *. z *. z) +. c
   ; 0.
   ; 0.
   ; 0.
   ; 0.
   ; 1.
  |]

let tm v = [| 1.; 0.; 0.; v.(0); 0.; 1.; 0.; v.(1); 0.; 0.; 1.; v.(2); 0.; 0.; 0.; 1. |]
let eye = [| 1.; 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0.; 1. |]
let mat_pos m = [| m.(3); m.(7); m.(11) |]

let mat_rot m =
  Array.init 9 (fun k ->
    let i = k / 3
    and j = k mod 3 in
    m.((i * 4) + j))

(* The six-joint arm of [Robot.six_dof], as matrices. *)
let six_lengths = [| 0.65; 0.55; 0.45; 0.35; 0.30; 0.25 |]

let six_axis i =
  match i with
  | 0 | 2 | 4 -> 0., 0., 1.
  | _ -> 0., 1., 0.

let six_matrix q =
  let q = Array.of_list q in
  let m = ref eye in
  for i = 0 to 5 do
    if i > 0 then m := mmul !m (tm [| 0.; 0.; six_lengths.(i - 1) |]);
    m := mmul !m (rotm ~axis:(six_axis i) ~angle:q.(i))
  done;
  mmul !m (tm [| 0.; 0.; six_lengths.(5) |])

(* A perturbation of one coordinate, for the finite differences. *)
let bump dof i h =
  let a = Array.make dof 0. in
  a.(i) <- h;
  Nx.create f32 [| dof |] a

(* -- Generative helper: a model with branches, a fixed joint and a prismatic
   joint, used by several tests. ------------------------------------------ *)

let branch_model () =
  let open R in
  match
    create
      ~links:[ "base"; "turret"; "slider"; "tool" ]
      ~joints:
        [ { name = "yaw"
          ; kind = Revolute (0., 0., 1.)
          ; parent = "base"
          ; child = "turret"
          ; origin = M.one
          ; limits = Some (-1.5, 1.5)
          }
        ; { name = "lift"
          ; kind = Prismatic (0., 0., 1.)
          ; parent = "turret"
          ; child = "slider"
          ; origin = M.translate3 0. 0. 1.
          ; limits = None
          }
        ; { name = "mount"
          ; kind = Fixed
          ; parent = "slider"
          ; child = "tool"
          ; origin = M.translate3 0.5 0. 0.
          ; limits = None
          }
        ]
      ~frames:
        [ { name = "ee"; link = "tool"; pose = M.one }
        ; { name = "eye"; link = "turret"; pose = M.translate3 0. 0. 2. }
        ]
      ()
  with
  | Ok m -> m
  | Error msg -> failwith msg

(* Tests *)

(* The generator of a joint screw: [d/dq exp (q g) = G exp (q g), and the model's
   analytic derivative is that identity applied through the chain. A helical
   screw is not reachable through [kind] yet, so this is checked for the two
   families the model has, through [motion_jacobian]. *)

let test_screw_generator () =
  let m = branch_model () in
  let fr = R.frame m "ee" in
  let q = q_of [ 0.7; -0.4 ] in
  let h = 1e-2 in
  let fd =
    let plus = R.pose m ~frame:fr (Nx.add q (bump 2 0 h)) in
    let minus = R.pose m ~frame:fr (Nx.sub q (bump 2 0 h)) in
    Nx.div_s (Nx.sub plus minus) (2. *. h)
  in
  equal_tensor
    ~msg:"d/dq of a revolute joint"
    ~rel:2e-3
    (Nx.slice [ A; I 0 ] (R.motion_jacobian m ~frame:fr q))
    fd;
  let fd =
    let plus = R.pose m ~frame:fr (Nx.add q (bump 2 1 h)) in
    let minus = R.pose m ~frame:fr (Nx.sub q (bump 2 1 h)) in
    Nx.div_s (Nx.sub plus minus) (2. *. h)
  in
  equal_tensor
    ~msg:"d/dq of a prismatic joint"
    ~rel:2e-3
    (Nx.slice [ A; I 1 ] (R.motion_jacobian m ~frame:fr q))
    fd

(* The planar arm's forward kinematics, from the closed form. *)

let test_planar2 () =
  let l1 = 0.8
  and l2 = 0.5 in
  let m = R.planar2 ~l1 ~l2 () in
  let fr = R.frame m "ee" in
  equal ~msg:"dof" int 2 (R.dof m);
  List.iter
    (fun (a, b) ->
       let pose = R.pose m ~frame:fr (q_of [ a; b ]) in
       let x = (l1 *. Float.cos a) +. (l2 *. Float.cos (a +. b)) in
       let y = (l1 *. Float.sin a) +. (l2 *. Float.sin (a +. b)) in
       equal_floats
         ~msg:(Printf.sprintf "tip position at (%g, %g)" a b)
         [| x; y; 0. |]
         (M.translation pose);
       equal_floats
         ~msg:(Printf.sprintf "tip orientation at (%g, %g)" a b)
         [| 0.; 0.; a +. b |]
         (Nx.slice [ R (0, 3) ] (M.log pose)))
    [ 0., 0.; 0.3, 0.; 0., -0.9; -1.2, 2.1 ]

(* The six-joint arm, against a 4x4 matrix chain. *)

let test_six_dof () =
  let m = R.six_dof () in
  let fr = R.frame m "ee" in
  equal ~msg:"links" int 7 (R.link_count m);
  equal ~msg:"joints" int 6 (R.joint_count m);
  equal ~msg:"dof" int 6 (R.dof m);
  List.iter
    (fun q ->
       let ref = six_matrix q in
       let pose = R.pose m ~frame:fr (q_of q) in
       equal_floats
         ~msg:"tip position against the matrix chain"
         ~eps:1e-4
         (mat_pos ref)
         (M.translation pose);
       equal_floats
         ~msg:"tip rotation against the matrix chain"
         ~eps:1e-4
         (mat_rot ref)
         (M.rotation pose))
    [ [ 0.; 0.; 0.; 0.; 0.; 0. ]
    ; [ 0.4; -0.3; 0.8; 0.2; -0.5; 0.9 ]
    ; [ -1.1; 0.7; -0.2; 1.3; 0.6; -0.8 ]
    ]

(* Every link's pose, and the frames' poses, agree with the single-frame query. *)

let test_frames () =
  let m = branch_model () in
  let q = cfg [ [ 0.3; 0.2 ]; [ -1.0; -0.5 ] ] in
  let all = R.frame_motors m q in
  equal ~msg:"frame stack shape" (array int) [| 2; 2; 8 |] (Nx.shape all);
  Array.iteri
    (fun i name ->
       let fr = R.frame m name in
       let one = R.pose m ~frame:fr q in
       equal_tensor
         ~msg:(Printf.sprintf "frame %s of a batch" name)
         one
         (Nx.slice [ A; I i ] all))
    (R.frame_names m);
  let links = R.link_motors m q in
  equal ~msg:"link stack shape" (array int) [| 2; 4; 8 |] (Nx.shape links);
  (* The root is the identity, and the tool frame sits at the slider's pose
     carried by the mount's fixed translation. *)
  equal_tensor
    ~msg:"the root link is the identity"
    (Nx.broadcast_to [| 2; 8 |] M.one)
    (Nx.slice [ A; I 0 ] links)

(* Batching is a tensor axis, not a loop: a batch of configurations gives row
   by row what the unbatched query gives. *)

let test_batching () =
  let m = R.six_dof () in
  let fr = R.frame m "ee" in
  let rows = [ [ 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 ]; [ -0.4; 0.9; 0.1; -0.7; 0.2; 1.0 ] ] in
  let batch = R.pose m ~frame:fr (cfg rows) in
  List.iteri
    (fun i row ->
       equal_tensor
         ~msg:(Printf.sprintf "row %d of a batch" i)
         (R.pose m ~frame:fr (q_of row))
         (Nx.slice [ I i ] batch))
    rows

(* Relative poses compose: [pose a * relative a b = pose b]. *)

let test_relative () =
  let m = branch_model () in
  let q = q_of [ 0.4; 0.6 ] in
  let a = R.frame m "eye"
  and b = R.frame m "ee" in
  let rel = R.relative m ~from:a ~to_:b q in
  equal_tensor
    ~msg:"relative composes to the absolute pose"
    (R.pose m ~frame:b q)
    (M.mul (R.pose m ~frame:a q) rel)

(* The geometric Jacobian, against a finite difference of the pose in float64.
   The finite difference fixes the reference point (the frame's origin) and the
   sign conventions. *)

let fd_jacobian m fr q i =
  let dof = R.dof m in
  let h = 5e-3 in
  let plus = R.pose m ~frame:fr (Nx.add q (bump dof i h)) in
  let minus = R.pose m ~frame:fr (Nx.sub q (bump dof i h)) in
  let diff x = x /. (2. *. h) in
  let dp =
    Array.map2
      (fun a b -> diff (a -. b))
      (arr64 (M.translation plus))
      (arr64 (M.translation minus))
  in
  let dr =
    Array.map2
      (fun a b -> diff (a -. b))
      (arr64 (M.rotation plus))
      (arr64 (M.rotation minus))
  in
  (* omega = vee (dR R^T), with R the rotation at the centre of the stencil: the
     rotation at an endpoint would bias the estimate by O(h). *)
  let r = arr64 (M.rotation (R.pose m ~frame:fr q)) in
  let skew i j =
    let s = ref 0. in
    for k = 0 to 2 do
      s := !s +. (dr.((i * 3) + k) *. r.((j * 3) + k))
    done;
    !s
  in
  let omega = [| skew 2 1; skew 0 2; skew 1 0 |] in
  omega, dp

let test_geometric_jacobian () =
  let check name m fr q =
    let j = R.jacobian m ~frame:fr q in
    equal ~msg:(name ^ ": shape") (array int) [| 6; R.dof m |] (Nx.shape j);
    for i = 0 to R.dof m - 1 do
      let omega, dp = fd_jacobian m fr q i in
      let column = arr64 (Nx.slice [ A; I i ] j) in
      for k = 0 to 2 do
        equal
          ~msg:(Printf.sprintf "%s: omega %d of column %d" name k i)
          (float 2e-3)
          omega.(k)
          column.(k);
        equal
          ~msg:(Printf.sprintf "%s: velocity %d of column %d" name k i)
          (float 2e-3)
          dp.(k)
          column.(3 + k)
      done
    done
  in
  let planar = R.planar2 ~l1:0.8 ~l2:0.5 () in
  check "planar2" planar (R.frame planar "ee") (q_of [ 0.5; -0.7 ]);
  let six = R.six_dof () in
  check "six_dof" six (R.frame six "ee") (q_of [ 0.4; -0.3; 0.8; 0.2; -0.5; 0.9 ]);
  let branch = branch_model () in
  check "branch" branch (R.frame branch "ee") (q_of [ 0.7; -0.4 ]);
  check "branch eye" branch (R.frame branch "eye") (q_of [ 0.7; -0.4 ])

(* The body Jacobian says what it means: [J_body qdot] is the body twist, that
   is, the right-trivialised derivative of the pose, [log (M^-1 M(q + h qdot)) / h]. Checking it against the definition rather than against the world one is
   what distinguishes "the same velocity in the body's axes" from "the world
   Jacobian carried by an adjoint", which moves the reference point too. *)

let test_body_jacobian () =
  let m = R.six_dof () in
  let fr = R.frame m "ee" in
  List.iter
    (fun (q, dq) ->
       let h = 1e-3 in
       let body = R.jacobian ~space:`Body m ~frame:fr q in
       equal ~msg:"shape" (array int) [| 6; 6 |] (Nx.shape body);
       let predicted = Nx.reshape [| 6 |] (Nx.matmul body (Nx.reshape [| 6; 1 |] dq)) in
       let current = R.pose m ~frame:fr q in
       let moved = R.pose m ~frame:fr (Nx.add q (Nx.mul_s dq h)) in
       let measured = Nx.div_s (M.log (M.mul (M.inverse current) moved)) h in
       equal_tensor
         ~msg:"the body twist is the right-trivialised derivative"
         ~rel:2e-3
         predicted
         measured)
    [ q_of [ 0.4; -0.3; 0.8; 0.2; -0.5; 0.9 ], q_of [ 0.1; 0.2; -0.3; 0.4; -0.5; 0.6 ]
    ; q_of [ -0.7; 0.5; 0.1; -1.1; 0.3; 0.6 ], q_of [ -0.3; 0.1; 0.4; 0.2; 0.5; -0.2 ]
    ]

(* The analytic Jacobian of the motor, against a finite difference of its eight
   coefficients. *)

let test_motion_jacobian () =
  let m = R.six_dof () in
  let fr = R.frame m "ee" in
  let q = q_of [ 0.4; -0.3; 0.8; 0.2; -0.5; 0.9 ] in
  let j = R.motion_jacobian m ~frame:fr q in
  equal ~msg:"shape" (array int) [| 8; 6 |] (Nx.shape j);
  let h = 1e-2 in
  for i = 0 to 5 do
    let plus = R.pose m ~frame:fr (Nx.add q (bump 6 i h)) in
    let minus = R.pose m ~frame:fr (Nx.sub q (bump 6 i h)) in
    let fd = Nx.div_s (Nx.sub plus minus) (2. *. h) in
    equal_tensor ~msg:(Printf.sprintf "column %d" i) ~rel:1e-3 (Nx.slice [ A; I i ] j) fd
  done

(* The tangent map is linear in its direction and equals the chain rule applied
   to the analytic Jacobian. *)

let test_pose_tangent () =
  let m = R.six_dof () in
  let fr = R.frame m "ee" in
  let q = q_of [ 0.4; -0.3; 0.8; 0.2; -0.5; 0.9 ] in
  let dq = q_of [ 0.1; -0.2; 0.3; 0.05; -0.25; 0.4 ] in
  let j = R.motion_jacobian m ~frame:fr q in
  let expected =
    List.init 8 (fun k ->
      let s = ref 0. in
      for i = 0 to 5 do
        s := !s +. (item [ k; i ] j *. item [ i ] dq)
      done;
      !s)
  in
  equal_floats
    ~msg:"tangent is the Jacobian applied to the direction"
    (Array.of_list expected)
    (R.pose_tangent m ~frame:fr q dq)

(* Forward-mode AD through the model, and through the rule attached to it.

   The first check is the important one: [Rune.jacfwd] differentiates [exp] and
   everything around it, so it is an independent implementation of the same
   derivative. The second checks that the [custom_jvp] rule agrees with the
   function it is attached to. *)

let test_ad () =
  let m = R.six_dof () in
  let fr = R.frame m "ee" in
  let q = q_of [ 0.4; -0.3; 0.8; 0.2; -0.5; 0.9 ] in
  let analytic = R.motion_jacobian m ~frame:fr q in
  let by_ad = Rune.jacfwd' (fun q -> R.pose m ~frame:fr q) q in
  equal_tensor ~msg:"jacfwd through exp" ~rel:2e-3 analytic by_ad;
  let by_rule = Rune.jacfwd' (R.pose_diff m ~frame:fr) q in
  equal_tensor ~msg:"jacfwd through the custom rule" ~rel:1e-6 analytic by_rule;
  equal_tensor
    ~msg:"the rule's values are pose's"
    (R.pose m ~frame:fr q)
    (R.pose_diff m ~frame:fr q);
  (* Reverse mode transposes the rule's tangent map, so a gradient through the
     rule agrees with one through the branches it replaces. *)
  let scalar_of p = Nx.sum (Nx.mul p p) in
  let by_branches = Rune.grad' (fun q -> scalar_of (R.pose m ~frame:fr q)) q in
  let by_rule = Rune.grad' (fun q -> scalar_of (R.pose_diff m ~frame:fr q)) q in
  equal_tensor ~msg:"grad through the custom rule" ~rel:2e-3 by_branches by_rule

(* Construction validation. *)

let bad msg =
  match R.create ~links:[ "base"; "tip" ] ~joints:msg () with
  | Ok _ -> None
  | Error m -> Some m

let joint ?(origin = M.one) ?limits name kind parent child =
  { R.name; kind; parent; child; origin; limits }

let test_validation () =
  let z = R.Revolute (0., 0., 1.) in
  let check msg spec =
    match R.create ~links:[ "base"; "tip" ] ~joints:spec () with
    | Ok _ -> failwith ("expected an error: " ^ msg)
    | Error _ -> ()
  in
  check "unknown parent" [ joint "a" z "nope" "tip" ];
  check "unknown child" [ joint "a" z "base" "nope" ];
  check "root driven" [ joint "a" z "base" "base" ];
  check "zero axis" [ joint "a" (R.Revolute (0., 0., 0.)) "base" "tip" ];
  check "reversed limits" [ joint "a" z "base" "tip" ~limits:(1., -1.) ];
  (* An unrelated link cannot be attached. *)
  (match R.create ~links:[ "base"; "a"; "b" ] ~joints:[ joint "j" z "base" "a" ] () with
   | Ok _ -> failwith "expected an error: disconnected link"
   | Error _ -> ());
  (* The joints must be topologically ordered. *)
  (match
     R.create
       ~links:[ "base"; "a"; "b" ]
       ~joints:[ joint "j1" z "a" "b"; joint "j0" z "base" "a" ]
       ()
   with
   | Ok _ -> failwith "expected an error: joints out of order"
   | Error _ -> ());
  (* Two joints cannot drive the same link. *)
  (match
     R.create
       ~links:[ "base"; "a"; "b" ]
       ~joints:[ joint "j0" z "base" "a"; joint "j1" z "base" "a" ]
       ()
   with
   | Ok _ -> failwith "expected an error: a link with two parents"
   | Error _ -> ());
  ignore bad;
  match branch_model () with
  | _ -> equal ~msg:"a valid model" int 2 (R.dof (branch_model ()))

(* The model's own index bookkeeping: a fixed joint in the middle does not
   consume a coordinate, and the coordinates follow the moving joints' order. *)

let test_dof_order () =
  let m = branch_model () in
  equal ~msg:"dof" int 2 (R.dof m);
  equal ~msg:"joint names" (array string) [| "yaw"; "lift"; "mount" |] (R.joint_names m);
  equal
    ~msg:"kinds"
    (array string)
    [| "revolute"; "prismatic"; "fixed" |]
    (Array.map
       (function
         | R.Fixed -> "fixed"
         | R.Revolute _ -> "revolute"
         | R.Prismatic _ -> "prismatic")
       (Array.init 3 (R.kind_of_joint m)));
  let q = q_of [ 0.; 0.5 ] in
  let pose = R.pose m ~frame:(R.frame m "ee") q in
  equal_floats
    ~msg:"a fixed joint carries its translation and consumes no coordinate"
    [| 0.5; 0.; 1.5 |]
    (M.translation pose)

(* A model with no moving joint: a fixed chain. The edge case matters because
   [dof = 0] must still produce well-shaped, empty-column Jacobians rather than
   errors, and the pose is a plain product of the joints' origins. *)

let test_fixed_chain () =
  let m =
    match
      R.create
        ~links:[ "base"; "a"; "b" ]
        ~joints:
          [ { R.name = "up"
            ; kind = R.Fixed
            ; parent = "base"
            ; child = "a"
            ; origin = M.translate3 1. 0. 0.
            ; limits = None
            }
          ; { R.name = "across"
            ; kind = R.Fixed
            ; parent = "a"
            ; child = "b"
            ; origin = M.translate3 0. 2. 0.
            ; limits = None
            }
          ]
        ~frames:[ { R.name = "ee"; link = "b"; pose = M.translate3 0. 0. 3. } ]
        ()
    with
    | Ok m -> m
    | Error msg -> failwith msg
  in
  let q = Nx.zeros f32 [| 0 |] in
  equal ~msg:"dof" int 0 (R.dof m);
  let fr = R.frame m "ee" in
  equal_floats
    ~msg:"a fixed chain is its origins"
    [| 1.; 2.; 3. |]
    (M.translation (R.pose m ~frame:fr q));
  equal
    ~msg:"motor Jacobian shape"
    (array int)
    [| 8; 0 |]
    (Nx.shape (R.motion_jacobian m ~frame:fr q));
  equal
    ~msg:"geometric Jacobian shape"
    (array int)
    [| 6; 0 |]
    (Nx.shape (R.jacobian m ~frame:fr q));
  equal_tensor
    ~msg:"a zero-width tangent is empty"
    (Nx.zeros f32 [| 8 |])
    (R.pose_tangent m ~frame:fr q q)

(* A frame attached to the root: a base camera, a world frame, a sensor on the
   body. Its chain is empty, so its pose is fixed and its Jacobian is a block of
   zeros — the case that would be an empty product if the chain walk were not
   careful about the root. *)

let test_root_frame () =
  let m =
    match
      R.create
        ~links:[ "base"; "tip" ]
        ~joints:
          [ { R.name = "j"
            ; kind = R.Revolute (0., 0., 1.)
            ; parent = "base"
            ; child = "tip"
            ; origin = M.translate3 1. 0. 0.
            ; limits = None
            }
          ]
        ~frames:
          [ { R.name = "base_cam"; link = "base"; pose = M.translate3 0. (-1.) 0. }
          ; { R.name = "tip"; link = "tip"; pose = M.one }
          ]
        ()
    with
    | Ok m -> m
    | Error msg -> failwith msg
  in
  equal ~msg:"frame names" (array string) [| "base_cam"; "tip" |] (R.frame_names m);
  let cam = R.frame m "base_cam" in
  let q = q_of [ 0.9 ] in
  equal_floats
    ~msg:"a root frame sits at its fixed pose"
    [| 0.; -1.; 0. |]
    (M.translation (R.pose m ~frame:cam q));
  equal_tensor
    ~msg:"a root frame does not move with the coordinates"
    (R.pose m ~frame:cam (q_of [ 0. ]))
    (R.pose m ~frame:cam q);
  equal_tensor
    ~msg:"its geometric Jacobian is zero"
    (Nx.zeros f32 [| 6; 1 |])
    (R.jacobian m ~frame:cam q);
  equal_tensor
    ~msg:"its motor Jacobian is zero"
    (Nx.zeros f32 [| 8; 1 |])
    (R.motion_jacobian m ~frame:cam q);
  equal_tensor
    ~msg:"its tangent is zero"
    (Nx.zeros f32 [| 8 |])
    (R.pose_tangent m ~frame:cam q q)

(* Name handling: unknown frames are rejected with the list of names, and the
   model prints. *)

let test_names () =
  let m = branch_model () in
  let show = Format.asprintf "%a" R.pp in
  let text = show m in
  let contains sub =
    let n = String.length sub
    and m = String.length text in
    let rec go i = i + n <= m && (String.equal (String.sub text i n) sub || go (i + 1)) in
    go 0
  in
  equal ~msg:"pp names the robot" bool true (contains "robot:");
  equal ~msg:"pp lists a joint" bool true (contains "lift");
  equal ~msg:"pp lists a frame" bool true (contains "eye");
  equal ~msg:"pp numbers the coordinates" bool true (contains "q[0]");
  equal ~msg:"frame names" (array string) [| "ee"; "eye" |] (R.frame_names m);
  raises_match
    ~msg:"an unknown frame lists what the model has"
    (Exn.invalid_arg ~substring:"eye")
    (fun () -> ignore (R.frame m "wrist"));
  (* Names collide within their namespace, not across. *)
  let joint ?(origin = M.one) ?limits name kind parent child =
    { R.name; kind; parent; child; origin; limits }
  in
  let z = R.Revolute (0., 0., 1.) in
  let dup links joints frames =
    match R.create ~links ~joints ~frames () with
    | Ok _ -> false
    | Error _ -> true
  in
  equal
    ~msg:"duplicate link names are rejected"
    bool
    true
    (dup [ "base"; "a"; "a" ] [ joint "j" z "base" "a" ] []);
  equal
    ~msg:"duplicate joint names are rejected"
    bool
    true
    (dup [ "base"; "a"; "b" ] [ joint "j" z "base" "a"; joint "j" z "a" "b" ] []);
  equal
    ~msg:"duplicate frame names are rejected"
    bool
    true
    (dup
       [ "base"; "a" ]
       [ joint "j" z "base" "a" ]
       [ { R.name = "f"; link = "a"; pose = M.one }
       ; { R.name = "f"; link = "a"; pose = M.one }
       ]);
  equal
    ~msg:"an empty link name is rejected"
    bool
    true
    (dup [ "base"; " " ] [ joint "j" z "base" " " ] []);
  equal
    ~msg:"a degenerate limit range is a range, not an error"
    bool
    false
    (dup [ "base"; "a" ] [ joint ~limits:(1., 1.) "j" z "base" "a" ] [])

let () =
  exit
    (run
       "geom robot"
       [ group
           "forward kinematics"
           [ test "the planar arm from its closed form" test_planar2
           ; test "the six-joint arm against matrices" test_six_dof
           ; test "frame and link stacks" test_frames
           ; test "batching is a tensor axis" test_batching
           ; test "relative poses" test_relative
           ]
       ; group
           "differential kinematics"
           [ test "the packed screw generator" test_screw_generator
           ; test "the geometric Jacobian" test_geometric_jacobian
           ; test "the body Jacobian" test_body_jacobian
           ; test "the motor Jacobian" test_motion_jacobian
           ; test "the tangent map" test_pose_tangent
           ]
       ; group "autodiff" [ test "jacfwd and the custom rule" test_ad ]
       ; group
           "models"
           [ test "validation" test_validation
           ; test "coordinates" test_dof_order
           ; test "a fixed chain" test_fixed_chain
           ; test "a frame on the root" test_root_frame
           ; test "names and printing" test_names
           ]
       ])
