(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Geom
module R = Geom_robot.Robot
module M = Motor
module P = Geom_viz.Primitive
module Draw = Geom_viz.Draw
module Camera = Geom_viz.Camera
module C = Hugin_gg.Color

type style =
  { link : Draw.style
  ; joint : Draw.style
  ; base : Draw.style
  ; tool : Draw.style
  ; target : Draw.style
  ; trail : Draw.style
  }

let default_style =
  { link =
      { Draw.default with fill = None; stroke = Some (C.v 0.22 0.27 0.33); width = 12. }
  ; joint =
      { Draw.default with
        fill = Some (C.v ~alpha:0.95 0.95 0.48 0.15)
      ; stroke = Some (C.v 0.45 0.25 0.05)
      ; width = 1.2
      }
  ; base =
      { Draw.default with
        fill = Some (C.v ~alpha:0.9 0.30 0.33 0.38)
      ; stroke = Some (C.v 0.15 0.17 0.20)
      ; width = 1.
      }
  ; tool = Draw.default
  ; target =
      { Draw.default with
        fill = None
      ; stroke = Some (C.v ~alpha:0.85 0.10 0.45 0.65)
      ; width = 2.
      ; dash = [ 5.; 4. ]
      }
  ; trail =
      { Draw.default with
        fill = None
      ; stroke = Some (C.v ~alpha:0.55 0.10 0.45 0.65)
      ; width = 2.5
      }
  }

type t =
  { model : R.t
  ; style : style
  ; joint_radius : float
  ; base_radius : float
  }

let make ?(style = default_style) ?(joint_radius = 0.07) ?(base_radius = 0.22) model =
  { model; style; joint_radius; base_radius }

let model t = t.model

(* {2 Numbers out of the model}

   One batched [link_motors] call per configuration; the renderer wants host
   floats, so this is the only place the tensor boundary is crossed. *)

let origins t q =
  let lm = R.link_motors t.model q in
  Array.init (R.link_count t.model) (fun i ->
    P.p3_of_tensor (M.translation (Nx.slice [ I i ] lm)))

let origin_of_frame t ~frame q = P.p3_of_tensor (M.translation (R.pose t.model ~frame q))

(* {2 Primitives} *)

let links t q =
  let o = origins t q in
  List.init (max 0 (Array.length o - 1)) (fun i -> P.Segment (o.(i), o.(i + 1)))

let joints t q =
  let radius i = if i = 0 then 1.4 *. t.joint_radius else t.joint_radius in
  Array.to_list
    (Array.mapi (fun i center -> P.Sphere { center; radius = radius i }) (origins t q))

let base t =
  P.Circle { center = P.p3 0. 0. 0.05; normal = P.p3 0. 0. 1.; radius = t.base_radius }

let tool t ~frame ?(len = 0.16) q =
  P.transform
    (R.pose t.model ~frame q)
    (P.Frame
       { origin = P.p3 0. 0. 0.
       ; x = P.p3 len 0. 0.
       ; y = P.p3 0. len 0.
       ; z = P.p3 0. 0. len
       })

let trail t ~frame qs =
  P.Polyline (Array.to_list (Array.map (fun q -> origin_of_frame t ~frame q) qs))

let target ?(radius = 0.13) pose =
  [ P.transform pose (P.Circle { center = P.p3 0. 0. 0.; normal = P.p3 0. 0. 1.; radius })
  ; P.transform pose (P.Segment (P.p3 0. 0. 0., P.p3 0. 0. (1.7 *. radius)))
  ; P.Sphere { center = P.p3_of_tensor (M.translation pose); radius = 0.2 *. radius }
  ]

let stage ?(radius = 2.4) _ =
  Draw.static
    ~layer:0
    ~style:
      { Draw.default with
        fill = Some (C.v ~alpha:0.10 0.45 0.50 0.55)
      ; stroke = Some (C.v ~alpha:0.25 0.25 0.35 0.40)
      ; width = 1.
      }
    (P.Plane
       { normal = P.p3 0. 0. 1.
       ; offset = 0.
       ; patch =
           [ P.p3 (-.radius) (-.radius) 0.
           ; P.p3 radius (-.radius) 0.
           ; P.p3 radius radius 0.
           ; P.p3 (-.radius) radius 0.
           ]
       })

(* {2 Entities} *)

let entities ?(layer = 1) t q =
  Draw.static ~layer ~style:t.style.base (base t)
  :: (List.map (Draw.static ~layer:(layer + 1) ~style:t.style.link) (links t q)
      @ List.map (Draw.static ~layer:(layer + 2) ~style:t.style.joint) (joints t q))

(* The animation recomputes the model at every time, once per picture: memoising
   on the time keeps that one [link_motors] call per frame rather than one per
   entity, since [Draw] asks every motion for the same time. *)
let memo f =
  let last = ref (Float.nan, f 0.) in
  fun time ->
    let previous, value = !last in
    if Float.equal previous time
    then value
    else (
      let value = f time in
      last := time, value;
      value)

let animated ?(layer = 1) t ~q ?frame ?trail:trail_qs ?target:target_pose ?(fps = 24) () =
  let positions = memo (fun time -> origins t (q time)) in
  let links =
    List.init
      (max 0 (R.link_count t.model - 1))
      (fun i ->
         Draw.moving_with
           ~layer:(layer + 1)
           ~style:t.style.link
           ~motion:(fun time _ ->
             let o = positions time in
             P.Segment (o.(i), o.(i + 1)))
           (P.Segment (P.p3 0. 0. 0., P.p3 0. 0. 1.)))
  in
  let joints =
    List.init (R.link_count t.model) (fun i ->
      let radius = if i = 0 then 1.4 *. t.joint_radius else t.joint_radius in
      Draw.moving_with
        ~layer:(layer + 2)
        ~style:t.style.joint
        ~motion:(fun time _ -> P.Sphere { center = (positions time).(i); radius })
        (P.Sphere { center = P.p3 0. 0. 0.; radius }))
  in
  let base = [ Draw.static ~layer ~style:t.style.base (base t) ] in
  let tool =
    match frame with
    | None -> []
    | Some frame ->
      [ Draw.moving_with
          ~layer:(layer + 3)
          ~style:t.style.tool
          ~motion:(fun time _ -> tool t ~frame (q time))
          (tool t ~frame (q 0.))
      ]
  in
  let trail =
    match trail_qs, frame with
    | Some qs, Some frame ->
      let last = Array.length qs - 1 in
      [ Draw.moving_with
          ~layer:(layer + 2)
          ~style:t.style.trail
          ~motion:(fun time _ ->
            let k = max 0 (min last (int_of_float (time *. float_of_int fps))) in
            P.Polyline (List.init (k + 1) (fun i -> origin_of_frame t ~frame qs.(i))))
          (trail t ~frame qs)
      ]
    | None, _ | Some _, None -> []
  in
  let target =
    match target_pose with
    | None -> []
    | Some pose ->
      List.map (Draw.static ~layer:(layer + 3) ~style:t.style.target) (target pose)
  in
  base @ links @ joints @ tool @ trail @ target

(* {2 Cameras and paths} *)

let camera
      ?(eye = P.p3 3. (-3.4) 2.1)
      ?(target = P.p3 0. 0. 1.1)
      ?(up = P.p3 0. 0. 1.)
      ?(size = 800., 620.)
      ()
  =
  let width, height = size in
  Camera.create ~eye ~target ~up ~fov_y:(Float.pi /. 4.) ~width ~height ()

let sample ?(fps = 24) qs time =
  let last = Array.length qs - 1 in
  qs.(max 0 (min last (int_of_float (time *. float_of_int fps))))

let linear_path from to_ ~n k =
  let alpha = float_of_int k /. float_of_int (max 1 (n - 1)) in
  Nx.add from (Nx.mul_s (Nx.sub to_ from) alpha)
